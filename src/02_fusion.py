"""
fusion.py — Construcción del modelo canónico e_drugDB.csv
===========================================================
Framework TDQM: Fase MEDIR + MEJORAR

Etapas:
  1. Carga de las tres fuentes crudas
  2. Perfilado de heterogeneidad (completitud y naming)
  3. Record linkage: detección de aliases drug_name ↔ iupac_name
  4. Fusión con resolución de conflictos
  5. Deduplicación y limpieza final
  6. Guardado del modelo canónico en datos/clean/e_drugDB.csv
"""

import os
import re
import pandas as pd
import recordlinkage
from thefuzz import fuzz

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW   = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
os.makedirs(CLEAN, exist_ok=True)

print("=" * 70)
print("FUSION.PY — Construcción del modelo canónico e_drugDB")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE FUENTES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Cargando fuentes crudas...")

se   = pd.read_csv(os.path.join(RAW, "sider_se_pt.csv"))
drug = pd.read_csv(os.path.join(RAW, "sider_drug_names.csv"))
smi  = pd.read_csv(os.path.join(RAW, "pubchem_smiles.csv"))

print(f"    sider_se_pt      : {len(se):>8,} filas  | {se['pubchem_cid'].nunique():,} fármacos | {se['side_effect_name'].nunique():,} reacciones")
print(f"    sider_drug_names : {len(drug):>8,} filas  | {drug['stitch_flat'].nunique():,} fármacos únicos")
print(f"    pubchem_smiles   : {len(smi):>8,} filas  | {smi['pubchem_cid'].nunique():,} CIDs únicos")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MEDICIÓN DE HETEROGENEIDAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Midiendo heterogeneidad entre fuentes...")

# 2a. Completitud: CIDs en SE sin nombre SIDER
cids_se   = set(se["pubchem_cid"].unique())
cids_drug = set(
    se.merge(drug, on="stitch_flat")["pubchem_cid"].unique()
)
sin_nombre_sider = cids_se - cids_drug
print(f"    CIDs en SE sin nombre SIDER  : {len(sin_nombre_sider):,}  "
      f"({100*len(sin_nombre_sider)/len(cids_se):.1f}%)")

# 2b. Completitud: CIDs en SE sin SMILES
cids_smi  = set(smi["pubchem_cid"].unique())
sin_smiles = cids_se - cids_smi
print(f"    CIDs en SE sin SMILES        : {len(sin_smiles):,}  "
      f"({100*len(sin_smiles)/len(cids_se):.1f}%)")

# 2c. Muestra de heterogeneidad naming: SIDER name vs IUPAC
joined_sample = (
    se[["pubchem_cid", "stitch_flat"]].drop_duplicates()
    .merge(drug, on="stitch_flat", how="inner")
    .merge(smi,  on="pubchem_cid",  how="inner")
    [["pubchem_cid", "drug_name", "iupac_name"]]
    .head(10)
)
print("\n    Muestra de heterogeneidad semántica drug_name vs iupac_name:")
print("    " + "-" * 60)
for _, row in joined_sample.iterrows():
    score = fuzz.token_sort_ratio(
        str(row["drug_name"]).lower(),
        str(row["iupac_name"]).lower()
    )
    print(f"    CID {row['pubchem_cid']:>8} | SIDER: {str(row['drug_name'])[:22]:<22} | "
          f"IUPAC: {str(row['iupac_name'])[:30]:<30} | sim={score}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. RECORD LINKAGE — matching entre SIDER names e IUPAC names
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Record linkage: SIDER drug_name ↔ PubChem iupac_name...")

# Construir tabla comparativa
cid_names = (
    se[["pubchem_cid", "stitch_flat"]].drop_duplicates()
    .merge(drug, on="stitch_flat", how="left")
    .merge(smi,  on="pubchem_cid",  how="left")
    [["pubchem_cid", "drug_name", "iupac_name"]]
    .drop_duplicates("pubchem_cid")
    .reset_index(drop=True)
)

# Preparar para recordlinkage (comparación de strings en el mismo CID)
left  = cid_names[["pubchem_cid", "drug_name"]].copy()
right = cid_names[["pubchem_cid", "iupac_name"]].copy()
left.index  = left["pubchem_cid"]
right.index = right["pubchem_cid"]

indexer = recordlinkage.Index()
indexer.add(recordlinkage.index.Full())
candidate_links = indexer.index(left, right)

compare = recordlinkage.Compare()
compare.string("drug_name", "iupac_name", method="levenshtein", label="sim_lev")
compare.string("drug_name", "iupac_name", method="jarowinkler",  label="sim_jw")

features = compare.compute(candidate_links, left, right)
# Quedarse solo con el par propio (mismo CID)
own_pairs = features[
    features.index.get_level_values(0) == features.index.get_level_values(1)
].copy()
own_pairs.index = own_pairs.index.get_level_values(0)
own_pairs.index.name = "pubchem_cid"

# Categorizar acuerdo de nombres
def categorizacion(sim):
    if pd.isna(sim):   return "sin_nombre_sider"
    if sim >= 0.85:    return "alta_similitud"
    if sim >= 0.50:    return "similitud_media"
    return "baja_similitud"

own_pairs["categoria"] = own_pairs["sim_jw"].apply(categorizacion)
dist = own_pairs["categoria"].value_counts()
print("    Distribución de similitud Jaro-Winkler (nombre SIDER vs IUPAC):")
for cat, n in dist.items():
    print(f"      {cat:<22}: {n:>5,}  ({100*n/len(own_pairs):.1f}%)")

# Guardar reporte de linkage
linkage_report = cid_names.join(own_pairs[["sim_lev","sim_jw","categoria"]],
                                on="pubchem_cid", how="left")
linkage_report.to_csv(os.path.join(CLEAN, "linkage_report.csv"), index=False)
print(f"    Reporte guardado: datos/clean/linkage_report.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FUSIÓN Y RESOLUCIÓN DE CONFLICTOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Fusionando fuentes y resolviendo conflictos de nombres...")

# JOIN principal
merged = (
    se
    .merge(drug, on="stitch_flat", how="left")
    .merge(smi,  on="pubchem_cid",  how="left")
)

# Estrategia de fusión de nombres:
#   - Si existe drug_name SIDER → usarlo (es el nombre comercial, más reconocible)
#   - Si no → usar iupac_name de PubChem
#   - Si ninguno → "desconocido"
def resolver_nombre(row):
    sider = str(row["drug_name"]).strip() if pd.notna(row["drug_name"]) else ""
    iupac = str(row["iupac_name"]).strip() if pd.notna(row["iupac_name"]) else ""
    if sider and sider.lower() not in ("nan", "none", ""):
        return sider
    if iupac and iupac.lower() not in ("nan", "none", ""):
        return iupac
    return "desconocido"

merged["name"] = merged.apply(resolver_nombre, axis=1)

# Contar conflictos resueltos
conflictos = merged[
    merged["drug_name"].notna() &
    merged["iupac_name"].notna() &
    (merged["drug_name"].str.lower() != merged["iupac_name"].str.lower())
]
print(f"    Pares con nombre diferente entre fuentes : {conflictos['pubchem_cid'].nunique():,} fármacos")
print(f"    Resolución: preferencia a nombre SIDER (más legible clínicamente)")

# Inferir severity desde meddra_freq si está disponible; si no → "unknown"
# (meddra_freq tiene frecuencias; sin ella marcamos unknown)
merged["severity"] = "unknown"

# source compuesto
merged["source"] = merged.apply(
    lambda r: "SIDER+PubChem" if pd.notna(r["smiles"]) else "SIDER_only",
    axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. DEDUPLICACIÓN Y LIMPIEZA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Deduplicando y construyendo modelo canónico...")

# Crear drug_id único basado en pubchem_cid
merged["drug_id"] = "DRUG_" + merged["pubchem_cid"].astype(str).str.zfill(8)

# Seleccionar y renombrar columnas del modelo canónico
canonical = merged[[
    "drug_id",
    "name",
    "smiles",
    "side_effect_name",
    "umls_meddra",
    "severity",
    "source",
    "pubchem_cid",
    "stitch_flat"
]].rename(columns={
    "side_effect_name": "adverse_reaction",
    "umls_meddra"     : "reaction_type"
})

# Normalizar texto
canonical["name"]             = canonical["name"].str.strip().str.title()
canonical["adverse_reaction"] = canonical["adverse_reaction"].str.strip().str.title()

# Eliminar duplicados exactos (misma droga + misma reacción)
before = len(canonical)
canonical = canonical.drop_duplicates(subset=["drug_id", "adverse_reaction"])
after  = len(canonical)
print(f"    Filas antes de deduplicar : {before:>8,}")
print(f"    Duplicados eliminados     : {before - after:>8,}  ({100*(before-after)/before:.1f}%)")
print(f"    Filas en modelo canónico  : {after:>8,}")

# Eliminar registros sin SMILES (no aportan al análisis molecular)
con_smiles    = canonical[canonical["smiles"].notna()]
sin_smiles_n  = canonical["smiles"].isna().sum()
print(f"    Filas sin SMILES (excluidas): {sin_smiles_n:>6,}")
print(f"    Filas finales con SMILES    : {len(con_smiles):>6,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. GUARDAR MODELO CANÓNICO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Guardando modelo canónico...")

out_all    = os.path.join(CLEAN, "e_drugDB.csv")
out_smiles = os.path.join(CLEAN, "e_drugDB_smiles.csv")

canonical.to_csv(out_all,    index=False)
con_smiles.to_csv(out_smiles, index=False)

size_all    = os.path.getsize(out_all)    / 1024
size_smiles = os.path.getsize(out_smiles) / 1024

print(f"    e_drugDB.csv        : {len(canonical):>7,} filas  {size_all:>8.1f} KB")
print(f"    e_drugDB_smiles.csv : {len(con_smiles):>7,} filas  {size_smiles:>8.1f} KB")

# ─────────────────────────────────────────────────────────────────────────────
# 7. RESUMEN DE CALIDAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN DE CALIDAD DEL MODELO CANÓNICO (e_drugDB_smiles.csv)")
print("=" * 70)

dims = {
    "Completitud (SMILES)":
        f"{100*con_smiles['smiles'].notna().mean():.1f}%",
    "Completitud (nombre)":
        f"{100*(con_smiles['name'] != 'Desconocido').mean():.1f}%",
    "Unicidad (drug+reacción)":
        f"{100*(1 - con_smiles.duplicated(['drug_id','adverse_reaction']).mean()):.1f}%",
    "Fármacos únicos":
        f"{con_smiles['drug_id'].nunique():,}",
    "Reacciones adversas únicas":
        f"{con_smiles['adverse_reaction'].nunique():,}",
    "Pares fármaco-reacción":
        f"{len(con_smiles):,}",
}
for k, v in dims.items():
    print(f"  {k:<35}: {v}")

print("\nListo. Ejecuta perfilado.py para el análisis exploratorio completo.")
