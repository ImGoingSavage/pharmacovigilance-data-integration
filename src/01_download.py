"""
Descarga y construcción del dataset integrado para el proyecto de farmacovigilancia.

Fuentes:
  - SIDER 4.1 : medicamento → reacción adversa (ya descargado como .tsv)
  - PubChem   : medicamento → estructura SMILES (via API REST)

Salida:
  - datos/raw/pubchem_smiles.csv   : CID, nombre IUPAC, SMILES
  - datos/raw/sider_se_pt.csv      : STITCH_flat, PubChem_CID, reacción (solo PT)
  - datos/raw/sider_drug_names.csv : STITCH_flat, nombre del medicamento
"""

import pandas as pd
import requests
import time
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)

# ─────────────────────────────────────────────
# 1. SIDER — cargar y filtrar efectos adversos
# ─────────────────────────────────────────────
print("=" * 60)
print("PASO 1: Cargando SIDER meddra_all_se.tsv")

cols_se = ["stitch_flat", "stitch_stereo", "umls_found", "meddra_type",
           "umls_meddra", "side_effect_name"]
se = pd.read_csv(os.path.join(RAW, "meddra_all_se.tsv"),
                 sep="\t", header=None, names=cols_se)

print(f"  Total registros : {len(se):,}")
print(f"  Tipos MedDRA    : {se['meddra_type'].value_counts().to_dict()}")

# Quedarse solo con PT (Preferred Term) — evita duplicados jerárquicos
se_pt = se[se["meddra_type"] == "PT"].copy()
print(f"  Solo PT         : {len(se_pt):,}")

# Extraer PubChem CID del stitch_stereo  (CID0XXXXXXXX → int)
se_pt["pubchem_cid"] = (
    se_pt["stitch_stereo"]
    .str.replace("CID0", "", regex=False)
    .astype(int)
)

se_pt = se_pt[["stitch_flat", "pubchem_cid", "umls_meddra", "side_effect_name"]]
se_pt.to_csv(os.path.join(RAW, "sider_se_pt.csv"), index=False)
print(f"  Guardado: sider_se_pt.csv  ({len(se_pt):,} filas)")

# ─────────────────────────────────────────────
# 2. SIDER — nombres de medicamentos
# ─────────────────────────────────────────────
print("\nPASO 2: Cargando SIDER drug_names.tsv")

names = pd.read_csv(os.path.join(RAW, "drug_names.tsv"),
                    sep="\t", header=None, names=["stitch_flat", "drug_name"])
names.to_csv(os.path.join(RAW, "sider_drug_names.csv"), index=False)

print(f"  Medicamentos únicos : {names['stitch_flat'].nunique():,}")
print(f"  Guardado: sider_drug_names.csv")

# ─────────────────────────────────────────────
# 3. PubChem — descargar SMILES por CID
# ─────────────────────────────────────────────
print("\nPASO 3: Descargando SMILES de PubChem")

unique_cids = sorted(se_pt["pubchem_cid"].unique())
print(f"  CIDs únicos a consultar : {len(unique_cids):,}")

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES,IUPACName/JSON"
BATCH_SIZE  = 200
results     = []
errors      = []

for i in range(0, len(unique_cids), BATCH_SIZE):
    batch = unique_cids[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    total_batches = (len(unique_cids) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"  Lote {batch_num}/{total_batches} — CIDs {batch[0]}...{batch[-1]}", end=" ")

    try:
        resp = requests.post(
            PUBCHEM_URL,
            data={"cid": ",".join(map(str, batch))},
            timeout=30
        )

        if resp.status_code == 200:
            props = resp.json()["PropertyTable"]["Properties"]
            results.extend(props)
            print(f"✓ ({len(props)} registros)")
        else:
            print(f"✗ HTTP {resp.status_code}")
            errors.extend(batch)

    except Exception as e:
        print(f"✗ Error: {e}")
        errors.extend(batch)

    time.sleep(0.3)  # respetar límite de rate de PubChem

# Guardar SMILES
smiles_df = pd.DataFrame(results)
smiles_df.columns = [c.lower() for c in smiles_df.columns]
smiles_df = smiles_df.rename(columns={
    "cid": "pubchem_cid",
    "isomericsmiles": "smiles",
    "iupacname": "iupac_name"
})
smiles_df.to_csv(os.path.join(RAW, "pubchem_smiles.csv"), index=False)

print(f"\n  Registros con SMILES   : {len(smiles_df):,}")
print(f"  CIDs sin respuesta     : {len(errors):,}")
print(f"  Guardado: pubchem_smiles.csv")

if errors:
    pd.DataFrame({"pubchem_cid": errors}).to_csv(
        os.path.join(RAW, "pubchem_errors.csv"), index=False
    )
    print(f"  CIDs fallidos guardados en pubchem_errors.csv")

# ─────────────────────────────────────────────
# 4. Resumen final
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN DE ARCHIVOS GENERADOS")
print("=" * 60)
for fname in ["sider_se_pt.csv", "sider_drug_names.csv", "pubchem_smiles.csv"]:
    path = os.path.join(RAW, fname)
    df   = pd.read_csv(path)
    size = os.path.getsize(path) / 1024
    print(f"  {fname:<30} {len(df):>8,} filas   {size:>8.1f} KB")

print("\nListo. Ahora ejecuta fusion.py para construir el modelo canónico.")
