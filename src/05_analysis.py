"""
análisis.py — Consultas y visualizaciones sobre el modelo canónico
===================================================================
Framework TDQM: Fase ANALIZAR + objetivo científico del proyecto

Contiene:
  ── SECCIÓN A: 5 consultas sobre SIDER (fuente individual)
  ── SECCIÓN B: 5 consultas sobre PubChem (fuente individual)
  ── SECCIÓN C: 10 consultas cross-source (integración SIDER + PubChem)
  ── SECCIÓN D: Análisis molecular con RDKit (fingerprints + Tanimoto)

Salida: datos/clean/graficas/  (21 gráficas + 1 CSV de fingerprints)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import DataStructs

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW     = os.path.join(ROOT, "data", "raw")
CLEAN   = os.path.join(ROOT, "data", "clean")
GRAFICAS = os.path.join(CLEAN, "graficas")
os.makedirs(GRAFICAS, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
PAL = sns.color_palette("tab10")

def guardar(fig, nombre):
    fig.savefig(os.path.join(GRAFICAS, nombre), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    {nombre}")

print("=" * 70)
print("ANÁLISIS.PY — TDQM Fase: ANALIZAR")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# CARGA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Carga] e_drugDB_clean.csv ...")
df = pd.read_csv(os.path.join(CLEAN, "e_drugDB_clean.csv"))
print(f"  {len(df):,} filas | {df['drug_id'].nunique():,} fármacos | "
      f"{df['adverse_reaction'].nunique():,} reacciones")

# Cargar fuente SIDER cruda para consultas fuente-individual
sider_raw = pd.read_csv(os.path.join(RAW, "sider_se_pt.csv"))
drug_names = pd.read_csv(os.path.join(RAW, "sider_drug_names.csv"))
smiles_raw = pd.read_csv(os.path.join(RAW, "pubchem_smiles.csv"))

# ─────────────────────────────────────────────────────────────────────────────
# PROPIEDADES MOLECULARES (RDKit) — base para secciones B y C
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Mol] Calculando propiedades moleculares con RDKit...")

drug_smiles = df[["drug_id","name","smiles"]].drop_duplicates("drug_id").copy()
rows = []
for _, r in drug_smiles.iterrows():
    try:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        rows.append({
            "drug_id"  : r["drug_id"],
            "name"     : r["name"],
            "smiles"   : r["smiles"],
            "MW"       : Descriptors.MolWt(mol),
            "logP"     : Descriptors.MolLogP(mol),
            "HBD"      : rdMolDescriptors.CalcNumHBD(mol),
            "HBA"      : rdMolDescriptors.CalcNumHBA(mol),
            "n_rings"  : rdMolDescriptors.CalcNumRings(mol),
            "n_atoms"  : mol.GetNumAtoms(),
            "TPSA"     : Descriptors.TPSA(mol),
        })
    except Exception:
        pass

mol_df = pd.DataFrame(rows)
print(f"  Propiedades calculadas para {len(mol_df):,} fármacos")

# Unir propiedades al df principal
df = df.merge(mol_df[["drug_id","MW","logP","HBD","HBA","n_rings","n_atoms","TPSA"]],
              on="drug_id", how="left")

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN A — 5 consultas sobre SIDER (fuente individual)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN A — Consultas sobre SIDER")
print("─" * 70)

# ── A1. Top 15 reacciones adversas más reportadas (barras horizontales) ──────
print("\n  A1: Top 15 reacciones adversas en SIDER")
top_reac = (sider_raw["side_effect_name"]
            .value_counts().head(15).reset_index()
            .rename(columns={"count":"n"}))
fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(data=top_reac, x="n", y="side_effect_name",
            ax=ax, palette="Blues_r")
ax.set_title("A1 — Top 15 reacciones adversas más reportadas en SIDER 4.1")
ax.set_xlabel("Número de fármacos asociados")
ax.set_ylabel("")
for p in ax.patches:
    ax.text(p.get_width()+3, p.get_y()+p.get_height()/2,
            f"{int(p.get_width()):,}", va="center", fontsize=8)
guardar(fig, "A1_top15_reacciones_sider.png")

# ── A2. Distribución de fármacos por número de reacciones (histograma) ───────
print("  A2: Distribución de reacciones por fármaco")
reac_x_drug = sider_raw.groupby("pubchem_cid")["side_effect_name"].nunique()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(reac_x_drug, bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(reac_x_drug.median(), color="red", linestyle="--",
           label=f"Mediana = {reac_x_drug.median():.0f}")
ax.axvline(reac_x_drug.mean(), color="orange", linestyle="--",
           label=f"Media = {reac_x_drug.mean():.0f}")
ax.set_title("A2 — Distribución de reacciones adversas por fármaco (SIDER)")
ax.set_xlabel("Número de reacciones adversas distintas")
ax.set_ylabel("Número de fármacos")
ax.legend()
guardar(fig, "A2_dist_reacciones_farmaco_sider.png")

# ── A3. Top 15 fármacos con más reacciones en SIDER (barras) ─────────────────
print("  A3: Top 15 fármacos con más reacciones (SIDER)")
top_drugs_sider = (
    sider_raw.merge(drug_names, on="stitch_flat", how="left")
    .groupby("drug_name")["side_effect_name"].nunique()
    .sort_values(ascending=False).head(15)
    .reset_index().rename(columns={"side_effect_name":"n"})
)
fig, ax = plt.subplots(figsize=(11, 6))
sns.barplot(data=top_drugs_sider, x="n", y="drug_name",
            ax=ax, palette="Oranges_r")
ax.set_title("A3 — Top 15 fármacos con más reacciones adversas (SIDER 4.1)")
ax.set_xlabel("Número de reacciones adversas únicas")
ax.set_ylabel("")
guardar(fig, "A3_top15_farmacos_sider.png")

# ── A4. Número de reacciones adversas únicas por CID (scatter ordenado) ──────
print("  A4: Reacciones por CID — scatter de densidad")
vals = reac_x_drug.sort_values(ascending=False).values
fig, ax = plt.subplots(figsize=(11, 5))
ax.scatter(range(len(vals)), vals, s=4, alpha=0.5, color="#55A868")
ax.set_title("A4 — Reacciones adversas únicas por fármaco (SIDER, ordenado)")
ax.set_xlabel("Fármaco (ordenado por # reacciones)")
ax.set_ylabel("# reacciones únicas")
ax.fill_between(range(len(vals)), vals, alpha=0.1, color="#55A868")
guardar(fig, "A4_scatter_reacciones_cid.png")

# ── A5. Matriz de co-ocurrencia de top-10 reacciones (heatmap) ───────────────
print("  A5: Co-ocurrencia de top 10 reacciones (SIDER)")
top10 = sider_raw["side_effect_name"].value_counts().head(10).index
sub   = sider_raw[sider_raw["side_effect_name"].isin(top10)]
pivot = sub.assign(v=1).pivot_table(
    index="pubchem_cid", columns="side_effect_name", values="v",
    aggfunc="max", fill_value=0)
co = pivot.T.dot(pivot)
np.fill_diagonal(co.values, 0)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(co, ax=ax, cmap="YlOrRd", annot=True, fmt="d",
            linewidths=0.5, cbar_kws={"label":"# fármacos en común"})
ax.set_title("A5 — Co-ocurrencia de las 10 reacciones más frecuentes (SIDER)")
plt.xticks(rotation=35, ha="right", fontsize=8)
plt.yticks(fontsize=8)
guardar(fig, "A5_coocurrencia_reacciones.png")

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN B — 5 consultas sobre PubChem (fuente individual)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN B — Consultas sobre PubChem")
print("─" * 70)

# ── B1. Distribución de peso molecular (histograma + KDE) ────────────────────
print("\n  B1: Distribución de peso molecular (PubChem)")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(mol_df["MW"], bins=60, kde=True, ax=ax, color="#4C72B0")
ax.axvline(500, color="red", linestyle="--", label="Regla de Lipinski (MW=500)")
ax.set_title("B1 — Distribución de peso molecular de fármacos (PubChem/RDKit)")
ax.set_xlabel("Peso molecular (Da)")
ax.set_ylabel("Número de fármacos")
ax.legend()
guardar(fig, "B1_dist_peso_molecular.png")

# ── B2. logP vs Peso Molecular — espacio farmacológico (scatter) ─────────────
print("  B2: logP vs MW — reglas de Lipinski")
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(mol_df["MW"], mol_df["logP"], s=15, alpha=0.5,
                c=mol_df["n_rings"], cmap="viridis")
plt.colorbar(sc, ax=ax, label="Número de anillos")
ax.axvline(500, color="red", linestyle="--", alpha=0.6, label="MW≤500")
ax.axhline(5,   color="orange", linestyle="--", alpha=0.6, label="logP≤5")
ax.set_title("B2 — Espacio farmacológico: MW vs logP (reglas de Lipinski)")
ax.set_xlabel("Peso molecular (Da)")
ax.set_ylabel("logP (lipofilicidad)")
ax.legend(fontsize=8)
lipinski_ok = ((mol_df["MW"]<=500) & (mol_df["logP"]<=5)).mean()
ax.text(0.02, 0.96, f"Cumplen Lipinski: {lipinski_ok*100:.0f}%",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
guardar(fig, "B2_lipinski_scatter.png")

# ── B3. Distribución de número de anillos (barras) ───────────────────────────
print("  B3: Distribución de número de anillos")
ring_counts = mol_df["n_rings"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(ring_counts.index, ring_counts.values, color="#DD8452", edgecolor="white")
ax.set_title("B3 — Distribución de número de anillos aromáticos/alicíclicos (PubChem)")
ax.set_xlabel("Número de anillos")
ax.set_ylabel("Número de fármacos")
for x, y in zip(ring_counts.index, ring_counts.values):
    ax.text(x, y+1, str(y), ha="center", fontsize=8)
guardar(fig, "B3_dist_anillos.png")

# ── B4. Violinplot HBD y HBA (donantes y aceptores de enlace hidrógeno) ──────
print("  B4: Distribución de HBD y HBA")
hb_data = pd.DataFrame({
    "Tipo": ["Donantes (HBD)"]*len(mol_df) + ["Aceptores (HBA)"]*len(mol_df),
    "Valor": list(mol_df["HBD"]) + list(mol_df["HBA"])
})
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=hb_data, x="Tipo", y="Valor", ax=ax,
               palette=["#4C72B0","#DD8452"], inner="box")
ax.set_title("B4 — Distribución de donantes (HBD) y aceptores (HBA) de enlace H (PubChem)")
ax.set_ylabel("Número de grupos")
ax.set_xlabel("")
guardar(fig, "B4_violin_hbd_hba.png")

# ── B5. TPSA vs número de átomos (scatter, coloreado por logP) ───────────────
print("  B5: TPSA vs número de átomos")
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(mol_df["n_atoms"], mol_df["TPSA"], s=15, alpha=0.5,
                c=mol_df["logP"], cmap="RdYlGn")
plt.colorbar(sc, ax=ax, label="logP")
ax.axhline(140, color="red", linestyle="--", alpha=0.7, label="TPSA=140 (permeabilidad)")
ax.set_title("B5 — TPSA vs número de átomos (PubChem/RDKit)")
ax.set_xlabel("Número de átomos pesados")
ax.set_ylabel("TPSA (Å²)")
ax.legend(fontsize=8)
guardar(fig, "B5_tpsa_natoms.png")

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN C — 10 consultas cross-source (SIDER + PubChem integradas)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN C — Consultas cross-source (SIDER + PubChem)")
print("─" * 70)

# ── C1. Completitud de SMILES por fármaco vs reacciones reportadas ────────────
print("\n  C1: Completitud SMILES × reacciones (cross-source)")
comp_data = df.groupby("drug_id").agg(
    n_reac=("adverse_reaction","nunique"),
    tiene_smiles=("smiles", lambda x: x.notna().any())
).reset_index()
counts = comp_data.groupby("tiene_smiles")["n_reac"].describe()
print(f"    Con SMILES: {comp_data['tiene_smiles'].sum()} fármacos | "
      f"Sin SMILES: {(~comp_data['tiene_smiles']).sum()}")

fig, ax = plt.subplots(figsize=(9, 5))
colors_cs = {True:"#4C72B0", False:"#DD8452"}
for has, grp in comp_data.groupby("tiene_smiles"):
    ax.hist(grp["n_reac"], bins=40, alpha=0.7,
            color=colors_cs[has],
            label=f"{'Con' if has else 'Sin'} SMILES")
ax.set_title("C1 — Completitud SMILES vs número de reacciones por fármaco")
ax.set_xlabel("Número de reacciones adversas únicas")
ax.set_ylabel("Número de fármacos")
ax.legend()
guardar(fig, "C1_completitud_smiles_vs_reacciones.png")

# ── C2. Peso molecular vs número de reacciones adversas (scatter + reg) ───────
print("  C2: MW vs número de reacciones (cross-source)")
drug_stats = (
    df.groupby("drug_id")
    .agg(n_reac=("adverse_reaction","nunique"),
         MW=("MW","first"), logP=("logP","first"),
         name=("name","first"))
    .dropna(subset=["MW"])
    .reset_index()
)
fig, ax = plt.subplots(figsize=(11, 6))
sc = ax.scatter(drug_stats["MW"], drug_stats["n_reac"], s=20, alpha=0.5,
                c=drug_stats["logP"], cmap="coolwarm")
plt.colorbar(sc, ax=ax, label="logP")
# línea de tendencia
z = np.polyfit(drug_stats["MW"], drug_stats["n_reac"], 1)
xline = np.linspace(drug_stats["MW"].min(), drug_stats["MW"].max(), 200)
ax.plot(xline, np.poly1d(z)(xline), "k--", lw=1.5, label="Tendencia lineal")
ax.set_title("C2 — Peso molecular vs reacciones adversas (SIDER + PubChem)")
ax.set_xlabel("Peso molecular (Da)")
ax.set_ylabel("Número de reacciones adversas únicas")
ax.legend(fontsize=8)
guardar(fig, "C2_MW_vs_reacciones.png")

# ── C3. Distribución de severity por número de reacciones (boxplot) ───────────
print("  C3: Severity × número de reacciones")
ORDER_SEV = ["very_common","common","uncommon","rare","very_rare","postmarketing"]
sub_sev = df[df["severity"].isin(ORDER_SEV)]
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=sub_sev, x="severity", y="n_reac", ax=ax,
            order=ORDER_SEV, palette="RdYlGn_r")
ax.set_title("C3 — Número de reacciones del fármaco por categoría de severity (cross-source)")
ax.set_xlabel("Categoría de frecuencia de la reacción")
ax.set_ylabel("# reacciones adversas del fármaco")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
guardar(fig, "C3_severity_vs_n_reacciones.png")

# ── C4. logP vs número de reacciones (scatter + violines laterales) ───────────
print("  C4: logP vs número de reacciones")
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(drug_stats["logP"], drug_stats["n_reac"], s=18, alpha=0.45,
                c=drug_stats["MW"], cmap="plasma")
plt.colorbar(sc, ax=ax, label="Peso Molecular (Da)")
z2 = np.polyfit(drug_stats["logP"], drug_stats["n_reac"], 1)
xline2 = np.linspace(drug_stats["logP"].min(), drug_stats["logP"].max(), 200)
ax.plot(xline2, np.poly1d(z2)(xline2), "k--", lw=1.5, label="Tendencia")
ax.set_title("C4 — logP vs reacciones adversas (SIDER + PubChem)")
ax.set_xlabel("logP (lipofilicidad)")
ax.set_ylabel("# reacciones adversas únicas")
ax.legend(fontsize=8)
guardar(fig, "C4_logP_vs_reacciones.png")

# ── C5. Top 10 reacciones: distribución de MW de fármacos (violín) ────────────
print("  C5: MW de fármacos por reacción adversa (top 10)")
top10_reac = df["adverse_reaction"].value_counts().head(10).index
sub_top = df[df["adverse_reaction"].isin(top10_reac)].dropna(subset=["MW"])
fig, ax = plt.subplots(figsize=(13, 6))
sns.violinplot(data=sub_top, x="adverse_reaction", y="MW",
               order=top10_reac, ax=ax, palette="Set2", inner="box")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax.set_title("C5 — Peso molecular de fármacos por reacción adversa (top 10) — SIDER + PubChem")
ax.set_xlabel("")
ax.set_ylabel("Peso Molecular (Da)")
guardar(fig, "C5_violin_MW_por_reaccion.png")

# ── C6. Fármacos únicos compartidos entre pares de reacciones (heatmap) ───────
print("  C6: Fármacos compartidos entre top 12 reacciones")
top12 = df["adverse_reaction"].value_counts().head(12).index
sub12 = df[df["adverse_reaction"].isin(top12)][["drug_id","adverse_reaction"]]
piv12 = sub12.assign(v=1).pivot_table(
    index="drug_id", columns="adverse_reaction", values="v",
    aggfunc="max", fill_value=0)
co12 = piv12.T.dot(piv12)
np.fill_diagonal(co12.values, 0)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(co12, ax=ax, cmap="Blues", annot=True, fmt="d",
            linewidths=0.5, cbar_kws={"label":"# fármacos compartidos"})
ax.set_title("C6 — Fármacos compartidos entre top 12 reacciones adversas (cross-source)")
plt.xticks(rotation=35, ha="right", fontsize=7)
plt.yticks(fontsize=7)
guardar(fig, "C6_heatmap_farmacos_compartidos.png")

# ── C7. Número de anillos vs reacciones adversas (barras agrupadas por cuartil)
print("  C7: Anillos moleculares vs reacciones adversas")
drug_stats2 = drug_stats.merge(
    mol_df[["drug_id","n_rings"]], on="drug_id", how="left"
)
drug_stats2["cuartil_reac"] = pd.qcut(
    drug_stats2["n_reac"], q=4,
    labels=["Q1 (bajo)","Q2","Q3","Q4 (alto)"]
)
ring_mean = drug_stats2.groupby("cuartil_reac", observed=True)["n_rings"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(ring_mean.index, ring_mean.values,
              color=sns.color_palette("Blues", 4), edgecolor="white")
ax.set_title("C7 — Promedio de anillos moleculares por cuartil de reacciones adversas")
ax.set_xlabel("Cuartil de reacciones adversas")
ax.set_ylabel("Promedio de anillos")
for bar, val in zip(bars, ring_mean.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{val:.2f}", ha="center", fontsize=9)
guardar(fig, "C7_anillos_vs_cuartil_reacciones.png")

# ── C8. Fármacos sin SMILES — impacto en cobertura del análisis (pie + tabla) ─
print("  C8: Cobertura del análisis molecular (cross-source)")
total_farmacos  = df["drug_id"].nunique()
con_smiles_valido = df.dropna(subset=["MW"])["drug_id"].nunique()
sin_smiles_n    = total_farmacos - con_smiles_valido
total_reac_cs   = df.groupby("drug_id").first().dropna(subset=["MW"])["n_reac"].sum()
total_reac_all  = df["drug_id"].nunique() * df.groupby("drug_id")["adverse_reaction"].nunique().mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.pie([con_smiles_valido, sin_smiles_n],
        labels=["Con propiedades\nmoleculares","Sin propiedades\nmoleculares"],
        autopct="%1.1f%%", colors=["#4C72B0","#DD8452"], startangle=90)
ax1.set_title("C8a — Cobertura de propiedades moleculares por fármaco")

cobertura_labels = ["Fármacos totales","Con SMILES válido","Sin propiedades"]
cobertura_vals   = [total_farmacos, con_smiles_valido, sin_smiles_n]
colors_cob = ["#7fcdbb","#4C72B0","#DD8452"]
ax2.barh(cobertura_labels, cobertura_vals, color=colors_cob)
ax2.set_xlabel("Número de fármacos")
ax2.set_title("C8b — Impacto de la completitud en el análisis molecular")
for i, v in enumerate(cobertura_vals):
    ax2.text(v+2, i, str(v), va="center", fontsize=9)
guardar(fig, "C8_cobertura_analisis_molecular.png")

# ── C9. Distribución de TPSA por reacciones adversas de tipo neurológico ──────
print("  C9: TPSA de fármacos por tipo de reacción (neurológico vs otros)")
neuro_keywords = ["headache","dizziness","tremor","somnolence","insomnia",
                  "seizure","neuropathy","confusion","anxiety","depression"]
df["es_neuro"] = df["adverse_reaction"].str.lower().apply(
    lambda x: any(k in str(x) for k in neuro_keywords)
)
neuro_drugs  = df[df["es_neuro"]]["drug_id"].unique()
drug_neuro   = mol_df[mol_df["drug_id"].isin(neuro_drugs)]["TPSA"]
drug_otros   = mol_df[~mol_df["drug_id"].isin(neuro_drugs)]["TPSA"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(drug_neuro, bins=40, alpha=0.6, color="#4C72B0",
        label=f"Fármacos con reacciones neurológicas (n={len(drug_neuro)})",
        density=True)
ax.hist(drug_otros, bins=40, alpha=0.6, color="#DD8452",
        label=f"Otros fármacos (n={len(drug_otros)})",
        density=True)
ax.axvline(90, color="green", linestyle="--", label="TPSA=90 (penetración CNS)")
ax.set_title("C9 — TPSA en fármacos con reacciones neurológicas vs otros (SIDER + PubChem)")
ax.set_xlabel("TPSA (Å²)")
ax.set_ylabel("Densidad")
ax.legend(fontsize=8)
guardar(fig, "C9_tpsa_neuro_vs_otros.png")

# ── C10. Top 20 reacciones adversas: promedio de MW y logP (tabla + scatter) ──
print("  C10: Propiedades moleculares por reacción adversa (top 20)")
top20_reac = df["adverse_reaction"].value_counts().head(20).index
props_reac = (
    df[df["adverse_reaction"].isin(top20_reac)]
    .dropna(subset=["MW","logP"])
    .groupby("adverse_reaction")
    .agg(n_drugs=("drug_id","nunique"),
         MW_mean=("MW","mean"),
         logP_mean=("logP","mean"))
    .reset_index()
)
fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter(props_reac["MW_mean"], props_reac["logP_mean"],
                s=props_reac["n_drugs"]*3, alpha=0.7,
                c=range(len(props_reac)), cmap="tab20")
for _, row in props_reac.iterrows():
    ax.annotate(row["adverse_reaction"],
                (row["MW_mean"], row["logP_mean"]),
                textcoords="offset points", xytext=(5,3), fontsize=6.5)
ax.set_title("C10 — MW vs logP de fármacos por reacción adversa (tamaño = # fármacos)")
ax.set_xlabel("Peso molecular promedio de fármacos (Da)")
ax.set_ylabel("logP promedio de fármacos")
ax.text(0.02, 0.97, "Cada punto = una reacción adversa\nTamaño = # fármacos que la causan",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
guardar(fig, "C10_MW_logP_por_reaccion.png")

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN D — Análisis molecular RDKit (fingerprints + Tanimoto)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("SECCIÓN D — Análisis molecular: fingerprints + Tanimoto (RDKit)")
print("─" * 70)

# Calcular fingerprints Morgan (radio=2, 1024 bits) para todos los fármacos
print("\n  D1: Calculando Morgan fingerprints...")
drug_fps = []
for _, r in drug_smiles.iterrows():
    try:
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            drug_fps.append({"drug_id": r["drug_id"], "name": r["name"], "fp": fp})
    except Exception:
        pass

print(f"  Fingerprints generados: {len(drug_fps):,}")

# ── D1. Similitud Tanimoto entre fármacos con misma reacción: Hepatotoxicidad ─
print("  D1: Tanimoto similarity — fármacos con hepatotoxicidad")
REAC_TARGET = "Hepatic Failure"
# Buscar la reacción más cercana al target
all_reacs = df["adverse_reaction"].unique()
hepa_reac = [r for r in all_reacs if "hepat" in r.lower() or "liver" in r.lower()]
print(f"    Reacciones hepáticas encontradas: {len(hepa_reac)}")
if not hepa_reac:
    hepa_reac = df["adverse_reaction"].value_counts().head(1).index.tolist()

hepa_drugs_ids = df[df["adverse_reaction"].isin(hepa_reac[:5])]["drug_id"].unique()
hepa_fps = [f for f in drug_fps if f["drug_id"] in hepa_drugs_ids][:20]

if len(hepa_fps) >= 3:
    n = len(hepa_fps)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(
                hepa_fps[i]["fp"], hepa_fps[j]["fp"]
            )
    names_hepa = [f["name"][:18] for f in hepa_fps]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=names_hepa, yticklabels=names_hepa,
                ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                annot=(n <= 12), fmt=".2f", linewidths=0.3,
                cbar_kws={"label":"Similitud Tanimoto"})
    ax.set_title(f"D1 — Similitud Tanimoto entre fármacos con reacciones hepáticas\n"
                 f"(Morgan fp, radio=2, n={n})")
    plt.xticks(rotation=35, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    guardar(fig, "D1_tanimoto_hepatotoxicidad.png")

# ── D2. Distribución de similitud Tanimoto dentro de grupos de reacción ────────
print("  D2: Distribución de Tanimoto por grupo de reacción (top 5)")
top5_reac = df["adverse_reaction"].value_counts().head(5).index

fp_dict = {f["drug_id"]: f["fp"] for f in drug_fps}
tanimoto_results = []

for reac in top5_reac:
    drug_ids = df[df["adverse_reaction"]==reac]["drug_id"].unique()
    fps_reac = [fp_dict[d] for d in drug_ids if d in fp_dict]
    if len(fps_reac) < 2:
        continue
    sample_fps = fps_reac[:30]
    sims = []
    for i in range(len(sample_fps)):
        for j in range(i+1, len(sample_fps)):
            sims.append(DataStructs.TanimotoSimilarity(sample_fps[i], sample_fps[j]))
    for s in sims:
        tanimoto_results.append({"adverse_reaction": reac, "tanimoto": s})

tan_df = pd.DataFrame(tanimoto_results)
fig, ax = plt.subplots(figsize=(11, 5))
sns.violinplot(data=tan_df, x="adverse_reaction", y="tanimoto",
               ax=ax, palette="Set1", inner="box")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
ax.set_title("D2 — Similitud Tanimoto entre fármacos del mismo grupo de reacción (top 5)")
ax.set_xlabel("")
ax.set_ylabel("Similitud Tanimoto (Morgan fp)")
ax.axhline(0.4, color="red", linestyle="--", label="Umbral similaridad (0.4)")
ax.legend(fontsize=8)
guardar(fig, "D2_tanimoto_por_reaccion.png")

# ── D3. PCA de fingerprints coloreado por número de reacciones ────────────────
print("  D3: PCA de fingerprints moleculares")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fp_mat = np.array([list(f["fp"]) for f in drug_fps])
pca_ids = [f["drug_id"] for f in drug_fps]

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(fp_mat)

pca_df = pd.DataFrame({
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "drug_id": pca_ids
}).merge(drug_stats[["drug_id","n_reac"]], on="drug_id", how="left")

fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(pca_df["PC1"], pca_df["PC2"],
                c=pca_df["n_reac"], cmap="plasma",
                s=15, alpha=0.6)
plt.colorbar(sc, ax=ax, label="# reacciones adversas")
ax.set_title(f"D3 — PCA de Morgan fingerprints moleculares\n"
             f"(PC1={pca.explained_variance_ratio_[0]*100:.1f}% | "
             f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
guardar(fig, "D3_pca_fingerprints.png")

# ── D4. Clustering KMeans de fármacos por fingerprint (scatter PCA) ───────────
print("  D4: KMeans clustering de fingerprints")
from sklearn.cluster import KMeans

K = 5
km = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = km.fit_predict(fp_mat)
pca_df["cluster"] = clusters

fig, ax = plt.subplots(figsize=(10, 7))
for k in range(K):
    mask = pca_df["cluster"] == k
    ax.scatter(pca_df.loc[mask,"PC1"], pca_df.loc[mask,"PC2"],
               s=18, alpha=0.6, label=f"Cluster {k+1}")
ax.set_title(f"D4 — KMeans (k={K}) sobre fingerprints moleculares (espacio PCA)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(title="Cluster", fontsize=8)
guardar(fig, "D4_kmeans_fingerprints.png")

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
graficas = sorted(os.listdir(GRAFICAS))
print("\n" + "=" * 70)
print("RESUMEN — GRÁFICAS GENERADAS")
print("=" * 70)
secciones = {"A":"SIDER","B":"PubChem","C":"Cross-source","D":"Molecular RDKit"}
for g in graficas:
    sec = g[0] if g[0] in secciones else "?"
    print(f"  [{sec}] {g}")

print(f"\n  Total: {len(graficas)} gráficas en datos/clean/graficas/")

print("\n  Problemáticas abordadas:")
print("  P1 Completitud  → C1, C8 — fármacos sin SMILES y su impacto")
print("  P2 Consistencia → A5, C6 — heterogeneidad de nombres y co-ocurrencia")
print("  P3 Exactitud    → D1, D2 — validación con similitud molecular Tanimoto")
print("  P4 Unicidad     → A2, A4 — distribución y deduplicación")
print("  P5 Validez      → D3, D4 — clustering predictivo por fingerprint")
print("\nProyecto completado. Todos los archivos listos para el reporte.")
