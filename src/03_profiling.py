"""
perfilado.py — Perfilado de datos antes y después de limpieza
==============================================================
Framework TDQM: Fase MEDIR

Genera:
  - Reporte HTML completo (ydata-profiling) del modelo canónico
  - Métricas por dimensión de calidad (tabla resumida)
  - Gráficas de perfilado guardadas en datos/clean/perfiles/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from ydata_profiling import ProfileReport

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN   = os.path.join(ROOT, "data", "clean")
PERFILES = os.path.join(CLEAN, "perfiles")
os.makedirs(PERFILES, exist_ok=True)

print("=" * 70)
print("PERFILADO.PY — TDQM Fase: MEDIR")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR MODELO CANÓNICO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Cargando e_drugDB_smiles.csv...")
df = pd.read_csv(os.path.join(CLEAN, "e_drugDB_smiles.csv"))
print(f"    {len(df):,} filas | {df['drug_id'].nunique():,} fármacos | "
      f"{df['adverse_reaction'].nunique():,} reacciones")

# ─────────────────────────────────────────────────────────────────────────────
# 2. REPORTE HTML — ydata-profiling
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Generando reporte HTML ydata-profiling (puede tardar ~1 min)...")

# Usamos un sample de 20k filas para el HTML (dataset completo es muy grande)
sample = df.sample(min(20_000, len(df)), random_state=42)

profile = ProfileReport(
    sample,
    title="Farmacovigilancia — Modelo canónico e_drugDB (muestra 20k)",
    explorative=True,
    minimal=False,
)

html_path = os.path.join(PERFILES, "perfil_antes_limpieza.html")
profile.to_file(html_path)
print(f"    Reporte guardado: {html_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MÉTRICAS POR DIMENSIÓN DE CALIDAD (sobre el dataset completo)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Calculando métricas de calidad por dimensión...")

metrics = {}

# Completitud
metrics["Completitud — SMILES"]            = df["smiles"].notna().mean()
metrics["Completitud — nombre"]            = (df["name"].notna() &
                                               (df["name"].str.lower() != "desconocido")).mean()
metrics["Completitud — adverse_reaction"]  = df["adverse_reaction"].notna().mean()
metrics["Completitud — reaction_type"]     = df["reaction_type"].notna().mean()
metrics["Completitud — severity"]          = (df["severity"] != "unknown").mean()

# Unicidad
dup_mask = df.duplicated(subset=["drug_id", "adverse_reaction"])
metrics["Unicidad — pares únicos"]         = 1 - dup_mask.mean()

# Consistencia
metrics["Consistencia — source válido"]    = df["source"].isin(
    ["SIDER+PubChem", "SIDER_only"]).mean()
metrics["Consistencia — CID numérico"]     = pd.to_numeric(
    df["pubchem_cid"], errors="coerce").notna().mean()

# Exactitud (proxy: SMILES válido via longitud > 2)
metrics["Exactitud — SMILES longitud>2"]   = (df["smiles"].str.len() > 2).mean()

# Validez (SMILES parseable con RDKit)
try:
    from rdkit import Chem
    def smiles_valido(s):
        try:
            return Chem.MolFromSmiles(str(s)) is not None
        except Exception:
            return False
    sample_smi = df["smiles"].dropna().sample(min(2000, len(df)), random_state=42)
    pct_validos = sample_smi.apply(smiles_valido).mean()
    metrics["Validez — SMILES parseable (muestra 2k)"] = pct_validos
except ImportError:
    metrics["Validez — SMILES parseable"]  = float("nan")

print("\n    Dimensión de Calidad                       Valor")
print("    " + "-" * 56)
for k, v in metrics.items():
    bar = "█" * int(v * 20) if not np.isnan(v) else "N/A"
    print(f"    {k:<45} {v*100:>5.1f}%  {bar}")

# Guardar métricas CSV
pd.DataFrame(
    {"metrica": list(metrics.keys()), "valor": list(metrics.values())}
).to_csv(os.path.join(PERFILES, "metricas_calidad.csv"), index=False)
print(f"\n    Métricas guardadas: perfiles/metricas_calidad.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4. GRÁFICAS DE PERFILADO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Generando gráficas de perfilado...")

sns.set_theme(style="whitegrid", palette="muted")
fig_size = (10, 5)

# ── 4a. Top 20 reacciones adversas más frecuentes ───────────────────────────
top_reac = (df["adverse_reaction"].value_counts().head(20).reset_index()
            .rename(columns={"count": "n"}))
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=top_reac, x="n", y="adverse_reaction", ax=ax, color="#4C72B0")
ax.set_title("Top 20 reacciones adversas más frecuentes (# fármacos)")
ax.set_xlabel("Número de fármacos")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "01_top20_reacciones.png"), dpi=150)
plt.close(fig)
print("    01_top20_reacciones.png")

# ── 4b. Distribución de reacciones por fármaco (histograma) ─────────────────
reac_per_drug = df.groupby("drug_id")["adverse_reaction"].nunique()
fig, ax = plt.subplots(figsize=fig_size)
ax.hist(reac_per_drug, bins=50, color="#DD8452", edgecolor="white")
ax.set_title("Distribución de reacciones adversas por fármaco")
ax.set_xlabel("Número de reacciones distintas")
ax.set_ylabel("Número de fármacos")
ax.axvline(reac_per_drug.median(), color="red", linestyle="--",
           label=f"Mediana = {reac_per_drug.median():.0f}")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "02_dist_reacciones_por_farmaco.png"), dpi=150)
plt.close(fig)
print("    02_dist_reacciones_por_farmaco.png")

# ── 4c. Completitud por columna (barras) ────────────────────────────────────
completitud = df.isnull().mean().sort_values()
fig, ax = plt.subplots(figsize=fig_size)
colors = ["#2ca02c" if v == 0 else "#d62728" for v in completitud]
bars = ax.barh(completitud.index, (1 - completitud) * 100, color=colors)
ax.set_xlim(0, 105)
ax.set_xlabel("% de valores no nulos")
ax.set_title("Completitud por columna")
for bar, val in zip(bars, (1 - completitud) * 100):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "03_completitud_columnas.png"), dpi=150)
plt.close(fig)
print("    03_completitud_columnas.png")

# ── 4d. Longitud de SMILES (boxplot por top-10 fuentes de reacciones) ────────
df["smiles_len"] = df["smiles"].str.len()
top10_reac = df["adverse_reaction"].value_counts().head(10).index
sub = df[df["adverse_reaction"].isin(top10_reac)]
fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=sub, x="adverse_reaction", y="smiles_len", ax=ax,
            order=top10_reac, color="#55A868")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
ax.set_title("Longitud de SMILES para los 10 efectos adversos más frecuentes")
ax.set_xlabel("")
ax.set_ylabel("Longitud del SMILES")
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "04_smiles_len_boxplot.png"), dpi=150)
plt.close(fig)
print("    04_smiles_len_boxplot.png")

# ── 4e. Dimensiones de calidad — radar/bar de métricas ──────────────────────
met_names  = [k.split(" — ")[0] for k in metrics.keys()]
met_values = [v * 100 if not np.isnan(v) else 0 for v in metrics.values()]

fig, ax = plt.subplots(figsize=(10, 5))
colors_met = ["#4C72B0" if v >= 95 else "#DD8452" if v >= 80 else "#C44E52"
              for v in met_values]
bars = ax.barh([k for k in metrics.keys()], met_values, color=colors_met)
ax.set_xlim(0, 110)
ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Porcentaje (%)")
ax.set_title("Métricas de calidad por dimensión — TDQM Fase: MEDIR")
for bar, val in zip(bars, met_values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "05_metricas_calidad.png"), dpi=150)
plt.close(fig)
print("    05_metricas_calidad.png")

# ── 4f. Top 20 fármacos con más reacciones (barras horizontales) ─────────────
top_drugs = (
    df.groupby("name")["adverse_reaction"].nunique()
    .sort_values(ascending=False)
    .head(20)
    .reset_index()
    .rename(columns={"adverse_reaction": "n_reacciones"})
)
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=top_drugs, x="n_reacciones", y="name", ax=ax, color="#8172B2")
ax.set_title("Top 20 fármacos con más reacciones adversas reportadas")
ax.set_xlabel("Número de reacciones adversas únicas")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "06_top20_farmacos.png"), dpi=150)
plt.close(fig)
print("    06_top20_farmacos.png")

# ── 4g. Distribución de fuente (pie) ─────────────────────────────────────────
src_counts = df["source"].value_counts()
fig, ax = plt.subplots(figsize=(6, 5))
ax.pie(src_counts, labels=src_counts.index, autopct="%1.1f%%",
       colors=["#4C72B0", "#DD8452"], startangle=90)
ax.set_title("Distribución por fuente de datos")
plt.tight_layout()
fig.savefig(os.path.join(PERFILES, "07_fuente_pie.png"), dpi=150)
plt.close(fig)
print("    07_fuente_pie.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TABLA RESUMEN DE PROBLEMAS DE CALIDAD ENCONTRADOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESUMEN DE PROBLEMAS DE CALIDAD ENCONTRADOS (TDQM: ANALIZAR)")
print("=" * 70)

problemas = [
    ("Completitud", "severity = 'unknown' en 100% de registros",
     "Campo no disponible en SIDER 4.1 base; requiere meddra_freq"),
    ("Consistencia", "54% de nombres SIDER ≠ IUPAC (sim JW < 0.50)",
     "Heterogeneidad semántica entre fuentes; resuelto con preferencia SIDER"),
    ("Unicidad",    f"10,447 duplicados eliminados (6.4%)",
     "Misma droga-reacción en múltiples STITCH IDs estéreo"),
    ("Exactitud",   "3 IUPAC names nulos en PubChem",
     "PubChem sin nombre IUPAC para esos CIDs"),
    ("Validez",     "SMILES parseables verificar con limpieza.py",
     "Pendiente validación con RDKit en limpieza.py"),
]

for dim, prob, causa in problemas:
    print(f"\n  [{dim}]")
    print(f"    Problema : {prob}")
    print(f"    Causa    : {causa}")

print("\n\nPrimera fase lista. Continuando con perfilado DESPUÉS de limpieza...")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PERFILADO "DESPUÉS" — ydata-profiling sobre e_drugDB_clean.csv
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PERFILADO DESPUÉS DE LIMPIEZA — TDQM Fase: MEJORAR (verificación)")
print("=" * 70)

clean_path = os.path.join(CLEAN, "e_drugDB_clean.csv")
if not os.path.exists(clean_path):
    print("  AVISO: e_drugDB_clean.csv no encontrado. Ejecuta limpieza.py primero.")
else:
    print("\n[6] Cargando e_drugDB_clean.csv...")
    df_clean = pd.read_csv(clean_path)
    print(f"    {len(df_clean):,} filas | {df_clean['drug_id'].nunique():,} fármacos")

    sample_clean = df_clean.sample(min(20_000, len(df_clean)), random_state=42)

    print("[7] Generando reporte HTML post-limpieza (puede tardar ~1 min)...")
    profile_clean = ProfileReport(
        sample_clean,
        title="Farmacovigilancia — e_drugDB DESPUÉS de limpieza (muestra 20k)",
        explorative=True,
        minimal=False,
    )
    html_clean = os.path.join(PERFILES, "perfil_despues_limpieza.html")
    profile_clean.to_file(html_clean)
    print(f"    Reporte guardado: {html_clean}")

    # ── Métricas "después" para comparación cuantificada ────────────────────
    print("\n[8] Calculando métricas post-limpieza por dimensión...")

    metrics_clean = {}
    metrics_clean["Completitud — SMILES"]           = df_clean["smiles"].notna().mean()
    metrics_clean["Completitud — nombre"]           = (
        df_clean["name"].notna() &
        (df_clean["name"].str.lower() != "desconocido")).mean()
    metrics_clean["Completitud — adverse_reaction"] = df_clean["adverse_reaction"].notna().mean()
    metrics_clean["Completitud — reaction_type"]    = df_clean["reaction_type"].notna().mean()
    metrics_clean["Completitud — severity"]         = (df_clean["severity"] != "unknown").mean()
    metrics_clean["Unicidad — pares únicos"]        = (
        1 - df_clean.duplicated(["drug_id","adverse_reaction"]).mean())
    metrics_clean["Consistencia — source válido"]   = df_clean["source"].isin(
        ["SIDER+PubChem","SIDER_only"]).mean()
    metrics_clean["Consistencia — CID numérico"]    = pd.to_numeric(
        df_clean["pubchem_cid"], errors="coerce").notna().mean()
    metrics_clean["Exactitud — SMILES longitud>2"]  = (
        df_clean["smiles"].str.len() > 2).mean()
    try:
        from rdkit import Chem as _Chem
        _smp = df_clean["smiles"].dropna().sample(min(2000, len(df_clean)), random_state=42)
        metrics_clean["Validez — SMILES parseable (muestra 2k)"] = _smp.apply(
            lambda s: _Chem.MolFromSmiles(str(s)) is not None).mean()
    except Exception:
        metrics_clean["Validez — SMILES parseable (muestra 2k)"] = float("nan")

    # ── Tabla comparativa antes/después ─────────────────────────────────────
    print("\n    Comparación cuantificada ANTES vs DESPUÉS de limpieza:")
    print(f"    {'Métrica':<45} {'ANTES':>8}  {'DESPUÉS':>8}  {'Δ':>8}")
    print("    " + "-" * 75)

    comp_rows = []
    for k in metrics:
        v_antes  = metrics[k]
        v_desp   = metrics_clean.get(k, float("nan"))
        delta    = v_desp - v_antes if not (np.isnan(v_antes) or np.isnan(v_desp)) else float("nan")
        signo    = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "=")
        antes_s  = f"{v_antes*100:>6.1f}%" if not np.isnan(v_antes) else "   N/A"
        desp_s   = f"{v_desp*100:>6.1f}%"  if not np.isnan(v_desp)  else "   N/A"
        delta_s  = f"{signo}{abs(delta)*100:>4.1f}%" if not np.isnan(delta) else "    N/A"
        print(f"    {k:<45} {antes_s}   {desp_s}   {delta_s}")
        comp_rows.append({"metrica": k, "antes": v_antes, "despues": v_desp, "delta": delta})

    pd.DataFrame(comp_rows).to_csv(
        os.path.join(PERFILES, "comparacion_antes_despues.csv"), index=False)
    print(f"\n    Tabla guardada: perfiles/comparacion_antes_despues.csv")

    # ── Gráfica comparativa horizontal ──────────────────────────────────────
    print("\n[9] Gráfica comparativa antes/después (todas las dimensiones)...")

    labels   = [k.split(" — ")[1] if " — " in k else k for k in metrics]
    v_a      = [metrics[k]*100 if not np.isnan(metrics[k]) else 0 for k in metrics]
    v_d      = [metrics_clean.get(k, 0)*100 if not np.isnan(metrics_clean.get(k,0)) else 0
                for k in metrics]

    x   = np.arange(len(labels))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.barh(x + w/2, v_a, w, label="Antes limpieza",  color="#DD8452", alpha=0.85)
    b2 = ax.barh(x - w/2, v_d, w, label="Después limpieza", color="#4C72B0", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 115)
    ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Perfilado ANTES vs DESPUÉS de limpieza — dimensiones de calidad TDQM")
    ax.legend(loc="lower right")
    for bar in list(b1) + list(b2):
        h = bar.get_width()
        if h > 1:
            ax.text(h + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{h:.1f}%", va="center", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(os.path.join(PERFILES, "12_comparacion_antes_despues.png"), dpi=150)
    plt.close(fig)
    print("    12_comparacion_antes_despues.png")

    # ── Gráfica de mejora por dimensión (delta como barras) ─────────────────
    print("[10] Gráfica de mejora neta por dimensión...")
    deltas   = [v_d[i] - v_a[i] for i in range(len(labels))]
    colors_d = ["#2ca02c" if d > 0 else "#d62728" if d < 0 else "#aec7e8"
                for d in deltas]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(labels, deltas, color=colors_d, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mejora en puntos porcentuales (Δ%)")
    ax.set_title("Mejora neta por dimensión de calidad tras limpieza (TDQM: MEJORAR)")
    for bar, val in zip(bars, deltas):
        xpos = bar.get_width() + (0.3 if val >= 0 else -0.3)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}%", va="center", ha=ha, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(PERFILES, "13_mejora_neta_dimensiones.png"), dpi=150)
    plt.close(fig)
    print("    13_mejora_neta_dimensiones.png")

    print("\n" + "=" * 70)
    print("PERFILADO COMPLETO — ARCHIVOS GENERADOS")
    print("=" * 70)
    for f in sorted(os.listdir(PERFILES)):
        ext = "HTML" if f.endswith(".html") else "CSV " if f.endswith(".csv") else "PNG "
        print(f"  [{ext}] {f}")

print("\nListo. Abre los reportes HTML en el navegador para la vista interactiva.")
