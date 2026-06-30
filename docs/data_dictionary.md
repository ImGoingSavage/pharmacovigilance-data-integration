# Data dictionary — `e_drugDB_clean.csv` (canonical model)

Final integrated model: **152,759** rows (drug–reaction pairs), **1,556** unique drugs, **4,251** unique MedDRA reactions.

| Column | Type | Description | Source / derivation |
|---|---|---|---|
| `drug_id` | string | Stable unified ID, `DRUG_` + zero-padded 8-digit PubChem CID | derived |
| `name` | string | Normalized generic name (Title case). Preference: SIDER name → IUPAC → "Desconocido" | SIDER / PubChem |
| `smiles` | string | Molecular structure (SMILES); 100% RDKit-parseable | PubChem |
| `adverse_reaction` | string | MedDRA Preferred Term of the adverse reaction | SIDER |
| `reaction_type` | string | Reaction classification (UMLS/MedDRA mapping) | SIDER |
| `severity` | string | Severity level; imputed from MedDRA frequency (coverage 0% → 41.4%; rest = "unknown") | SIDER meddra_freq |
| `source` | string | `SIDER+PubChem` if SMILES present, else `SIDER_only` | derived |
| `pubchem_cid` | int | PubChem Compound ID (join key) | SIDER/PubChem |
| `stitch_flat` | string | SIDER STITCH flat identifier | SIDER |
| `n_reac` | int | Number of distinct adverse reactions for the drug | derived |

## Quality dimensions measured (TDQM)

| Dimension | Metric | Result |
|---|---|---|
| Completeness (SMILES) | % rows with valid SMILES | high (analysis subset filtered to SMILES) |
| Consistency (naming) | Jaro-Winkler SIDER↔IUPAC | 53.9% of drugs < 0.50 (high semantic heterogeneity) |
| Accuracy (SMILES) | % RDKit-parseable | 100% |
| Uniqueness | duplicates removed | 10,447 (6.4%) |
| Validity | structure→ADR predictive signal | weak/partial (exploratory) |
