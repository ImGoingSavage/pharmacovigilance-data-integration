-- schema.sql — Canonical pharmacovigilance model (PostgreSQL / SQLite compatible)
-- Table built from the pipeline output: data/clean/e_drugDB_clean.csv
-- (regenerate that file with src/01_download.py ... src/05_analysis.py)

DROP TABLE IF EXISTS drug_reactions;
CREATE TABLE drug_reactions (
    drug_id           TEXT,        -- unified id: 'DRUG_' + zero-padded PubChem CID
    name              TEXT,        -- normalized generic drug name
    smiles            TEXT,        -- molecular structure (SMILES)
    adverse_reaction  TEXT,        -- MedDRA Preferred Term
    reaction_type     TEXT,        -- UMLS / MedDRA mapping
    severity          TEXT,        -- imputed severity or 'unknown'
    source            TEXT,        -- 'SIDER+PubChem' | 'SIDER_only'
    pubchem_cid       INTEGER,     -- PubChem Compound ID (join key)
    stitch_flat       TEXT,        -- SIDER STITCH flat identifier
    n_reac            INTEGER      -- number of distinct reactions for the drug
);

-- Load — PostgreSQL:
--   \copy drug_reactions FROM 'data/clean/e_drugDB_clean.csv' WITH (FORMAT csv, HEADER true);
-- Load — SQLite:
--   .mode csv
--   .import --skip 1 data/clean/e_drugDB_clean.csv drug_reactions

CREATE INDEX IF NOT EXISTS idx_dr_drug     ON drug_reactions (drug_id);
CREATE INDEX IF NOT EXISTS idx_dr_reaction ON drug_reactions (adverse_reaction);
CREATE INDEX IF NOT EXISTS idx_dr_cid      ON drug_reactions (pubchem_cid);
