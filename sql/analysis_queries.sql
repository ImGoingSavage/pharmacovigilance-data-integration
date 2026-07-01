-- analysis_queries.sql — Analytical SQL over the canonical pharmacovigilance model.
-- Demonstrates: aggregations, GROUP BY/HAVING, CTEs, window functions, self-joins,
-- duplicate detection, drug–reaction frequency, and molecule-level safety exploration.
-- Tested on SQLite (152,759 rows) and written to be PostgreSQL-compatible.

-- 1) Top 15 most frequently reported adverse reactions -----------------------
SELECT adverse_reaction,
       COUNT(*)                AS n_pairs,
       COUNT(DISTINCT drug_id) AS n_drugs
FROM drug_reactions
GROUP BY adverse_reaction
ORDER BY n_pairs DESC
LIMIT 15;

-- 2) Drugs with the broadest adverse-reaction profile ------------------------
SELECT name,
       COUNT(DISTINCT adverse_reaction) AS distinct_reactions
FROM drug_reactions
GROUP BY drug_id, name
ORDER BY distinct_reactions DESC
LIMIT 15;

-- 3) Duplicate detection: same drug + same reaction more than once -----------
SELECT drug_id, adverse_reaction, COUNT(*) AS occurrences
FROM drug_reactions
GROUP BY drug_id, adverse_reaction
HAVING COUNT(*) > 1
ORDER BY occurrences DESC
LIMIT 20;

-- 4) Reaction frequency with share of total (window function) ----------------
SELECT adverse_reaction,
       COUNT(*)                                                   AS n_pairs,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)         AS pct_of_all_pairs
FROM drug_reactions
GROUP BY adverse_reaction
ORDER BY n_pairs DESC
LIMIT 15;

-- 5) Drugs above the dataset-average reaction count (CTE) --------------------
WITH per_drug AS (
    SELECT drug_id, name, COUNT(DISTINCT adverse_reaction) AS reac
    FROM drug_reactions
    GROUP BY drug_id, name
),
stats AS (SELECT AVG(reac) AS avg_reac FROM per_drug)
SELECT p.name, p.reac, ROUND((SELECT avg_reac FROM stats), 1) AS dataset_avg
FROM per_drug p
WHERE p.reac > (SELECT avg_reac FROM stats)
ORDER BY p.reac DESC
LIMIT 20;

-- 6) Data-quality KPI: severity completeness by source ----------------------
SELECT source,
       COUNT(*)                                                          AS n_pairs,
       SUM(CASE WHEN severity <> 'unknown' THEN 1 ELSE 0 END)            AS with_severity,
       ROUND(100.0 * SUM(CASE WHEN severity <> 'unknown' THEN 1 ELSE 0 END)
             / COUNT(*), 1)                                              AS pct_known
FROM drug_reactions
GROUP BY source;

-- 7) Molecule-level safety exploration: drugs flagged for a hepatic reaction -
SELECT DISTINCT name, smiles
FROM drug_reactions
WHERE LOWER(adverse_reaction) LIKE '%hepat%'
   OR LOWER(adverse_reaction) LIKE '%liver%'
ORDER BY name
LIMIT 25;

-- 8) Per-source top reactions by drug coverage (RANK window, partitioned) ----
WITH ranked AS (
    SELECT source,
           adverse_reaction,
           COUNT(DISTINCT drug_id)                                             AS n_drugs,
           RANK() OVER (PARTITION BY source ORDER BY COUNT(DISTINCT drug_id) DESC) AS rnk
    FROM drug_reactions
    GROUP BY source, adverse_reaction
)
SELECT source, adverse_reaction, n_drugs
FROM ranked
WHERE rnk <= 3
ORDER BY source, n_drugs DESC;
