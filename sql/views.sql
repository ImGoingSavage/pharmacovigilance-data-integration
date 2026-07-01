-- views.sql — Reusable analytical views over the canonical model.

DROP VIEW IF EXISTS v_reaction_summary;
CREATE VIEW v_reaction_summary AS
SELECT adverse_reaction,
       COUNT(*)                AS n_pairs,
       COUNT(DISTINCT drug_id) AS n_drugs
FROM drug_reactions
GROUP BY adverse_reaction;

DROP VIEW IF EXISTS v_drug_summary;
CREATE VIEW v_drug_summary AS
SELECT drug_id,
       name,
       COUNT(DISTINCT adverse_reaction) AS distinct_reactions,
       MAX(source)                      AS source
FROM drug_reactions
GROUP BY drug_id, name;

-- Example use:
--   SELECT * FROM v_reaction_summary ORDER BY n_drugs DESC LIMIT 10;
--   SELECT * FROM v_drug_summary     ORDER BY distinct_reactions DESC LIMIT 10;
