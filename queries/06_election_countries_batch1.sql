-- 06_election_countries_batch1.sql
-- Binned scatter data for election countries (AL through NZ)
-- Countries from Cárdenas-Sánchez et al. (2022) Table 1
-- NOTE: BigQuery download limit may truncate results;
-- split into two batches if needed

SELECT
  ActionGeo_CountryCode AS country,
  Year,
  ROUND(GoldsteinScale * 2) / 2 AS goldstein_bin,
  ROUND(AvgTone * 2) / 2 AS tone_bin,
  COUNT(*) AS n
FROM `gdelt-bq.full.events`
WHERE Year >= 2017
  AND GoldsteinScale IS NOT NULL
  AND AvgTone IS NOT NULL
  AND ActionGeo_CountryCode IN (
    'AL','AM','AR','BL','BO','CI','DJ','DR','EC','FR',
    'GH','GM','GV','HO','IS','IV','IZ','JA','JM','MD',
    'MY','NL','NZ'
  )
GROUP BY country, Year, goldstein_bin, tone_bin
ORDER BY country, Year, goldstein_bin, tone_bin;
