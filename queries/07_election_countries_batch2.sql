-- 07_election_countries_batch2.sql
-- Binned scatter data for remaining election countries (PE through ZA)

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
    'PE','PL','PO','SP','TD','TG','TH','TU','TZ','UG','UV','ZA'
  )
GROUP BY country, Year, goldstein_bin, tone_bin
ORDER BY country, Year, goldstein_bin, tone_bin;
