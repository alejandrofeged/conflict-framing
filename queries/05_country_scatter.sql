-- 05_country_scatter.sql
-- Country-level binned scatter data for core countries
-- Used for: country-level β estimation, floor-effect test

SELECT
  ActionGeo_CountryCode AS country,
  Year,
  ROUND(GoldsteinScale * 2) / 2 AS goldstein_bin,
  ROUND(AvgTone * 2) / 2 AS tone_bin,
  COUNT(*) AS n
FROM `gdelt-bq.full.events`
WHERE GoldsteinScale IS NOT NULL
  AND AvgTone IS NOT NULL
  AND ActionGeo_CountryCode IN ('VE','CO','US','BR','MX','UK')
GROUP BY country, Year, goldstein_bin, tone_bin
ORDER BY country, Year, goldstein_bin, tone_bin;
