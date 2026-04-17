-- 02_country_annual.sql
-- Country-level annual tone and Goldstein statistics
-- Nine countries with long GDELT coverage history
-- Used for: z-score correction, country β time series

SELECT
  ActionGeo_CountryCode AS country,
  Year,
  AVG(AvgTone) AS mean_tone,
  STDDEV(AvgTone) AS sd_tone,
  AVG(GoldsteinScale) AS mean_goldstein,
  STDDEV(GoldsteinScale) AS sd_goldstein,
  CORR(AvgTone, GoldsteinScale) AS r_tone_gold,
  COUNT(*) AS n_events
FROM `gdelt-bq.full.events`
WHERE GoldsteinScale IS NOT NULL
  AND AvgTone IS NOT NULL
  AND ActionGeo_CountryCode IN ('VE','CO','US','BR','MX','UK','IN','TU','RS')
GROUP BY country, Year
ORDER BY country, Year;
