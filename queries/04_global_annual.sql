-- 04_global_annual.sql
-- Global annual summary statistics
-- Used for: z-score normalization denominators, global β trend

SELECT
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
GROUP BY Year
ORDER BY Year;
