-- 03_topic_annual.sql
-- CAMEO topic-level annual statistics (17 event root codes)
-- Used for: topic-level elasticity analysis, synchronicity diagnostic

SELECT
  EventRootCode AS topic_code,
  Year,
  AVG(AvgTone) AS mean_tone,
  STDDEV(AvgTone) AS sd_tone,
  AVG(GoldsteinScale) AS mean_goldstein,
  CORR(AvgTone, GoldsteinScale) AS r_tone_gold,
  COUNT(*) AS n_events
FROM `gdelt-bq.full.events`
WHERE GoldsteinScale IS NOT NULL
  AND AvgTone IS NOT NULL
  AND EventRootCode IS NOT NULL
GROUP BY topic_code, Year
ORDER BY topic_code, Year;
