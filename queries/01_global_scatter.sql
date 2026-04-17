-- 01_global_scatter.sql
-- Global tone×Goldstein binned scatter data
-- Returns ~80K rows covering 1979–2026
-- Used for: Figure 1 (centroid trajectory), global β estimation, TDA

SELECT
  Year,
  ROUND(GoldsteinScale * 2) / 2 AS goldstein_bin,
  ROUND(AvgTone * 2) / 2 AS tone_bin,
  COUNT(*) AS n
FROM `gdelt-bq.full.events`
WHERE GoldsteinScale IS NOT NULL
  AND AvgTone IS NOT NULL
GROUP BY Year, goldstein_bin, tone_bin
ORDER BY Year, goldstein_bin, tone_bin;
