# The Conflict Framing Elasticity

**A Three-Layer Diagnostic of Media Coverage and Democratic Erosion in 40 Countries, 2017–2026**

Alejandro Feged-Rivadeneira  
Tecnológico de Monterrey  
[ORCID: 0000-0001-8649-4213](https://orcid.org/0000-0001-8649-4213)

## Overview

This repository contains the data extraction queries, analysis scripts, and figure-generation code for the paper *"The Conflict Framing Elasticity: A Three-Layer Diagnostic of Media Coverage and Democratic Erosion in 40 Countries, 2017–2026."*

The paper proposes three complementary measures of media coverage health derived from the GDELT Event Database (872 million news events, 1979–2026):

1. **Layer 1 — Conflict Framing Elasticity (β):** The regression slope of tone on the Goldstein conflict–cooperation scale, measuring how reactively media frame conflict events.
2. **Layer 2 — Regime Position:** The position of each country in standardized tone×Goldstein space, characterized via a 1-skeleton (Rips) graph.
3. **Layer 3 — Rate of Change (Δβ):** The linear trend in β across 2017–2026, identifying countries whose media environments are deteriorating fastest.

## Repository Structure

```
├── queries/
│   ├── 01_global_scatter.sql          # Global tone×Goldstein binned scatter (N=872M)
│   ├── 02_country_annual.sql          # Country-level annual statistics (9 countries)
│   ├── 03_topic_annual.sql            # CAMEO topic-level annual statistics
│   ├── 04_global_annual.sql           # Global annual summary statistics
│   ├── 05_country_scatter.sql         # Country-level binned scatter (6 countries)
│   ├── 06_election_countries_batch1.sql  # Election countries (AL–NZ)
│   └── 07_election_countries_batch2.sql  # Election countries (PE–ZA)
├── scripts/
│   ├── 01_artifact_analysis.py        # Synchronicity diagnostic & z-score correction
│   ├── 02_elasticity_estimation.py    # β computation & floor-effect test
│   ├── 03_tda_analysis.py             # Persistent homology
│   ├── 04_skeleton_graph.py           # 1-skeleton in tone×Goldstein space
│   ├── 05_dynamic_analysis.py         # Δβ trajectories & canary identification
│   ├── 06_democracy_integration.py    # EIU Democracy Index correlations
│   └── 07_publication_figures.py      # All main and supplementary figures
├── figures/                           # Generated publication figures
├── README.md
└── LICENSE
```

## Data Access

All GDELT data are publicly available via Google BigQuery:

- **Project:** `gdelt-bq`
- **Table:** `full.events`
- **Access:** Free tier available at [console.cloud.google.com](https://console.cloud.google.com)

Run the SQL queries in `queries/` to reproduce the data extracts. Each query is designed to run within BigQuery's free tier limits when executed individually.

## Requirements

```
python >= 3.9
pandas
numpy
scipy
matplotlib
scikit-learn
ripser
persim
networkx
adjustText
```

Install:
```bash
pip install pandas numpy scipy matplotlib scikit-learn ripser persim networkx adjustText
```

## Reproducing the Analysis

1. Run the SQL queries in `queries/` via Google BigQuery and download the CSV outputs
2. Place the CSVs in a `data/` directory
3. Run the scripts in `scripts/` sequentially (01 through 07)
4. Publication figures will be saved to `figures/`

## Key Findings

- **GDELT Discontinuity:** Mean tone collapses 6.47 points at the v1→v2 transition (2015), confirmed as artifact by synchronicity across 9 countries and 17 event types
- **Rising β:** Global conflict framing elasticity increases +0.004/yr (p = .022) within the consistent post-2015 regime
- **Canary Cases:** Honduras, Uganda, Ecuador, Malaysia, Tanzania, Bolivia, Turkey, and Mexico show the fastest-rising Δβ — all hybrid regimes or flawed democracies
- **Floor Effect:** Authoritarian regimes (Venezuela, Belarus, Djibouti) show stable or declining β, consistent with media environments already at maximum negativity

## Citation

```
Feged-Rivadeneira, A. (2026). The conflict framing elasticity: A three-layer 
diagnostic of media coverage and democratic erosion in 40 countries, 2017–2026. 
[Manuscript submitted for publication]. Tecnológico de Monterrey.
```

## License

MIT License. See [LICENSE](LICENSE).
