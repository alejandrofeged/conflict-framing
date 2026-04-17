import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy import stats

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT = '/home/claude/'

# ── Load data ──
global_scatter = pd.read_csv('/mnt/user-data/uploads/script_job_0a8fef8becb1f4ae12a43b22ed3ef81b_0.csv')
country_annual = pd.read_csv('/mnt/user-data/uploads/script_job_49917d342750619023dc837ce8834d8d_1.csv')
country_scatter = pd.read_csv('/mnt/user-data/uploads/script_job_c4e16328bd50bee1c97535b85c063849_2.csv')
topic_annual = pd.read_csv('/mnt/user-data/uploads/script_job_8c5f2bb731747f1cdc29db5047502d3f_3.csv')
annual_global = pd.read_csv('/mnt/user-data/uploads/script_job_54cdee2baba05b459ada7ddb2d3696b7_2.csv')

COUNTRIES = ['VE', 'CO', 'US', 'BR', 'MX', 'UK']
LABELS = {
    'VE': 'Venezuela', 'CO': 'Colombia', 'US': 'United States',
    'BR': 'Brazil', 'MX': 'Mexico', 'UK': 'United Kingdom',
    'IN': 'India', 'TU': 'Turkey', 'RS': 'Russia'
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Z-SCORE CORRECTION
# Standardize tone within each year to remove level shift
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("STEP 1: Z-SCORE CORRECTION")
print("="*70)

# For country annual data: z_tone = (country_mean_tone - global_mean_tone) / global_sd_tone
# This tells us: how many SDs above/below the global mean is this country?
global_lookup = annual_global.set_index('Year')[['mean_tone', 'sd_tone']].to_dict('index')

ca = country_annual.copy()
ca['global_mean'] = ca['Year'].map(lambda y: global_lookup.get(y, {}).get('mean_tone', np.nan))
ca['global_sd'] = ca['Year'].map(lambda y: global_lookup.get(y, {}).get('sd_tone', np.nan))
ca['z_tone'] = (ca['mean_tone'] - ca['global_mean']) / ca['global_sd']

print("\nZ-score tone by country (post-2015 average, relative to global):")
print("  Negative = more negative than global average")
post = ca[ca.Year >= 2016].groupby('country')['z_tone'].mean().sort_values()
for cc, z in post.items():
    print(f"  {LABELS.get(cc, cc):20s}: z = {z:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Z-SCORE CORRECTED TONE TIME SERIES
# Now shows relative positioning free of the level shift
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5.5))
all_countries = sorted(ca.country.unique())
colors = dict(zip(all_countries, plt.cm.tab10(np.linspace(0, 1, len(all_countries)))))

for cc in all_countries:
    df = ca[ca.country == cc].sort_values('Year')
    if len(df) < 10:
        continue
    ax.plot(df['Year'], df['z_tone'], color=colors[cc], lw=1.8,
            alpha=0.85, label=LABELS.get(cc, cc))

ax.axhline(0, color='black', lw=1, ls='-', alpha=0.3, label='Global mean')
ax.axvline(2015, color='gray', ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Z-Score (Tone relative to global annual mean)')
ax.set_title('Country Tone Relative to Global Average (Z-Score Corrected)',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=8, ncol=3, loc='lower left')
ax.set_ylim(-1.5, 1.5)
plt.tight_layout()
plt.savefig(OUT + 'fig_zscore_tone.png', dpi=300)
plt.savefig(OUT + 'fig_zscore_tone.pdf')
print('\n✓ fig_zscore_tone')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: ELASTICITY ESTIMATION
# β = slope of OLS regression tone ~ Goldstein, per country per year
# This is the "conflict framing elasticity" — how much does tone change
# per unit of Goldstein? Robust to baseline shifts.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: ELASTICITY (β) ESTIMATION")
print("="*70)

# We need per-year per-country regression. Use the binned scatter data
# to compute weighted regression: tone_bin ~ goldstein_bin, weights = n
def weighted_ols(df):
    """Weighted OLS of tone ~ goldstein from binned data."""
    x = df['goldstein_bin'].values
    y = df['tone_bin'].values
    w = df['n'].values.astype(float)

    # Weighted means
    sw = w.sum()
    if sw == 0 or len(df) < 5:
        return {'beta': np.nan, 'r': np.nan, 'n_events': 0}

    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)

    # Weighted covariance and variance
    cov_xy = np.average((x - xw) * (y - yw), weights=w)
    var_x = np.average((x - xw)**2, weights=w)
    var_y = np.average((y - yw)**2, weights=w)

    if var_x == 0:
        return {'beta': np.nan, 'r': np.nan, 'n_events': int(sw)}

    beta = cov_xy / var_x
    r = cov_xy / np.sqrt(var_x * var_y) if var_y > 0 else np.nan

    return {'beta': beta, 'r': r, 'n_events': int(sw)}


# Global elasticity by year
print("\nGlobal elasticity (β) by year:")
global_betas = []
for yr in sorted(global_scatter.Year.unique()):
    sub = global_scatter[global_scatter.Year == yr]
    res = weighted_ols(sub)
    res['Year'] = yr
    global_betas.append(res)
    if yr % 5 == 0 or yr >= 2012:
        print(f"  {yr}: β = {res['beta']:.4f}, r = {res['r']:.4f}, N = {res['n_events']:,}")

global_betas = pd.DataFrame(global_betas)

# Country-level elasticity by year
country_betas = []
for cc in country_scatter.country.unique():
    for yr in sorted(country_scatter[country_scatter.country == cc].Year.unique()):
        sub = country_scatter[(country_scatter.country == cc) & (country_scatter.Year == yr)]
        res = weighted_ols(sub)
        res['Year'] = yr
        res['country'] = cc
        country_betas.append(res)

country_betas = pd.DataFrame(country_betas)
country_betas = country_betas[country_betas.n_events > 100]  # filter sparse years


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: GLOBAL ELASTICITY (β) OVER TIME
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel A: Beta (slope)
ax1.plot(global_betas['Year'], global_betas['beta'], color='#8e44ad',
         lw=2.5, marker='o', markersize=4, zorder=3)
ax1.axvline(2013, color='gray', ls=':', lw=1, alpha=0.5)
ax1.axvline(2015, color='gray', ls=':', lw=1, alpha=0.5)
ax1.annotate('GKG 1.0\nlaunches', xy=(2013, ax1.get_ylim()[0]),
             fontsize=8, ha='center', color='gray')
ax1.set_ylabel('β (Tone–Goldstein Slope)')
ax1.set_title('A. Conflict Framing Elasticity: How Much Does Tone Change Per Unit of Conflict?',
              fontweight='bold')
ax1.grid(True, alpha=0.2)

# Panel B: Correlation (r)
ax2.plot(global_betas['Year'], global_betas['r'], color='#2c3e50',
         lw=2.5, marker='s', markersize=4, zorder=3)
ax2.axvline(2013, color='gray', ls=':', lw=1, alpha=0.5)
ax2.axvline(2015, color='gray', ls=':', lw=1, alpha=0.5)
ax2.set_ylabel('r (Tone–Goldstein Correlation)')
ax2.set_xlabel('Year')
ax2.set_title('B. Tone–Conflict Coupling Strength', fontweight='bold')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(OUT + 'fig_global_elasticity.png', dpi=300)
plt.savefig(OUT + 'fig_global_elasticity.pdf')
print('\n✓ fig_global_elasticity')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: COUNTRY ELASTICITY (β) OVER TIME
# This should show REAL variation — unlike the raw tone which was artifact
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

for cc in COUNTRIES:
    df = country_betas[country_betas.country == cc].sort_values('Year')
    if len(df) < 10:
        continue
    color = colors.get(cc, 'gray')

    ax1.plot(df['Year'], df['beta'], color=color, lw=1.8,
             alpha=0.85, label=LABELS.get(cc, cc))
    ax2.plot(df['Year'], df['r'], color=color, lw=1.8, alpha=0.85)

# Global reference
ax1.plot(global_betas['Year'], global_betas['beta'], color='black',
         lw=2.5, ls='--', alpha=0.5, label='Global')
ax2.plot(global_betas['Year'], global_betas['r'], color='black',
         lw=2.5, ls='--', alpha=0.5)

for ax in [ax1, ax2]:
    ax.axvline(2015, color='gray', ls=':', lw=1, alpha=0.5)
    ax.grid(True, alpha=0.2)

ax1.set_ylabel('β (Tone–Goldstein Slope)')
ax1.set_title('A. Conflict Framing Elasticity by Country', fontweight='bold')
ax1.legend(fontsize=8, ncol=4, loc='upper left')

ax2.set_ylabel('r (Tone–Goldstein Correlation)')
ax2.set_xlabel('Year')
ax2.set_title('B. Tone–Conflict Coupling Strength by Country', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT + 'fig_country_elasticity.png', dpi=300)
plt.savefig(OUT + 'fig_country_elasticity.pdf')
print('✓ fig_country_elasticity')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: YEAR-COLORED SCATTER — ANNUAL CENTROIDS
# This shows the "migration" of the centroid in tone×goldstein space
# ══════════════════════════════════════════════════════════════════════════════
cmap_year = cm.plasma
norm_year = Normalize(vmin=1979, vmax=2026)

fig, ax = plt.subplots(figsize=(10, 8))

years = annual_global['Year'].values
tones = annual_global['mean_tone'].values
golds = annual_global['mean_goldstein'].values

# Draw trajectory line
for i in range(len(years)-1):
    ax.plot([golds[i], golds[i+1]], [tones[i], tones[i+1]],
            color=cmap_year(norm_year(years[i])), lw=1, alpha=0.5)

# Plot centroids
sc = ax.scatter(golds, tones, c=years, cmap=cmap_year, norm=norm_year,
                s=80, edgecolors='white', linewidths=0.5, zorder=3)

# Label key years
for yr in [1979, 1990, 2000, 2010, 2012, 2013, 2015, 2020, 2026]:
    row = annual_global[annual_global.Year == yr]
    if len(row) > 0:
        ax.annotate(str(yr), xy=(row['mean_goldstein'].values[0],
                    row['mean_tone'].values[0]),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('Mean Goldstein Scale (Conflict ← → Cooperation)')
ax.set_ylabel('Mean Tone')
ax.set_title('Annual Centroid Trajectory in Tone × Goldstein Space',
             fontweight='bold', fontsize=13)
ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
plt.colorbar(sc, ax=ax, label='Year', shrink=0.8)
plt.tight_layout()
plt.savefig(OUT + 'fig_centroid_trajectory.png', dpi=300)
plt.savefig(OUT + 'fig_centroid_trajectory.pdf')
print('✓ fig_centroid_trajectory')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: COUNTRY CENTROID TRAJECTORIES (faceted)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, cc in zip(axes.flat, COUNTRIES):
    df = country_annual[country_annual.country == cc].sort_values('Year')
    yrs = df['Year'].values
    ts = df['mean_tone'].values
    gs = df['mean_goldstein'].values

    for i in range(len(yrs)-1):
        ax.plot([gs[i], gs[i+1]], [ts[i], ts[i+1]],
                color=cmap_year(norm_year(yrs[i])), lw=1, alpha=0.5)

    sc = ax.scatter(gs, ts, c=yrs, cmap=cmap_year, norm=norm_year,
                    s=40, edgecolors='white', linewidths=0.3, zorder=3)

    # Label a few key years
    for yr in [1979, 2000, 2012, 2015, 2026]:
        row = df[df.Year == yr]
        if len(row) > 0:
            ax.annotate(str(yr), xy=(row['mean_goldstein'].values[0],
                        row['mean_tone'].values[0]),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

    ax.set_title(LABELS[cc], fontweight='bold')
    ax.axhline(0, color='gray', lw=0.4, alpha=0.5)

for ax in axes[-1, :]:
    ax.set_xlabel('Mean Goldstein')
for ax in axes[:, 0]:
    ax.set_ylabel('Mean Tone')

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(cm.ScalarMappable(norm=norm_year, cmap=cmap_year),
             cax=cbar_ax, label='Year')
fig.suptitle('Country Centroid Trajectories in Tone × Goldstein Space',
             fontweight='bold', fontsize=14, y=1.01)
plt.savefig(OUT + 'fig_country_trajectories.png', dpi=300, bbox_inches='tight')
plt.savefig(OUT + 'fig_country_trajectories.pdf', bbox_inches='tight')
print('✓ fig_country_trajectories')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: TOPIC ELASTICITY — cooperative vs conflictual
# ══════════════════════════════════════════════════════════════════════════════
topic = topic_annual.copy()
topic['code_str'] = topic['topic_code'].astype(str).str.zfill(2)

# Compute elasticity proxy from the correlation and SDs
# β = r * (sd_tone / sd_goldstein) -- but we don't have sd_goldstein per topic
# We'll use r_tone_gold directly as the coupling measure

coop_codes = ['03', '04', '05', '06', '07']
conf_codes = ['13', '14', '17', '18', '19', '20']

fig, ax = plt.subplots(figsize=(12, 5))

# Average r across cooperative topics per year
for label, codes, color, marker in [
    ('Cooperative events', coop_codes, '#2980b9', 'o'),
    ('Conflictual events', conf_codes, '#c0392b', 's'),
]:
    sub = topic[topic.code_str.isin(codes)]
    avg = sub.groupby('Year')['r_tone_gold'].mean()
    ax.plot(avg.index, avg.values, color=color, lw=2.5, marker=marker,
            markersize=4, label=label, zorder=3)

ax.axvline(2015, color='gray', ls=':', lw=1, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean r (Tone × Goldstein)')
ax.set_title('Tone–Conflict Coupling: Cooperative vs. Conflictual Event Types',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(OUT + 'fig_topic_elasticity.png', dpi=300)
plt.savefig(OUT + 'fig_topic_elasticity.pdf')
print('✓ fig_topic_elasticity')
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: CONVERGENCE FRAMEWORK — λ₂ CONNECTION
# Print the analytical framework for connecting GDELT β to
# mention-network λ₂ from Cárdenas-Sánchez et al.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: CONVERGENCE FRAMEWORK")
print("="*70)

print("""
PROPOSED TWO-LAYER ANALYSIS STRUCTURE:

Layer 1: GDELT Tone-Conflict Elasticity (β)
  ├── Global β trajectory (1979–2026)
  ├── Country-level β (VE, CO, US, BR, MX, UK)
  ├── Topic-level β (cooperative vs conflictual events)
  └── Z-score corrected relative tone positioning

Layer 2: Twitter Mention-Network Convergence (from Cárdenas-Sánchez et al.)
  ├── λ₂ (second eigenvalue) = resistance to convergence
  ├── Influence centrality distribution = who drives vs resists convergence
  └── Temporal evolution of network structure across electoral cycles

CONNECTION HYPOTHESIS:
  Countries/elections where β is HIGH (strong conflict-negativity coupling)
  should also show HIGH λ₂ (resistance to convergence).

  The mechanism: algorithmic amplification of negative-conflict content
  creates feedback loops that reinforce factional structure, which in
  turn is measured by λ₂ in mention networks.

TO COMPLETE THIS LAYER:
  Alejo needs to provide λ₂ values and influence centrality measures
  from the Twitter electoral data analyzed in the 2022 paper.
  Ideally for elections where GDELT country-level β can be computed
  for the same country and time window.

CANDIDATE ELECTIONS FOR JOINT ANALYSIS:
  - Colombia 2018, 2022 (λ₂ from paper + GDELT CO β)
  - Brazil 2018, 2022 (if Twitter data available)
  - Mexico 2018, 2024 (if Twitter data available)
  - USA 2016, 2020, 2024 (if Twitter data available)
  - Venezuela (special case: no free elections, but protest networks)
""")

# Compute β for election-year windows
print("\nGDELT β for key election years (country-level):")
election_years = {
    'CO': [2014, 2018, 2022],
    'BR': [2014, 2018, 2022],
    'US': [2016, 2020, 2024],
    'MX': [2018, 2024],
    'VE': [2013, 2018, 2024],
    'UK': [2015, 2017, 2019, 2024],
}

for cc, years in election_years.items():
    print(f"\n  {LABELS.get(cc, cc)}:")
    for yr in years:
        row = country_betas[(country_betas.country == cc) & (country_betas.Year == yr)]
        if len(row) > 0:
            b = row.iloc[0]
            print(f"    {yr}: β = {b['beta']:.4f}, r = {b['r']:.4f}, N = {b['n_events']:,}")
        else:
            print(f"    {yr}: [no data]")


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: β synchronicity test
# Does β shift at the same time across countries? (artifact test for slope)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DIAGNOSTIC: Is β shift synchronous? (artifact test)")
print("="*70)

for cc in COUNTRIES:
    df = country_betas[country_betas.country == cc].sort_values('Year')
    if len(df) < 10:
        continue
    df['beta_diff'] = df['beta'].diff()
    biggest = df.loc[df['beta_diff'].abs().idxmax()]
    pre = df[df.Year < 2012]['beta'].mean()
    post = df[df.Year >= 2016]['beta'].mean()
    print(f"  {LABELS.get(cc, cc):20s}: pre-2012 β={pre:.4f}, post-2016 β={post:.4f}, "
          f"largest Δβ in {int(biggest['Year'])} ({biggest['beta_diff']:+.4f})")


print("\n" + "="*70)
print("ALL ANALYSIS COMPLETE")
print("="*70)
