import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
})
OUT = '/home/claude/'

# Load
country_scatter = pd.read_csv('/mnt/user-data/uploads/script_job_c4e16328bd50bee1c97535b85c063849_2.csv')
country_annual = pd.read_csv('/mnt/user-data/uploads/script_job_49917d342750619023dc837ce8834d8d_1.csv')
global_scatter = pd.read_csv('/mnt/user-data/uploads/script_job_0a8fef8becb1f4ae12a43b22ed3ef81b_0.csv')
annual_global = pd.read_csv('/mnt/user-data/uploads/script_job_54cdee2baba05b459ada7ddb2d3696b7_2.csv')

LABELS = {'VE': 'Venezuela', 'CO': 'Colombia', 'US': 'United States',
          'BR': 'Brazil', 'MX': 'Mexico', 'UK': 'United Kingdom'}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FLOOR-EFFECT TEST
# Is Venezuela's low β due to tone compression (low SD) rather than
# a distinct framing regime?
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("FLOOR-EFFECT ANALYSIS")
print("="*70)

# Compute conditional tone SD by Goldstein bin, per country (post-2016)
for cc in ['VE', 'CO', 'US', 'BR', 'MX', 'UK']:
    sub = country_scatter[(country_scatter.country == cc) &
                          (country_scatter.Year >= 2016)]
    # For each Goldstein bin, compute weighted mean and weighted variance of tone
    gold_bins = sub.groupby('goldstein_bin').apply(
        lambda g: pd.Series({
            'w_mean_tone': np.average(g['tone_bin'], weights=g['n']),
            'w_sd_tone': np.sqrt(np.average((g['tone_bin'] - np.average(g['tone_bin'], weights=g['n']))**2, weights=g['n'])),
            'total_n': g['n'].sum()
        })
    ).reset_index()

    # Overall weighted SD of tone conditional on Goldstein
    overall_cond_sd = gold_bins['w_sd_tone'].mean()

    # Also get overall tone SD from annual data
    ca_sub = country_annual[(country_annual.country == cc) & (country_annual.Year >= 2016)]
    overall_sd = ca_sub['sd_tone'].mean()

    # Ratio of conditional SD to overall SD (if close to 1, Goldstein doesn't
    # explain much tone variance; if low, Goldstein is a strong predictor)
    print(f"\n  {LABELS[cc]}:")
    print(f"    Overall tone SD (post-2016):    {overall_sd:.3f}")
    print(f"    Mean conditional tone SD:       {overall_cond_sd:.3f}")
    print(f"    Ratio (cond/overall):           {overall_cond_sd/overall_sd:.3f}")

# Also compute: what is the tone SD for conflictual vs cooperative events?
print("\n\n  Tone SD by event valence (conflict vs cooperation Goldstein bins):")
for cc in ['VE', 'US']:
    sub = country_scatter[(country_scatter.country == cc) &
                          (country_scatter.Year >= 2016)]
    # Conflict: Goldstein < -2, Cooperation: Goldstein > 2
    conflict = sub[sub.goldstein_bin < -2]
    coop = sub[sub.goldstein_bin > 2]

    if len(conflict) > 0 and len(coop) > 0:
        c_mean = np.average(conflict['tone_bin'], weights=conflict['n'])
        c_sd = np.sqrt(np.average((conflict['tone_bin'] - c_mean)**2, weights=conflict['n']))
        co_mean = np.average(coop['tone_bin'], weights=coop['n'])
        co_sd = np.sqrt(np.average((coop['tone_bin'] - co_mean)**2, weights=coop['n']))
        print(f"\n  {LABELS[cc]}:")
        print(f"    Conflict events (G<-2): mean tone = {c_mean:.2f}, SD = {c_sd:.2f}")
        print(f"    Cooperative events (G>2): mean tone = {co_mean:.2f}, SD = {co_sd:.2f}")
        print(f"    Tone gap (coop - conflict): {co_mean - c_mean:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TDA: PERSISTENT HOMOLOGY ON TONE × GOLDSTEIN POINT CLOUDS
# Compare topological features of pre-2012 vs post-2015 regimes
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("TDA: PERSISTENT HOMOLOGY")
print("="*70)

def prepare_point_cloud(df, max_points=3000):
    """Convert binned data to a weighted point cloud, then sample."""
    # Expand bins to approximate point cloud (sample proportional to n)
    df = df.copy()
    df = df[(df.tone_bin >= -15) & (df.tone_bin <= 20) &
            (df.goldstein_bin >= -10.5) & (df.goldstein_bin <= 10.5)]
    # Use log(n) as weight to avoid extreme skew
    df['log_n'] = np.log10(df['n'] + 1)
    # Sample with replacement, probability ∝ log_n
    probs = df['log_n'] / df['log_n'].sum()
    idx = np.random.choice(len(df), size=min(max_points, len(df)),
                           replace=True, p=probs)
    points = df.iloc[idx][['goldstein_bin', 'tone_bin']].values
    # Standardize
    scaler = StandardScaler()
    return scaler.fit_transform(points)

np.random.seed(42)

# Global pre-2012 vs post-2016
pre_cloud = prepare_point_cloud(global_scatter[global_scatter.Year <= 2011])
post_cloud = prepare_point_cloud(global_scatter[global_scatter.Year >= 2016])

print(f"\nPre-2012 point cloud: {pre_cloud.shape[0]} points")
print(f"Post-2016 point cloud: {post_cloud.shape[0]} points")

# Compute persistent homology
result_pre = ripser(pre_cloud, maxdim=1)
result_post = ripser(post_cloud, maxdim=1)

# Plot persistence diagrams
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

plot_diagrams(result_pre['dgms'], ax=axes[0], show=False)
axes[0].set_title('A. Pre-2012 Regime: Persistence Diagram', fontweight='bold')
axes[0].set_xlim(-0.1, 3)
axes[0].set_ylim(-0.1, 3)

plot_diagrams(result_post['dgms'], ax=axes[1], show=False)
axes[1].set_title('B. Post-2016 Regime: Persistence Diagram', fontweight='bold')
axes[1].set_xlim(-0.1, 3)
axes[1].set_ylim(-0.1, 3)

plt.tight_layout()
plt.savefig(OUT + 'fig_tda_persistence.png', dpi=300)
plt.savefig(OUT + 'fig_tda_persistence.pdf')
print('✓ fig_tda_persistence')
plt.close()

# Compute summary statistics of persistence
for label, result in [('Pre-2012', result_pre), ('Post-2016', result_post)]:
    h0 = result['dgms'][0]  # Connected components
    h1 = result['dgms'][1]  # Loops/cycles

    # Filter out infinite persistence
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
    h1_lifetimes = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([0])

    print(f"\n  {label}:")
    print(f"    H0 (components): {len(h0_finite)} features, "
          f"max lifetime = {h0_lifetimes.max():.3f}, "
          f"mean = {h0_lifetimes.mean():.3f}")
    print(f"    H1 (loops):      {len(h1)} features, "
          f"max lifetime = {h1_lifetimes.max():.3f}, "
          f"mean = {h1_lifetimes.mean():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. TDA PER COUNTRY: Do different countries have different topological
# signatures in their tone×Goldstein space?
# ══════════════════════════════════════════════════════════════════════════════
print("\n\nTDA by country (post-2016):")

country_tda = {}
for cc in ['VE', 'CO', 'US', 'BR', 'MX', 'UK']:
    sub = country_scatter[(country_scatter.country == cc) &
                          (country_scatter.Year >= 2016)]
    if len(sub) < 100:
        continue
    cloud = prepare_point_cloud(sub, max_points=2000)
    result = ripser(cloud, maxdim=1)

    h0 = result['dgms'][0]
    h1 = result['dgms'][1]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h0_life = h0_finite[:, 1] - h0_finite[:, 0]
    h1_life = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([0])

    country_tda[cc] = {
        'h0_n': len(h0_finite), 'h0_max': h0_life.max(), 'h0_mean': h0_life.mean(),
        'h1_n': len(h1), 'h1_max': h1_life.max(), 'h1_mean': h1_life.mean(),
    }
    print(f"  {LABELS[cc]:20s}: H0={len(h0_finite)} feat (max={h0_life.max():.3f}), "
          f"H1={len(h1)} feat (max={h1_life.max():.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ODDS RATIO FRAMEWORK FOR β–λ₂ CONNECTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("ODDS RATIO FRAMEWORK")
print("="*70)

# From our GDELT analysis + Cárdenas-Sánchez et al. paper findings
# The paper studied: Colombia 2018, Spain 2019, UK 2019
# Finding: incumbent elections → slower convergence (higher λ₂)
# Colombia 2018 had incumbent (Santos party continuity via Duque) → higher λ₂
# UK 2019 had incumbent (Johnson) → higher λ₂

# We have GDELT β for these election-country-years:
election_data = pd.DataFrame([
    {'election': 'Colombia 2014', 'country': 'CO', 'year': 2014, 'beta': 0.0619, 'incumbent': False},
    {'election': 'Colombia 2018', 'country': 'CO', 'year': 2018, 'beta': 0.2179, 'incumbent': True},
    {'election': 'Colombia 2022', 'country': 'CO', 'year': 2022, 'beta': 0.2804, 'incumbent': False},
    {'election': 'Brazil 2014', 'country': 'BR', 'year': 2014, 'beta': 0.0499, 'incumbent': True},
    {'election': 'Brazil 2018', 'country': 'BR', 'year': 2018, 'beta': 0.2323, 'incumbent': True},
    {'election': 'Brazil 2022', 'country': 'BR', 'year': 2022, 'beta': 0.2555, 'incumbent': True},
    {'election': 'US 2016', 'country': 'US', 'year': 2016, 'beta': 0.2740, 'incumbent': False},
    {'election': 'US 2020', 'country': 'US', 'year': 2020, 'beta': 0.2566, 'incumbent': True},
    {'election': 'US 2024', 'country': 'US', 'year': 2024, 'beta': 0.3157, 'incumbent': False},
    {'election': 'Mexico 2018', 'country': 'MX', 'year': 2018, 'beta': 0.2372, 'incumbent': False},
    {'election': 'Mexico 2024', 'country': 'MX', 'year': 2024, 'beta': 0.2862, 'incumbent': True},
    {'election': 'UK 2017', 'country': 'UK', 'year': 2017, 'beta': 0.2598, 'incumbent': True},
    {'election': 'UK 2019', 'country': 'UK', 'year': 2019, 'beta': 0.2541, 'incumbent': True},
    {'election': 'UK 2024', 'country': 'UK', 'year': 2024, 'beta': 0.2889, 'incumbent': True},
    {'election': 'Venezuela 2018', 'country': 'VE', 'year': 2018, 'beta': 0.2062, 'incumbent': True},
    {'election': 'Venezuela 2024', 'country': 'VE', 'year': 2024, 'beta': 0.2606, 'incumbent': True},
])

# Only post-2015 elections (consistent measurement)
ed = election_data[election_data.year >= 2016].copy()
ed['high_beta'] = ed['beta'] > ed['beta'].median()

print(f"\nPost-2016 elections (N={len(ed)}):")
print(f"  Median β = {ed['beta'].median():.4f}")
print(f"  High β elections: {ed[ed.high_beta]['election'].tolist()}")
print(f"  Low β elections: {ed[~ed.high_beta]['election'].tolist()}")

# Cárdenas-Sánchez finding: incumbent → higher λ₂
# Our framework: high β → higher λ₂
# Odds ratio: P(high_β | incumbent) / P(high_β | non-incumbent)
inc_high = len(ed[(ed.incumbent) & (ed.high_beta)])
inc_low = len(ed[(ed.incumbent) & (~ed.high_beta)])
non_high = len(ed[(~ed.incumbent) & (ed.high_beta)])
non_low = len(ed[(~ed.incumbent) & (~ed.high_beta)])

print(f"\n  2×2 Table (incumbent × high β):")
print(f"                    High β    Low β")
print(f"    Incumbent:       {inc_high}          {inc_low}")
print(f"    Non-incumbent:   {non_high}          {non_low}")

if inc_low > 0 and non_high > 0:
    OR = (inc_high * non_low) / (inc_low * non_high)
    print(f"\n  Odds Ratio = {OR:.2f}")
else:
    print(f"\n  Odds Ratio: cell with 0, use Haldane correction")
    OR = ((inc_high + 0.5) * (non_low + 0.5)) / ((inc_low + 0.5) * (non_high + 0.5))
    print(f"  Haldane-corrected OR = {OR:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("STATISTICAL TESTS")
print("="*70)

# Test 1: Is the post-2015 β trend significantly different from zero?
post_global = annual_global[annual_global.Year >= 2016].copy()
# Compute β per year from global scatter
from scipy.stats import linregress

def weighted_ols(df):
    x = df['goldstein_bin'].values
    y = df['tone_bin'].values
    w = df['n'].values.astype(float)
    sw = w.sum()
    if sw == 0 or len(df) < 5:
        return np.nan
    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    cov_xy = np.average((x - xw) * (y - yw), weights=w)
    var_x = np.average((x - xw)**2, weights=w)
    return cov_xy / var_x if var_x > 0 else np.nan

betas_post = []
for yr in range(2016, 2027):
    sub = global_scatter[global_scatter.Year == yr]
    b = weighted_ols(sub)
    if not np.isnan(b):
        betas_post.append({'Year': yr, 'beta': b})

bp = pd.DataFrame(betas_post)
slope, intercept, r_value, p_value, std_err = linregress(bp['Year'], bp['beta'])
print(f"\nTest 1: Post-2016 trend in global β")
print(f"  Slope = {slope:.5f} per year")
print(f"  R² = {r_value**2:.4f}")
print(f"  p = {p_value:.4f}")
print(f"  Interpretation: β increases by {slope:.4f} per year ({slope*10:.3f} per decade)")

# Test 2: Is β pre-2012 significantly different from post-2016? (Welch's t)
betas_pre = []
for yr in range(1979, 2012):
    sub = global_scatter[global_scatter.Year == yr]
    b = weighted_ols(sub)
    if not np.isnan(b):
        betas_pre.append(b)

betas_post_vals = bp['beta'].values
t_stat, p_val = stats.ttest_ind(betas_pre, betas_post_vals, equal_var=False)
print(f"\nTest 2: Welch's t-test, pre-2012 β vs post-2016 β")
print(f"  Pre-2012 mean β = {np.mean(betas_pre):.4f} (SD = {np.std(betas_pre):.4f})")
print(f"  Post-2016 mean β = {np.mean(betas_post_vals):.4f} (SD = {np.std(betas_post_vals):.4f})")
print(f"  t = {t_stat:.2f}, p = {p_val:.2e}")

# Test 3: Country-level β variation — ANOVA within post-2016
print(f"\nTest 3: One-way ANOVA on country β (post-2016)")
country_betas_post = {}
for cc in ['VE', 'CO', 'US', 'BR', 'MX', 'UK']:
    cb = []
    for yr in range(2016, 2027):
        sub = country_scatter[(country_scatter.country == cc) & (country_scatter.Year == yr)]
        b = weighted_ols(sub)
        if not np.isnan(b):
            cb.append(b)
    if len(cb) > 3:
        country_betas_post[cc] = cb
        print(f"  {LABELS[cc]:20s}: mean β = {np.mean(cb):.4f} ± {np.std(cb):.4f} (n={len(cb)})")

groups = list(country_betas_post.values())
f_stat, p_val = stats.f_oneway(*groups)
print(f"  F = {f_stat:.2f}, p = {p_val:.4f}")


print("\n\nAll analyses complete.")
