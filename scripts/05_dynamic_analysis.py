import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'legend.frameon': False,
})

C_BLUE='#4C72B0'; C_RED='#C44E52'; C_GREEN='#55A868'; C_ORANGE='#DD8452'
C_GRAY='#8C8C8C'; C_LGRAY='#D5D5D5'; C_BLACK='#333333'; C_PURPLE='#8172B3'

# Load all data
df_bq1 = pd.read_csv('/mnt/user-data/uploads/bquxjob_5ba3abf4_19d930676f3.csv')
df_bq2 = pd.read_csv('/mnt/user-data/uploads/bigquery2.csv')
df = pd.concat([df_bq1, df_bq2], ignore_index=True)

elections_raw = [
    ('TD', 2020, 33.771, 'Trinidad & Tobago', 7.09),
    ('DJ', 2021, 25.903, 'Djibouti', 2.37),
    ('TH', 2019, 25.716, 'Thailand', 6.27),
    ('MD', 2021, 20.135, 'Moldova', 6.04),
    ('AL', 2021, 18.819, 'Albania', 6.20),
    ('IZ', 2021, 17.092, 'Iraq', 3.13),
    ('GV', 2020, 15.952, 'Guinea', 2.31),
    ('UV', 2020, 14.839, 'Burkina Faso', 2.62),
    ('AM', 2021, 13.631, 'Armenia', 5.35),
    ('JA', 2021, 13.445, 'Japan', 8.48),
    ('ZA', 2021, 10.890, 'Zambia', 5.73),
    ('PO', 2022, 10.073, 'Portugal', 8.08),
    ('HO', 2021, 9.586, 'Honduras', 4.98),
    ('JM', 2020, 9.368, 'Jamaica', 6.74),
    ('BO', 2020, 9.275, 'Belarus', 1.99),
    ('MY', 2018, 8.958, 'Malaysia', 7.11),
    ('IV', 2020, 8.614, "Côte d'Ivoire", 4.22),
    ('IS', 2021, 6.692, 'Israel', 7.80),
    ('BL', 2020, 5.383, 'Bolivia', 4.26),
    ('NZ', 2020, 5.227, 'New Zealand', 8.50),
    ('PE', 2021, 5.188, 'Peru', 5.69),
    ('FR', 2017, 5.114, 'France', 7.99),
    ('TU', 2018, 5.095, 'Turkey', 4.26),
    ('DR', 2020, 4.634, 'Dom. Rep.', 6.62),
    ('NL', 2021, 4.413, 'Netherlands', 9.00),
    ('TZ', 2020, 4.036, 'Tanzania', 5.20),
    ('GH', 2020, 3.843, 'Ghana', 6.24),
    ('UG', 2021, 3.318, 'Uganda', 4.49),
    ('EC', 2021, 2.683, 'Ecuador', 5.24),
    ('PL', 2020, 2.081, 'Poland', 7.40),
    ('CI', 2021, 1.672, 'Chile', 7.83),
    ('SP', 2019, 1.669, 'Spain', 8.13),
    ('AR', 2019, 1.114, 'Argentina', 6.51),
    ('GM', 2021, 0.543, 'Germany', 8.73),
]

def weighted_stats(sub):
    """Compute weighted mean tone, mean Goldstein, and β from binned data."""
    x, y, w = sub['goldstein_bin'].values, sub['tone_bin'].values, sub['n'].values.astype(float)
    if w.sum() < 100 or len(sub) < 10:
        return np.nan, np.nan, np.nan
    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    cov = np.average((x - xw) * (y - yw), weights=w)
    vx = np.average((x - xw)**2, weights=w)
    beta = cov / vx if vx > 0 else np.nan
    return yw, xw, beta  # mean_tone, mean_goldstein, beta


# ══════════════════════════════════════════════════════════════════════════
# PART 1: Compute Tone, Goldstein, β for election year AND full trajectories
# ══════════════════════════════════════════════════════════════════════════
results = []
trajectories = {}

for cc, elec_yr, nse, name, di in elections_raw:
    # Election year snapshot
    sub_elec = df[(df.country == cc) & (df.Year == elec_yr)]
    if len(sub_elec) == 0:
        continue
    tone_e, gold_e, beta_e = weighted_stats(sub_elec)
    if np.isnan(beta_e):
        continue
    results.append({
        'cc': cc, 'name': name, 'year': elec_yr,
        'tone': tone_e, 'goldstein': gold_e, 'beta': beta_e,
        'NSE': nse, 'DI': di
    })

    # Full trajectory 2017-2026
    traj = []
    for yr in range(2017, 2027):
        sub_yr = df[(df.country == cc) & (df.Year == yr)]
        if len(sub_yr) > 0:
            t, g, b = weighted_stats(sub_yr)
            if not np.isnan(b):
                traj.append({'year': yr, 'tone': t, 'goldstein': g, 'beta': b})
    if len(traj) >= 5:
        trajectories[cc] = pd.DataFrame(traj)

res = pd.DataFrame(results)
print(f"Elections with tone/goldstein/β: {len(res)}")
print(f"Countries with trajectories (≥5 years): {len(trajectories)}")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: 1-SKELETON IN TONE × GOLDSTEIN SPACE
# "Which countries SOUND alike?"
# ══════════════════════════════════════════════════════════════════════════

X_tg = res[['tone', 'goldstein']].values
X_tg_std = StandardScaler().fit_transform(X_tg)
D_tg = squareform(pdist(X_tg_std, 'euclidean'))

# Find good threshold
for eps in [0.6, 0.8, 1.0, 1.3, 1.6]:
    n_e = ((D_tg < eps).sum() - len(res)) / 2
    print(f"  Tone×Gold ε={eps:.1f}: {int(n_e)} edges, avg deg = {2*n_e/len(res):.1f}")

EPS_TG = 1.0

# Regime colors
def regime(di):
    if di >= 8.01: return 'Full democracy'
    if di >= 6.01: return 'Flawed democracy'
    if di >= 4.01: return 'Hybrid regime'
    return 'Authoritarian'
res['regime'] = res.DI.apply(regime)
rcol = {'Full democracy': C_GREEN, 'Flawed democracy': C_BLUE,
        'Hybrid regime': C_ORANGE, 'Authoritarian': C_RED}

G_tg = nx.Graph()
for i in range(len(res)):
    G_tg.add_node(i)
for i in range(len(res)):
    for j in range(i+1, len(res)):
        if D_tg[i, j] < EPS_TG:
            G_tg.add_edge(i, j, weight=D_tg[i, j])

print(f"\nTone×Goldstein skeleton (ε={EPS_TG}): {G_tg.number_of_edges()} edges, "
      f"{nx.number_connected_components(G_tg)} components")

fig, ax = plt.subplots(figsize=(7, 5.5))

# Edges
for i, j, data in G_tg.edges(data=True):
    w = data['weight']
    alpha = max(0.06, 0.4 * (1 - w / EPS_TG))
    ax.plot([res.iloc[i].goldstein, res.iloc[j].goldstein],
            [res.iloc[i].tone, res.iloc[j].tone],
            color=C_LGRAY, lw=0.5, alpha=alpha, zorder=1)

# Nodes
for i, row in res.iterrows():
    deg = G_tg.degree(i)
    ax.scatter(row.goldstein, row.tone, s=20 + deg * 5,
              c=rcol[row.regime], alpha=0.85,
              edgecolors='white', linewidths=0.3, zorder=3)

texts = []
for i, row in res.iterrows():
    texts.append(ax.text(row.goldstein, row.tone, row['name'], fontsize=5, color=C_BLACK))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=C_LGRAY, lw=0.3),
           force_points=(0.3, 0.3), force_text=(0.6, 0.6))

# Legend
for rtype in ['Full democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian']:
    n = len(res[res.regime == rtype])
    ax.scatter([], [], c=rcol[rtype], s=30, label=f'{rtype} ({n})')
ax.legend(fontsize=6, loc='upper left')

ax.set_xlabel('Mean Goldstein scale (conflict ← → cooperation)', fontsize=9)
ax.set_ylabel('Mean tone', fontsize=9)
ax.text(0.97, 0.03, f'1-skeleton (ε = {EPS_TG})\n{G_tg.number_of_edges()} edges\nDual distance: standardized (tone, Goldstein)',
        transform=ax.transAxes, fontsize=6, va='bottom', ha='right', color=C_GRAY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_LGRAY, alpha=0.9))

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_tg_skeleton.png', dpi=300, bbox_inches='tight')
print('✓ tone×goldstein skeleton')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# PART 3: DYNAMIC ANALYSIS — How is β CHANGING?
# Compute slope of β over 2017-2026 for each country
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("DYNAMIC ANALYSIS: Rate of change of β (2017-2026)")
print("="*60)

dynamics = []
for cc, traj_df in trajectories.items():
    if len(traj_df) < 5:
        continue
    slope, intercept, r_val, p_val, se = stats.linregress(traj_df.year, traj_df.beta)
    name = [n for c, y, nse, n, di in elections_raw if c == cc][0]
    nse_val = [nse for c, y, nse, n, di in elections_raw if c == cc][0]
    di_val = [di for c, y, nse, n, di in elections_raw if c == cc][0]
    beta_mean = traj_df.beta.mean()
    dynamics.append({
        'cc': cc, 'name': name,
        'beta_slope': slope,  # Δβ per year
        'beta_mean': beta_mean,
        'beta_start': traj_df.iloc[0].beta,
        'beta_end': traj_df.iloc[-1].beta,
        'r2': r_val**2, 'p': p_val,
        'NSE': nse_val, 'DI': di_val,
        'regime': regime(di_val),
        'n_years': len(traj_df),
    })

dyn = pd.DataFrame(dynamics).sort_values('beta_slope', ascending=False)
print(f"\nCountries with β trajectory data: {len(dyn)}")
print(dyn[['name', 'beta_slope', 'beta_mean', 'beta_start', 'beta_end', 'p', 'DI']].to_string(index=False))

# Correlate Δβ with NSE and DI
print(f"\n  Δβ vs NSE: ρ = {stats.spearmanr(dyn.beta_slope, dyn.NSE)[0]:.3f}, p = {stats.spearmanr(dyn.beta_slope, dyn.NSE)[1]:.3f}")
print(f"  Δβ vs DI:  ρ = {stats.spearmanr(dyn.beta_slope, dyn.DI)[0]:.3f}, p = {stats.spearmanr(dyn.beta_slope, dyn.DI)[1]:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: β trajectories for selected countries (small multiples)
# ══════════════════════════════════════════════════════════════════════════
# Pick countries that are interesting: fastest rising, fastest falling, stable
top_rising = dyn.nlargest(4, 'beta_slope').cc.tolist()
top_falling = dyn.nsmallest(4, 'beta_slope').cc.tolist()
names_map = {r.cc: r['name'] for _, r in dyn.iterrows()}

fig, axes = plt.subplots(2, 4, figsize=(12, 4.5), sharex=True, sharey=True)

for i, cc in enumerate(top_rising):
    ax = axes[0, i]
    t = trajectories[cc]
    color = rcol[dyn[dyn.cc == cc].regime.values[0]]
    ax.plot(t.year, t.beta, color=color, lw=1.5, marker='o', markersize=3, zorder=3)
    # Trend
    sl = dyn[dyn.cc == cc].beta_slope.values[0]
    p = dyn[dyn.cc == cc].p.values[0]
    x_line = np.array([t.year.min(), t.year.max()])
    ax.plot(x_line, sl * x_line + (t.beta.mean() - sl * t.year.mean()),
            color=color, lw=0.7, ls=':', alpha=0.5)
    ax.set_title(f'{names_map[cc]}\nΔβ = {sl:+.004f}/yr', fontsize=7, color=color)
    ax.axhline(0.25, color=C_LGRAY, lw=0.3, ls='--')

for i, cc in enumerate(top_falling):
    ax = axes[1, i]
    t = trajectories[cc]
    color = rcol[dyn[dyn.cc == cc].regime.values[0]]
    ax.plot(t.year, t.beta, color=color, lw=1.5, marker='o', markersize=3, zorder=3)
    sl = dyn[dyn.cc == cc].beta_slope.values[0]
    x_line = np.array([t.year.min(), t.year.max()])
    ax.plot(x_line, sl * x_line + (t.beta.mean() - sl * t.year.mean()),
            color=color, lw=0.7, ls=':', alpha=0.5)
    ax.set_title(f'{names_map[cc]}\nΔβ = {sl:+.004f}/yr', fontsize=7, color=color)
    ax.axhline(0.25, color=C_LGRAY, lw=0.3, ls='--')

axes[0, 0].set_ylabel('β', fontsize=8)
axes[1, 0].set_ylabel('β', fontsize=8)
fig.text(0.5, -0.01, 'Year', ha='center', fontsize=9)
fig.text(-0.01, 0.75, 'Fastest rising β →', ha='center', va='center', rotation=90,
         fontsize=8, color=C_RED, fontstyle='italic')
fig.text(-0.01, 0.25, '← Fastest falling β', ha='center', va='center', rotation=90,
         fontsize=8, color=C_GREEN, fontstyle='italic')

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_dynamics.png', dpi=300, bbox_inches='tight')
print('\n✓ dynamics small multiples')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: Δβ vs DI (are democracies deteriorating faster or slower?)
# ══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# Panel A: Δβ vs DI
for rtype in ['Full democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian']:
    sub = dyn[dyn.regime == rtype]
    ax1.scatter(sub.DI, sub.beta_slope * 100, s=40, c=rcol[rtype], alpha=0.8,
               edgecolors='white', linewidths=0.3, label=rtype)

# Label extremes
for _, row in dyn.iterrows():
    if abs(row.beta_slope) > dyn.beta_slope.quantile(0.85) or abs(row.beta_slope) > dyn.beta_slope.abs().quantile(0.85):
        ax1.annotate(row['name'], xy=(row.DI, row.beta_slope * 100),
                    fontsize=5, color=C_GRAY, xytext=(3, 3), textcoords='offset points')

ax1.axhline(0, color=C_GRAY, lw=0.5)
rho, p = stats.spearmanr(dyn.beta_slope, dyn.DI)
ax1.text(0.05, 0.95, f'ρ = {rho:.2f}, p = {p:.3f}', transform=ax1.transAxes, fontsize=7, va='top', color=C_GRAY)
ax1.set_xlabel('EIU Democracy Index')
ax1.set_ylabel('Δβ per year (×100)')
ax1.set_title('Rate of change in β vs. democratic quality', fontsize=9, fontweight='bold')
ax1.legend(fontsize=5.5, loc='lower right')

# Panel B: Δβ vs NSE
for rtype in ['Full democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian']:
    sub = dyn[dyn.regime == rtype]
    ax2.scatter(sub.NSE, sub.beta_slope * 100, s=40, c=rcol[rtype], alpha=0.8,
               edgecolors='white', linewidths=0.3)

for _, row in dyn.iterrows():
    if abs(row.beta_slope) > dyn.beta_slope.abs().quantile(0.85) or row.NSE > 15:
        ax2.annotate(row['name'], xy=(row.NSE, row.beta_slope * 100),
                    fontsize=5, color=C_GRAY, xytext=(3, 3), textcoords='offset points')

ax2.axhline(0, color=C_GRAY, lw=0.5)
rho2, p2 = stats.spearmanr(dyn.beta_slope, dyn.NSE)
ax2.text(0.05, 0.95, f'ρ = {rho2:.2f}, p = {p2:.3f}', transform=ax2.transAxes, fontsize=7, va='top', color=C_GRAY)
ax2.set_xlabel('Convergence resistance (NSE)')
ax2.set_ylabel('Δβ per year (×100)')
ax2.set_title('Rate of change in β vs. convergence resistance', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_delta_beta.png', dpi=300, bbox_inches='tight')
print('✓ Δβ panels')
plt.close()

print("\nDone.")
