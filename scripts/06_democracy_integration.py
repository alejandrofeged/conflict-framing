import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'axes.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'legend.frameon': False,
})

C_BLUE='#4C72B0'; C_RED='#C44E52'; C_GREEN='#55A868'; C_ORANGE='#DD8452'
C_GRAY='#8C8C8C'; C_LGRAY='#D5D5D5'; C_BLACK='#333333'; C_PURPLE='#8172B3'

# ══════════════════════════════════════════════════════════════════════════
# DATA: Cárdenas countries (have NSE) + additional GDELT-only countries
# ══════════════════════════════════════════════════════════════════════════
df_bq1 = pd.read_csv('/mnt/user-data/uploads/bquxjob_5ba3abf4_19d930676f3.csv')
df_bq2 = pd.read_csv('/mnt/user-data/uploads/bigquery2.csv')
df_card = pd.concat([df_bq1, df_bq2], ignore_index=True)

# Also load VE, CO, US, BR, MX, UK scatter data
df_extra = pd.read_csv('/mnt/user-data/uploads/script_job_c4e16328bd50bee1c97535b85c063849_2.csv')

# Combine
df_all = pd.concat([df_card, df_extra], ignore_index=True)

# EIU Democracy Index 2024 (adding the extra countries)
eiu = {
    # Cárdenas countries
    'Germany': 8.73, 'New Zealand': 8.50, 'Japan': 8.48, 'Netherlands': 9.00,
    'Spain': 8.13, 'Portugal': 8.08, 'France': 7.99, 'Chile': 7.83,
    'Israel': 7.80, 'Poland': 7.40, 'Malaysia': 7.11, 'Trinidad & Tobago': 7.09,
    'Jamaica': 6.74, 'Dom. Rep.': 6.62, 'Argentina': 6.51, 'Thailand': 6.27,
    'Ghana': 6.24, 'Albania': 6.20, 'Moldova': 6.04, 'Zambia': 5.73,
    'Peru': 5.69, 'Armenia': 5.35, 'Ecuador': 5.24, 'Tanzania': 5.20,
    'Honduras': 4.98, 'Uganda': 4.49, 'Turkey': 4.26, 'Bolivia': 4.26,
    "Côte d'Ivoire": 4.22,
    'Iraq': 3.13, 'Belarus': 1.99, 'Guinea': 2.31, 'Burkina Faso': 2.62,
    'Djibouti': 2.37,
    # GDELT-only additions
    'Venezuela': 2.74, 'Colombia': 6.35, 'United States': 7.85,
    'Brazil': 6.49, 'Mexico': 5.32, 'United Kingdom': 8.34,
}

# Elections with NSE (from Cárdenas)
elections_card = [
    ('TD', 2020, 33.771, 'Trinidad & Tobago'), ('DJ', 2021, 25.903, 'Djibouti'),
    ('TH', 2019, 25.716, 'Thailand'), ('MD', 2021, 20.135, 'Moldova'),
    ('AL', 2021, 18.819, 'Albania'), ('IZ', 2021, 17.092, 'Iraq'),
    ('GV', 2020, 15.952, 'Guinea'), ('UV', 2020, 14.839, 'Burkina Faso'),
    ('AM', 2021, 13.631, 'Armenia'), ('JA', 2021, 13.445, 'Japan'),
    ('ZA', 2021, 10.890, 'Zambia'), ('PO', 2022, 10.073, 'Portugal'),
    ('HO', 2021, 9.586, 'Honduras'), ('JM', 2020, 9.368, 'Jamaica'),
    ('BO', 2020, 9.275, 'Belarus'), ('MY', 2018, 8.958, 'Malaysia'),
    ('IV', 2020, 8.614, "Côte d'Ivoire"), ('IS', 2021, 6.692, 'Israel'),
    ('BL', 2020, 5.383, 'Bolivia'), ('NZ', 2020, 5.227, 'New Zealand'),
    ('PE', 2021, 5.188, 'Peru'), ('FR', 2017, 5.114, 'France'),
    ('TU', 2018, 5.095, 'Turkey'), ('DR', 2020, 4.634, 'Dom. Rep.'),
    ('NL', 2021, 4.413, 'Netherlands'), ('TZ', 2020, 4.036, 'Tanzania'),
    ('GH', 2020, 3.843, 'Ghana'), ('UG', 2021, 3.318, 'Uganda'),
    ('EC', 2021, 2.683, 'Ecuador'), ('PL', 2020, 2.081, 'Poland'),
    ('CI', 2021, 1.672, 'Chile'), ('SP', 2019, 1.669, 'Spain'),
    ('AR', 2019, 1.114, 'Argentina'), ('GM', 2021, 0.543, 'Germany'),
]

# GDELT-only countries (no NSE) with representative election years
# For these, we don't have a specific NSE. We'll use 2022 as reference year
elections_extra = [
    ('VE', 2024, None, 'Venezuela'),
    ('CO', 2022, None, 'Colombia'),
    ('US', 2024, None, 'United States'),
    ('BR', 2022, None, 'Brazil'),
    ('MX', 2024, None, 'Mexico'),
    ('UK', 2024, None, 'United Kingdom'),
]

def weighted_stats(sub):
    x, y, w = sub['goldstein_bin'].values, sub['tone_bin'].values, sub['n'].values.astype(float)
    if w.sum() < 100 or len(sub) < 10:
        return np.nan, np.nan, np.nan
    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    cov = np.average((x-xw)*(y-yw), weights=w)
    vx = np.average((x-xw)**2, weights=w)
    return yw, xw, cov/vx if vx > 0 else np.nan

def regime(di):
    if di >= 8.01: return 'Full democracy'
    if di >= 6.01: return 'Flawed democracy'
    if di >= 4.01: return 'Hybrid regime'
    return 'Authoritarian'

rcol = {'Full democracy': C_GREEN, 'Flawed democracy': C_BLUE,
        'Hybrid regime': C_ORANGE, 'Authoritarian': C_RED}

# ══════════════════════════════════════════════════════════════════════════
# Compute everything
# ══════════════════════════════════════════════════════════════════════════
all_results = []
trajectories = {}

# Cárdenas countries
for cc, elec_yr, nse, name in elections_card:
    sub = df_all[(df_all.country == cc) & (df_all.Year == elec_yr)]
    if len(sub) == 0: continue
    t, g, b = weighted_stats(sub)
    if np.isnan(b): continue
    di = eiu.get(name, np.nan)
    all_results.append({
        'cc': cc, 'name': name, 'year': elec_yr, 'tone': t, 'goldstein': g,
        'beta': b, 'NSE': nse, 'DI': di, 'has_NSE': True,
        'regime': regime(di) if not np.isnan(di) else 'Unknown'
    })

# Extra countries
for cc, elec_yr, _, name in elections_extra:
    sub = df_all[(df_all.country == cc) & (df_all.Year == elec_yr)]
    if len(sub) == 0: continue
    t, g, b = weighted_stats(sub)
    if np.isnan(b): continue
    di = eiu.get(name, np.nan)
    all_results.append({
        'cc': cc, 'name': name, 'year': elec_yr, 'tone': t, 'goldstein': g,
        'beta': b, 'NSE': np.nan, 'DI': di, 'has_NSE': False,
        'regime': regime(di)
    })

# Trajectories for all countries
all_countries = list(set([r['cc'] for r in all_results]))
for cc in all_countries:
    traj = []
    for yr in range(2017, 2027):
        sub = df_all[(df_all.country == cc) & (df_all.Year == yr)]
        if len(sub) > 0:
            t, g, b = weighted_stats(sub)
            if not np.isnan(b):
                traj.append({'year': yr, 'tone': t, 'goldstein': g, 'beta': b})
    if len(traj) >= 5:
        trajectories[cc] = pd.DataFrame(traj)

res = pd.DataFrame(all_results)
print(f"Total countries/elections: {len(res)}")
print(f"  With NSE (Cárdenas): {res.has_NSE.sum()}")
print(f"  GDELT-only: {(~res.has_NSE).sum()}")

# Compute Δβ for all
dyn = []
for cc, traj in trajectories.items():
    sl, icpt, r, p, se = stats.linregress(traj.year, traj.beta)
    info = [r for r in all_results if r['cc'] == cc][0]
    dyn.append({
        'cc': cc, 'name': info['name'], 'regime': info['regime'],
        'beta_slope': sl, 'beta_mean': traj.beta.mean(),
        'beta_start': traj.iloc[0].beta, 'beta_end': traj.iloc[-1].beta,
        'p': p, 'DI': info['DI'], 'NSE': info['NSE'], 'has_NSE': info['has_NSE'],
    })
dyn = pd.DataFrame(dyn).sort_values('beta_slope', ascending=False)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE A: EXPANDED QUADRANT (β vs NSE, with GDELT-only countries as annotations)
# ══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.5, 6))

res_nse = res[res.has_NSE].copy()
res_extra = res[~res.has_NSE].copy()

med_b = res_nse.beta.median()
med_n = res_nse.NSE.median()

ax.axvline(med_b, color=C_LGRAY, lw=0.5, ls='--', zorder=1)
ax.axhline(med_n, color=C_LGRAY, lw=0.5, ls='--', zorder=1)

# NSE-bounded countries
for rtype in ['Full democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian']:
    sub = res_nse[res_nse.regime == rtype]
    ax.scatter(sub.beta, sub.NSE, s=45, c=rcol[rtype], alpha=0.8,
              edgecolors='white', linewidths=0.4, zorder=3,
              label=f'{rtype} (n={len(sub)})')

# GDELT-only countries: plot on bottom edge with arrows pointing up to their β
for _, row in res_extra.iterrows():
    color = rcol[row.regime]
    # Plot at bottom with open marker
    ax.scatter(row.beta, -1.5, s=60, facecolors='none', edgecolors=color,
              linewidths=1.2, zorder=3, marker='s')
    ax.annotate(row['name'], xy=(row.beta, -1.5),
               xytext=(0, -12), textcoords='offset points',
               fontsize=6, ha='center', color=color, fontweight='bold')
    # Vertical dashed line showing β
    ax.plot([row.beta, row.beta], [-1.5, 1], color=color, lw=0.3, ls=':', alpha=0.4)

# Labels for NSE-bounded
texts = []
for _, row in res_nse.iterrows():
    texts.append(ax.text(row.beta, row.NSE, row['name'], fontsize=5.5, color=C_BLACK))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=C_LGRAY, lw=0.3),
           force_points=(0.5, 0.5), force_text=(0.8, 0.8))

# Quadrant labels
ax.text(0.185, 34, 'Fragmented\nconversation', fontsize=7, color=C_GRAY,
        fontstyle='italic', fontweight='bold', ha='left', va='top')
ax.text(0.47, 34, 'Dual\npathology', fontsize=7, color=C_RED,
        fontstyle='italic', fontweight='bold', ha='right', va='top')
ax.text(0.185, 0.5, 'Healthy\ndiscourse', fontsize=7, color=C_GREEN,
        fontstyle='italic', fontweight='bold', ha='left', va='bottom')
ax.text(0.47, 0.5, 'Reactive\nmedia', fontsize=7, color=C_ORANGE,
        fontstyle='italic', fontweight='bold', ha='right', va='bottom')

ax.text(0.03, -3.5, 'GDELT-only (no NSE):', fontsize=6, color=C_GRAY,
        transform=ax.transData, fontstyle='italic')
ax.scatter([], [], marker='s', facecolors='none', edgecolors=C_GRAY,
          linewidths=1, s=50, label='GDELT-only (bottom axis)')

ax.legend(fontsize=6.5, loc='upper right')
ax.set_xlabel('Conflict framing elasticity (β)', fontsize=9)
ax.set_ylabel('Convergence resistance (NSE)', fontsize=9)
ax.set_xlim(0.15, 0.48)
ax.set_ylim(-5, 36)

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_quadrant_expanded.png', dpi=300, bbox_inches='tight')
print('✓ expanded quadrant')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE B: EXPANDED TONE×GOLDSTEIN SKELETON (includes all countries)
# ══════════════════════════════════════════════════════════════════════════
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

X_tg = res[['tone', 'goldstein']].values
X_tg_std = StandardScaler().fit_transform(X_tg)
D_tg = squareform(pdist(X_tg_std, 'euclidean'))

EPS = 1.0
G = nx.Graph()
for i in range(len(res)):
    G.add_node(i)
for i in range(len(res)):
    for j in range(i+1, len(res)):
        if D_tg[i, j] < EPS:
            G.add_edge(i, j, weight=D_tg[i, j])

print(f"\nExpanded skeleton: {G.number_of_edges()} edges, "
      f"{nx.number_connected_components(G)} components, "
      f"N = {len(res)} countries")

fig, ax = plt.subplots(figsize=(8, 6))

for i, j, data in G.edges(data=True):
    w = data['weight']
    alpha = max(0.06, 0.4 * (1 - w/EPS))
    ax.plot([res.iloc[i].goldstein, res.iloc[j].goldstein],
            [res.iloc[i].tone, res.iloc[j].tone],
            color=C_LGRAY, lw=0.5, alpha=alpha, zorder=1)

for i, row in res.iterrows():
    deg = G.degree(i)
    marker = 's' if not row.has_NSE else 'o'
    edge_w = 1.2 if not row.has_NSE else 0.3
    ax.scatter(row.goldstein, row.tone, s=20 + deg * 5,
              c=rcol[row.regime], alpha=0.85, marker=marker,
              edgecolors=C_BLACK if not row.has_NSE else 'white',
              linewidths=edge_w, zorder=3)

texts = []
for i, row in res.iterrows():
    weight = 'bold' if not row.has_NSE else 'normal'
    texts.append(ax.text(row.goldstein, row.tone, row['name'],
                        fontsize=5.2, color=C_BLACK, fontweight=weight))
adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=C_LGRAY, lw=0.3),
           force_points=(0.3, 0.3), force_text=(0.6, 0.6))

for rtype in ['Full democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian']:
    n = len(res[res.regime == rtype])
    ax.scatter([], [], c=rcol[rtype], s=30, label=f'{rtype} ({n})')
ax.scatter([], [], marker='s', facecolors='white', edgecolors=C_BLACK,
          linewidths=1, s=30, label='GDELT-only')

ax.legend(fontsize=6.5, loc='lower left')
ax.set_xlabel('Mean Goldstein scale (conflict ← → cooperation)', fontsize=9)
ax.set_ylabel('Mean tone', fontsize=9)
ax.text(0.97, 0.97, f'1-skeleton ε={EPS}\n{G.number_of_edges()} edges · N={len(res)}',
        transform=ax.transAxes, fontsize=6, va='top', ha='right', color=C_GRAY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_LGRAY, alpha=0.9))

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_tg_skeleton_expanded.png', dpi=300, bbox_inches='tight')
print('✓ expanded tone×goldstein skeleton')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE C: Δβ DOT PLOT WITH ALL COUNTRIES — ranked, with significance
# ══════════════════════════════════════════════════════════════════════════
dyn_sorted = dyn.sort_values('beta_slope')
fig, ax = plt.subplots(figsize=(6, 9))

y_pos = range(len(dyn_sorted))
for i, (_, row) in enumerate(dyn_sorted.iterrows()):
    color = rcol[row.regime]
    marker = 's' if not row.has_NSE else 'o'
    edge_w = 1.2 if not row.has_NSE else 0.4
    # Lollipop
    ax.plot([0, row.beta_slope * 100], [i, i], color=color, lw=0.6, alpha=0.6, zorder=1)
    sig_alpha = 1.0 if row.p < 0.05 else 0.5
    ax.scatter(row.beta_slope * 100, i, c=color, marker=marker, s=50,
              alpha=sig_alpha, edgecolors=C_BLACK if not row.has_NSE else 'white',
              linewidths=edge_w, zorder=3)
    # Star for significant
    if row.p < 0.05:
        ax.text(row.beta_slope * 100 + (0.1 if row.beta_slope > 0 else -0.1), i,
               '★', fontsize=7, color=color, va='center',
               ha='left' if row.beta_slope > 0 else 'right')

ax.axvline(0, color=C_BLACK, lw=0.5, zorder=2)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(dyn_sorted['name'].values, fontsize=6.5)
ax.set_xlabel('Δβ per year (× 100)', fontsize=9)
ax.set_title('Rate of change in conflict framing elasticity (2017–2026)',
             fontsize=10, fontweight='bold')
ax.tick_params(axis='y', length=0)
ax.spines['left'].set_visible(False)

# Legend
for rtype in ['Full democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian']:
    ax.scatter([], [], c=rcol[rtype], s=40, label=rtype)
ax.scatter([], [], c='white', edgecolors=C_BLACK, marker='s', s=40, label='GDELT-only', linewidths=1.2)
ax.text(1.5, -2, '★ = p < 0.05', fontsize=6, color=C_GRAY, fontstyle='italic')
ax.legend(fontsize=6, loc='lower right', ncol=2)

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_dbeta_ranked.png', dpi=300, bbox_inches='tight')
print('✓ Δβ ranked dot plot')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY: All countries with their three-layer metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("THREE-LAYER METRICS — ALL 40 COUNTRIES")
print("="*80)

# Merge β snapshot with dynamics
summary = res.merge(dyn[['cc', 'beta_slope', 'p']], on='cc', how='left')
summary = summary[['name', 'regime', 'DI', 'tone', 'goldstein', 'beta', 'beta_slope', 'NSE', 'has_NSE']]
summary.columns = ['Country', 'Regime', 'DI', 'Tone', 'Goldstein', 'β', 'Δβ/yr', 'NSE', 'has_NSE']
summary = summary.sort_values('DI', ascending=False)

print(summary.to_string(index=False))

# Correlations including all countries
print("\n" + "="*60)
print("CORRELATIONS (all countries with trajectories)")
print("="*60)
for a, b, la, lb in [
    ('beta_slope', 'DI', 'Δβ', 'Democracy Index'),
    ('beta_mean', 'DI', 'β (mean)', 'Democracy Index'),
]:
    sub = dyn.dropna(subset=[a, b])
    rho, p = stats.spearmanr(sub[a], sub[b])
    print(f"  {la} vs {lb}: ρ = {rho:.3f}, p = {p:.4f}  (n = {len(sub)})")

# β and Δβ summary for the new countries
print("\n" + "="*60)
print("EXTRA COUNTRIES (GDELT-only)")
print("="*60)
extra_dyn = dyn[~dyn.has_NSE]
print(extra_dyn[['name', 'regime', 'DI', 'beta_mean', 'beta_slope', 'p']].to_string(index=False))

print("\nDone.")
