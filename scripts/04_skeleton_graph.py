import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': False,
})

C_BLUE = '#4C72B0'; C_RED = '#C44E52'; C_GREEN = '#55A868'
C_ORANGE = '#DD8452'; C_GRAY = '#8C8C8C'; C_LGRAY = '#D5D5D5'; C_BLACK = '#333333'

# ── Load and compute β for all 34 elections ──
df_bq1 = pd.read_csv('/mnt/user-data/uploads/bquxjob_5ba3abf4_19d930676f3.csv')
df_bq2 = pd.read_csv('/mnt/user-data/uploads/bigquery2.csv')
df = pd.concat([df_bq1, df_bq2], ignore_index=True)

elections = [
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

def weighted_ols(sub):
    x, y, w = sub['goldstein_bin'].values, sub['tone_bin'].values, sub['n'].values.astype(float)
    if w.sum() < 100 or len(sub) < 10: return np.nan
    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    cov = np.average((x-xw)*(y-yw), weights=w)
    vx = np.average((x-xw)**2, weights=w)
    return cov/vx if vx > 0 else np.nan

results = []
for cc, yr, nse, name in elections:
    sub = df[(df.country == cc) & (df.Year == yr)]
    if len(sub) > 0:
        b = weighted_ols(sub)
        if not np.isnan(b):
            results.append({'name': name, 'cc': cc, 'beta': b, 'NSE': nse})

res = pd.DataFrame(results)
med_b = res.beta.median()
med_n = res.NSE.median()

# Assign quadrant
def quadrant(row):
    if row.beta < med_b and row.NSE < med_n: return 'Healthy'
    if row.beta >= med_b and row.NSE < med_n: return 'Reactive'
    if row.beta < med_b and row.NSE >= med_n: return 'Fragmented'
    return 'Dual'

res['quadrant'] = res.apply(quadrant, axis=1)
quad_colors = {'Healthy': C_GREEN, 'Reactive': C_ORANGE, 'Fragmented': C_GRAY, 'Dual': C_RED}

# ══════════════════════════════════════════════════════════════════════════
# STANDARDIZE AND COMPUTE DUAL DISTANCE
# ══════════════════════════════════════════════════════════════════════════
X = res[['beta', 'NSE']].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Pairwise Euclidean in standardized space
D = squareform(pdist(X_std, metric='euclidean'))

print(f"Distance matrix: {D.shape}")
print(f"Min non-zero: {D[D > 0].min():.3f}, Median: {np.median(D[D > 0]):.3f}, Max: {D.max():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# 1-SKELETON AT MULTIPLE THRESHOLDS (filtration view)
# ══════════════════════════════════════════════════════════════════════════

# Find a good threshold: ~2 edges per node on average → sparse but connected
for eps in [0.8, 1.0, 1.2, 1.5, 2.0]:
    n_edges = (D < eps).sum() / 2 - len(res) / 2  # subtract self-loops
    avg_deg = n_edges * 2 / len(res)
    components = 0
    G_test = nx.Graph()
    for i in range(len(res)):
        G_test.add_node(i)
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if D[i, j] < eps:
                G_test.add_edge(i, j)
    components = nx.number_connected_components(G_test)
    print(f"  ε = {eps:.1f}: {int(n_edges)} edges, avg degree = {avg_deg:.1f}, components = {components}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: 1-SKELETON GRAPH (single optimal threshold)
# Using ε = 1.2 for a readable graph
# ══════════════════════════════════════════════════════════════════════════
EPS = 1.2

G = nx.Graph()
for i, row in res.iterrows():
    G.add_node(i, name=row['name'], quadrant=row['quadrant'],
               beta=row['beta'], NSE=row['NSE'])

for i in range(len(res)):
    for j in range(i+1, len(res)):
        if D[i, j] < EPS:
            G.add_edge(i, j, weight=D[i, j])

print(f"\n1-Skeleton at ε = {EPS}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Components: {nx.number_connected_components(G)}")
print(f"  Avg degree: {2 * G.number_of_edges() / G.number_of_nodes():.1f}")

# Layout: use original (β, NSE) coordinates for position
pos = {}
for i, row in res.iterrows():
    pos[i] = (row['beta'], row['NSE'])

fig, ax = plt.subplots(figsize=(7, 6))

# Subtle quadrant shading
ax.axvspan(0.17, med_b, ymin=0, ymax=0.5, color='#f4faf4', alpha=0.6, zorder=0)
ax.axvspan(med_b, 0.48, ymin=0, ymax=0.5, color='#fdfaf0', alpha=0.6, zorder=0)
ax.axvspan(0.17, med_b, ymin=0.5, ymax=1.0, color='#f0f0f0', alpha=0.6, zorder=0)
ax.axvspan(med_b, 0.48, ymin=0.5, ymax=1.0, color='#fdf0f0', alpha=0.6, zorder=0)

ax.axvline(med_b, color=C_LGRAY, lw=0.5, ls='--', zorder=1)
ax.axhline(med_n, color=C_LGRAY, lw=0.5, ls='--', zorder=1)

# Draw edges — thinner for longer distances
for i, j, data in G.edges(data=True):
    w = data['weight']
    alpha = max(0.08, 0.5 * (1 - w / EPS))
    lw = max(0.3, 1.2 * (1 - w / EPS))
    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
            color=C_LGRAY, lw=lw, alpha=alpha, zorder=1)

# Draw nodes
for i, row in res.iterrows():
    color = quad_colors[row['quadrant']]
    deg = G.degree(i)
    size = 20 + deg * 8  # larger nodes = more connected
    ax.scatter(row['beta'], row['NSE'], s=size, c=color, alpha=0.85,
              edgecolors='white', linewidths=0.4, zorder=3)

# Labels
texts = []
for i, row in res.iterrows():
    texts.append(ax.text(row['beta'], row['NSE'], row['name'],
                        fontsize=5.5, color=C_BLACK, zorder=4))
adjust_text(texts, ax=ax,
           arrowprops=dict(arrowstyle='-', color=C_LGRAY, lw=0.3),
           force_points=(0.5, 0.5), force_text=(0.8, 0.8),
           expand_points=(1.4, 1.4), expand_text=(1.2, 1.2))

# Quadrant labels
ax.text(0.185, 34, 'Fragmented\nconversation', fontsize=7.5, color=C_GRAY,
        fontstyle='italic', fontweight='bold', ha='left', va='top')
ax.text(0.44, 34, 'Dual\npathology', fontsize=7.5, color=C_RED,
        fontstyle='italic', fontweight='bold', ha='right', va='top')
ax.text(0.185, 0.3, 'Healthy\ndiscourse', fontsize=7.5, color=C_GREEN,
        fontstyle='italic', fontweight='bold', ha='left', va='bottom')
ax.text(0.44, 0.3, 'Reactive\nmedia', fontsize=7.5, color=C_ORANGE,
        fontstyle='italic', fontweight='bold', ha='right', va='bottom')

ax.set_xlabel('Conflict framing elasticity (β)', fontsize=9)
ax.set_ylabel('Convergence resistance (NSE)', fontsize=9)
ax.set_xlim(0.17, 0.48)
ax.set_ylim(-1, 36)

# Legend for threshold
ax.text(0.97, 0.88, f'1-skeleton (ε = {EPS})\n{G.number_of_edges()} edges\nDual distance:\nstandardized (β, NSE)',
        transform=ax.transAxes, fontsize=6, va='top', ha='right', color=C_GRAY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_LGRAY, alpha=0.9))

plt.tight_layout()
plt.savefig('/home/claude/pub_fig_skeleton.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/claude/pub_fig_skeleton.pdf', bbox_inches='tight')
print('✓ 1-skeleton')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE: FILTRATION — 4 panels at increasing ε
# Shows how the graph builds up as threshold grows
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharex=True, sharey=True)

for ax, eps in zip(axes, [0.6, 1.0, 1.5, 2.5]):
    # Quadrant shading
    ax.axvline(med_b, color=C_LGRAY, lw=0.3, ls='--')
    ax.axhline(med_n, color=C_LGRAY, lw=0.3, ls='--')

    # Edges at this threshold
    n_edges = 0
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if D[i, j] < eps:
                alpha = max(0.05, 0.4 * (1 - D[i,j] / eps))
                ax.plot([res.iloc[i].beta, res.iloc[j].beta],
                       [res.iloc[i].NSE, res.iloc[j].NSE],
                       color=C_LGRAY, lw=0.5, alpha=alpha, zorder=1)
                n_edges += 1

    # Nodes
    for i, row in res.iterrows():
        color = quad_colors[row['quadrant']]
        ax.scatter(row['beta'], row['NSE'], s=15, c=color, alpha=0.85,
                  edgecolors='white', linewidths=0.2, zorder=3)

    ax.set_title(f'ε = {eps:.1f}  ({n_edges} edges)', fontsize=8)
    ax.set_xlim(0.17, 0.48)
    ax.set_ylim(-1, 36)
    ax.tick_params(labelsize=6)

axes[0].set_ylabel('NSE', fontsize=8)
for ax in axes:
    ax.set_xlabel('β', fontsize=8)

fig.suptitle('Rips Filtration: 1-Skeleton at Increasing Threshold',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/pub_fig_filtration.png', dpi=300, bbox_inches='tight')
plt.savefig('/home/claude/pub_fig_filtration.pdf', bbox_inches='tight')
print('✓ filtration')
plt.close()


# ══════════════════════════════════════════════════════════════════════════
# NETWORK STATS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("NETWORK STATISTICS (ε = 1.2)")
print("="*60)

# Degree distribution by quadrant
for q in ['Healthy', 'Reactive', 'Fragmented', 'Dual']:
    nodes_in_q = [i for i, row in res.iterrows() if row['quadrant'] == q]
    degs = [G.degree(n) for n in nodes_in_q]
    print(f"\n  {q} ({len(nodes_in_q)} countries):")
    for n in nodes_in_q:
        name = res.loc[n, 'name']
        deg = G.degree(n)
        neighbors = [res.loc[nb, 'name'] for nb in G.neighbors(n)]
        print(f"    {name}: degree {deg} → {', '.join(neighbors[:5])}")

# Cross-quadrant edges
print(f"\n  Cross-quadrant edges:")
same = 0; cross = 0
for i, j in G.edges():
    q_i = res.iloc[i]['quadrant']
    q_j = res.iloc[j]['quadrant']
    if q_i == q_j:
        same += 1
    else:
        cross += 1
print(f"    Same quadrant: {same}")
print(f"    Cross quadrant: {cross}")
print(f"    Ratio cross/total: {cross/(same+cross):.2f}")
