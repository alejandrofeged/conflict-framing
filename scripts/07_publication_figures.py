import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════
# NATURE/SCIENCE STYLE SETUP
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'legend.frameon': False,
    'legend.fontsize': 7,
})

# Nature palette
C_BLUE = '#4C72B0'
C_RED = '#C44E52'
C_GREEN = '#55A868'
C_ORANGE = '#DD8452'
C_PURPLE = '#8172B3'
C_GRAY = '#8C8C8C'
C_LGRAY = '#D5D5D5'
C_BLACK = '#333333'

OUT = '/home/claude/pub_'

# ── Load data ──
global_scatter = pd.read_csv('/mnt/user-data/uploads/script_job_0a8fef8becb1f4ae12a43b22ed3ef81b_0.csv')
country_annual = pd.read_csv('/mnt/user-data/uploads/script_job_49917d342750619023dc837ce8834d8d_1.csv')
annual_global = pd.read_csv('/mnt/user-data/uploads/script_job_54cdee2baba05b459ada7ddb2d3696b7_2.csv')
country_scatter = pd.read_csv('/mnt/user-data/uploads/script_job_c4e16328bd50bee1c97535b85c063849_2.csv')
topic_annual = pd.read_csv('/mnt/user-data/uploads/script_job_8c5f2bb731747f1cdc29db5047502d3f_3.csv')

# Combine BigQuery election data
df_bq1 = pd.read_csv('/mnt/user-data/uploads/bquxjob_5ba3abf4_19d930676f3.csv')
df_bq2 = pd.read_csv('/mnt/user-data/uploads/bigquery2.csv')
df_elections = pd.concat([df_bq1, df_bq2], ignore_index=True)

LABELS = {'VE': 'Venezuela', 'CO': 'Colombia', 'US': 'United States',
          'BR': 'Brazil', 'MX': 'Mexico', 'UK': 'United Kingdom',
          'IN': 'India', 'TU': 'Turkey', 'RS': 'Russia'}

def weighted_ols(sub):
    x, y, w = sub['goldstein_bin'].values, sub['tone_bin'].values, sub['n'].values.astype(float)
    if w.sum() < 100 or len(sub) < 10: return np.nan
    xw = np.average(x, weights=w)
    yw = np.average(y, weights=w)
    cov = np.average((x-xw)*(y-yw), weights=w)
    vx = np.average((x-xw)**2, weights=w)
    return cov/vx if vx > 0 else np.nan

# Compute global β per year
global_betas = []
for yr in sorted(global_scatter.Year.unique()):
    b = weighted_ols(global_scatter[global_scatter.Year == yr])
    global_betas.append({'Year': yr, 'beta': b})
global_betas = pd.DataFrame(global_betas)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: "THE JUMP" — Centroid trajectory
# Redesign: vertical layout emphasizing the DROP, connected path,
# key years annotated, minimal chrome
# ══════════════════════════════════════════════════════════════════════════
def fig1_the_jump():
    years = annual_global['Year'].values
    tones = annual_global['mean_tone'].values
    golds = annual_global['mean_goldstein'].values

    fig, ax = plt.subplots(figsize=(3.5, 5))  # tall, narrow — emphasizes vertical drop

    # Draw path segments colored by era
    for i in range(len(years)-1):
        color = C_BLUE if years[i] < 2012 else (C_GRAY if years[i] < 2016 else C_RED)
        ax.plot([golds[i], golds[i+1]], [tones[i], tones[i+1]],
                color=color, lw=0.6, alpha=0.5, zorder=1)

    # Points
    pre = years < 2012
    trans = (years >= 2012) & (years < 2016)
    post = years >= 2016

    ax.scatter(golds[pre], tones[pre], c=C_BLUE, s=18, zorder=3,
              edgecolors='white', linewidths=0.3, label='1979–2011')
    ax.scatter(golds[trans], tones[trans], c=C_GRAY, s=18, zorder=3,
              edgecolors='white', linewidths=0.3, marker='D', label='2012–2015 (transition)')
    ax.scatter(golds[post], tones[post], c=C_RED, s=18, zorder=3,
              edgecolors='white', linewidths=0.3, label='2016–2026')

    # Annotate key years only
    for yr, ha, va, dx, dy in [
        (1979, 'left', 'bottom', -0.02, 0.15),
        (2000, 'right', 'bottom', 0.02, 0.15),
        (2012, 'left', 'top', -0.02, -0.2),
        (2015, 'right', 'top', 0.02, -0.15),
        (2026, 'left', 'top', -0.02, -0.2),
    ]:
        idx = np.where(years == yr)[0]
        if len(idx) > 0:
            i = idx[0]
            ax.annotate(str(yr), xy=(golds[i], tones[i]),
                       xytext=(golds[i]+dx, tones[i]+dy),
                       fontsize=6.5, color=C_BLACK, ha=ha, va=va,
                       arrowprops=dict(arrowstyle='-', color=C_GRAY, lw=0.4))

    # Annotation: the drop
    ax.annotate('', xy=(0.42, -1.8), xytext=(0.42, 5.0),
               arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2, ls='--'))
    ax.text(0.33, 1.5, 'Δ = −6.47', fontsize=7, color=C_RED, rotation=90,
            ha='right', va='center', fontstyle='italic')

    ax.set_xlabel('Mean Goldstein scale', fontsize=8)
    ax.set_ylabel('Mean tone', fontsize=8)
    ax.legend(loc='upper right', fontsize=6, markerscale=0.8)

    ax.set_xlim(0.25, 1.25)
    plt.tight_layout()
    plt.savefig(OUT + 'fig1_jump.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig1_jump.pdf', bbox_inches='tight')
    print('✓ fig1')
    plt.close()

fig1_the_jump()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: "THE SLOPE IS RISING" — β over time
# Global trend + VE/US/CO highlighted. Single clean panel.
# ══════════════════════════════════════════════════════════════════════════
def fig2_slope():
    fig, ax = plt.subplots(figsize=(5.5, 3))

    # Global
    gb = global_betas.copy()
    ax.plot(gb.Year, gb.beta, color=C_BLACK, lw=1.5, zorder=3, label='Global')

    # Highlighted countries
    for cc, color, label in [('US', C_BLUE, 'United States'), ('VE', C_RED, 'Venezuela'), ('CO', C_ORANGE, 'Colombia')]:
        betas = []
        for yr in range(1979, 2027):
            sub = country_scatter[(country_scatter.country == cc) & (country_scatter.Year == yr)]
            if len(sub) > 0:
                b = weighted_ols(sub)
                betas.append({'Year': yr, 'beta': b})
        if betas:
            bdf = pd.DataFrame(betas).dropna()
            ax.plot(bdf.Year, bdf.beta, color=color, lw=0.9, alpha=0.8, label=label)

    # Post-2016 regression line
    post = gb[gb.Year >= 2016]
    slope, intercept, r_val, p_val, se = stats.linregress(post.Year, post.beta)
    x_line = np.array([2016, 2026])
    ax.plot(x_line, slope * x_line + intercept, color=C_BLACK, lw=0.7, ls=':', alpha=0.6)

    # Annotation
    ax.text(2022, 0.33, f'Post-2016 trend:\n+0.004/yr, p = .022',
            fontsize=6.5, color=C_BLACK, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_LGRAY, alpha=0.9))

    # Regime break
    ax.axvspan(2012.5, 2015.5, color=C_LGRAY, alpha=0.3, zorder=0)
    ax.text(2014, 0.05, 'v1→v2', fontsize=6, color=C_GRAY, ha='center', fontstyle='italic')

    ax.set_xlabel('Year')
    ax.set_ylabel('Conflict framing elasticity (β)')
    ax.legend(loc='upper left', fontsize=6.5)
    ax.set_xlim(1978, 2027)
    ax.set_ylim(0, 0.36)

    plt.tight_layout()
    plt.savefig(OUT + 'fig2_slope.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig2_slope.pdf', bbox_inches='tight')
    print('✓ fig2')
    plt.close()

fig2_slope()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: "WHO'S WORST" — Cleveland dot plot of z-scores
# Horizontal, ranked, with thin connecting lines to zero
# ══════════════════════════════════════════════════════════════════════════
def fig3_zscore():
    # Compute post-2016 z-scores
    global_lookup = annual_global.set_index('Year')[['mean_tone', 'sd_tone']].to_dict('index')
    ca = country_annual.copy()
    ca['global_mean'] = ca.Year.map(lambda y: global_lookup.get(y, {}).get('mean_tone', np.nan))
    ca['global_sd'] = ca.Year.map(lambda y: global_lookup.get(y, {}).get('sd_tone', np.nan))
    ca['z_tone'] = (ca.mean_tone - ca.global_mean) / ca.global_sd

    post = ca[ca.Year >= 2016].groupby('country').agg(
        z_mean=('z_tone', 'mean'),
        z_se=('z_tone', lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()
    post['label'] = post.country.map(LABELS)
    post = post.sort_values('z_mean')

    fig, ax = plt.subplots(figsize=(3.5, 3))

    y_pos = range(len(post))

    # Lollipop stems
    for i, (_, row) in enumerate(post.iterrows()):
        color = C_RED if row.z_mean < -0.1 else (C_GREEN if row.z_mean > 0.1 else C_GRAY)
        ax.plot([0, row.z_mean], [i, i], color=color, lw=0.6, zorder=1)
        ax.scatter(row.z_mean, i, color=color, s=30, zorder=3, edgecolors='white', linewidths=0.3)
        # Error bar
        ax.plot([row.z_mean - 1.96*row.z_se, row.z_mean + 1.96*row.z_se], [i, i],
                color=color, lw=2, alpha=0.25, solid_capstyle='round')

    ax.axvline(0, color=C_BLACK, lw=0.5, zorder=2)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(post.label.values, fontsize=7)
    ax.set_xlabel('Tone relative to global mean (z-score)', fontsize=8)
    ax.set_xlim(-0.55, 0.35)

    # Subtle annotation
    ax.text(-0.50, -0.8, '← more negative', fontsize=5.5, color=C_RED, fontstyle='italic')
    ax.text(0.15, -0.8, 'more positive →', fontsize=5.5, color=C_GREEN, fontstyle='italic')

    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    plt.tight_layout()
    plt.savefig(OUT + 'fig3_zscore.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig3_zscore.pdf', bbox_inches='tight')
    print('✓ fig3')
    plt.close()

fig3_zscore()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: "THE QUADRANT" — β vs NSE, the money shot
# Smart label placement, region shading, clean axes
# ══════════════════════════════════════════════════════════════════════════
def fig4_quadrant():
    elections = [
        ('TD', 2020, 33.771, 'Trinidad & Tobago'), ('DJ', 2021, 25.903, 'Djibouti'),
        ('TH', 2019, 25.716, 'Thailand'), ('TG', 2020, 21.375, 'Togo'),
        ('MD', 2021, 20.135, 'Moldova'), ('AL', 2021, 18.819, 'Albania'),
        ('IZ', 2021, 17.092, 'Iraq'), ('GV', 2020, 15.952, 'Guinea'),
        ('UV', 2020, 14.839, 'Burkina Faso'), ('AM', 2021, 13.631, 'Armenia'),
        ('JA', 2021, 13.445, 'Japan'), ('ZA', 2021, 10.890, 'Zambia'),
        ('PO', 2022, 10.073, 'Portugal'), ('HO', 2021, 9.586, 'Honduras'),
        ('JM', 2020, 9.368, 'Jamaica'), ('BO', 2020, 9.275, 'Belarus'),
        ('MY', 2018, 8.958, 'Malaysia'), ('IV', 2020, 8.614, "Côte d'Ivoire"),
        ('IS', 2021, 6.692, 'Israel'), ('BL', 2020, 5.383, 'Bolivia'),
        ('NZ', 2020, 5.227, 'New Zealand'), ('PE', 2021, 5.188, 'Peru'),
        ('FR', 2017, 5.114, 'France'), ('TU', 2018, 5.095, 'Turkey'),
        ('DR', 2020, 4.634, 'Dom. Rep.'), ('NL', 2021, 4.413, 'Netherlands'),
        ('TZ', 2020, 4.036, 'Tanzania'), ('GH', 2020, 3.843, 'Ghana'),
        ('UG', 2021, 3.318, 'Uganda'), ('EC', 2021, 2.683, 'Ecuador'),
        ('PL', 2020, 2.081, 'Poland'), ('CI', 2021, 1.672, 'Chile'),
        ('SP', 2019, 1.669, 'Spain'), ('AR', 2019, 1.114, 'Argentina'),
        ('GM', 2021, 0.543, 'Germany'),
    ]

    results = []
    for cc, yr, nse, name in elections:
        sub = df_elections[(df_elections.country == cc) & (df_elections.Year == yr)]
        if len(sub) > 0:
            b = weighted_ols(sub)
            if not np.isnan(b):
                results.append({'name': name, 'beta': b, 'NSE': nse})
    res = pd.DataFrame(results)

    med_b = res.beta.median()
    med_n = res.NSE.median()

    fig, ax = plt.subplots(figsize=(5.5, 5))

    # Subtle quadrant shading
    ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < med_b else 0.17, med_b, ymin=0.5, ymax=1.0, color='#f0f0f0', alpha=0.5, zorder=0)  # top-left
    ax.axvspan(med_b, 0.48, ymin=0.5, ymax=1.0, color='#fdf0f0', alpha=0.5, zorder=0)  # top-right
    ax.axvspan(0.17, med_b, ymin=0, ymax=0.5, color='#f0fdf0', alpha=0.5, zorder=0)  # bottom-left
    ax.axvspan(med_b, 0.48, ymin=0, ymax=0.5, color='#fdfaf0', alpha=0.5, zorder=0)  # bottom-right

    ax.axvline(med_b, color=C_LGRAY, lw=0.7, ls='--', zorder=1)
    ax.axhline(med_n, color=C_LGRAY, lw=0.7, ls='--', zorder=1)

    # Points
    ax.scatter(res.beta, res.NSE, s=25, c=C_BLACK, alpha=0.7,
              edgecolors='white', linewidths=0.3, zorder=3)

    # Labels with adjustText
    texts = []
    for _, row in res.iterrows():
        texts.append(ax.text(row.beta, row.NSE, row['name'], fontsize=5.5, color=C_BLACK))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color=C_LGRAY, lw=0.3),
                force_points=(0.5, 0.5), force_text=(0.8, 0.8),
                expand_points=(1.5, 1.5), expand_text=(1.2, 1.2))

    # Quadrant labels
    ax.text(0.185, 32, 'Fragmented\nconversation', fontsize=7, color=C_GRAY,
            fontstyle='italic', fontweight='bold', ha='left', va='top')
    ax.text(0.42, 32, 'Dual\npathology', fontsize=7, color=C_RED,
            fontstyle='italic', fontweight='bold', ha='right', va='top')
    ax.text(0.185, 1.0, 'Healthy\ndiscourse', fontsize=7, color=C_GREEN,
            fontstyle='italic', fontweight='bold', ha='left', va='bottom')
    ax.text(0.42, 1.0, 'Reactive\nmedia', fontsize=7, color=C_ORANGE,
            fontstyle='italic', fontweight='bold', ha='right', va='bottom')

    # Stats
    rho, p = stats.spearmanr(res.beta, res.NSE)
    ax.text(0.97, 0.97, f'ρ = {rho:.2f}, p = {p:.2f}\nn = {len(res)}',
            transform=ax.transAxes, fontsize=6.5, va='top', ha='right', color=C_GRAY)

    ax.set_xlabel('Conflict framing elasticity (β)', fontsize=9)
    ax.set_ylabel('Convergence resistance (NSE)', fontsize=9)
    ax.set_xlim(0.17, 0.48)
    ax.set_ylim(-1, 36)

    plt.tight_layout()
    plt.savefig(OUT + 'fig4_quadrant.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUT + 'fig4_quadrant.pdf', bbox_inches='tight')
    print('✓ fig4')
    plt.close()

fig4_quadrant()


# ══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ══════════════════════════════════════════════════════════════════════════

# S1: Raw tone time series
def figS1():
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(annual_global.Year, annual_global.mean_tone, color=C_BLACK, lw=1.5, label='Global')
    ax.fill_between(annual_global.Year,
                    annual_global.mean_tone - annual_global.sd_tone,
                    annual_global.mean_tone + annual_global.sd_tone,
                    color=C_BLACK, alpha=0.06)
    ax.axhline(0, color=C_GRAY, lw=0.4)
    ax.axvspan(2012.5, 2015.5, color=C_LGRAY, alpha=0.3, zorder=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean tone')
    ax.set_xlim(1978, 2027)
    plt.tight_layout()
    plt.savefig(OUT + 'figS1_tone_timeseries.png', dpi=300, bbox_inches='tight')
    print('✓ figS1')
    plt.close()

figS1()

# S2: Country β time series (all countries)
def figS2():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors_c = {'VE': C_RED, 'CO': C_ORANGE, 'US': C_BLUE, 'BR': C_GREEN, 'MX': C_PURPLE, 'UK': '#CCB974'}
    for cc in ['VE', 'CO', 'US', 'BR', 'MX', 'UK']:
        betas = []
        for yr in range(1979, 2027):
            sub = country_scatter[(country_scatter.country == cc) & (country_scatter.Year == yr)]
            if len(sub) > 0:
                b = weighted_ols(sub)
                betas.append({'Year': yr, 'beta': b})
        bdf = pd.DataFrame(betas).dropna()
        ax.plot(bdf.Year, bdf.beta, color=colors_c[cc], lw=0.8, alpha=0.8, label=LABELS[cc])
    ax.plot(global_betas.Year, global_betas.beta, color=C_BLACK, lw=1.2, ls='--', alpha=0.5, label='Global')
    ax.axvspan(2012.5, 2015.5, color=C_LGRAY, alpha=0.3, zorder=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('β')
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim(1978, 2027)
    plt.tight_layout()
    plt.savefig(OUT + 'figS2_country_beta.png', dpi=300, bbox_inches='tight')
    print('✓ figS2')
    plt.close()

figS2()

# S3: Topic elasticity
def figS3():
    df = topic_annual.copy()
    df['code_str'] = df.topic_code.astype(str).str.zfill(2)
    coop = df[df.code_str.isin(['03','04','05','06','07'])]
    conf = df[df.code_str.isin(['13','14','17','18','19','20'])]

    fig, ax = plt.subplots(figsize=(5.5, 3))
    for label, sub, color in [('Cooperative', coop, C_BLUE), ('Conflictual', conf, C_RED)]:
        avg = sub.groupby('Year')['r_tone_gold'].mean()
        ax.plot(avg.index, avg.values, color=color, lw=1.2, label=label)
    ax.axvspan(2012.5, 2015.5, color=C_LGRAY, alpha=0.3, zorder=0)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean r (tone × Goldstein)')
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT + 'figS3_topic.png', dpi=300, bbox_inches='tight')
    print('✓ figS3')
    plt.close()

figS3()

# S4: TDA (rerun with Nature style)
def figS4():
    from ripser import ripser
    from persim import plot_diagrams
    from sklearn.preprocessing import StandardScaler

    def sample_cloud(df, n=3000):
        df = df[(df.tone_bin >= -15) & (df.tone_bin <= 20) & (df.goldstein_bin >= -10.5) & (df.goldstein_bin <= 10.5)].copy()
        df['lw'] = np.log10(df['n']+1)
        p = df.lw / df.lw.sum()
        idx = np.random.choice(len(df), size=min(n, len(df)), replace=True, p=p)
        return StandardScaler().fit_transform(df.iloc[idx][['goldstein_bin','tone_bin']].values)

    np.random.seed(42)
    pre = sample_cloud(global_scatter[global_scatter.Year <= 2011])
    post = sample_cloud(global_scatter[global_scatter.Year >= 2016])

    r_pre = ripser(pre, maxdim=1)
    r_post = ripser(post, maxdim=1)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    for ax, result, title in [(axes[0], r_pre, '1979–2011'), (axes[1], r_post, '2016–2026')]:
        h0 = result['dgms'][0]
        h1 = result['dgms'][1]
        h0f = h0[np.isfinite(h0[:,1])]
        ax.scatter(h0f[:,0], h0f[:,1], s=8, c=C_BLUE, alpha=0.5, label='H₀', zorder=2)
        if len(h1) > 0:
            ax.scatter(h1[:,0], h1[:,1], s=8, c=C_ORANGE, alpha=0.5, label='H₁', zorder=2)
        lim = max(h0f[:,1].max(), h1[:,1].max() if len(h1)>0 else 0) * 1.1
        ax.plot([0, lim], [0, lim], color=C_LGRAY, lw=0.5, ls='--', zorder=1)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Birth', fontsize=7)
        ax.set_ylabel('Death', fontsize=7)
        ax.legend(fontsize=6, loc='lower right')
        ax.set_xlim(-0.02, lim)
        ax.set_ylim(-0.02, lim)
    plt.tight_layout()
    plt.savefig(OUT + 'figS4_tda.png', dpi=300, bbox_inches='tight')
    print('✓ figS4')
    plt.close()

figS4()

# S5: Venezuela deep dive
def figS5():
    ve = country_annual[country_annual.country == 'VE'].sort_values('Year')
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(annual_global.Year, annual_global.mean_tone, color=C_LGRAY, lw=1.5, label='Global')
    ax.plot(ve.Year, ve.mean_tone, color=C_RED, lw=1.5, label='Venezuela')
    ax.fill_between(ve.Year, ve.mean_tone - ve.sd_tone, ve.mean_tone + ve.sd_tone,
                    color=C_RED, alpha=0.06)
    ax.axhline(0, color=C_GRAY, lw=0.3)

    events = [(1999, 'Chávez'), (2002, 'Coup attempt'), (2013, 'Maduro'),
              (2017, 'Const. Assembly'), (2019, 'Guaidó'), (2024, 'Disputed\nelection')]
    for yr, txt in events:
        if yr in ve.Year.values:
            tone = ve[ve.Year == yr].mean_tone.values[0]
            ax.annotate(txt, xy=(yr, tone), xytext=(0, -15 if yr % 2 == 0 else 12),
                       textcoords='offset points', fontsize=5, ha='center', color=C_RED,
                       arrowprops=dict(arrowstyle='-', color=C_RED, lw=0.3))

    ax.set_xlabel('Year')
    ax.set_ylabel('Mean tone')
    ax.legend(fontsize=6)
    ax.set_xlim(1978, 2027)
    plt.tight_layout()
    plt.savefig(OUT + 'figS5_venezuela.png', dpi=300, bbox_inches='tight')
    print('✓ figS5')
    plt.close()

figS5()

print('\nAll publication figures complete.')
