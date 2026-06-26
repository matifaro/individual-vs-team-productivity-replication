"""
FILE: metric_category_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes survey responses about metric categories used to assess productivity,
    comparing perspectives between managers and developers. This script generates
    three visualizations:
    
    Output 1 — Managers: what they USE vs what they find USEFUL
               (% using each category vs mean usefulness score)
    
    Output 2 — Managers vs Developers: perceived usefulness per category
               (Mann-Whitney U with significance, gap chart)
    
    Output 3 — Usefulness distribution heatmaps (full Likert breakdown)
    
    The script:
    1. Filters respondents into Managers and Developers
    2. Calculates usage percentages and usefulness scores for 9 metric categories
    3. Performs Mann-Whitney U tests comparing managers vs. developers
    4. Generates three complementary visualizations

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/metrics_output1_use_vs_useful.png - Dual-axis chart: manager use vs usefulness
    results/metrics_output2_mgr_vs_dev_usefulm.png - Gap chart: manager vs developer usefulness
    results/metrics_output3_useful_heatmap.png - Heatmaps: usefulness distribution by role

USAGE:
    python3 code/metric_category_analysis.py

NOTES:
    - Metric categories analyzed:
      Satisfaction & Wellbeing, Communication & Collaboration, Output & Efficiency,
      Quality, Value & Outcomes, Learning & Growth, Technical Debt & Maintainability,
      Developer Experience, Process & Predictability
    - Usefulness scale: 1=Not useful at all ... 5=Very useful
    - Mann-Whitney U test (two-sided), no multiple testing correction
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."
MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

USEFUL_ORDER = ["Not useful at all", "Not very useful", "Neutral",
                "Somewhat useful", "Very useful"]
USEFUL_SCORE = {v: i + 1 for i, v in enumerate(USEFUL_ORDER)}

# CATEGORIES
CATEGORIES = [
    "Satisfaction and Wellbeing",
    "Communication and Collaboration",
    "Output & Efficiency",
    "Quality",
    "Value & Outcomes",
    "Learning & Growth",
    "Technical Debt & Maintainability",
    "Developer Experience",
    "Process & Predictability",
]
# HTML-encoded names as they appear in columns
CAT_RAW = [c.replace("&", "&amp;") for c in CATEGORIES]

USE_PREFIX    = "For each metric category below, indicate which you use to assess productivity.  ["
USEFUL_PREFIX = "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  ["

def use_col(cat):    return f"{USE_PREFIX}{cat.replace('&', '&amp;')}]"
def useful_col(cat): return f"{USEFUL_PREFIX}{cat.replace('&', '&amp;')}]"

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

N_MGR = len(mgr); N_DEV = len(dev); N_ALL = len(df)
print(f"Total: {N_ALL}  |  Managers: {N_MGR}  |  Developers: {N_DEV}")

# ADDITIONAL FUNCTIONS
def use_pct(subset, cat):
    return (subset[use_col(cat)] == "Yes").mean() * 100

def useful_score(subset, cat):
    return subset[useful_col(cat)].map(USEFUL_SCORE).dropna()

def mean_useful(subset, cat):
    return useful_score(subset, cat).mean()

def pct_positive_useful(subset, cat):
    vals = subset[useful_col(cat)].dropna()
    return (vals.isin(["Somewhat useful", "Very useful"])).mean() * 100

def rank_biserial(g1, g2):
    u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
    return 1 - (2 * u) / (len(g1) * len(g2))

def mwu(cat):
    m = useful_score(mgr, cat); d = useful_score(dev, cat)
    u, p = mannwhitneyu(m, d, alternative="two-sided")
    rb   = rank_biserial(m, d)
    return u, p, rb

def sig(p):
    #Convert p-value to significance stars (no correction)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(ns)"


# CONSOLE
print("\n" + "=" * 75)
print("METRIC CATEGORY SUMMARY")
print("=" * 75)
print(f"\n{'Category':<35} {'Use%':>6} {'MgrUtil':>8} {'DevUtil':>8} {'Gap':>6} "
      f"{'p':>8} {'sig':>5} {'r':>6}")
print("-" * 82)

stats = []
for cat in CATEGORIES:
    u_pct  = use_pct(mgr, cat)
    m_util = mean_useful(mgr, cat)
    d_util = mean_useful(dev, cat)
    gap    = m_util - d_util
    u, p, rb = mwu(cat)
    s      = sig(p)
    print(f"{cat:<35} {u_pct:>5.1f}% {m_util:>8.2f} {d_util:>8.2f} "
          f"{gap:>+6.2f} {p:>8.4f} {s:>5} {rb:>6.3f}")
    stats.append(dict(cat=cat, use_pct=u_pct, mgr_util=m_util, dev_util=d_util,
                      gap=gap, U=u, p=p, sig=s, r=rb))

print("\nSignificance: * p<0.05  ** p<0.01  *** p<0.001  (ns) = not significant")
print("Usefulness scale: 1=Not useful at all … 5=Very useful")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 1 — Managers: USE rate vs USEFULNESS (dual-axis bar + scatter)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output 1] Use vs Usefulness (managers) …")

# Sort by use rate
s1 = sorted(stats, key=lambda x: x["use_pct"], reverse=True)
cats_s   = [r["cat"] for r in s1]
use_pcts = [r["use_pct"] for r in s1]
util_mgr = [r["mgr_util"] for r in s1]

fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(cats_s))
w = 0.45

bars = ax1.bar(x, use_pcts, w, color=MGR_COLOR, alpha=0.8,
               label="% Managers using", zorder=2)
for bar, v in zip(bars, use_pcts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{v:.0f}%", ha="center", va="bottom", fontsize=8.5, color=MGR_COLOR,
             fontweight="bold")

ax1.set_ylabel("% Managers using this category", color=MGR_COLOR, fontsize=11)
ax1.set_ylim(0, 120)
ax1.tick_params(axis="y", labelcolor=MGR_COLOR)
ax1.set_xticks(x)
ax1.set_xticklabels(cats_s, rotation=30, ha="right", fontsize=9)
ax1.axhline(50, color=MGR_COLOR, linestyle="--", linewidth=0.7, alpha=0.4)
ax1.spines[["top"]].set_visible(False)

ax2 = ax1.twinx()
ax2.plot(x, util_mgr, color="darkorange", marker="D", linewidth=2,
         markersize=7, label="Mean usefulness (managers)", zorder=3)
for xi, v in zip(x, util_mgr):
    ax2.text(xi + 0.05, v + 0.07, f"{v:.1f}", fontsize=8, color="darkorange",
             fontweight="bold")
ax2.set_ylabel("Mean usefulness score (1–5)", color="darkorange", fontsize=11)
ax2.set_ylim(1, 6)
ax2.set_yticks([1, 2, 3, 4, 5])
ax2.set_yticklabels(["Not useful\nat all", "Not very\nuseful", "Neutral",
                      "Somewhat\nuseful", "Very\nuseful"], fontsize=8)
ax2.tick_params(axis="y", labelcolor="darkorange")
ax2.spines[["top"]].set_visible(False)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

plt.tight_layout()
path = OUT_DIR + "metrics_output1_use_vs_useful.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 2 — Managers vs Developers: usefulness comparison
# ══════════════════════════════════════════════════════════════════════════════
print("[Output 2] Managers vs Developers usefulness …")

# Sort by overall mean usefulness
s2 = sorted(stats, key=lambda x: (x["mgr_util"] + x["dev_util"]) / 2, reverse=True)

fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                         gridspec_kw={"width_ratios": [3, 1]})

# Left — grouped bars (% Somewhat/Very useful)
ax = axes[0]
y  = np.arange(len(s2))
w  = 0.35
mgr_pos = [pct_positive_useful(mgr, r["cat"]) for r in s2]
dev_pos = [pct_positive_useful(dev, r["cat"]) for r in s2]

ax.barh(y - w/2, mgr_pos, w, color=MGR_COLOR, alpha=0.85,
        label=f"Managers (n={N_MGR})")
ax.barh(y + w/2, dev_pos, w, color=DEV_COLOR, alpha=0.85,
        label=f"Developers (n={N_DEV})")

for i, (m, d, r) in enumerate(zip(mgr_pos, dev_pos, s2)):
    s = r["sig"]
    if s != "(ns)":
        ax.text(max(m, d) + 1.5, i, s, va="center", fontsize=10,
                color="darkred", fontweight="bold")
    # Connect with line
    ax.plot([m, d], [i - w/2, i + w/2], color="black",
            linewidth=1, alpha=0.35, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([r["cat"] for r in s2], fontsize=9)
ax.set_xlim(0, 115)
ax.set_xlabel('% "Somewhat useful" or "Very useful"')
ax.axvline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# Right — mean score gap
ax2 = axes[1]
gaps  = [r["gap"] for r in s2]
gcols = [MGR_COLOR if g > 0 else DEV_COLOR for g in gaps]
bars  = ax2.barh(y, gaps, color=gcols, alpha=0.75, edgecolor="white")
ax2.axvline(0, color="black", linewidth=0.8)
for bar, val, r in zip(bars, gaps, s2):
    s = r["sig"]
    xpos = val + 0.02 if val >= 0 else val - 0.02
    ha   = "left" if val >= 0 else "right"
    label_txt = f"{val:+.2f}"
    if s != "(ns)":
        label_txt += f" {s}"
    ax2.text(xpos, bar.get_y() + bar.get_height()/2,
             label_txt, va="center", ha=ha, fontsize=8)
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel("Mean score gap\n(Mgr − Dev)")
ax2.spines[["top", "right"]].set_visible(False)

# Add note about significance (no Bonferroni)
note = "* p<0.05  ** p<0.01  *** p<0.001 "
fig.text(0.98, 0.01, note, ha="right", fontsize=7.5, color="gray", style="italic")
plt.tight_layout()
path = OUT_DIR + "metrics_output2_mgr_vs_dev_useful.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 3 — Full Likert distribution heatmaps
# (managers: top half = use+useful; developers: bottom = useful only)
# ══════════════════════════════════════════════════════════════════════════════
print("[Output 3] Usefulness distribution heatmaps …")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, subset, role, color in [
    (axes[0], mgr, f"Managers (n={N_MGR})", MGR_COLOR),
    (axes[1], dev, f"Developers (n={N_DEV})", DEV_COLOR),
]:
    # Sort categories by % Very useful
    sorted_cats = sorted(CATEGORIES,
                         key=lambda c: (subset[useful_col(c)] == "Very useful").mean(),
                         reverse=True)
    mat = np.array([
        [(subset[useful_col(c)] == lvl).mean() * 100 for lvl in USEFUL_ORDER]
        for c in sorted_cats
    ])
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=70)
    ax.set_xticks(range(5))
    ax.set_xticklabels([l.replace(" ", "\n") for l in USEFUL_ORDER], fontsize=8.5)
    ax.set_yticks(range(len(sorted_cats)))
    ax.set_yticklabels(sorted_cats, fontsize=9)
    ax.set_title(role, fontweight="bold", fontsize=11)
    for i in range(len(sorted_cats)):
        for j in range(5):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    color="white" if v > 50 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="%", shrink=0.85)

plt.tight_layout()
path = OUT_DIR + "metrics_output3_useful_heatmap.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Analysis complete.")