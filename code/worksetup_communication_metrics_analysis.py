"""
FILE: worksetup_communication_metrics_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Examines whether a respondent's work setup (Hybrid / Remote / On-site)
    influences how they classify Communication & Collaboration metrics:
    
    - USE: "Do you use C&C metrics to assess productivity?" (Yes/No — managers only)
    - USEFULNESS: "How useful are C&C metrics?" (5-point Likert, all respondents)
    
    This script:
    1. Filters respondents by role (Managers for USE, all for USEFULNESS)
    2. Calculates descriptive statistics per setup group
    3. Performs Fisher's Exact Test for USE (2×3 binary × 3 groups → pairwise)
    4. Performs Kruskal-Wallis + pairwise Mann-Whitney U for USEFULNESS
    5. Generates a heatmap showing usefulness distribution by setup × role

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/worksetup_fig3_heatmap.png - Heatmap: usefulness by setup × role

USAGE:
    python3 code/worksetup_communication_metrics_analysis.py

NOTES:
    - Work setup categories: Remote, Hybrid, On-site
    - USE analysis: Managers only (Fisher's Exact Test)
    - USEFULNESS analysis: All respondents (Kruskal-Wallis + MWU)
    - Usefulness scale: 1=Not useful at all ... 5=Very useful
    - Pairwise tests: no multiple testing correction
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kruskal, mannwhitneyu, fisher_exact
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
SETUP_COL = "What is the work set up for collaboration with your team members?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."

USE_COL  = ("For each metric category below, indicate which you use to assess "
            "productivity.  [Communication and Collaboration]")
UTIL_COL = ("For each metric category below, indicate whether you consider "
            "metrics in this category to be useful for assessing productivity. "
            " [Communication and Collaboration]")

UTIL_SCORE = {
    "Not useful at all": 1,
    "Not very useful":   2,
    "Neutral":           3,
    "Somewhat useful":   4,
    "Very useful":       5,
}

SETUP_ORDER  = ["Remote", "Hybrid", "On-site"]
SETUP_COLORS = {"Remote": "#7C3AED", "Hybrid": "#059669", "On-site": "#D97706"}

# LOAD
df = pd.read_csv(DATA_PATH)

mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()


# ADDITIONAL FUNCTIONS
def rank_biserial(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
    return 1 - (2 * u) / (n1 * n2)

def sig_label(p):
    #Convert p-value to significance stars (no correction)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(ns)"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE — USE (managers only)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SECTION 1 — USE of C&C Metrics (managers only)")
print("=" * 65)

mgr_use = mgr[[SETUP_COL, USE_COL]].dropna()
print(f"\nTotal managers with both fields: {len(mgr_use)}")
use_ct = pd.crosstab(mgr_use[SETUP_COL], mgr_use[USE_COL])
use_pct = use_ct.div(use_ct.sum(axis=1), axis=0) * 100
print("\nCounts:\n", use_ct.reindex(SETUP_ORDER, fill_value=0))
print("\n% Yes per setup:")
for s in SETUP_ORDER:
    if s in use_pct.index and "Yes" in use_pct.columns:
        print(f"  {s:10s}: {use_pct.loc[s,'Yes']:.1f}%  (n={int(use_ct.loc[s].sum())})")

# Pairwise Fisher's Exact (Yes vs No, two groups at a time) - NO Bonferroni
print("\nPairwise Fisher's Exact Tests (no correction):")
setups_present = [s for s in SETUP_ORDER if s in use_ct.index]
for s1, s2 in combinations(setups_present, 2):
    table = use_ct.reindex([s1, s2]).fillna(0)[["Yes", "No"]].values
    if table.shape == (2, 2):
        _, p = fisher_exact(table)
        print(f"  {s1} vs {s2}: p={p:.4f} {sig_label(p)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DESCRIPTIVE — USEFULNESS (all respondents)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("SECTION 2 — USEFULNESS of C&C Metrics (all respondents)")
print(f"{'=' * 65}")

df_util = df[[SETUP_COL, ROLE_COL, UTIL_COL]].dropna().copy()
df_util["score"] = df_util[UTIL_COL].map(UTIL_SCORE)

print(f"\nTotal respondents with both fields: {len(df_util)}")
print(f"\n{'Setup':<12} {'n':>4} {'Mean':>6} {'Median':>7} {'SD':>6}")
print("-" * 38)
groups = {}
for s in SETUP_ORDER:
    g = df_util[df_util[SETUP_COL] == s]["score"].dropna()
    groups[s] = g
    if len(g):
        print(f"{s:<12} {len(g):>4} {g.mean():>6.2f} {g.median():>7.1f} {g.std():>6.2f}")

# Kruskal-Wallis
valid_groups = [g for g in groups.values() if len(g) > 0]
if len(valid_groups) >= 2:
    h, p_kw = kruskal(*valid_groups)
    print(f"\nKruskal-Wallis H = {h:.3f}, p = {p_kw:.4f} {sig_label(p_kw)}")

# Pairwise Mann-Whitney U
print(f"\nPairwise Mann-Whitney U (no correction):")
print(f"  {'Pair':<22} {'U':>7} {'p':>8} {'sig':>5} {'r':>6}")
print("  " + "-" * 52)
for s1, s2 in combinations(SETUP_ORDER, 2):
    g1, g2 = groups.get(s1, pd.Series([])), groups.get(s2, pd.Series([]))
    if len(g1) > 0 and len(g2) > 0:
        u, p = mannwhitneyu(g1, g2, alternative="two-sided")
        rb = rank_biserial(g1, g2)
        print(f"  {s1+' vs '+s2:<22} {u:>7.1f} {p:>8.4f} "
              f"{sig_label(p):>5} {rb:>6.3f}")

# Distribution breakdown
print(f"\n{'─' * 65}")
print("USEFULNESS DISTRIBUTION BY SETUP (counts)")
print(f"{'─' * 65}")
order_labels = ["Not useful at all", "Not very useful", "Neutral",
                "Somewhat useful", "Very useful"]
for s in SETUP_ORDER:
    subset = df_util[df_util[SETUP_COL] == s]
    counts = {k: (subset[UTIL_COL] == k).sum() for k in order_labels}
    print(f"\n  {s} (n={len(subset)}): {counts}")

# By role & setup
print(f"\n{'─' * 65}")
print("USEFULNESS MEAN BY SETUP × ROLE")
print(f"{'─' * 65}")
print(f"\n{'Setup':<12} {'Role':<12} {'n':>4} {'Mean':>6} {'Median':>7}")
print("-" * 44)
for s in SETUP_ORDER:
    for role_lbl, role_val in [("Manager", MANAGER), ("Developer", DEVELOPER)]:
        g = df_util[(df_util[SETUP_COL] == s) & (df_util[ROLE_COL] == role_val)]["score"]
        if len(g):
            print(f"{s:<12} {role_lbl:<12} {len(g):>4} {g.mean():>6.2f} {g.median():>7.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Heat-map: usefulness by setup × role
# ══════════════════════════════════════════════════════════════════════════════
print("[Figure 3] Heatmap: usefulness by setup × role …")

roles       = [("Managers", MANAGER), ("Developers", DEVELOPER)]
grid_labels = [f"{s}\n{r}" for s in SETUP_ORDER for r, _ in roles]
grid_means  = []
grid_ns     = []
for s in SETUP_ORDER:
    for _, rv in roles:
        g = df_util[(df_util[SETUP_COL] == s) & (df_util[ROLE_COL] == rv)]["score"]
        grid_means.append(g.mean() if len(g) else np.nan)
        grid_ns.append(len(g))

matrix = np.array(grid_means).reshape(len(SETUP_ORDER), len(roles))
ns_mat = np.array(grid_ns).reshape(len(SETUP_ORDER), len(roles))

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
cbar = plt.colorbar(im, ax=ax, label="Mean Usefulness Score (1–5)")
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Mean Usefulness Score (1–5)', fontsize=13)

ax.set_xticks(range(len(roles)))
ax.set_xticklabels([r for r, _ in roles], fontsize=14)
ax.set_yticks(range(len(SETUP_ORDER)))
ax.set_yticklabels(SETUP_ORDER, fontsize=14)

for i in range(len(SETUP_ORDER)):
    for j in range(len(roles)):
        val = matrix[i, j]
        n   = ns_mat[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}\n(n={n})", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if val < 2.5 or val > 4.2 else "black")

ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
path = OUT_DIR + "worksetup_fig3_heatmap.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Done.")