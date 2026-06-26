"""
FILE: contextualization_consensus_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Compares managers' and developers' agreement on whether 
    "Productivity metrics must be contextualized." This script:
    1. Filters respondents into Managers and Developers
    2. Maps Likert responses to numeric scores (1-5)
    3. Calculates descriptive statistics and percentages
    4. Performs Mann-Whitney U test comparing managers vs. developers
    5. Generates a two-panel figure:
       - Left: Likert distribution bar chart
       - Right: Boxplot with jittered points and MWU annotation

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/analysis6_contextualization_consensus.png - Two-panel figure: distribution + boxplot

USAGE:
    python3 code/contextualization_consensus_analysis.py

NOTES:
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses from scoring
    - Mann-Whitney U test (two-sided)
    - Rank-biserial correlation for effect size
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

LIKERT_ORDER = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
LIKERT_SCORE = {v: i + 1 for i, v in enumerate(LIKERT_ORDER)}

# COLUMN DEFINITIONS
C = {
    "mgr_ctx": " [Productivity metrics must be contextualized.]",
    "dev_ctx": " [Productivity metrics must be contextualized.].1",
}

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()
N_MGR, N_DEV = len(mgr), len(dev)

# ADDITIONAL FUNCTIONS
def score(subset, key, exclude_dna=True):
    #Convert Likert responses to numeric scores, excluding DNA
    s = subset[C[key]].map(LIKERT_SCORE)
    if exclude_dna:
        s = s[subset[C[key]] != "Does Not Apply"]
    return s.dropna()

def pct_agree(subset, key):
    #Calculate percentage of Agree or Strongly Agree responses
    vals = subset[C[key]].dropna()
    vals = vals[vals != "Does Not Apply"]
    return (vals.isin(["Agree", "Strongly Agree"])).mean() * 100

def dist(subset, key):
    #Calculate percentage distribution across Likert levels
    vals = subset[C[key]]
    vals = vals[vals != "Does Not Apply"].dropna()
    n = len(vals)
    return {l: (vals == l).sum() / n * 100 for l in LIKERT_ORDER}

def mwu(s1, s2):
    #Mann-Whitney U test with rank-biserial correlation
    u, p = mannwhitneyu(s1, s2, alternative="two-sided")
    rb = 1 - (2 * u) / (len(s1) * len(s2))
    return u, p, rb

def sig(p):
    #Convert p-value to significance stars
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def annotate(ax, u, p, rb, pos=(0.5, 0.97)):
    #Add annotation box with test statistics
    txt = f"U={u:.0f}, p={p:.3f} ({sig(p)})\nr={rb:.3f}"
    ax.text(*pos, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF9C3",
                      edgecolor="#CA8A04"))

def likert_bar(ax, data_dict, title, colors_list, labels_list, ylabel="%"):
    #Create grouped Likert bar chart
    x = np.arange(len(LIKERT_ORDER))
    w = 0.35
    for i, (vals, color, label) in enumerate(zip(data_dict, colors_list, labels_list)):
        offset = (i - (len(data_dict) - 1) / 2) * w
        bars = ax.bar(x + offset, [vals.get(l, 0) for l in LIKERT_ORDER],
                      w, color=color, alpha=0.85, label=label, edgecolor="white")
        for bar in bars:
            v = bar.get_height()
            if v > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n") for l in LIKERT_ORDER], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

def boxplot_pair(ax, s1, s2, labels, colors, title, u=None, p=None, rb=None):
    #Create paired boxplot with jittered points
    bp = ax.boxplot([s1.values, s2.values], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="gray",
                                    markersize=4, alpha=0.5))
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.8)
    for i, data in enumerate([s1, s2]):
        ax.scatter(np.random.normal(i + 1, 0.06, len(data)),
                   data, alpha=0.35, s=18, color="gray", zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["SD", "D", "N", "A", "SA"], fontsize=8)
    ax.set_ylim(0.3, 5.9)
    ax.spines[["top", "right"]].set_visible(False)
    if u is not None:
        annotate(ax, u, p, rb)


# CONSOLE
print("=" * 65)
print("CONTEXTUALIZATION CONSENSUS — Output 6")
print("=" * 65)
print(f"\nManagers:   {N_MGR}")
print(f"Developers: {N_DEV}")

# Calculate scores and test
s_mgr_c = score(mgr, "mgr_ctx")
s_dev_c = score(dev, "dev_ctx")
u4, p4, r4 = mwu(s_mgr_c, s_dev_c)

print(f"\nSpearman correlation (all respondents):")
print(f"  Managers (n={len(s_mgr_c)}): mean={s_mgr_c.mean():.2f}, median={s_mgr_c.median():.1f}")
print(f"  Developers (n={len(s_dev_c)}): mean={s_dev_c.mean():.2f}, median={s_dev_c.median():.1f}")

print(f"\nMann-Whitney U test:")
print(f"  U = {u4:.0f}, p = {p4:.4f} ({sig(p4)}), r = {r4:.3f}")

print(f"\n% Agree or Strongly Agree:")
print(f"  Managers:   {pct_agree(mgr, 'mgr_ctx'):.1f}%")
print(f"  Developers: {pct_agree(dev, 'dev_ctx'):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE — Contextualization consensus
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output] Generating contextualization consensus figure …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: distribution
ax = axes[0]
dist_mc = dist(mgr, "mgr_ctx")
dist_dc = dist(dev, "dev_ctx")
likert_bar(ax, [dist_mc, dist_dc],
           '"Productivity metrics must be contextualized"',
           [MGR_COLOR, DEV_COLOR],
           [f"Managers (n={N_MGR})", f"Developers (n={N_DEV})"])

# Right: boxplot
boxplot_pair(axes[1], s_mgr_c, s_dev_c,
             [f"Managers\n(n={len(s_mgr_c)})", f"Developers\n(n={len(s_dev_c)})"],
             [MGR_COLOR, DEV_COLOR],
             f"MWU: U={u4:.0f}, p={p4:.3f} ({sig(p4)}), r={r4:.3f}",
             u4, p4, r4)
axes[1].set_ylabel("Agreement level")

plt.tight_layout()
path = OUT_DIR + "analysis6_contextualization_consensus.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Done.")