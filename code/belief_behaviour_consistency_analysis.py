"""
FILE: belief_behaviour_consistency_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes belief-behaviour consistency among managers regarding productivity 
    metrics. This script:
    1. Filters respondents to Managers only
    2. Examines whether managers who believe metrics must be contextualized 
       actually use contextualized categories (vs. Output & Efficiency)
    3. Tests if "productivity is hard to assess" correlates with breadth of metric use
    4. Generates two visualizations:
       - Stacked bar chart: Belief in contextualization vs. actual metric category use
       - Headline inconsistency chart: Belief in contextualization vs. use of Output & Efficiency

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/consist_output1_belief_vs_use.png - Stacked bar: belief × category use
    results/consist_output3_headline.png - Headline inconsistency chart

USAGE:
    python3 code/belief_behaviour_consistency_analysis.py

NOTES:
    - Managers only (developers are filtered out)
    - Belief columns analyzed:
      - [45] "Productivity metrics must be contextualized."
      - [48] "Productivity is hard to assess."
    - Metric use categories: cols [74]–[82] (Output & Efficiency, Quality, etc.)
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Output 1: Stacked bar by belief level (Neutral, Agree, Strongly Agree)
    - Output 3: Two-panel chart showing the key inconsistency finding
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL = "Which description best fits your role?"
MANAGER  = "I manage other software development professionals."
MGR_COLOR = "#2563EB"
LIKERT_SCORE = {"Strongly Disagree":1,"Disagree":2,"Neutral":3,"Agree":4,"Strongly Agree":5}
USE_PREFIX = "For each metric category below, indicate which you use to assess productivity.  ["

# LOAD MANAGERS ONLY
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
N   = len(mgr)
print(f"Managers: n={N}")

# COLUMN DEFINITION
CTX_COL    = " [Productivity metrics must be contextualized.]"
GAMING_COL = " [Providing productivity metrics explicitly incentivizes developers to  gamify metrics.]"
HARD_COL   = " [Productivity is hard to assess.]"
QUANT_COL  = " [Productivity assessment should be quantitative.]"

USE_COLS = {
    "Output & Efficiency":              f"{USE_PREFIX}Output &amp; Efficiency]",
    "Quality":                          f"{USE_PREFIX}Quality]",
    "Value & Outcomes":                 f"{USE_PREFIX}Value &amp; Outcomes]",
    "Satisfaction & Wellbeing":         f"{USE_PREFIX}Satisfaction and Wellbeing]",
    "Comm. & Collaboration":            f"{USE_PREFIX}Communication and Collaboration]",
    "Learning & Growth":                f"{USE_PREFIX}Learning &amp; Growth]",
    "Tech Debt & Maintainability":      f"{USE_PREFIX}Technical Debt &amp; Maintainability]",
    "Developer Experience":             f"{USE_PREFIX}Developer Experience]",
    "Process & Predictability":         f"{USE_PREFIX}Process &amp; Predictability]",
}

# Breadth = number of categories used
mgr["n_categories"] = sum(
    (mgr[col] == "Yes").astype(int) for col in USE_COLS.values()
)

# ADDITIONAL FUNCTIONS
def score(series):
    return series.map(LIKERT_SCORE).dropna()

def uses_pct(belief_col, use_col, belief_level):
    """% of managers at a belief level who use a given metric category."""
    subset = mgr[mgr[belief_col] == belief_level]
    if len(subset) == 0:
        return np.nan, 0
    return (subset[use_col] == "Yes").mean() * 100, len(subset)

def spear(belief_col, numeric_outcome):
    """Spearman r between a Likert belief and a numeric outcome."""
    b = mgr[belief_col].map(LIKERT_SCORE).dropna()
    o = numeric_outcome.reindex(b.index).dropna()
    b = b.reindex(o.index)
    if len(b) < 5:
        return np.nan, np.nan
    r, p = spearmanr(b, o)
    return r, p


# CONSOLE
print("\n" + "=" * 65)
print("BELIEF–BEHAVIOUR CONSISTENCY  (Managers only)")
print("=" * 65)

# Check A: contextualization belief × Output & Efficiency use
print("\n── A. 'Metrics must be contextualized' × Output & Efficiency use ──")
out_col = USE_COLS["Output & Efficiency"]
for level in ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]:
    pct, n_lev = uses_pct(CTX_COL, out_col, level)
    if n_lev > 0:
        print(f"  {level:<20} → {pct:>6.1f}% use Output & Efficiency  (n={n_lev})")

r_a, p_a = spear(CTX_COL, (mgr[out_col] == "Yes").astype(float))
print(f"  Spearman r = {r_a:.3f}, p = {p_a:.4f}")

# Check C: "hard to assess" × breadth
print("\n── C. 'Productivity is hard to assess' × breadth of metric use ──")
for level in ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]:
    subset = mgr[mgr[HARD_COL] == level]
    if len(subset) > 0:
        mean_n = subset["n_categories"].mean()
        print(f"  {level:<20} → mean categories used = {mean_n:.1f}  (n={len(subset)})")

r_c, p_c = spear(HARD_COL, mgr["n_categories"])
print(f"  Spearman r = {r_c:.3f}, p = {p_c:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 1 — Stacked bar: contextualization belief × use of each category
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output 1] Belief × category use stacked bars …")

# Group by belief level, compute % using each category
belief_order = ["Neutral", "Agree", "Strongly Agree"]  # only levels present
cat_names = list(USE_COLS.keys())
cat_cols  = list(USE_COLS.values())

# Build matrix: rows = belief levels, cols = categories
mat_ctx = np.zeros((len(belief_order), len(cat_names)))
ns_ctx  = []
for i, level in enumerate(belief_order):
    subset = mgr[mgr[CTX_COL] == level]
    ns_ctx.append(len(subset))
    for j, col in enumerate(cat_cols):
        mat_ctx[i, j] = (subset[col] == "Yes").mean() * 100 if len(subset) > 0 else 0

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(cat_names))
w = 0.25
colors_levels = ["#93C5FD", "#3B82F6", "#1D4ED8"]
for i, (level, color) in enumerate(zip(belief_order, colors_levels)):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, mat_ctx[i], w, label=f"{level} (n={ns_ctx[i]})",
                  color=color, alpha=0.85, edgecolor="white")

ax.axhline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
ax.set_xticks(x)
ax.set_xticklabels(cat_names, rotation=35, ha="right", fontsize=9)
ax.set_ylim(0, 115)
ax.set_ylabel('% managers using this category', fontsize=11)
ax.legend(title="Belief level", fontsize=9, title_fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
path = OUT_DIR + "consist_output1_belief_vs_use.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 3 — The headline inconsistency: contextualization vs Output use
# (single focused chart — most citable finding)
# ══════════════════════════════════════════════════════════════════════════════
print("[Output 3] Headline inconsistency chart …")

# For each manager: belief score + whether they use Output & Efficiency
mgr_plot = mgr[[CTX_COL, USE_COLS["Output & Efficiency"]]].copy()
mgr_plot["belief_score"] = mgr_plot[CTX_COL].map(LIKERT_SCORE)
mgr_plot["uses_output"]  = (mgr_plot[USE_COLS["Output & Efficiency"]] == "Yes").astype(int)
mgr_plot = mgr_plot.dropna(subset=["belief_score"])

# Summary: % using Output & Efficiency at each belief level
summary = mgr_plot.groupby("belief_score")["uses_output"].agg(["mean","count"]).reset_index()
summary["pct"] = summary["mean"] * 100
summary["label"] = summary["belief_score"].map({
    1:"SD", 2:"D", 3:"N", 4:"A", 5:"SA"
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: % using Output & Efficiency by belief level
ax = axes[0]
bar_colors = ["#BFDBFE","#93C5FD","#60A5FA","#3B82F6","#1D4ED8"]
bars = ax.bar(summary["label"], summary["pct"],
              color=[bar_colors[int(s)-1] for s in summary["belief_score"]],
              edgecolor="white", alpha=0.9, width=0.6)
for bar, (_, row) in zip(bars, summary.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{row['pct']:.0f}%\n(n={int(row['count'])})",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(100, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax.set_ylim(0, 120)
ax.set_xlabel('"Metrics must be contextualized" (SD=1 … SA=5)', fontsize=10)
ax.set_ylabel("% using Output & Efficiency", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# Right: same data for all 9 categories — % using by strong believers vs rest
ax2 = axes[1]
strong = mgr[mgr[CTX_COL].isin(["Agree", "Strongly Agree"])]
other  = mgr[mgr[CTX_COL].isin(["Neutral", "Disagree", "Strongly Disagree"])]

cat_pct_strong = [(strong[col] == "Yes").mean() * 100 for col in cat_cols]
cat_pct_other  = [(other[col]  == "Yes").mean() * 100 for col in cat_cols]

y2 = np.arange(len(cat_names))
w2 = 0.35
ax2.barh(y2 - w2/2, cat_pct_strong, w2, color="#1D4ED8", alpha=0.85,
         label=f"Agree/Strongly Agree (n={len(strong)})")
ax2.barh(y2 + w2/2, cat_pct_other,  w2, color="#93C5FD", alpha=0.85,
         label=f"Neutral or below (n={len(other)})")
ax2.set_yticks(y2)
ax2.set_yticklabels(cat_names, fontsize=8.5)
ax2.set_xlim(0, 115)
ax2.axvline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
ax2.set_xlabel("% using this category")
ax2.legend(fontsize=8.5, loc="lower right")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
path = OUT_DIR + "consist_output3_headline.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n" + "=" * 55)
print("KEY FINDINGS SUMMARY")
print("=" * 55)
strong_ctx = mgr[mgr[CTX_COL].isin(["Agree","Strongly Agree"])]
pct_strong_uses_output = (strong_ctx[USE_COLS["Output & Efficiency"]] == "Yes").mean() * 100
print(f"\n  Managers who Agree/Strongly Agree 'metrics must be contextualized': n={len(strong_ctx)}")
print(f"  Of those, % who still use Output & Efficiency: {pct_strong_uses_output:.1f}%")
print(f"\n  Contextualization belief × Output use:  r={r_a:.3f}, p={p_a:.4f}")
print(f"  'Hard to assess' × breadth of use:      r={r_c:.3f}, p={p_c:.4f}")
print("\n✓ Done.")