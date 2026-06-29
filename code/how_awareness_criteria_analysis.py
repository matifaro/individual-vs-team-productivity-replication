"""
FILE: how_awareness_criteria_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Tests whether knowing HOW your manager assesses productivity (the criteria 
    and methods used) predicts optimisation toward those criteria — a distinct 
    mechanism from knowing the purpose (WHY). This script:
    1. Filters respondents to Developers only
    2. Maps Likert responses to numeric scores (1-5)
    3. Calculates Spearman correlations for:
       - HOW awareness → criteria improvement
       - WHY awareness → criteria improvement (comparison)
       - Assessed at all → criteria improvement (comparison)
    4. Generates a two-panel scatter plot comparing HOW vs. WHY awareness
    5. Prints detailed statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/how_gaming_output1_scatter.png - Two-panel scatter plot: 
                                                 HOW vs. WHY awareness → criteria improvement

USAGE:
    python3 code/how_awareness_criteria_analysis.py

NOTES:
    - Developers only (managers are filtered out)
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses
    - Spearman correlation (two-sided)
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
DEVELOPER = "I develop software under a tech lead or manager role."
MANAGER   = "I manage other software development professionals."
DEV_COLOR = "#DC2626"
MGR_COLOR = "#2563EB"

LIKERT_SCORE = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}

# LOAD
df  = pd.read_csv(DATA_PATH)
dev = df[df[ROLE_COL] == DEVELOPER].copy()
N_DEV = len(dev)

# COLUMNS
HOW_COL      = " [I am aware of how my manager assesses productivity.]"
WHY_COL      = " [I am aware of why my productivity is being assessed.]"
ASSESSED_COL = " [My individual productivity is being assessed in my team.]"
CRITERIA_COL = " [I try to improve my performance in the criteria my manager uses for productivity assessment. ]"

# ADDITIONAL FUNCTIONS
def score(series):
    s = series.map(LIKERT_SCORE)
    s = s[series != "Does Not Apply"]
    return s.dropna()

def spear(x, y):
    common = x.index.intersection(y.index)
    xc, yc = x.loc[common], y.loc[common]
    r, p = spearmanr(xc, yc)
    return r, p, len(xc)

def sig_label(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

# SCORES
dev_how      = score(dev[HOW_COL])
dev_why      = score(dev[WHY_COL])
dev_assessed = score(dev[ASSESSED_COL])
dev_criteria = score(dev[CRITERIA_COL])

r_how, p_how, n_how = spear(dev_how,      dev_criteria)
r_why, p_why, n_why = spear(dev_why,      dev_criteria)
r_ass, p_ass, n_ass = spear(dev_assessed, dev_criteria)

# CONSOLE
print("=" * 65)
print(f"HOW AWARENESS → CRITERIA IMPROVEMENT  |  Developers n={N_DEV}")
print("=" * 65)
print(f"\n  HOW assessed → criteria:        r={r_how:.3f}, p={p_how:.4f} "
      f"{sig_label(p_how)}  (n={n_how})")
print(f"  WHY assessed → criteria:        r={r_why:.3f}, p={p_why:.4f} "
      f"{sig_label(p_why)}  (n={n_why})")
print(f"  ASSESSED AT ALL → criteria:     r={r_ass:.3f}, p={p_ass:.4f} "
      f"{sig_label(p_ass)}  (n={n_ass})")

print("\n  Distribution (HOW awareness × criteria improvement):")
for level in ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]:
    subset = dev[(dev[HOW_COL] == level) & (dev[CRITERIA_COL] != "Does Not Apply")]
    crit = subset[CRITERIA_COL].map(LIKERT_SCORE).dropna()
    if len(crit) > 0:
        print(f"    HOW={level:<20} → criteria mean={crit.mean():.2f}  (n={len(crit)})")

dna_how  = (dev[HOW_COL]      == "Does Not Apply").sum()
dna_crit = (dev[CRITERIA_COL] == "Does Not Apply").sum()
print(f"\n  Does Not Apply — HOW col: {dna_how}, CRITERIA col: {dna_crit}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 1 — Scatter: HOW → criteria 
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output 1] HOW → criteria scatter …")

common_h = dev_how.index.intersection(dev_criteria.index)
xh = dev_how.loc[common_h]
yh = dev_criteria.loc[common_h]

common_w = dev_why.index.intersection(dev_criteria.index)
xw = dev_why.loc[common_w]
yw = dev_criteria.loc[common_w]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

jitter = 0.12
tick_labels = ["SD", "D", "N", "A", "SA"]

for ax, x_vals, y_vals, r_val, p_val, n, xlabel, color in [
    (axes[0], xh, yh, r_how, p_how, n_how,
     '"I am aware of HOW my manager assesses productivity"', DEV_COLOR),
    (axes[1], xw, yw, r_why, p_why, n_why,
     '"I am aware of WHY my productivity is being assessed"', "#DC7633"),
]:
    ax.scatter(x_vals + np.random.uniform(-jitter, jitter, len(x_vals)),
               y_vals + np.random.uniform(-jitter, jitter, len(y_vals)),
               alpha=0.65, s=70, color=color, edgecolors="white", linewidth=0.5)
    z = np.polyfit(x_vals, y_vals, 1)
    ax.plot([1, 5], np.poly1d(z)([1, 5]), "k--", linewidth=2, alpha=0.7,
            label=f"Trend (slope={z[0]:.2f})")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(tick_labels, fontsize=13)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(tick_labels, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('"I try to improve on manager\'s criteria"', fontsize=14)
    ax.set_title(f"Spearman r = {r_val:.3f}, p = {p_val:.4f} ({sig_label(p_val)}, n={n})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

axes[0].set_title(axes[0].get_title(), color=DEV_COLOR)
axes[1].set_title(axes[1].get_title(), color="#DC7633")

plt.tight_layout()
path = OUT_DIR + "how_gaming_output1_scatter.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Done.")