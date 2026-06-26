"""
FILE: individual_team_assessment_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Tests whether developer awareness that their INDIVIDUAL productivity is being 
    assessed, and separately whether their TEAM productivity is being assessed, 
    predicts the tendency to optimise behaviour toward the manager's assessment 
    criteria (Goodhart's Law dynamic). This script:
    1. Filters respondents to Developers only
    2. Maps Likert responses to numeric scores (1-5)
    3. Calculates Spearman correlations between assessment awareness and criteria optimization
    4. Generates two scatter plots side-by-side showing:
       - Individual assessment awareness vs. criteria optimization
       - Team assessment awareness vs. criteria optimization
    5. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/indteam_gaming_scatter.png - Two-panel scatter plot showing 
                                                     correlations between assessment awareness 
                                                     and criteria optimization

USAGE:
    python3 code/individual_team_assessment_analysis.py

NOTES:
    - Developers only (managers are filtered out)
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses
    - Spearman correlation (two-sided), no correction applied
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

ROLE_COL  = "Which description best fits your role?"
DEVELOPER = "I develop software under a tech lead or manager role."
DEV_COLOR_IND  = "#DC2626"
DEV_COLOR_TEAM = "#2563EB"

LIKERT_SCORE = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
dev = df[df[ROLE_COL] == DEVELOPER].copy()
N_DEV = len(dev)

# COLUMNS
IND_ASSESSED_COL  = " [My individual productivity is being assessed in my team.]"
TEAM_ASSESSED_COL = " [Team productivity is being assessed in my team.]"
CRITERIA_COL      = " [I try to improve my performance in the criteria my manager uses for productivity assessment. ]"

# ADDITIONAL FUNCTIONS
def score(series):
    #Map Likert labels to 1-5, excluding 'Does Not Apply' and dropping NaN
    s = series.map(LIKERT_SCORE)
    s = s[series != "Does Not Apply"]
    return s.dropna()

def spear(x, y):
    #Spearman correlation on the common (aligned) index of two series
    common = x.index.intersection(y.index)
    xc, yc = x.loc[common], y.loc[common]
    r, p = spearmanr(xc, yc)
    return r, p, len(xc)

def significance_label(p):
    #Return significance descriptor for a p-value (uncorrected, two-sided)
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

# COMPUTE SCORES
dev_ind      = score(dev[IND_ASSESSED_COL])
dev_team     = score(dev[TEAM_ASSESSED_COL])
dev_criteria = score(dev[CRITERIA_COL])

r_ind,  p_ind,  n_ind  = spear(dev_ind,  dev_criteria)
r_team, p_team, n_team = spear(dev_team, dev_criteria)

# CONSOLE
print("=" * 65)
print(f"INDIVIDUAL/TEAM ASSESSMENT AWARENESS → CRITERIA OPTIMISATION")
print(f"Developers n={N_DEV}")
print("=" * 65)
print(f"\n  Individual productivity assessed → criteria improvement:")
print(f"    r={r_ind:.3f}, p={p_ind:.4f} ({significance_label(p_ind)}), n={n_ind}")
print(f"\n  Team productivity assessed → criteria improvement:")
print(f"    r={r_team:.3f}, p={p_team:.4f} ({significance_label(p_team)}), n={n_team}")

print("\n  Distribution — criteria improvement by individual assessment awareness:")
for level in ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]:
    subset = dev[(dev[IND_ASSESSED_COL] == level) & (dev[CRITERIA_COL] != "Does Not Apply")]
    crit = subset[CRITERIA_COL].map(LIKERT_SCORE).dropna()
    if len(crit) > 0:
        print(f"    IND={level:<20} → criteria mean={crit.mean():.2f}  (n={len(crit)})")

print("\n  Distribution — criteria improvement by team assessment awareness:")
for level in ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]:
    subset = dev[(dev[TEAM_ASSESSED_COL] == level) & (dev[CRITERIA_COL] != "Does Not Apply")]
    crit = subset[CRITERIA_COL].map(LIKERT_SCORE).dropna()
    if len(crit) > 0:
        print(f"    TEAM={level:<20} → criteria mean={crit.mean():.2f}  (n={len(crit)})")

dna_ind  = (dev[IND_ASSESSED_COL]  == "Does Not Apply").sum()
dna_team = (dev[TEAM_ASSESSED_COL] == "Does Not Apply").sum()
dna_crit = (dev[CRITERIA_COL]      == "Does Not Apply").sum()
print(f"\n  Does Not Apply excluded — Individual: {dna_ind}, Team: {dna_team}, Criteria: {dna_crit}")

# SCATTER PLOTS (single output, two panels)
print("\n[Output] Generating scatter plots …")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

jitter = 0.12
tick_labels = ["SD", "D", "N", "A", "SA"]

configs = [
    (axes[0], dev_ind, dev_criteria, r_ind, p_ind, n_ind,
     '"My individual productivity is being assessed in my team"', DEV_COLOR_IND),
    (axes[1], dev_team, dev_criteria, r_team, p_team, n_team,
     '"Team productivity is being assessed in my team"', DEV_COLOR_TEAM),
]

for ax, x_series, y_series, r_val, p_val, n, xlabel, color in configs:
    common = x_series.index.intersection(y_series.index)
    x_vals = x_series.loc[common]
    y_vals = y_series.loc[common]

    ax.scatter(x_vals + np.random.uniform(-jitter, jitter, len(x_vals)),
               y_vals + np.random.uniform(-jitter, jitter, len(y_vals)),
               alpha=0.65, s=60, color=color, edgecolors="white", linewidth=0.5)

    z = np.polyfit(x_vals, y_vals, 1)
    ax.plot([1, 5], np.poly1d(z)([1, 5]), "k--", linewidth=1.8, alpha=0.7,
            label=f"Trend (slope={z[0]:.2f})")

    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(tick_labels)
    ax.set_yticks([1, 2, 3, 4, 5]); ax.set_yticklabels(tick_labels)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('"I try to improve on manager\'s criteria"', fontsize=10)
    ax.set_title(f"Spearman r = {r_val:.3f}, p = {p_val:.4f} "
                 f"({significance_label(p_val)}, n={n})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

note = "* p<0.05  ** p<0.01  *** p<0.001 (Spearman correlation, two-sided)"
fig.text(0.98, 0.01, note, ha="right", fontsize=8, color="gray", style="italic")

plt.tight_layout()
path = OUT_DIR + "indteam_gaming_scatter.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  ✓ Saved → {path}")
plt.close()

print("\n✓ Analysis complete.")