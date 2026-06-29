"""
FILE: gaming_belief_criteria_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Tests whether developers who believe that providing productivity metrics 
    incentivizes gaming also try to improve on the manager's assessment criteria.
    This script:
    1. Maps Likert responses to numeric scores (1-5)
    2. Calculates Spearman correlation between gaming belief and criteria optimization
    3. Generates a scatter plot with trend line
    4. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/output1_right_panel_gaming_belief.png - Scatter plot: gaming belief vs. 
                                                 criteria optimization

USAGE:
    python3 code/gaming_belief_criteria_analysis.py

NOTES:
    - All respondents (not filtered by role)
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses
    - Spearman correlation (two-sided)
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

LIKERT_SCORE = {
    "Strongly Disagree":1, "Disagree":2, "Neutral":3,
    "Agree":4, "Strongly Agree":5,
}

# LOAD
df = pd.read_csv(DATA_PATH)

# COLUMN DEFINTIONS
DEV_GAMING_COL  = " [Providing productivity metrics explicitly incentivizes developers to  gamify metrics.].1"
CRITERIA_COL    = " [I try to improve my performance in the criteria my manager uses for productivity assessment. ]"

# ADDITIONAL FUNCTIONS
def score(series):
    s = series.map(LIKERT_SCORE)
    s = s[series != "Does Not Apply"]
    return s.dropna()

# SCORES
dev_gaming   = score(df[DEV_GAMING_COL])
dev_criteria = score(df[CRITERIA_COL])

common = dev_gaming.index.intersection(dev_criteria.index)
xg = dev_gaming.loc[common]
yg = dev_criteria.loc[common]

r, p = spearmanr(xg, yg)
z = np.polyfit(xg, yg, 1)

# CONSOLE
print("=" * 55)
print("GAMING BELIEF → CRITERIA IMPROVEMENT")
print("=" * 55)
print(f"  Spearman r = {r:.3f}, p = {p:.4f}  (n={len(xg)})")
print(f"  Regression slope = {z[0]:.3f}")
print(f"  Intercept = {z[1]:.3f}")

# PLOT
fig, ax = plt.subplots(figsize=(7.5, 6.5))

jitter = 0.12
tick_labels = ["SD", "D", "N", "A", "SA"]

ax.scatter(xg + np.random.uniform(-jitter, jitter, len(xg)),
           yg + np.random.uniform(-jitter, jitter, len(yg)),
           alpha=0.65, s=70, color="#DC7633", edgecolors="white", linewidth=0.5)

ax.plot([1, 5], np.poly1d(z)([1, 5]), "k--", linewidth=2, alpha=0.7,
        label=f"Trend (slope={z[0]:.2f})")

ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(tick_labels, fontsize=13)
ax.set_yticks([1,2,3,4,5])
ax.set_yticklabels(tick_labels, fontsize=13)
ax.set_xlabel('"Providing metrics incentivizes gaming" (developer view)', fontsize=15)
ax.set_ylabel('"I try to improve on manager\'s criteria"', fontsize=15)
ax.legend(fontsize=13)
ax.grid(True, alpha=0.2)
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
path = OUT_DIR + "output1_right_panel_gaming_belief.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n  Saved → {path}")
plt.close()

print("\n✓ Done.")