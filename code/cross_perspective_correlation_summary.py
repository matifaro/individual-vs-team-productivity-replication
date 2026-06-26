"""
FILE: cross_perspective_correlation_summary.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Creates a horizontal bar chart summarizing all correlations in the 
    awareness-gaming chain across developer and manager perspectives. This script:
    1. Calculates Spearman correlations for the developer-side chain:
       - Awareness of assessment → criteria improvement
       - Awareness of why assessed → criteria improvement
       - Gaming belief → criteria improvement
       - Fairness perception → criteria improvement
       - Misuse perception → criteria improvement
    2. Calculates manager-side correlation:
       - Gaming belief → breadth of metric use
    3. Generates a horizontal bar chart comparing all correlations
    4. Prints detailed statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/gaming_output4_summary.png - Horizontal bar chart: 
                                                        all correlations in the gaming chain

USAGE:
    python3 code/cross_perspective_correlation_summary.py

NOTES:
    - Developers and Managers (split by role)
    - Spearman correlations (two-sided)
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Excludes "Does Not Apply" responses
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
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."
MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

LIKERT_SCORE = {
    "Strongly Disagree":1, "Disagree":2, "Neutral":3,
    "Agree":4, "Strongly Agree":5,
}
USE_PREFIX = "For each metric category below, indicate which you use to assess productivity.  ["

# LOAD AND
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()
N_MGR = len(mgr); N_DEV = len(dev)

# COLUMN DEFINITIONS
# Developer-side
AWARE_ASSESSED_COL = " [My individual productivity is being assessed in my team.]"
AWARE_WHY_COL      = " [I am aware of why my productivity is being assessed.]"
CRITERIA_COL       = " [I try to improve my performance in the criteria my manager uses for productivity assessment. ]"
DEV_GAMING_COL     = " [Providing productivity metrics explicitly incentivizes developers to  gamify metrics.].1"
FAIR_COL           = " [I consider my manager\u2019s assessment of my productivity fair.]"
MISUSE_COL         = " [My manager misuses productivity metrics.]"

# Manager-side
MGR_GAMING_COL     = " [Providing productivity metrics explicitly incentivizes developers to  gamify metrics.]"

USE_COLS = {
    "Output & Efficiency":         f"{USE_PREFIX}Output &amp; Efficiency]",
    "Quality":                     f"{USE_PREFIX}Quality]",
    "Value & Outcomes":            f"{USE_PREFIX}Value &amp; Outcomes]",
    "Satisfaction & Wellbeing":    f"{USE_PREFIX}Satisfaction and Wellbeing]",
    "Comm. & Collaboration":       f"{USE_PREFIX}Communication and Collaboration]",
    "Learning & Growth":           f"{USE_PREFIX}Learning &amp; Growth]",
    "Tech Debt":                   f"{USE_PREFIX}Technical Debt &amp; Maintainability]",
    "Developer Experience":        f"{USE_PREFIX}Developer Experience]",
    "Process & Predictability":    f"{USE_PREFIX}Process &amp; Predictability]",
}

mgr["n_categories"] = sum(
    (mgr[col] == "Yes").astype(int) for col in USE_COLS.values()
)

# ADDITIONAL FUNCTIONS
def score(series, exclude_dna=True):
    """Map to 1-5, drop NaN and 'Does Not Apply'."""
    s = series.map(LIKERT_SCORE)
    if exclude_dna:
        s = s[series != "Does Not Apply"]
    return s.dropna()

def spear_print(label, x, y):
    common = x.index.intersection(y.index)
    xc, yc = x.loc[common], y.loc[common]
    if len(xc) < 5:
        print(f"  {label}: insufficient data (n={len(xc)})")
        return np.nan, np.nan
    r, p = spearmanr(xc, yc)
    sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "(ns)"
    print(f"  {label}: r={r:.3f}, p={p:.4f} {sig}  (n={len(xc)})")
    return r, p

#CONSOLE
print("=" * 65)
print(f"AWARENESS → GAMING CHAIN  |  Developers n={N_DEV}  Managers n={N_MGR}")
print("=" * 65)

dev_criteria = score(dev[CRITERIA_COL])
dev_aware    = score(dev[AWARE_ASSESSED_COL])
dev_why      = score(dev[AWARE_WHY_COL])
dev_gaming   = score(dev[DEV_GAMING_COL])
dev_fair     = score(dev[FAIR_COL])
dev_misuse   = score(dev[MISUSE_COL])

print("\n── Developer-side chain ──")
r1, p1 = spear_print("Assessed (awareness)   → criteria improvement", dev_aware,   dev_criteria)
r2, p2 = spear_print("Why assessed (awareness) → criteria improvement", dev_why,   dev_criteria)
r3, p3 = spear_print("Gaming belief          → criteria improvement",   dev_gaming, dev_criteria)
r4, p4 = spear_print("Fairness perception    → criteria improvement",   dev_fair,   dev_criteria)
r5, p5 = spear_print("Misuse perception      → criteria improvement",   dev_misuse, dev_criteria)

print("\n── Manager-side: gaming belief × metric breadth ──")
mgr_gaming = score(mgr[MGR_GAMING_COL])
mgr_breadth = mgr["n_categories"]
r6, p6 = spear_print("Gaming belief (mgr) → breadth of metric use", mgr_gaming, mgr_breadth)

print("\n── Distribution: criteria improvement by awareness level ──")
print("  'I try to improve on manager's criteria':")
for level in ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]:
    n = (dev[AWARE_WHY_COL] == level).sum()
    if n > 0:
        subset_crit = dev[dev[AWARE_WHY_COL] == level][CRITERIA_COL].map(LIKERT_SCORE).dropna()
        print(f"  Aware why={level:<20} → criteria mean={subset_crit.mean():.2f}  (n={n})")

print("\n── Does Not Apply (excluded from analysis) ──")
for col, label in [(CRITERIA_COL, "Criteria improvement"),
                   (AWARE_WHY_COL, "Aware why assessed"),
                   (AWARE_ASSESSED_COL, "Assessed in team")]:
    dna = (dev[col] == "Does Not Apply").sum()
    print(f"  {label}: {dna} DNAs excluded ({dna/N_DEV*100:.0f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 4 — Cross-perspective summary heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[Output 4] Cross-perspective summary …")

# Correlations to display
rows = [
    ("Dev: aware why assessed → improves on criteria",    r2, p2, N_DEV),
    ("Dev: aware assessed → improves on criteria",        r1, p1, N_DEV),
    ("Dev: believes gaming → improves on criteria",       r3, p3, N_DEV),
    ("Dev: perceives fairness → improves on criteria",    r4, p4, N_DEV),
    ("Dev: perceives misuse → improves on criteria",      r5, p5, N_DEV),
    ("Mgr: believes gaming → uses more categories",       r6, p6, N_MGR),
]

fig, ax = plt.subplots(figsize=(11, 4))
y_pos = np.arange(len(rows))

r_vals = [r[1] for r in rows]
p_vals = [r[2] for r in rows]
ns     = [r[3] for r in rows]
labels_r = [r[0] for r in rows]

bar_colors = ["#DC2626" if r < 0 else "#2563EB" for r in r_vals]
bars = ax.barh(y_pos, r_vals, color=bar_colors, alpha=0.8, edgecolor="white",
               height=0.6)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline( 0.3, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
ax.axvline(-0.3, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

for bar, r_val, p_val, n in zip(bars, r_vals, p_vals, ns):
    sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else ""
    xpos = r_val + 0.02 if r_val >= 0 else r_val - 0.02
    ha   = "left" if r_val >= 0 else "right"
    label_txt = f"r={r_val:.2f} {sig}  (n={n})"
    ax.text(xpos, bar.get_y() + bar.get_height()/2, label_txt,
            va="center", ha=ha, fontsize=8.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_r, fontsize=9)
ax.set_xlim(-0.8, 0.9)
ax.set_xlabel("Spearman r", fontsize=10)
mgr_patch = mpatches.Patch(color=MGR_COLOR, label="Manager")
dev_patch = mpatches.Patch(color=DEV_COLOR, label="Developer")
ax.legend(handles=[mgr_patch, dev_patch], fontsize=9, loc="lower right")
ax.spines[["top","right"]].set_visible(False)

note = "* p<0.05  ** p<0.01  *** p<0.001"
fig.text(0.98, 0.01, note, ha="right", fontsize=7.5, color="gray", style="italic")
plt.tight_layout()
path = OUT_DIR + "gaming_output4_summary.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Done.")