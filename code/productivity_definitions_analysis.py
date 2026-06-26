"""
FILE: productivity_definitions_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Compares how managers vs. developers define "productive" work in software 
    development. This script:
    1. Loads survey data from Qualtrics
    2. Splits respondents into Managers and Developers
    3. Calculates percentage of 'Yes' responses for 10 productivity definitions
    4. Performs Fisher's exact tests comparing managers vs. developers for each definition
    5. Generates a grouped bar chart with:
       - Left panel: % Yes for managers and developers (sorted by overall popularity)
       - Right panel: Gap (Manager % - Developer %) with significance stars
    6. Prints a summary table to console with p-values and significance stars

DEPENDENCIES:
    Python 3.x
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/proddef_barchart.png - Bar chart comparing managers vs. developers
                                   with Fisher's exact test p-values

USAGE:
    python3 code/productivity_definitions_analysis.py

NOTES:
    - Assumes the CSV file is located at data/results-survey331585.csv
    - The script excludes "None of the above" responses from analysis
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
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

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

# OPTIONS (EXCLUDING NONE OF THE ABOVE)
OPTIONS_RAW = [
    ("What do you consider as being productive? [Reaching business objectives.]",
     "Reaching business\nobjectives"),
    ("What do you consider as being productive? [Delivering business value.]",
     "Delivering business\nvalue"),
    ("What do you consider as being productive? [Individual progress.]",
     "Individual\nprogress"),
    ("What do you consider as being productive? [Delivering tasks on time.]",
     "Delivering tasks\non time"),
    ("What do you consider as being productive? [Producing quality artifacts that need little rework.]",
     "Producing quality\nartifacts"),
    ("What do you consider as being productive? [Meeting stakeholder expectations.]",
     "Meeting stakeholder\nexpectations"),
    ("What do you consider as being productive? [Transforming ideas into software effectively.]",
     "Transforming ideas\ninto software"),
    ("What do you consider as being productive? [Reducing developer effort while increasing output.]",
     "Reducing effort,\nincreasing output"),
    ("What do you consider as being productive? [Solving complex problems.]",
     "Solving complex\nproblems"),
    ("What do you consider as being productive? [Helping others on the team.]",
     "Helping others\non the team"),
]

COLS   = [o[0] for o in OPTIONS_RAW]
LABELS = [o[1] for o in OPTIONS_RAW]

# ADDITIONAL FUNCTIONS
def yes_pct(subset, col):
    #Calculate percentage of 'Yes' responses for a given column
    return (subset[col] == "Yes").mean() * 100

def yes_n(subset, col):
    #Count number of 'Yes' responses for a given column
    return (subset[col] == "Yes").sum()

def fisher_test(col):
    """
    Perform Fisher's exact test comparing managers vs developers.
    Returns p-value.
    """
    m_yes = yes_n(mgr, col)
    m_no = len(mgr) - m_yes
    d_yes = yes_n(dev, col)
    d_no = len(dev) - d_yes
    _, p = fisher_exact([[m_yes, m_no], [d_yes, d_no]], alternative="two-sided")
    return p

def significance_star(p):
    """
    Convert p-value to significance stars.
    * p < 0.05, ** p < 0.01, *** p < 0.001
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

# CONSOLE
print("=" * 72)
print("PRODUCTIVITY DEFINITIONS — SUMMARY (Managers vs Developers)")
print(f"Total respondents: {len(df)}")
print(f"  Managers:  {len(mgr)}")
print(f"  Developers: {len(dev)}")
print("=" * 72)
print(f"\n{'Definition':<40} {'Mgr':>8} {'Dev':>8} {'p (Fisher)':>12} {'sig':>5}")
print("-" * 75)

for col, label in zip(COLS, LABELS):
    label_short = label.replace("\n", " ")
    m_pct = yes_pct(mgr, col)
    d_pct = yes_pct(dev, col)
    p = fisher_test(col)
    sig = significance_star(p)
    print(f"{label_short:<40} {m_pct:>7.1f}% {d_pct:>7.1f}% {p:>12.4f} {sig:>5}")

print("\nSignificance: * p<0.05  ** p<0.01  *** p<0.001")

# BAR CHART
print("\n[Output 1] Generating bar chart …")

# Build sorted data (by overall percentage)
rows = []
for col, label in zip(COLS, LABELS):
    rows.append({
        "label": label,
        "col": col,
        "mgr": yes_pct(mgr, col),
        "dev": yes_pct(dev, col),
        "p": fisher_test(col),
    })
rows.sort(key=lambda x: x["mgr"] + x["dev"], reverse=True)  # Sort by combined popularity

labels_sorted = [r["label"] for r in rows]
mgr_pcts = [r["mgr"] for r in rows]
dev_pcts = [r["dev"] for r in rows]
p_values = [r["p"] for r in rows]

# Create figure with two panels: main chart + gap panel
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5),
                         gridspec_kw={"width_ratios": [3, 1]})

y = np.arange(len(rows))
width = 0.35

# Left panel: Grouped bar chart
ax = axes[0]
bars1 = ax.barh(y - width/2, mgr_pcts, width, color=MGR_COLOR, 
                alpha=0.85, label=f"Managers (n={len(mgr)})")
bars2 = ax.barh(y + width/2, dev_pcts, width, color=DEV_COLOR, 
                alpha=0.85, label=f"Developers (n={len(dev)})")

# Add significance stars
for i, (m, d, p) in enumerate(zip(mgr_pcts, dev_pcts, p_values)):
    sig = significance_star(p)
    if sig:
        max_val = max(m, d)
        ax.text(max_val + 1.5, i, sig, va="center", fontsize=10,
                color="darkred", fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels_sorted, fontsize=9)
ax.set_xlim(0, 105)
ax.set_xlabel('% who selected "Yes"', fontsize=10)
ax.axvline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# Right panel: Gap (Manager - Developer)
ax2 = axes[1]
gaps = [m - d for m, d in zip(mgr_pcts, dev_pcts)]
bar_colors = [MGR_COLOR if g > 0 else DEV_COLOR for g in gaps]
bars_gap = ax2.barh(y, gaps, color=bar_colors, alpha=0.75, edgecolor="white")
ax2.axvline(0, color="black", linewidth=0.8)

# Add gap values with significance
for bar, val, p in zip(bars_gap, gaps, p_values):
    sig = significance_star(p)
    xpos = val + 1 if val >= 0 else val - 1
    ha = "left" if val >= 0 else "right"
    label = f"{val:+.0f}%"
    if sig:
        label += f" {sig}"
    ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
             label, va="center", ha=ha, fontsize=8)

ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("Gap (Managers − Developers)", fontsize=10)
ax2.spines[["top", "right"]].set_visible(False)

# Add note about significance
fig.text(0.98, 0.01, "* p<0.05  ** p<0.01  *** p<0.001 (Fisher's exact test, two-sided)",
         ha="right", fontsize=8, color="gray", style="italic")

plt.tight_layout()
path = OUT_DIR + "proddef_barchart.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  ✓ Saved → {path}")
plt.close()

print("\n✓ Analysis complete.")