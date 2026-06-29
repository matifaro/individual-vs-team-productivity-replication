"""
FILE: manager_gaming_output_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Tests whether managers who believe that providing productivity metrics 
    incentivizes gaming still use decontextualised Output & Efficiency metrics.
    This script:
    1. Filters respondents to Managers only
    2. Calculates the percentage of managers at each gaming belief level who use Output & Efficiency metrics
    3. Generates a bar chart showing the relationship
    4. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/output3_right_panel_manager_output_metrics.png - Bar chart: gaming belief vs. Output & Efficiency use

USAGE:
    python3 code/manager_gaming_output_analysis.py

NOTES:
    - Managers only (developers are filtered out)
    - Gaming belief levels: Disagree, Neutral, Agree, Strongly Agree
    - Output & Efficiency metric use: Yes/No
    - Run from the project root directory
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL = "Which description best fits your role?"
MANAGER  = "I manage other software development professionals."

USE_PREFIX = "For each metric category below, indicate which you use to assess productivity.  ["
USE_COLS = {
    "Output & Efficiency": f"{USE_PREFIX}Output &amp; Efficiency]",
}

# LOAD
df = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()

# COLUMN DEFINITIONS
MGR_GAMING_COL = " [Providing productivity metrics explicitly incentivizes developers to  gamify metrics.]"

# PERCENTYAGES
gaming_levels = ["Disagree", "Neutral", "Agree", "Strongly Agree"]
out_pcts = []
n_per_level = []

for level in gaming_levels:
    subset = mgr[mgr[MGR_GAMING_COL] == level]
    n_per_level.append(len(subset))
    pct = (subset[USE_COLS["Output & Efficiency"]] == "Yes").mean() * 100
    out_pcts.append(pct if len(subset) > 0 else 0)

# CONSOLE
print("=" * 65)
print("MANAGER GAMING BELIEF → USE OF OUTPUT & EFFICIENCY METRICS")
print("=" * 65)
for level, pct, n in zip(gaming_levels, out_pcts, n_per_level):
    print(f"  {level:<15}: {pct:.0f}% use Output metrics  (n={n})")

# PLOT
fig, ax = plt.subplots(figsize=(8, 6))

colors = ["#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"][:len(gaming_levels)]
bars = ax.bar(range(len(gaming_levels)), out_pcts,
              color=colors, edgecolor="white", alpha=0.9, width=0.6)

for bar, pct, n in zip(bars, out_pcts, n_per_level):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{pct:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=14,
            fontweight="bold")

ax.set_xticks(range(len(gaming_levels)))
ax.set_xticklabels(gaming_levels, fontsize=14)
ax.set_ylim(0, 120)
ax.axhline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.4,
           label="50% reference")

ax.set_xlabel('"Metrics incentivize gaming" (manager agreement)', fontsize=15)
ax.set_ylabel("% still using Output & Efficiency", fontsize=15)

ax.legend(fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
path = OUT_DIR + "output3_right_panel_manager_output_metrics.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\n  Saved → {path}")
plt.close()

print("\n✓ Done.")