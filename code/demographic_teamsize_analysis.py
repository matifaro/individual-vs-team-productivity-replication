"""
FILE: demographic_teamsize_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes the demographic profile of survey respondents by immediate team size,
    split by role (Managers vs. Developers). This script:
    1. Filters respondents into Managers and Developers
    2. Calculates the distribution of team size categories for each group
    3. Generates a grouped bar chart showing team size by role
    4. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/demo_teamsize.png - Grouped bar chart: team size distribution by role

USAGE:
    python3 code/demographic_teamsize_analysis.py

NOTES:
    - Team size categories: 1-5, 6-10, 11-15, 16-20, 20+
    - Role split: Managers vs. Developers
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

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."
TEAM_COL  = "How many people are in your immediate team?"

MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

TEAM_ORDER = ["1-5", "6-10", "11-15", "16-20", "20+"]

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

print("=" * 65)
print("DEMOGRAPHIC PROFILE — IMMEDIATE TEAM SIZE")
print("=" * 65)
print(f"\nTotal respondents: {len(df)}")
print(f"  Managers:   {len(mgr)}")
print(f"  Developers: {len(dev)}")

# CONSOLE
print(f"\nTeam Size Distribution:")
print(f"  {'Category':<12} {'Managers':>10} {'Developers':>12}")
print("-" * 36)
for cat in TEAM_ORDER:
    mgr_count = mgr[TEAM_COL].value_counts().get(cat, 0)
    dev_count = dev[TEAM_COL].value_counts().get(cat, 0)
    print(f"  {cat:<12} {mgr_count:>10} {dev_count:>12}")


fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(TEAM_ORDER))
w = 0.35

mgr_ts = mgr[TEAM_COL].value_counts().reindex(TEAM_ORDER, fill_value=0)
dev_ts = dev[TEAM_COL].value_counts().reindex(TEAM_ORDER, fill_value=0)

ax.bar(x - w/2, mgr_ts, w, color=MGR_COLOR, alpha=0.85, label=f"Managers (n={len(mgr)})")
ax.bar(x + w/2, dev_ts, w, color=DEV_COLOR, alpha=0.85, label=f"Developers (n={len(dev)})")

# Add count labels on bars
for xi, (mv, dv) in enumerate(zip(mgr_ts, dev_ts)):
    if mv > 0:
        ax.text(xi - w/2, mv + 0.1, str(mv), ha="center", va="bottom", fontsize=13, fontweight="bold")
    if dv > 0:
        ax.text(xi + w/2, dv + 0.1, str(dv), ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(TEAM_ORDER, fontsize=14)
ax.set_ylabel("Count", fontsize=15)
ax.legend(fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
path = OUT_DIR + "demo_teamsize.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {path}")
plt.close()

print("\n✓ Done.")