"""
FILE: demographic_worksetup_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes the demographic profile of survey respondents by work setup
    (Remote, Hybrid, On-site), split by role (Managers vs. Developers). This script:
    1. Filters respondents into Managers and Developers
    2. Calculates the distribution of work setup categories for each group
    3. Generates a grouped bar chart showing work setup by role
    4. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/demo_worksetup.png - Grouped bar chart: work setup distribution by role

USAGE:
    python3 code/demographic_worksetup_analysis.py

NOTES:
    - Work setup categories: Remote, Hybrid, On-site
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
SETUP_COL = "What is the work set up for collaboration with your team members?"

MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

SETUP_ORDER = ["Remote", "Hybrid", "On-site"]

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

print("=" * 65)
print("DEMOGRAPHIC PROFILE — WORK SETUP")
print("=" * 65)
print(f"\nTotal respondents: {len(df)}")
print(f"  Managers:   {len(mgr)}")
print(f"  Developers: {len(dev)}")

# CONSOLE
print(f"\nWork Setup Distribution:")
print(f"  {'Category':<12} {'Managers':>10} {'Developers':>12}")
print("-" * 36)
for cat in SETUP_ORDER:
    mgr_count = mgr[SETUP_COL].value_counts().get(cat, 0)
    dev_count = dev[SETUP_COL].value_counts().get(cat, 0)
    print(f"  {cat:<12} {mgr_count:>10} {dev_count:>12}")

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(SETUP_ORDER))
w = 0.35

mgr_ws = mgr[SETUP_COL].value_counts().reindex(SETUP_ORDER, fill_value=0)
dev_ws = dev[SETUP_COL].value_counts().reindex(SETUP_ORDER, fill_value=0)

ax.bar(x - w/2, mgr_ws, w, color=MGR_COLOR, alpha=0.85, label=f"Managers (n={len(mgr)})")
ax.bar(x + w/2, dev_ws, w, color=DEV_COLOR, alpha=0.85, label=f"Developers (n={len(dev)})")

# Add count labels on bars
for xi, (mv, dv) in enumerate(zip(mgr_ws, dev_ws)):
    if mv > 0:
        ax.text(xi - w/2, mv + 0.1, str(mv), ha="center", va="bottom", fontsize=13, fontweight="bold")
    if dv > 0:
        ax.text(xi + w/2, dv + 0.1, str(dv), ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(SETUP_ORDER, fontsize=14)
ax.set_ylabel("Count", fontsize=15)
ax.legend(fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
path = OUT_DIR + "demo_worksetup.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {path}")
plt.close()

print("\n✓ Done.")