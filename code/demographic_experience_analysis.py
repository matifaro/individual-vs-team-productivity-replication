"""
FILE: demographic_experience_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes the demographic profile of survey respondents by professional 
    experience bands, split by role (Managers vs. Developers). This script:
    1. Filters respondents into Managers and Developers
    2. Creates experience bands from continuous years of experience
    3. Calculates the distribution of experience bands for each group
    4. Generates a grouped bar chart showing experience by role
    5. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/demo_expbands.png - Grouped bar chart: experience bands by role

USAGE:
    python3 code/demographic_experience_analysis.py

NOTES:
    - Experience bands: 0-5, 6-10, 11-15, 16-20, 21+ years
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
EXP_COL   = "How many years of professional experience do you have?"

MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

# Experience bands
BINS_EXP = [0, 5, 10, 15, 20, 35]
LABELS_EXP = ["0–5", "6–10", "11–15", "16–20", "21+"]

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

# Create experience bands
df["exp_bin"] = pd.cut(df[EXP_COL], bins=BINS_EXP, labels=LABELS_EXP, right=True)
mgr["exp_bin"] = pd.cut(mgr[EXP_COL], bins=BINS_EXP, labels=LABELS_EXP, right=True)
dev["exp_bin"] = pd.cut(dev[EXP_COL], bins=BINS_EXP, labels=LABELS_EXP, right=True)

print("=" * 65)
print("DEMOGRAPHIC PROFILE — EXPERIENCE BANDS")
print("=" * 65)
print(f"\nTotal respondents: {len(df)}")
print(f"  Managers:   {len(mgr)}")
print(f"  Developers: {len(dev)}")

# CONSOLE
print(f"\nExperience Distribution:")
print(f"  {'Category':<12} {'Managers':>10} {'Developers':>12}")
print("-" * 36)
for cat in LABELS_EXP:
    mgr_count = mgr["exp_bin"].value_counts().get(cat, 0)
    dev_count = dev["exp_bin"].value_counts().get(cat, 0)
    print(f"  {cat:<12} {mgr_count:>10} {dev_count:>12}")

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(LABELS_EXP))
w = 0.35

exp_mgr = mgr["exp_bin"].value_counts().reindex(LABELS_EXP, fill_value=0)
exp_dev = dev["exp_bin"].value_counts().reindex(LABELS_EXP, fill_value=0)

ax.bar(x - w/2, exp_mgr, w, color=MGR_COLOR, alpha=0.85, label=f"Managers (n={len(mgr)})")
ax.bar(x + w/2, exp_dev, w, color=DEV_COLOR, alpha=0.85, label=f"Developers (n={len(dev)})")

# Add count labels on bars
for xi, (mv, dv) in enumerate(zip(exp_mgr, exp_dev)):
    if mv > 0:
        ax.text(xi - w/2, mv + 0.1, str(mv), ha="center", va="bottom", fontsize=13, fontweight="bold")
    if dv > 0:
        ax.text(xi + w/2, dv + 0.1, str(dv), ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([f"{l} yrs" for l in LABELS_EXP], fontsize=14)
ax.set_ylabel("Count", fontsize=15)
ax.legend(fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=13)

plt.tight_layout()
path = OUT_DIR + "demo_expbands.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {path}")
plt.close()

print("\n✓ Done.")