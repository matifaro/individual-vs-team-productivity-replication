"""
FILE: demographic_companysize_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes the demographic profile of survey respondents by company size,
    split by role (Managers vs. Developers). This script:
    1. Filters respondents into Managers and Developers
    2. Calculates the distribution of company size categories for each group
    3. Generates a grouped bar chart showing company size by role
    4. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/demo_companysize.png - Grouped bar chart: company size distribution by role

USAGE:
    python3 code/demographic_companysize_analysis.py

NOTES:
    - Company size categories: 1-10, 11-50, 51-250, 250+ employees
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
COMPANY_COL = "What is the size of your company?\xa0"

MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

COMPANY_ORDER = ["1-10 employees", "11-50 employees", "51-250 employees", "250+ employees"]

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

print("=" * 65)
print("DEMOGRAPHIC PROFILE — COMPANY SIZE")
print("=" * 65)
print(f"\nTotal respondents: {len(df)}")
print(f"  Managers:   {len(mgr)}")
print(f"  Developers: {len(dev)}")

# CONSOLE
print(f"\nCompany Size Distribution:")
print(f"  {'Category':<18} {'Managers':>10} {'Developers':>12}")
print("-" * 42)
for cat in COMPANY_ORDER:
    mgr_count = mgr[COMPANY_COL].value_counts().get(cat, 0)
    dev_count = dev[COMPANY_COL].value_counts().get(cat, 0)
    print(f"  {cat:<18} {mgr_count:>10} {dev_count:>12}")

fig, ax = plt.subplots(figsize=(8, 5))

short_labels = ["1–10", "11–50", "51–250", "250+"]
x = np.arange(len(COMPANY_ORDER))
w = 0.35

mgr_co = mgr[COMPANY_COL].value_counts().reindex(COMPANY_ORDER, fill_value=0)
dev_co = dev[COMPANY_COL].value_counts().reindex(COMPANY_ORDER, fill_value=0)

ax.bar(x - w/2, mgr_co, w, color=MGR_COLOR, alpha=0.85, label=f"Managers (n={len(mgr)})")
ax.bar(x + w/2, dev_co, w, color=DEV_COLOR, alpha=0.85, label=f"Developers (n={len(dev)})")

# Add count labels on bars
for xi, (mv, dv) in enumerate(zip(mgr_co, dev_co)):
    if mv > 0:
        ax.text(xi - w/2, mv + 0.1, str(mv), ha="center", va="bottom", fontsize=9, fontweight="bold")
    if dv > 0:
        ax.text(xi + w/2, dv + 0.1, str(dv), ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(short_labels, fontsize=11)
ax.set_xlabel("Number of Employees", fontsize=10)
ax.set_ylabel("Count", fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
path = OUT_DIR + "demo_companysize.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {path}")
plt.close()

print("\n✓ Done.")