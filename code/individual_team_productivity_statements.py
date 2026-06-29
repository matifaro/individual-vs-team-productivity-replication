"""
FILE: individual_team_productivity_statements.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes survey responses about statements regarding individual vs. team 
    productivity, comparing perspectives between managers and developers. 
    This script:
    1. Filters respondents into Managers and Developers
    2. Maps Likert responses to numeric scores (1-5)
    3. Calculates percentage of 'Agree/Strongly Agree' responses for each statement
    4. Performs Mann-Whitney U tests comparing managers vs. developers
    5. Generates a gap chart with:
       - Left panel: % Agree for managers and developers
       - Right panel: Gap (Manager % - Developer %) with significance stars
    6. Prints a summary table to console with p-values and significance stars

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/indteam_gap_chart.png - Gap chart comparing managers vs. developers
                                            with Mann-Whitney U test p-values

USAGE:
    python3 code/individual_team_productivity_statements.py

NOTES:
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Statements analyzed:
      - Team productivity more important than individual
      - Should be assessed for different purposes
      - Productive individuals → productive team
      - Conflict between individual & team productivity
      - Individual harder to assess than team
    - Mann-Whitney U test (two-sided), no multiple testing correction
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."
LIKERT_SCORE = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}
MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()

# STATEMENTS
STATEMENTS = [
    (
        " [Team productivity is more important than individual productivity.]",
        "Team productivity more\nimportant than individual",
    ),
    (
        " [Team productivity and individual productivity should be assessed for different purposes.]",
        "Should be assessed\nfor different purposes",
    ),
    (
        " [A group of productive individuals makes a productive team.]",
        "Productive individuals →\nproductive team",
    ),
    (
        " [ There is a conflict between productivity of individuals and of the whole team.]",
        "Conflict between individual\n& team productivity",
    ),
    (
        " [It is harder to assess individual productivity over team productivity.]",
        "Individual harder\nto assess than team",
    ),
]

# ADDITIONAL FUNCTIONS
def score(series):
    #Convert Likert responses to numeric scores.
    return series.map(LIKERT_SCORE).dropna()

def pct_agree(series):
    #Calculate percentage of 'Agree' or 'Strongly Agree' responses.
    vals = series.dropna()
    if len(vals) == 0:
        return np.nan
    return (vals.isin(["Agree", "Strongly Agree"])).mean() * 100

def rank_biserial(g1, g2):
    #Calculate rank-biserial correlation (effect size).
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
    return 1 - (2 * u) / (n1 * n2)

def significance_star(p):
    """
    Convert p-value to significance stars (no correction).
    * p < 0.05, ** p < 0.01, *** p < 0.001
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(ns)"

# ANALYSIS
print("=" * 72)
print("INDIVIDUAL VS TEAM PRODUCTIVITY — GAP ANALYSIS")
print(f"Total: {len(df)}  |  Managers: {len(mgr)}  |  Developers: {len(dev)}")
print("=" * 72)

print(f"\n{'Statement':<45} {'Mgr%':>8} {'Dev%':>8} {'Gap':>7} {'p':>9} {'sig':>6} {'r':>7}")
print("-" * 95)

results = []
for col, label in STATEMENTS:
    mgr_s = score(mgr[col])
    dev_s = score(dev[col])
    mgr_pct = pct_agree(mgr[col])
    dev_pct = pct_agree(dev[col])
    gap = mgr_pct - dev_pct
    u, p = mannwhitneyu(mgr_s, dev_s, alternative="two-sided")
    rb = rank_biserial(mgr_s, dev_s)
    sig = significance_star(p)
    lbl_short = label.replace("\n", " ")
    print(f"{lbl_short:<45} {mgr_pct:>7.1f}% {dev_pct:>7.1f}% "
          f"{gap:>+6.1f} {p:>8.4f} {sig:>6} {rb:>7.3f}")
    results.append({
        "label": label,
        "col": col,
        "mgr_pct": mgr_pct,
        "dev_pct": dev_pct,
        "gap": gap,
        "U": u,
        "p": p,
        "sig": sig,
        "r": rb,
        "mgr_scores": mgr_s,
        "dev_scores": dev_s,
    })

print("\nSignificance: * p<0.05  ** p<0.01  *** p<0.001  (ns) = not significant")

# GAP CHART
print("\n[Output] Generating gap chart …")

# Sort by gap (largest positive = managers agree more)
results_sorted = sorted(results, key=lambda x: x["gap"], reverse=True)
labels   = [r["label"] for r in results_sorted]
mgr_pcts = [r["mgr_pct"] for r in results_sorted]
dev_pcts = [r["dev_pct"] for r in results_sorted]
gaps     = [r["gap"] for r in results_sorted]
sigs     = [r["sig"] for r in results_sorted]

y = np.arange(len(labels))

# Create figure with two panels: main chart + gap panel
fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                         gridspec_kw={"width_ratios": [3, 1]})

# Left panel: Grouped bar chart
ax = axes[0]
ax.barh(y - 0.2, mgr_pcts, 0.35, color=MGR_COLOR, alpha=0.85,
        label=f"Managers (n={len(mgr)})")
ax.barh(y + 0.2, dev_pcts, 0.35, color=DEV_COLOR, alpha=0.85,
        label=f"Developers (n={len(dev)})")

# Add connecting lines between manager and developer bars
for i, (m, d) in enumerate(zip(mgr_pcts, dev_pcts)):
    ax.plot([m, d], [i - 0.2, i + 0.2], color="black",
            linewidth=1.2, alpha=0.4, zorder=3)

# Add significance stars
for i, (sig, m, d) in enumerate(zip(sigs, mgr_pcts, dev_pcts)):
    if sig != "(ns)":
        ax.text(max(m, d) + 1.5, i, sig, va="center",
                fontsize=13, color="darkred", fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=13)
ax.set_xlim(0, 110)
ax.set_xlabel('% Agree or Strongly Agree', fontsize=15)
ax.axvline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.legend(loc="lower right", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

# Right panel: Gap (Manager - Developer)
ax2 = axes[1]
bar_colors = [MGR_COLOR if g > 0 else DEV_COLOR for g in gaps]
bars = ax2.barh(y, gaps, color=bar_colors, alpha=0.75, edgecolor="white")
ax2.axvline(0, color="black", linewidth=0.8)

# Add gap values with significance
for bar, val, sig in zip(bars, gaps, sigs):
    xpos = val + 1 if val >= 0 else val - 1
    ha = "left" if val >= 0 else "right"
    label_txt = f"{val:+.0f}%"
    if sig != "(ns)":
        label_txt += f" {sig}"
    ax2.text(xpos, bar.get_y() + bar.get_height() / 2,
             label_txt, va="center", ha=ha, fontsize=12)

ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("Gap (Mgr − Dev) in %", fontsize=15)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis='both', which='major', labelsize=12)

note = "* p<0.05  ** p<0.01  *** p<0.001"
fig.text(0.98, 0.01, note, ha="right", fontsize=11, color="gray", style="italic")

plt.tight_layout()
path = OUT_DIR + "indteam_gap_chart.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  ✓ Saved → {path}")
plt.close()

print("\n✓ Analysis complete.")