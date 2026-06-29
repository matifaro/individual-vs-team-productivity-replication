"""
FILE: scenario_tradeoff_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Compares managers' (M1) and developers' (D1) actual views on scenario 
    trade-offs between individually efficient behaviour (Scenario A) and 
    team-oriented behaviour (Scenario B). This script:
    1. Filters respondents into Managers and Developers
    2. Calculates percentage choosing Scenario B (team-oriented) for each scenario
    3. Performs Fisher's exact tests comparing M1 vs. D1 responses
    4. Generates a grouped bar chart with gap panel showing manager-developer differences
    5. Prints detailed statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/scenarios_output1_M1vsD1.png - Bar chart comparing M1 vs D1 with gap panel

USAGE:
    python3 code/scenario_tradeoff_analysis.py

NOTES:
    - M1 = Managers' own views on scenario trade-offs
    - D1 = Developers' own views on scenario trade-offs
    - 14 scenario trade-offs analyzed:
      Independent delivery vs collaboration, Skip docs vs thorough docs,
      Personal tooling vs shared tooling, Deliver own scope vs help team,
      Decline mentoring vs mentor juniors, Deepen expertise vs broaden skills,
      Implement fast vs share knowledge, Ship fast vs refactor,
      Async focus vs alignment meetings, Solve alone vs team discussion,
      Calendar blocks vs stay available, Minimal tests vs full tests,
      Optimistic estimates vs realistic buffer, Avoid feedback vs honest feedback
    - Scenario A = individually efficient, Scenario B = team-oriented
    - Fisher's exact test (two-sided), no multiple testing correction
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
N_MGR = len(mgr)
N_DEV = len(dev)

# COLUMN DEFINITIONS
all_sc = [c for c in df.columns if "behaviour" in c.lower()]

M1_cols = [c for c in all_sc
           if c.endswith("[Which behaviour do you think is more productive?]")]
D1_cols = [c for c in all_sc
           if c.endswith("[Which behaviour do you think is more productive?].1")]

# Short scenario labels
LABELS = [
    "Independent delivery\nvs collaboration",
    "Skip docs\nvs thorough docs",
    "Personal tooling\nvs shared tooling",
    "Deliver own scope\nvs help team",
    "Decline mentoring\nvs mentor juniors",
    "Deepen expertise\nvs broaden skills",
    "Implement fast\nvs share knowledge",
    "Ship fast\nvs refactor",
    "Async focus\nvs alignment meetings",
    "Solve alone\nvs team discussion",
    "Calendar blocks\nvs stay available",
    "Minimal tests\nvs full tests",
    "Optimistic estimates\nvs realistic buffer",
    "Avoid feedback\nvs honest feedback",
]
SHORT = [l.replace("\n", " ") for l in LABELS]

# ADDITIONAL FUNCTIONS
def pct_B(subset, cols):
    """% choosing Scenario B per column."""
    return [(subset[c] == "Scenario B").mean() * 100 for c in cols]

def fisher_p(col_a, subset_a, col_b, subset_b):
    """Fisher's exact for two binary columns from two subsets."""
    a_B = (subset_a[col_a] == "Scenario B").sum()
    a_A = (subset_a[col_a] == "Scenario A").sum()
    b_B = (subset_b[col_b] == "Scenario B").sum()
    b_A = (subset_b[col_b] == "Scenario A").sum()
    if a_A + a_B == 0 or b_A + b_B == 0:
        return np.nan
    _, p = fisher_exact([[a_B, a_A], [b_B, b_A]], alternative="two-sided")
    return p

def sig(p):
    """Convert p-value to significance stars (no correction)."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(ns)"


# CONSOLE
print("=" * 75)
print(f"SCENARIO TRADE-OFF — OUTPUT 1: M1 vs D1")
print(f"Managers n={N_MGR}  Developers n={N_DEV}")
print("A = individually efficient  |  B = team-oriented")
print("=" * 75)
print(f"\n{'Scenario':<35} {'M1%B':>6} {'D1%B':>6} {'p(M1vD1)':>12} {'sig':>6}")
print("-" * 70)

m1_pcts = pct_B(mgr, M1_cols)
d1_pcts = pct_B(dev, D1_cols)

scenario_stats = []
for i, label in enumerate(SHORT):
    p = fisher_p(M1_cols[i], mgr, D1_cols[i], dev)
    s = sig(p)
    print(f"{label:<35} {m1_pcts[i]:>5.1f}% {d1_pcts[i]:>5.1f}% {p:>12.4f} {s:>6}")
    scenario_stats.append(dict(
        label=label,
        label_nl=LABELS[i],
        m1=m1_pcts[i],
        d1=d1_pcts[i],
        p_m1d1=p,
        sig_m1d1=s,
    ))

print("\nSignificance: * p<0.05  ** p<0.01  *** p<0.001  (ns) = not significant")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 1 — M1 vs D1: actual views compared
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output 1] Generating M1 vs D1 chart …")

s1 = sorted(scenario_stats, key=lambda x: (x["m1"] + x["d1"]) / 2, reverse=True)
y  = np.arange(len(s1))
w  = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                         gridspec_kw={"width_ratios": [3, 1]})

# Left panel: Grouped bars
ax = axes[0]
ax.barh(y - w/2, [r["m1"] for r in s1], w, color=MGR_COLOR,
        alpha=0.85, label=f"Managers — own view M1 (n={N_MGR})")
ax.barh(y + w/2, [r["d1"] for r in s1], w, color=DEV_COLOR,
        alpha=0.85, label=f"Developers — own view D1 (n={N_DEV})")

# Add connecting lines and significance stars
for i, r in enumerate(s1):
    s = r["sig_m1d1"]
    if s and s != "(ns)":
        ax.text(max(r["m1"], r["d1"]) + 1.5, i, s, va="center",
                fontsize=14, color="darkred", fontweight="bold")
    ax.plot([r["m1"], r["d1"]], [i - w/2, i + w/2],
            color="black", linewidth=1.2, alpha=0.3, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([r["label_nl"] for r in s1], fontsize=12)
ax.set_xlim(0, 115)
ax.axvline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
ax.set_xlabel("% choosing Scenario B (team-oriented behaviour)", fontsize=14)
ax.legend(loc="lower right", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

# Right panel: Gap
ax2 = axes[1]
gaps = [r["m1"] - r["d1"] for r in s1]
gcols = [MGR_COLOR if g > 0 else DEV_COLOR for g in gaps]
bars = ax2.barh(y, gaps, color=gcols, alpha=0.75, edgecolor="white")
ax2.axvline(0, color="black", linewidth=0.8)

for bar, val, r in zip(bars, gaps, s1):
    s = r["sig_m1d1"]
    xpos = val + 1 if val >= 0 else val - 1
    ha = "left" if val >= 0 else "right"
    label_txt = f"{val:+.0f}%"
    if s and s != "(ns)":
        label_txt += f" {s}"
    ax2.text(xpos, bar.get_y() + bar.get_height()/2,
             label_txt, va="center", ha=ha, fontsize=12)

ax2.set_yticks(y)
ax2.set_yticklabels([])
ax2.set_xlabel("Gap (M1 − D1)", fontsize=14)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis='both', which='major', labelsize=12)

note = "* p<0.05  ** p<0.01  *** p<0.001"
fig.text(0.98, 0.01, note, ha="right", fontsize=11, color="gray", style="italic")
plt.tight_layout()

path = OUT_DIR + "scenarios_output1_M1vsD1.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Done.")