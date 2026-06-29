"""
FILE: scenario_perspective_taking_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Examines whether managers and developers can accurately predict each other's 
    preferences on scenario trade-offs between individually efficient behaviour 
    (Scenario A) and team-oriented behaviour (Scenario B). This script:
    1. Filters respondents into Managers and Developers
    2. Calculates percentage choosing Scenario B for:
       - M1: Managers' own views
       - M2: Managers predicting developers' views
       - D1: Developers' own views
       - D2: Developers predicting managers' views
    3. Generates a two-panel chart showing:
       - Left: M2 vs D1 (managers predicting developers)
       - Right: D2 vs M1 (developers predicting managers)
    4. Calculates perspective-taking accuracy (mean absolute error and bias)
    5. Identifies scenarios with largest misperceptions

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/scenarios_output2_perspectivetaking.png - Two-panel chart showing 
                                              perspective-taking accuracy

USAGE:
    python3 code/scenario_perspective_taking_analysis.py

NOTES:
    - M1 = Managers' own views on scenario trade-offs
    - M2 = Managers predicting developers' views
    - D1 = Developers' own views on scenario trade-offs
    - D2 = Developers predicting managers' views
    - Scenario A = individually efficient, Scenario B = team-oriented
    - 14 scenario trade-offs analyzed
    - Mean absolute error and bias calculated for each role
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
M2_cols = [c for c in all_sc if "developers perspective" in c.lower()]
D1_cols = [c for c in all_sc
           if c.endswith("[Which behaviour do you think is more productive?].1")]
D2_cols = [c for c in all_sc if "manager perspective" in c.lower()]

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

# ADDITIONAL FUNCTIONS
def pct_B(subset, cols):
    """% choosing Scenario B per column."""
    return [(subset[c] == "Scenario B").mean() * 100 for c in cols]


# CONSOLE
print("=" * 75)
print(f"SCENARIO TRADE-OFF — OUTPUT 2: PERSPECTIVE-TAKING")
print(f"Managers n={N_MGR}  Developers n={N_DEV}")
print("A = individually efficient  |  B = team-oriented")
print("=" * 75)

m1_pcts = pct_B(mgr, M1_cols)
m2_pcts = pct_B(mgr, M2_cols)
d1_pcts = pct_B(dev, D1_cols)
d2_pcts = pct_B(dev, D2_cols)

print(f"\n{'Scenario':<35} {'M2%':>6} {'D1%':>6} {'M2-D1':>8} {'D2%':>6} {'M1%':>6} {'D2-M1':>8}")
print("-" * 80)

scenario_stats = []
for i, label in enumerate(LABELS):
    short_label = label.replace("\n", " ")
    m2_minus_d1 = m2_pcts[i] - d1_pcts[i]
    d2_minus_m1 = d2_pcts[i] - m1_pcts[i]
    print(f"{short_label:<35} {m2_pcts[i]:>5.1f}% {d1_pcts[i]:>5.1f}% "
          f"{m2_minus_d1:>+7.1f} {d2_pcts[i]:>5.1f}% {m1_pcts[i]:>5.1f}% "
          f"{d2_minus_m1:>+7.1f}")
    scenario_stats.append(dict(
        label=short_label,
        label_nl=LABELS[i],
        m1=m1_pcts[i],
        m2=m2_pcts[i],
        d1=d1_pcts[i],
        d2=d2_pcts[i],
    ))


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 2 — Perspective-taking accuracy: M2 vs D1, and M1 vs D2
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output 2] Generating perspective-taking chart …")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, actual_key, guess_key, actual_label, guess_label, actual_color, guess_color in [
    (axes[0], "d1", "m2", "D1 — Dev actual view", "M2 — Mgr guesses dev",
     DEV_COLOR, MGR_COLOR),
    (axes[1], "m1", "d2", "M1 — Mgr actual view", "D2 — Dev guesses mgr",
     MGR_COLOR, DEV_COLOR),
]:
    # Sort by actual view for readability
    s_sorted = sorted(scenario_stats,
                      key=lambda x: x[actual_key], reverse=True)
    y = np.arange(len(s_sorted))
    w = 0.35

    # Actual view (solid)
    ax.barh(y - w/2, [r[actual_key] for r in s_sorted], w,
            color=actual_color, alpha=0.85, label=actual_label)
    
    # Guess view (hatched)
    ax.barh(y + w/2, [r[guess_key] for r in s_sorted], w,
            color=guess_color, alpha=0.5, label=guess_label,
            hatch="//", edgecolor="white")

    # Connecting lines and error annotations
    for i, r in enumerate(s_sorted):
        actual = r[actual_key]
        guess = r[guess_key]
        ax.plot([actual, guess], [i - w/2, i + w/2],
                color="black", linewidth=1.2, alpha=0.3, zorder=3)
        # Annotate direction of error if >15 percentage points
        diff = guess - actual
        if abs(diff) >= 15:
            arrow = "→" if diff > 0 else "←"
            ax.text(max(actual, guess) + 1.5, i,
                    f"{arrow}{abs(diff):.0f}pp", va="center",
                    fontsize=11, color="gray")

    ax.set_yticks(y)
    ax.set_yticklabels([r["label_nl"] for r in s_sorted], fontsize=12)
    ax.set_xlim(0, 115)
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_xlabel("% choosing Scenario B (team-oriented)", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
path = OUT_DIR + "scenarios_output2_perspectivetaking.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

# CONSOLE
print("\n" + "=" * 65)
print("PERSPECTIVE-TAKING ACCURACY SUMMARY")
print("=" * 65)

mgr_error = np.mean([abs(r["m2"] - r["d1"]) for r in scenario_stats])
dev_error = np.mean([abs(r["d2"] - r["m1"]) for r in scenario_stats])
mgr_bias = np.mean([r["m2"] - r["d1"] for r in scenario_stats])
dev_bias = np.mean([r["d2"] - r["m1"] for r in scenario_stats])

print(f"\nManagers guessing developers (M2 vs D1):")
print(f"  Mean absolute error : {mgr_error:.1f} pp")
print(f"  Mean bias           : {mgr_bias:+.1f} pp  "
      f"({'managers underestimate dev B preference' if mgr_bias < 0 else 'managers overestimate dev B preference'})")

print(f"\nDevelopers guessing managers (D2 vs M1):")
print(f"  Mean absolute error : {dev_error:.1f} pp")
print(f"  Mean bias           : {dev_bias:+.1f} pp  "
      f"({'developers underestimate mgr B preference' if dev_bias < 0 else 'developers overestimate mgr B preference'})")

# Worst errors
print(f"\nScenarios where managers most underestimate developer B preference (M2 < D1 by >20pp):")
for r in sorted(scenario_stats, key=lambda x: x["m2"] - x["d1"]):
    diff = r["m2"] - r["d1"]
    if diff < -20:
        print(f"  {r['label']:<40} M2={r['m2']:.0f}% D1={r['d1']:.0f}% ({diff:+.0f}pp)")

print(f"\nScenarios where developers most underestimate manager B preference (D2 < M1 by >20pp):")
for r in sorted(scenario_stats, key=lambda x: x["d2"] - x["m1"]):
    diff = r["d2"] - r["m1"]
    if diff < -20:
        print(f"  {r['label']:<40} D2={r['d2']:.0f}% M1={r['m1']:.0f}% ({diff:+.0f}pp)")

print("\n✓ Done.")