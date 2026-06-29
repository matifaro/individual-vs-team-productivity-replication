"""
FILE: scenario_four_perspective_heatmap.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Creates a heatmap showing all four perspectives per scenario trade-off:
    - M1: Managers' own views
    - M2: Managers predicting developers' views
    - D1: Developers' own views
    - D2: Developers predicting managers' views
    
    This script:
    1. Filters respondents into Managers and Developers
    2. Calculates percentage choosing Scenario B (team-oriented) for all four perspectives
    3. Generates a heatmap with all perspectives per scenario
    4. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/scenarios_output4_heatmap.png - Heatmap of all 4 perspectives

USAGE:
    python3 code/scenario_four_perspective_heatmap.py

NOTES:
    - M1 = Managers' own views on scenario trade-offs
    - M2 = Managers predicting developers' views
    - D1 = Developers' own views on scenario trade-offs
    - D2 = Developers predicting managers' views
    - Scenario A = individually efficient, Scenario B = team-oriented
    - 14 scenario trade-offs analyzed
    - Color scale: Red (low % B) → Green (high % B)
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

# COLUMN DEFINIITONS
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
print(f"SCENARIO TRADE-OFF — OUTPUT 4: FOUR-PERSPECTIVE HEATMAP")
print(f"Managers n={N_MGR}  Developers n={N_DEV}")
print("A = individually efficient  |  B = team-oriented")
print("=" * 75)

m1_pcts = pct_B(mgr, M1_cols)
m2_pcts = pct_B(mgr, M2_cols)
d1_pcts = pct_B(dev, D1_cols)
d2_pcts = pct_B(dev, D2_cols)

print(f"\n{'Scenario':<35} {'M1%':>6} {'M2%':>6} {'D1%':>6} {'D2%':>6}")
print("-" * 65)

scenario_stats = []
for i, label in enumerate(LABELS):
    short_label = label.replace("\n", " ")
    print(f"{short_label:<35} {m1_pcts[i]:>5.1f}% {m2_pcts[i]:>5.1f}% "
          f"{d1_pcts[i]:>5.1f}% {d2_pcts[i]:>5.1f}%")
    scenario_stats.append(dict(
        label=short_label,
        label_nl=LABELS[i],
        m1=m1_pcts[i],
        m2=m2_pcts[i],
        d1=d1_pcts[i],
        d2=d2_pcts[i],
    ))


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 4 — Heatmap of all 4 perspectives per scenario
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Output 4] Generating heatmap …")

# Sort scenarios by M1 for readability
s4 = sorted(scenario_stats, key=lambda x: x["m1"], reverse=True)
mat = np.array([[r["m1"], r["m2"], r["d1"], r["d2"]] for r in s4])
col_labels = ["M1\nMgr own", "M2\nMgr→Dev\nguess",
              "D1\nDev own", "D2\nDev→Mgr\nguess"]

fig, ax = plt.subplots(figsize=(10.5, 9.5))
im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label="% choosing Scenario B", shrink=0.85)
cbar.ax.tick_params(labelsize=12)
cbar.set_label('% choosing Scenario B', fontsize=13)

# Labels
ax.set_xticks(range(4))
ax.set_xticklabels(col_labels, fontsize=13)
ax.set_yticks(range(len(s4)))
ax.set_yticklabels([r["label_nl"] for r in s4], fontsize=11)

# Add percentage labels in each cell
for i in range(len(s4)):
    for j in range(4):
        v = mat[i, j]
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                color="white" if (v > 75 or v < 25) else "black", fontsize=12)

# Add vertical separator between M and D sections
ax.axvline(1.5, color="white", linewidth=2)
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
path = OUT_DIR + "scenarios_output4_heatmap.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

# CONSOLE
print("\n" + "=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)

overall = {
    "M1 (Mgr own)": np.mean(m1_pcts),
    "M2 (Mgr→Dev guess)": np.mean(m2_pcts),
    "D1 (Dev own)": np.mean(d1_pcts),
    "D2 (Dev→Mgr guess)": np.mean(d2_pcts),
}

print("\nOverall % choosing Scenario B across all 14 scenarios:")
for key, val in overall.items():
    print(f"  {key}: {val:.1f}%")

print("\nScenarios with strongest B preference (D1):")
for r in sorted(scenario_stats, key=lambda x: x["d1"], reverse=True)[:3]:
    print(f"  {r['label']:<40} {r['d1']:.0f}%")

print("\nScenarios with strongest A preference (D1):")
for r in sorted(scenario_stats, key=lambda x: x["d1"])[:3]:
    print(f"  {r['label']:<40} {r['d1']:.0f}%")

print("\n✓ Done.")