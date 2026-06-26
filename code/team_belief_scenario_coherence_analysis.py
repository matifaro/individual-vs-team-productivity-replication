"""
FILE: team_belief_scenario_coherence_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes the coherence between the belief that "Team productivity is more 
    important than individual productivity" and actual scenario choices where 
    Scenario B always favours team-oriented behaviour. This script:
    1. Computes per-respondent team-orientation score (fraction of scenarios choosing B)
    2. Calculates Spearman correlation: belief score ↔ team-orientation score
    3. Groups respondents by belief level (Disagree/Neutral/Agree)
    4. Compares team-orientation scores across belief groups using Kruskal-Wallis
    5. Performs pairwise Mann-Whitney U tests
    6. Generates three visualizations:
       - Scatter: belief score vs team-orientation score (by role)
       - Boxplot: team-orientation distribution by belief level
       - Heatmap: per-scenario % choosing B by belief group

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/teambelief_fig1_scatter.png - Scatter plot: belief vs team-orientation by role

USAGE:
    python3 code/team_belief_scenario_coherence_analysis.py

NOTES:
    - Statement: "Team productivity is more important than individual productivity"
    - 14 scenario trade-offs analyzed
    - Scenario B always favours team-oriented behaviour
    - Belief groups: Disagree (1-2), Neutral (3), Agree (4-5)
    - Spearman correlation, Kruskal-Wallis, and Mann-Whitney U tests
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, kruskal, mannwhitneyu
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."

BELIEF_COL = " [Team productivity is more important than individual productivity.]"
LIKERT_SCORE = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}
MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

# LOAD
df = pd.read_csv(DATA_PATH)

# ── Scenario columns: each respondent answers their own role's column
#    Managers → no-suffix "more productive?]" cols (not .1, not "from your")
#    Developers → ".1" cols
# Scenario B is always the team-oriented behaviour.
SCENARIO_MGR_COLS = [c for c in df.columns
                     if "Which behaviour do you think is more productive?]" in c
                     and ".1" not in c and "from your" not in c]
SCENARIO_DEV_COLS = [c for c in df.columns
                     if "Which behaviour do you think is more productive?].1" in c]

assert len(SCENARIO_MGR_COLS) == 14
assert len(SCENARIO_DEV_COLS) == 14


def short_label(col, maxlen=38):
    """Extract a short scenario label from the column text."""
    # Find A) … B) split; grab just A)'s opening action phrase
    a_start = col.find("A) ") + 3
    # Cut at first comma, 'but', or 60 chars
    candidates = []
    for needle in [",", " but", " and avoids", " while"]:
        idx = col.find(needle, a_start)
        if idx > a_start:
            candidates.append(idx)
    cut = min(candidates) if candidates else a_start + maxlen
    label = col[a_start:cut].strip()
    return label[:maxlen]


SCENARIO_LABELS = [short_label(c) for c in SCENARIO_MGR_COLS]

BELIEF_GROUPS = {
    "Disagree\n(1–2)": [1, 2],
    "Neutral\n(3)":   [3],
    "Agree\n(4–5)":   [4, 5],
}
BG_COLORS = {
    "Disagree\n(1–2)": "#DC2626",
    "Neutral\n(3)":    "#D97706",
    "Agree\n(4–5)":    "#059669",
}


# ══════════════════════════════════════════════════════════════════════════════
# BUILD ANALYSIS DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for _, row in df.iterrows():
    role = row[ROLE_COL]
    if role not in (MANAGER, DEVELOPER):
        continue
    belief_raw = row[BELIEF_COL]
    belief_sc  = LIKERT_SCORE.get(belief_raw, np.nan)
    if pd.isna(belief_sc):
        continue

    # Pick the right scenario columns for this respondent's role
    if role == MANAGER:
        scen_cols = SCENARIO_MGR_COLS
    else:
        scen_cols = SCENARIO_DEV_COLS

    choices = [row[c] for c in scen_cols]
    b_count = sum(1 for c in choices if c == "Scenario B")
    valid   = sum(1 for c in choices if c in ("Scenario A", "Scenario B"))
    team_score = b_count / valid if valid > 0 else np.nan

    # Individual scenario choices (1 = B, 0 = A, nan = missing)
    indiv = {f"S{i+1}": (1 if row[scen_cols[i]] == "Scenario B"
                          else 0 if row[scen_cols[i]] == "Scenario A"
                          else np.nan)
             for i in range(14)}

    rows.append({
        "role": role,
        "belief_raw": belief_raw,
        "belief_sc": belief_sc,
        "team_score": team_score,
        **indiv,
    })

adf = pd.DataFrame(rows)
adf["role_short"] = adf["role"].map({MANAGER: "Manager", DEVELOPER: "Developer"})

# Assign belief group
def assign_group(sc):
    for grp, vals in BELIEF_GROUPS.items():
        if sc in vals:
            return grp
    return np.nan

adf["belief_group"] = adf["belief_sc"].apply(assign_group)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SPEARMAN CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("SECTION 1 — SPEARMAN CORRELATION: Belief ↔ Team-Orientation")
print("=" * 65)

clean = adf[["belief_sc", "team_score", "role_short"]].dropna()
r_all, p_all = spearmanr(clean["belief_sc"], clean["team_score"])
print(f"\nAll respondents (n={len(clean)}): r={r_all:.3f}, p={p_all:.4f} "
      f"{'*** p<0.001' if p_all<0.001 else '** p<0.01' if p_all<0.01 else '* p<0.05' if p_all<0.05 else '(ns)'}")

for role in ["Manager", "Developer"]:
    sub = clean[clean["role_short"] == role]
    if len(sub) >= 4:
        r, p = spearmanr(sub["belief_sc"], sub["team_score"])
        print(f"  {role:<12} (n={len(sub):>2}): r={r:.3f}, p={p:.4f} "
              f"{'*** p<0.001' if p<0.001 else '** p<0.01' if p<0.01 else '* p<0.05' if p<0.05 else '(ns)'}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. GROUP COMPARISON: team-orientation by belief group
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("SECTION 2 — TEAM-ORIENTATION SCORE by BELIEF GROUP")
print(f"{'=' * 65}")
print(f"\n{'Group':<18} {'n':>4} {'Mean':>6} {'Median':>7} {'SD':>6}")
print("-" * 42)

group_data = {}
for grp in BELIEF_GROUPS:
    g = adf[adf["belief_group"] == grp]["team_score"].dropna()
    group_data[grp] = g
    if len(g):
        print(f"{grp.replace(chr(10),' '):<18} {len(g):>4} {g.mean():>6.3f} "
              f"{g.median():>7.3f} {g.std():>6.3f}")

valid_groups = [g for g in group_data.values() if len(g) > 0]
if len(valid_groups) >= 2:
    h, p_kw = kruskal(*valid_groups)
    print(f"\nKruskal-Wallis H={h:.3f}, p={p_kw:.4f} "
          f"{'*** p<0.001' if p_kw<0.001 else '** p<0.01' if p_kw<0.01 else '* p<0.05' if p_kw<0.05 else '(ns)'}")

pairs = list(combinations(list(BELIEF_GROUPS.keys()), 2))
print(f"\nPairwise Mann-Whitney U (no correction):")
print(f"  {'Pair':<34} {'U':>6} {'p':>8} {'sig':>5}")
print("  " + "-" * 58)
for g1k, g2k in pairs:
    g1, g2 = group_data.get(g1k, pd.Series([])), group_data.get(g2k, pd.Series([]))
    if len(g1) > 0 and len(g2) > 0:
        u, p = mannwhitneyu(g1, g2, alternative="two-sided")
        lbl = g1k.replace("\n", " ") + " vs " + g2k.replace("\n", " ")
        sig = ("***" if p < 0.001 else "**" if p < 0.01
               else "*" if p < 0.05 else "(ns)")
        print(f"  {lbl:<34} {u:>6.1f} {p:>8.4f} {sig:>5}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PER-SCENARIO % CHOOSING B by BELIEF GROUP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("SECTION 3 — % CHOOSING TEAM (B) PER SCENARIO × BELIEF GROUP")
print(f"{'=' * 65}")
print(f"\n{'Scenario':<42} " +
      " ".join(f"{g.replace(chr(10),' '):>14}" for g in BELIEF_GROUPS))
print("-" * 86)

scenario_heatmap = np.zeros((14, len(BELIEF_GROUPS)))
for si in range(14):
    col = f"S{si+1}"
    row_parts = [f"S{si+1}: {SCENARIO_LABELS[si]:<38}"[:42]]
    for gi, grp in enumerate(BELIEF_GROUPS):
        sub = adf[adf["belief_group"] == grp][col].dropna()
        pct = sub.mean() * 100 if len(sub) else np.nan
        scenario_heatmap[si, gi] = pct if not np.isnan(pct) else -1
        row_parts.append(f"{pct:>13.1f}%" if not np.isnan(pct) else f"{'n/a':>14}")
    print(" ".join(row_parts))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Scatter: belief score vs team-orientation (by role)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))

for role, color in [("Manager", MGR_COLOR), ("Developer", DEV_COLOR)]:
    sub = adf[adf["role_short"] == role][["belief_sc", "team_score"]].dropna()
    jitter = np.random.default_rng(99).uniform(-0.08, 0.08, size=len(sub))
    ax.scatter(sub["belief_sc"] + jitter, sub["team_score"],
               color=color, alpha=0.65, s=55, label=f"{role} (n={len(sub)})")

# Trend line (all)
clean2 = adf[["belief_sc", "team_score"]].dropna()
if len(clean2) > 3:
    z = np.polyfit(clean2["belief_sc"], clean2["team_score"], 1)
    xline = np.linspace(0.8, 5.2, 100)
    ax.plot(xline, np.polyval(z, xline), color="black",
            linewidth=1.5, linestyle="--", alpha=0.6, label="Trend (all)")

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["Strongly\nDisagree", "Disagree", "Neutral",
                    "Agree", "Strongly\nAgree"], fontsize=9)
ax.set_ylabel("Team-orientation score\n(fraction of scenarios choosing B)", fontsize=10)
ax.set_xlim(0.5, 5.5)
ax.set_ylim(-0.05, 1.1)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
path = OUT_DIR + "teambelief_fig1_scatter.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {path}")
plt.close()

print("\n✓ Done.")