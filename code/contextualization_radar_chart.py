"""
FILE: contextualization_radar_chart.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Visualizes how contextualization beliefs relate to perceived usefulness of 
    different metric categories using a radar/spider chart. This script:
    1. Builds a contextualization score per respondent (using role-appropriate columns)
    2. Groups respondents by context level (Low ≤3 / High ≥4)
    3. Calculates mean usefulness ratings for each metric category per group
    4. Generates a radar chart comparing high-context vs. low-context groups
    5. Prints supporting statistics (Spearman correlations and MWU tests) to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/ctx_fig2_radar.png - Radar chart comparing mean usefulness 
                                          ratings between high and low context groups

USAGE:
    python3 code/contextualization_radar_chart.py

NOTES:
    - Uses role-appropriate contextualization columns:
      - Managers: [45] "Productivity metrics must be contextualized."
      - Developers: [57] "Productivity metrics must be contextualized." (.1)
    - Metric usefulness categories: cols [82]–[90] (Satisfaction & Wellbeing, 
      Communication & Collaboration, Output & Efficiency, Quality, 
      Value & Outcomes, Learning & Growth, Technical Debt & Maintainability, 
      Developer Experience, Process & Predictability)
    - Radar chart shows mean usefulness scores on a 1-5 Likert scale
    - High context = scores 4-5, Low context = scores 1-3
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - No Bonferroni correction applied (exploratory analysis)
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."

MGR_CONTEXT_COL = " [Productivity metrics must be contextualized.]"
DEV_CONTEXT_COL = " [Productivity metrics must be contextualized.].1"

LIKERT_SCORE = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}
USEFUL_SCORE = {
    "Not useful at all": 1,
    "Not very useful":   2,
    "Neutral":           3,
    "Somewhat useful":   4,
    "Very useful":       5,
}
MGR_COLOR  = "#2563EB"
DEV_COLOR  = "#DC2626"
LOW_COLOR  = "#F87171"
HIGH_COLOR = "#34D399"

# LOAD
df  = pd.read_csv(DATA_PATH)

USEFULNESS_COLS = {
    "Satisfaction &\nWellbeing":           "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Satisfaction and Wellbeing]",
    "Communication &\nCollaboration":      "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Communication and Collaboration]",
    "Output &\nEfficiency":                "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Output &amp; Efficiency]",
    "Quality":                             "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Quality]",
    "Value &\nOutcomes":                   "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Value &amp; Outcomes]",
    "Learning &\nGrowth":                  "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Learning &amp; Growth]",
    "Technical Debt &\nMaintainability":   "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Technical Debt &amp; Maintainability]",
    "Developer\nExperience":               "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Developer Experience]",
    "Process &\nPredictability":           "For each metric category below, indicate whether you consider metrics in this category to be useful for assessing productivity.  [Process &amp; Predictability]",
}

# ADDITIONAL FUNCTIONS
def rank_biserial(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
    return 1 - (2 * u) / (n1 * n2)

def sig_label(p):
    """Convert p-value to significance stars (no correction)."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "(ns)"



rows = []
for _, row in df.iterrows():
    role = row[ROLE_COL]
    if role == MANAGER:
        ctx_raw = row[MGR_CONTEXT_COL]
    elif role == DEVELOPER:
        ctx_raw = row[DEV_CONTEXT_COL]
    else:
        continue
    ctx_sc = LIKERT_SCORE.get(ctx_raw, np.nan)
    if pd.isna(ctx_sc):
        continue

    util_scores = {lbl: USEFUL_SCORE.get(row[col], np.nan)
                   for lbl, col in USEFULNESS_COLS.items()}

    rows.append({
        "role":    role,
        "ctx_sc":  ctx_sc,
        "ctx_raw": ctx_raw,
        **util_scores,
    })

adf = pd.DataFrame(rows)
adf["composite_util"] = adf[list(USEFULNESS_COLS.keys())].mean(axis=1)
adf["ctx_group"] = adf["ctx_sc"].apply(lambda x: "High (4–5)" if x >= 4 else "Low (1–3)")

print("=" * 70)
print("SAMPLE SIZES")
print("=" * 70)
print(f"\nTotal analysed: {len(adf)}")
print(f"Context group distribution:")
print(adf["ctx_group"].value_counts().to_string())
print(f"\nRole distribution:")
print(adf["role"].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE USEFULNESS BY CONTEXT GROUP
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("DESCRIPTIVE: Mean Usefulness by Context Group")
print(f"{'=' * 70}")

for grp in ["High (4–5)", "Low (1–3)"]:
    sub = adf[adf["ctx_group"] == grp]
    composite = sub["composite_util"].dropna()
    print(f"\n  {grp} (n={len(sub)}): composite mean={composite.mean():.2f}, "
          f"SD={composite.std():.2f}")
    for lbl in USEFULNESS_COLS:
        s = sub[lbl].dropna()
        if len(s):
            print(f"    {lbl.replace(chr(10),' '):<35}: mean={s.mean():.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPEARMAN CORRELATION: context_score ↔ each usefulness category (NO Bonferroni)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SPEARMAN CORRELATIONS (no correction)")
print(f"{'=' * 70}")
print(f"\n  {'Category':<38} {'n':>4} {'r':>7} {'p':>9} {'sig':>6}")
print("  " + "-" * 68)

corr_results = {}
for lbl in USEFULNESS_COLS:
    clean = adf[["ctx_sc", lbl]].dropna()
    if len(clean) >= 5:
        r, p = spearmanr(clean["ctx_sc"], clean[lbl])
        corr_results[lbl] = (r, p, len(clean))
        print(f"  {lbl.replace(chr(10),' '):<38} {len(clean):>4} "
              f"{r:>7.3f} {p:>9.4f} {sig_label(p):>6}")

# Composite
clean_comp = adf[["ctx_sc", "composite_util"]].dropna()
r_comp, p_comp = spearmanr(clean_comp["ctx_sc"], clean_comp["composite_util"])
print(f"\n  {'Composite (all 9 categories)':<38} {len(clean_comp):>4} "
      f"{r_comp:>7.3f} {p_comp:>9.4f} {sig_label(p_comp):>6}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MWU: High vs Low context group per usefulness category (NO Bonferroni)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("MWU: High-Context vs Low-Context Group per Usefulness Category (no correction)")
print(f"{'=' * 70}")
print(f"\n  {'Category':<38} {'U':>7} {'p':>9} {'sig':>6} {'r':>7}")
print("  " + "-" * 68)

high_grp = adf[adf["ctx_group"] == "High (4–5)"]
low_grp  = adf[adf["ctx_group"] == "Low (1–3)"]

for lbl in list(USEFULNESS_COLS.keys()) + ["composite_util"]:
    g_high = high_grp[lbl].dropna()
    g_low  = low_grp[lbl].dropna()
    if len(g_high) > 0 and len(g_low) > 0:
        u, p = mannwhitneyu(g_high, g_low, alternative="two-sided")
        rb   = rank_biserial(g_high, g_low)
        display_lbl = lbl.replace("\n", " ")
        print(f"  {display_lbl:<38} {u:>7.1f} {p:>9.4f} "
              f"{sig_label(p):>6} {rb:>7.3f}")

cat_labels = list(USEFULNESS_COLS.keys())
all_r = []
all_p = []

for lbl in cat_labels:
    r, p, n = corr_results.get(lbl, (np.nan, np.nan, 0))
    all_r.append(r)
    all_p.append(p)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Radar chart: mean usefulness by context group
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Figure 2] Generating radar chart …")

short_labels = [
    "Satisf.\n& Well.",
    "Comm. &\nCollab.",
    "Output &\nEffic.",
    "Quality",
    "Value &\nOutc.",
    "Learn. &\nGrowth",
    "Tech\nDebt",
    "Dev\nExp.",
    "Process &\nPredict.",
]

N = len(cat_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for grp, color, ls in [("High (4–5)", HIGH_COLOR, "-"),
                        ("Low (1–3)",  LOW_COLOR,  "--")]:
    sub = adf[adf["ctx_group"] == grp]
    vals = [sub[lbl].dropna().mean() for lbl in cat_labels]
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2.5, linestyle=ls,
            label=f"Context {grp} (n={len(sub)})")
    ax.fill(angles, vals, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(short_labels, fontsize=12)
ax.set_ylim(1, 5)
ax.set_yticks([2, 3, 4, 5])
ax.set_yticklabels(["2", "3", "4", "5"], fontsize=11)

ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=13)
ax.tick_params(axis='both', which='major', labelsize=11)
plt.tight_layout()
path = OUT_DIR + "ctx_fig2_radar.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"  Saved → {path}")
plt.close()

print("\n✓ Done.")