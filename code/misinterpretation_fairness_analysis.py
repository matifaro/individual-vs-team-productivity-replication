"""
FILE: misinterpretation_fairness_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 30 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes perceptions of misinterpretation and fairness difficulty in 
    productivity assessment, comparing managers and developers. This script:
    1. Filters respondents into Managers and Developers
    2. Calculates descriptive statistics for four perception variables:
       - Managers: "My assessment has been misinterpreted"
       - Managers: "Difficult to know fair assessment"
       - Developers: "Manager misuses metrics"
       - Developers: "Assessment feels fair"
    3. Performs Spearman correlations within roles
    4. Performs Wilcoxon signed-rank test for managers (misinterpret vs. fairness difficulty)
    5. Performs cross-construct MWU test (manager misinterpret vs. developer misuse)
    6. Generates a four-panel diverging bar chart showing full Likert distributions

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/misfair_fig3_diverging_all.png - Four-panel diverging bar chart

USAGE:
    python3 code/misinterpretation_fairness_analysis.py

NOTES:
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses from scoring
    - Spearman correlation (two-sided)
    - Wilcoxon signed-rank test (two-sided)
    - Mann-Whitney U test (two-sided)
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."
MGR_MISINTERP_COL = " [My assessment of productivity has been misinterpreted in the past.]"
MGR_FAIRHARD_COL  = " [It is difficult for me to know what is fair productivity assessment.]"
DEV_MISUSE_COL    = " [My manager misuses productivity metrics.]"
DEV_FAIR_COL      = " [I consider my manager\u2019s assessment of my productivity fair.]"

LIKERT_SCORE = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}
LIKERT_ORDER = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

COLORS_DIV = {
    "Strongly Disagree": "#991B1B",
    "Disagree":          "#FCA5A5",
    "Neutral":           "#D1D5DB",
    "Agree":             "#6EE7B7",
    "Strongly Agree":    "#065F46",
}
MGR_COLOR = "#2563EB"
DEV_COLOR = "#DC2626"

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)
mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()



def score(series):
    """Map Likert labels → 1–5; 'Does Not Apply' → NaN."""
    return series.map(LIKERT_SCORE).dropna()

def pct_agree(series):
    vals = series.dropna()
    return (vals.isin(["Agree", "Strongly Agree"])).mean() * 100 if len(vals) else np.nan

def rank_biserial(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
    return 1 - (2 * u) / (n1 * n2)

def sig_label(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "(ns)"


# ══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("DESCRIPTIVE SUMMARY")
print("=" * 70)
print(f"\n{'Variable':<38} {'n':>4} {'Mean':>6} {'Median':>7} {'SD':>6} {'%Agree':>8}")
print("-" * 68)
entries = [
    ("MGR  Misinterpreted",     MGR_MISINTERP_COL, mgr),
    ("MGR  Fairness Difficult", MGR_FAIRHARD_COL,  mgr),
    ("DEV  Manager Misuse",     DEV_MISUSE_COL,    dev),
    ("DEV  Fair Assessment",    DEV_FAIR_COL,      dev),
]
for lbl, col, subset in entries:
    s = score(subset[col])
    print(f"{lbl:<38} {len(s):>4} {s.mean():>6.2f} {s.median():>7.1f} "
          f"{s.std():>6.2f} {pct_agree(subset[col]):>7.1f}%")

print(f"\n  Note: 'Does Not Apply' responses excluded from all scoring.")
print(f"\nDistribution counts (all Likert levels):")
for lbl, col, subset in entries:
    counts = {k: (subset[col] == k).sum() for k in LIKERT_ORDER}
    dna    = (subset[col] == "Does Not Apply").sum()
    print(f"  {lbl}: {counts}  DNA={dna}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SPEARMAN CORRELATIONS WITHIN ROLES
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("SPEARMAN CORRELATION WITHIN ROLES")
print(f"{'=' * 70}")

# Manager: misinterpret ↔ fairness-difficulty
mgr_df = mgr[[MGR_MISINTERP_COL, MGR_FAIRHARD_COL]].copy()
mgr_df["misinterp"] = mgr_df[MGR_MISINTERP_COL].map(LIKERT_SCORE)
mgr_df["fairhard"]  = mgr_df[MGR_FAIRHARD_COL].map(LIKERT_SCORE)
mgr_df = mgr_df.dropna()
r_mgr, p_mgr = spearmanr(mgr_df["misinterp"], mgr_df["fairhard"])
print(f"\n  Managers (n={len(mgr_df)}): Misinterpreted ↔ Fairness Difficult")
print(f"    Spearman r = {r_mgr:.3f},  p = {p_mgr:.4f}  {sig_label(p_mgr)}")

# Developer: misuse ↔ fairness perception (fairness reversed: high = NOT fair)
dev_df = dev[[DEV_MISUSE_COL, DEV_FAIR_COL]].copy()
dev_df["misuse"]   = dev_df[DEV_MISUSE_COL].map(LIKERT_SCORE)
dev_df["fairness"] = dev_df[DEV_FAIR_COL].map(LIKERT_SCORE)
dev_df = dev_df.dropna()
r_dev, p_dev = spearmanr(dev_df["misuse"], dev_df["fairness"])
print(f"\n  Developers (n={len(dev_df)}): Manager Misuse ↔ Fairness Perception")
print(f"    Spearman r = {r_dev:.3f},  p = {p_dev:.4f}  {sig_label(p_dev)}")
print(f"    (negative r expected: more misuse → less fair)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. WILCOXON WITHIN MANAGERS: misinterpret vs fairness difficulty
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("WILCOXON SIGNED-RANK (Managers): Misinterpreted vs Fairness Difficulty")
print(f"{'=' * 70}")

diffs = mgr_df["fairhard"] - mgr_df["misinterp"]
print(f"\n  n paired = {len(mgr_df)}")
print(f"  Misinterpreted:     mean={mgr_df['misinterp'].mean():.2f}, median={mgr_df['misinterp'].median():.1f}")
print(f"  Fairness Difficult: mean={mgr_df['fairhard'].mean():.2f}, median={mgr_df['fairhard'].median():.1f}")
print(f"  Difference (Fair-Hard − Misinterp): mean={diffs.mean():+.2f}, median={diffs.median():+.1f}")
print(f"  Shifted up={( diffs>0).sum()}, no change={( diffs==0).sum()}, down={( diffs<0).sum()}")

if not (diffs == 0).all() and len(diffs.dropna()) >= 4:
    w, p_wil = wilcoxon(mgr_df["misinterp"], mgr_df["fairhard"])
    print(f"\n  Wilcoxon W = {w:.1f},  p = {p_wil:.4f}  {sig_label(p_wil)}")
else:
    p_wil = 1.0
    print("\n  Wilcoxon: insufficient variation or sample.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-CONSTRUCT MWU: MGR misinterpret vs DEV misuse (parallel concepts)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CROSS-CONSTRUCT MWU: MGR Misinterpret vs DEV Manager-Misuse")
print(f"{'=' * 70}")

mg_mi = score(mgr[MGR_MISINTERP_COL])
dv_mu = score(dev[DEV_MISUSE_COL])
u_cross, p_cross = mannwhitneyu(mg_mi, dv_mu, alternative="two-sided")
rb_cross = rank_biserial(mg_mi, dv_mu)
print(f"\n  MGR Misinterpreted (n={len(mg_mi)}): mean={mg_mi.mean():.2f}, median={mg_mi.median():.1f}")
print(f"  DEV Manager Misuse (n={len(dv_mu)}): mean={dv_mu.mean():.2f}, median={dv_mu.median():.1f}")
print(f"\n  MWU U = {u_cross:.1f},  p = {p_cross:.4f}  {sig_label(p_cross)},  r = {rb_cross:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT 3 — Four-panel diverging bars
# ══════════════════════════════════════════════════════════════════════════════
panel_meta = [
    ("MGR — Misinterpreted",     MGR_MISINTERP_COL, mgr, MGR_COLOR),
    ("MGR — Fairness Difficult", MGR_FAIRHARD_COL,  mgr, MGR_COLOR),
    ("DEV — Manager Misuse",     DEV_MISUSE_COL,    dev, DEV_COLOR),
    ("DEV — Fair Assessment",    DEV_FAIR_COL,      dev, DEV_COLOR),
]

fig, ax = plt.subplots(figsize=(13, 6))
y_ticks = []

for i, (lbl, col, subset, _) in enumerate(panel_meta):
    n = score(subset[col]).count()
    left_neg = left_pos = 0.0
    for level in LIKERT_ORDER:
        pct = (subset[col] == level).sum() / n * 100 if n else 0
        if level in ("Strongly Disagree", "Disagree"):
            ax.barh(i, -pct, left=-left_neg, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            if pct > 5:
                ax.text(-left_neg - pct / 2, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=12, color="white")
            left_neg += pct
        elif level == "Neutral":
            ax.barh(i,  pct / 2, left= left_pos, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            ax.barh(i, -pct / 2, left=-left_neg, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            if pct > 5:
                ax.text(left_pos + pct / 4, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=12, color="black")
            left_neg += pct / 2
            left_pos += pct / 2
        else:
            ax.barh(i, pct, left=left_pos, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            if pct > 5:
                ax.text(left_pos + pct / 2, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=12, color="white")
            left_pos += pct
    y_ticks.append(lbl)

# Divider between manager and developer blocks
ax.axhline(1.5, color="black", linewidth=0.7, linestyle=":")

ax.set_yticks(range(4))
ax.set_yticklabels(y_ticks, fontsize=14)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("← Disagree    |    Agree →", fontsize=15)
legend_patches = [mpatches.Patch(color=COLORS_DIV[l], label=l) for l in LIKERT_ORDER]
ax.legend(handles=legend_patches, loc="lower right", fontsize=12,
          ncol=5, bbox_to_anchor=(1.0, -0.22))
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
path = OUT_DIR + "misfair_fig3_diverging_all.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"Saved → {path}")
plt.close()

print("\n✓ Done.")