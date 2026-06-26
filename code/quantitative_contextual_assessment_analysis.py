"""
FILE: quantitative_contextual_assessment_analysis.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Compares managers' and developers' agreement on quantitative vs. contextual 
    productivity assessment. Both roles answered conceptually identical statements:
    
    Quantitative:
      MGR [44] "Productivity assessment should be quantitative."
      DEV [56] "Productivity assessment should be quantitative." (.1)
    
    Contextual:
      MGR [45] "Productivity metrics must be contextualized."
      DEV [57] "Productivity metrics must be contextualized." (.1)
    
    This script:
    1. Filters respondents into Managers and Developers
    2. Maps Likert responses to numeric scores (1-5)
    3. Calculates descriptive statistics for all four series
    4. Performs within-role Wilcoxon tests (Quant vs. Context)
    5. Performs cross-role Mann-Whitney U tests
    6. Calculates tension scores (Context − Quant) per respondent
    7. Generates a 2x2 diverging Likert bar chart

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/qvsc_fig1_diverging_bar.png - 2x2 diverging bar chart

USAGE:
    python3 code/quantitative_contextual_assessment_analysis.py

NOTES:
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses from scoring
    - Within-role Wilcoxon signed-rank test (two-sided)
    - Cross-role Mann-Whitney U test (two-sided)
    - Rank-biserial correlation for effect size
    - Tension score = Context − Quant (positive = favors contextual)
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
MANAGER   = "I manage other software development professionals."
DEVELOPER = "I develop software under a tech lead or manager role."
MGR_QUANT_COL   = " [Productivity assessment should be quantitative.]"
DEV_QUANT_COL   = " [Productivity assessment should be quantitative.].1"
MGR_CONTEXT_COL = " [Productivity metrics must be contextualized.]"
DEV_CONTEXT_COL = " [Productivity metrics must be contextualized.].1"

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
MGR_COLOR   = "#2563EB"
DEV_COLOR   = "#DC2626"
QUANT_COLOR = "#D97706"
CTX_COLOR   = "#059669"

# LOAD AND SPLIT
df  = pd.read_csv(DATA_PATH)

mgr = df[df[ROLE_COL] == MANAGER].copy()
dev = df[df[ROLE_COL] == DEVELOPER].copy()


# ADDITIONAL FUNCTIONS
def score(series):
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
print(f"\n{'Variable':<32} {'n':>4} {'Mean':>6} {'Median':>7} {'SD':>6} {'%Agree':>8}")
print("-" * 60)
entries = [
    ("MGR  Quantitative",  MGR_QUANT_COL,   mgr),
    ("MGR  Contextual",    MGR_CONTEXT_COL, mgr),
    ("DEV  Quantitative",  DEV_QUANT_COL,   dev),
    ("DEV  Contextual",    DEV_CONTEXT_COL, dev),
]
for lbl, col, subset in entries:
    s = score(subset[col])
    print(f"{lbl:<32} {len(s):>4} {s.mean():>6.2f} {s.median():>7.1f} "
          f"{s.std():>6.2f} {pct_agree(subset[col]):>7.1f}%")

print("\nDistribution counts:")
for lbl, col, subset in entries:
    counts = {k: (subset[col] == k).sum() for k in LIKERT_ORDER}
    dna    = (subset[col] == "Does Not Apply").sum()
    print(f"  {lbl}: {counts}  DNA={dna}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. WITHIN-ROLE WILCOXON: Quant vs Context
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("WITHIN-ROLE WILCOXON: Quantitative vs Contextual")
print(f"{'=' * 70}")

for role_lbl, subset, qcol, ccol in [
    ("Managers",   mgr, MGR_QUANT_COL, MGR_CONTEXT_COL),
    ("Developers", dev, DEV_QUANT_COL, DEV_CONTEXT_COL),
]:
    tmp = subset[[qcol, ccol]].copy()
    tmp["q"] = tmp[qcol].map(LIKERT_SCORE)
    tmp["c"] = tmp[ccol].map(LIKERT_SCORE)
    paired = tmp.dropna()
    diffs  = paired["c"] - paired["q"]

    print(f"\n  {role_lbl} (n={len(paired)} paired):")
    print(f"    Quantitative: mean={paired['q'].mean():.2f}, median={paired['q'].median():.1f}")
    print(f"    Contextual:   mean={paired['c'].mean():.2f}, median={paired['c'].median():.1f}")
    print(f"    Context−Quant: mean={diffs.mean():+.2f}, median={diffs.median():+.1f}")
    print(f"    Context > Quant: {(diffs>0).sum()},  Equal: {(diffs==0).sum()},  Quant > Context: {(diffs<0).sum()}")
    if not (diffs == 0).all() and len(diffs.dropna()) >= 4:
        w, p = wilcoxon(paired["q"], paired["c"])
        print(f"    Wilcoxon W={w:.1f}, p={p:.4f}  {sig_label(p)}")
    else:
        print("    Wilcoxon: insufficient variation.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-ROLE MWU
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("CROSS-ROLE MWU: Managers vs Developers")
print(f"{'=' * 70}")

for lbl, mgr_col, dev_col in [
    ("Quantitative", MGR_QUANT_COL,   DEV_QUANT_COL),
    ("Contextual",   MGR_CONTEXT_COL, DEV_CONTEXT_COL),
]:
    mg = score(mgr[mgr_col])
    dv = score(dev[dev_col])
    u, p = mannwhitneyu(mg, dv, alternative="two-sided")
    rb   = rank_biserial(mg, dv)
    print(f"\n  {lbl}:")
    print(f"    MGR (n={len(mg)}): mean={mg.mean():.2f}, median={mg.median():.1f}")
    print(f"    DEV (n={len(dv)}): mean={dv.mean():.2f}, median={dv.median():.1f}")
    print(f"    MWU U={u:.1f}, p={p:.4f}  {sig_label(p)},  r={rb:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. TENSION SCORE: Context − Quant per respondent
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TENSION SCORE (Context − Quant) per Respondent")
print(f"{'=' * 70}")

for role_lbl, subset, qcol, ccol in [
    ("Managers",   mgr, MGR_QUANT_COL, MGR_CONTEXT_COL),
    ("Developers", dev, DEV_QUANT_COL, DEV_CONTEXT_COL),
]:
    tmp = subset[[qcol, ccol]].copy()
    tmp["q"] = tmp[qcol].map(LIKERT_SCORE)
    tmp["c"] = tmp[ccol].map(LIKERT_SCORE)
    paired = tmp.dropna()
    tension = paired["c"] - paired["q"]
    print(f"\n  {role_lbl} (n={len(paired)}):")
    print(f"    Tension mean={tension.mean():+.2f}, median={tension.median():+.1f}, SD={tension.std():.2f}")
    print(f"    Context > Quant: {(tension>0).sum()},  Equal: {(tension==0).sum()},  Quant > Context: {(tension<0).sum()}")

# Cross-role tension MWU
mgr_t = mgr[[MGR_QUANT_COL, MGR_CONTEXT_COL]].copy()
mgr_t["q"] = mgr_t[MGR_QUANT_COL].map(LIKERT_SCORE)
mgr_t["c"] = mgr_t[MGR_CONTEXT_COL].map(LIKERT_SCORE)
mgr_t = mgr_t.dropna(); mgr_tension = mgr_t["c"] - mgr_t["q"]

dev_t = dev[[DEV_QUANT_COL, DEV_CONTEXT_COL]].copy()
dev_t["q"] = dev_t[DEV_QUANT_COL].map(LIKERT_SCORE)
dev_t["c"] = dev_t[DEV_CONTEXT_COL].map(LIKERT_SCORE)
dev_t = dev_t.dropna(); dev_tension = dev_t["c"] - dev_t["q"]

u_t, p_t = mannwhitneyu(mgr_tension, dev_tension, alternative="two-sided")
print(f"\n  Cross-role tension MWU: U={u_t:.1f}, p={p_t:.4f}  {sig_label(p_t)}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2×2 diverging Likert bar
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5))

panel_meta = [
    ("MGR — Quantitative",  MGR_QUANT_COL,   mgr),
    ("MGR — Contextual",    MGR_CONTEXT_COL, mgr),
    ("DEV — Quantitative",  DEV_QUANT_COL,   dev),
    ("DEV — Contextual",    DEV_CONTEXT_COL, dev),
]

for i, (lbl, col, subset) in enumerate(panel_meta):
    n = score(subset[col]).count()
    left_neg = left_pos = 0.0
    for level in LIKERT_ORDER:
        pct = (subset[col] == level).sum() / n * 100 if n else 0
        if level in ("Strongly Disagree", "Disagree"):
            ax.barh(i, -pct, left=-left_neg, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            if pct > 5:
                ax.text(-left_neg - pct / 2, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8.5, color="white")
            left_neg += pct
        elif level == "Neutral":
            ax.barh(i,  pct / 2, left= left_pos, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            ax.barh(i, -pct / 2, left=-left_neg, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            if pct > 5:
                ax.text(left_pos + pct / 4, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8.5, color="black")
            left_neg += pct / 2
            left_pos += pct / 2
        else:
            ax.barh(i, pct, left=left_pos, color=COLORS_DIV[level],
                    edgecolor="white", height=0.6)
            if pct > 5:
                ax.text(left_pos + pct / 2, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8.5, color="white")
            left_pos += pct

ax.axhline(1.5, color="black", linewidth=0.7, linestyle=":")
ax.set_yticks(range(4))
ax.set_yticklabels([m[0] for m in panel_meta], fontsize=10)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("← Disagree    |    Agree →", fontsize=10)
legend_patches = [mpatches.Patch(color=COLORS_DIV[l], label=l) for l in LIKERT_ORDER]
ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
          ncol=5, bbox_to_anchor=(1.0, -0.22))
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
path = OUT_DIR + "qvsc_fig1_diverging_bar.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {path}")
plt.close()

print("\n✓ Done.")
