"""
FILE: quantitative_contextual_paired_arrows.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Visualizes the shift from quantitative to contextual assessment preferences
    within each role using paired arrow plots. Both roles answered conceptually 
    identical statements:
    
    Quantitative:
      MGR [44] "Productivity assessment should be quantitative."
      DEV [56] "Productivity assessment should be quantitative." (.1)
    
    Contextual:
      MGR [45] "Productivity metrics must be contextualized."
      DEV [57] "Productivity metrics must be contextualized." (.1)
    
    This script:
    1. Filters respondents into Managers and Developers
    2. Maps Likert responses to numeric scores (1-5)
    3. Creates paired arrow plots showing individual shifts from Quant to Context
    4. Includes mean trend lines and descriptive statistics
    5. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/qvsc_fig2_paired_arrows.png - Two-panel paired arrow plot

USAGE:
    python3 code/quantitative_contextual_paired_arrows.py

NOTES:
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Excludes "Does Not Apply" responses from scoring
    - Arrows: Green = shift toward contextual, Orange = shift toward quantitative,
              Gray = no change
    - Dashed line shows mean shift for each role
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
# FIGURE 2 — Paired arrow: Quant → Context within each role
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, (role_lbl, subset, qcol, ccol, role_color) in zip(axes, [
    ("Managers",   mgr, MGR_QUANT_COL, MGR_CONTEXT_COL, MGR_COLOR),
    ("Developers", dev, DEV_QUANT_COL, DEV_CONTEXT_COL, DEV_COLOR),
]):
    tmp = subset[[qcol, ccol]].copy()
    tmp["q"] = tmp[qcol].map(LIKERT_SCORE)
    tmp["c"] = tmp[ccol].map(LIKERT_SCORE)
    paired = tmp.dropna()
    jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(paired))

    for idx, (_, row_) in enumerate(paired.iterrows()):
        qs, cs = row_["q"], row_["c"]
        color = CTX_COLOR if cs > qs else (QUANT_COLOR if cs < qs else "#9CA3AF")
        ax.annotate("", xy=(1 + jitter[idx], cs), xytext=(0 + jitter[idx], qs),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.7))
    ax.scatter(np.zeros(len(paired)) + jitter, paired["q"],
               color=QUANT_COLOR, s=45, zorder=5, label="Quantitative")
    ax.scatter(np.ones(len(paired)) + jitter, paired["c"],
               color=CTX_COLOR, s=45, zorder=5, label="Contextual")
    ax.plot([0, 1], [paired["q"].mean(), paired["c"].mean()],
            color="black", linewidth=2, linestyle="--", zorder=6,
            label=f"Mean ({paired['q'].mean():.2f} → {paired['c'].mean():.2f})")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Quantitative", "Contextual"], fontsize=11)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["Strongly\nDisagree", "Disagree", "Neutral",
                        "Agree", "Strongly\nAgree"], fontsize=9)
    ax.set_title(f"{role_lbl} (n={len(paired)})", fontweight="bold", fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
path = OUT_DIR + "qvsc_fig2_paired_arrows.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
print(f"Saved → {path}")
plt.close()

print("\n✓ Done.")
