"""
FILE: misuse_fairness_gaming_triad.py
PROJECT: Measuring What Really Matters: Individual vs Team Productivity in Software Engineering
AUTHOR: Matilde Faro Martins Castelo Pires
CONTACT: matildefaro.work@gmail.com
DATE CREATED: 26 May 2026
DATE LAST MODIFIED: 26 June 2026
VERSION: 1.0

DESCRIPTION:
    Analyzes the relationships between three developer perceptions:
    1. Manager misuses productivity metrics
    2. Assessment is fair
    3. Metrics incentivize gaming
    
    This script:
    1. Filters respondents to Developers only
    2. Maps Likert responses to numeric scores (1-5)
    3. Calculates Spearman correlations between all three pairs
    4. Generates three scatter plots with correlation statistics
    5. Prints summary statistics to console

DEPENDENCIES:
    Python 3.7+
    Packages: pandas, numpy, matplotlib, scipy

INPUT:
    data/results-survey331585.csv - Raw survey data from Qualtrics export

OUTPUT:
    results/analysis5_misuse_fairness_gaming_triad.png - Three-panel scatter plot matrix

USAGE:
    python3 code/misuse_fairness_gaming_triad.py

NOTES:
    - Developers only (managers are filtered out)
    - Likert scale: 1=Strongly Disagree ... 5=Strongly Agree
    - Spearman correlations (two-sided)
    - Significance stars: * p<0.05, ** p<0.01, *** p<0.001
    - Run from the project root directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "data/results-survey331585.csv"
OUT_DIR   = "results/"

ROLE_COL  = "Which description best fits your role?"
DEVELOPER = "I develop software under a tech lead or manager role."

LIKERT_ORDER = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
LIKERT_SCORE = {v: i + 1 for i, v in enumerate(LIKERT_ORDER)}

# LOAD DEVELOPERS ONLY
df  = pd.read_csv(DATA_PATH)
dev = df[df[ROLE_COL] == DEVELOPER].copy()
N_DEV = len(dev)
print(f"Developers: n={N_DEV}")

# COLUMN DEFINITIONS
C = {
    "dev_misuse": " [My manager misuses productivity metrics.]",
    "dev_fair":   " [I consider my manager\u2019s assessment of my productivity fair.]",
    "dev_gaming": " [Providing productivity metrics explicitly incentivizes developers to  gamify metrics.].1",
}

# ADDITIONAL FUNCTIONS
def score(subset, key):
    """Convert Likert responses to numeric scores."""
    s = subset[C[key]].map(LIKERT_SCORE)
    return s.dropna()

def sig(p):
    #Convert p-value to significance star.
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# GENERATE OUTPUT
print("\n" + "=" * 65)
print("ANALYSIS 5: MISUSE–FAIRNESS–GAMING TRIAD (Developers)")
print("=" * 65)

# Get scores
s_mis  = score(dev, "dev_misuse")
s_fair = score(dev, "dev_fair")
s_gam  = score(dev, "dev_gaming")

# Print summary
print(f"\n  Developer responses (n≈{N_DEV}):")
print(f"    'Manager misuses metrics' - mean: {s_mis.mean():.2f}, SD: {s_mis.std():.2f}")
print(f"    'Assessment is fair'      - mean: {s_fair.mean():.2f}, SD: {s_fair.std():.2f}")
print(f"    'Metrics incentivize gaming' - mean: {s_gam.mean():.2f}, SD: {s_gam.std():.2f}")

# Create figure with 3 panels
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

pairs = [
    (s_mis,  s_fair, "dev_misuse",  "dev_fair",
     '"Manager misuses metrics"', '"Assessment is fair"',
     "Misuse vs Fairness", "#7C3AED"),
    (s_mis,  s_gam,  "dev_misuse",  "dev_gaming",
     '"Manager misuses metrics"', '"Metrics incentivize gaming"',
     "Misuse vs Gaming belief", "#DC7633"),
    (s_fair, s_gam,  "dev_fair",    "dev_gaming",
     '"Assessment is fair"', '"Metrics incentivize gaming"',
     "Fairness vs Gaming belief", "#16A34A"),
]

print("\n  Correlations:")
for ax, (sa, sb, ka, kb, xlabel, ylabel, title, color) in zip(axes, pairs):
    # Get common respondents
    common = sa.index.intersection(sb.index)
    xc, yc = sa.loc[common], sb.loc[common]
    n = len(xc)
    
    # Spearman correlation
    r, p = spearmanr(xc, yc)
    print(f"    {title}: r={r:.3f}, p={p:.4f} ({sig(p)}), n={n}")
    
    # Scatter plot with jitter
    jitter = 0.12
    ax.scatter(xc + np.random.uniform(-jitter, jitter, n),
               yc + np.random.uniform(-jitter, jitter, n),
               alpha=0.6, s=55, color=color, edgecolors="white", linewidth=0.5)
    
    # Regression line
    z = np.polyfit(xc, yc, 1)
    ax.plot([1, 5], np.poly1d(z)([1, 5]), "k--", linewidth=1.8, alpha=0.6,
            label=f"slope={z[0]:.2f}")
    
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["SD","D","N","A","SA"], fontsize=8)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["SD","D","N","A","SA"], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
path = OUT_DIR + "analysis5_misuse_fairness_gaming_triad.png"
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Saved → {path}")
print("\n✓ Done.")