# Replication Package - Measuring What Matters: Individual vs Team Productivity in Software Development

**Author:** Matilde Faro Martins Castelo Pires

**Contact:** matildefaro.work@gmail.com

**Repository:** Measuring What Matters: Individual vs Team Productivity in Software Development

**DOI:**

---
## Overview

This repository contains the complete replication package for the study:
> **"Measuring What Matters: Individual vs Team Productivity in Software Development"**

The package includes all materials, data, and code necessary to replicate the analysis presented in the study.

---
## Abstract

Productivity assessment sits at the centre of software project management, yet it remains one of the most contested and ambiguous concepts in software engineering. Existing research has produced frameworks and qualitative accounts of how productivity should be defined and measured, but empirical evidence on how practitioners actually experience and navigate assessment, especially the tension between individual- and team-level assessment, remains limited.

This study investigates how managers and developers conceptualise and assess productivity at individual and team levels, employing an exploratory sequential mixed-methods design: 7 pilot interviews with software engineering managers, followed by a cross-sectional survey administered to 20 managers and 36 developers. Four research questions guided the study, examining how productivity definitions diverge between roles, how each role perceives metric use and misuse, in what contexts metrics are considered useful and reliable, and under what conditions individual or team productivity is favoured.

The findings reveal that managers define productivity in output-oriented terms, while developers place significantly greater weight on collaborative activities. The greatest divergence in perceived metric usefulness occurs among metrics applicable to both individual and team analysis, where managers consistently rate them higher. Developers' awareness of the purpose of assessment is the primary predictor of criteria optimisation behaviour, while managers who acknowledge that metrics incentivise gaming do not reduce their use of the most gameable metric categories. Both roles strongly favour team-oriented behaviour in concrete trade-offs, but both also underestimate the other's collaborative orientation. The only consistent exception to team primacy is protecting individual focus time, which both roles prioritise over team availability.

These findings suggest that the misalignment documented in the literature is primarily perceptual rather than substantive. Responsible productivity assessment requires not just better metrics, but also clearer governance of how metrics are communicated and used across roles, and structures that protect focus time as a team norm.

**Keywords:** Software Engineering Productivity, Individual Productivity, Team Productivity, Productivity Metrics, Metric Misuse

---
## Data Availability

The data used in this study come from an original survey conducted between May and June 2026.

**Data Sharing:**
- Survey data is fully anonymised and is included in this package (`data/results-survey331585.csv`)
- No personally identifiable information is present

**Interview Data:**
- Interview transcripts are not shared to protect participant confidentiality
- The interview guide is provided in `materials/Interview_Guide.pdf` for procedural transparency

---
## Software Requirements

### Python Version
- Python 3.7 or higher

### Required Packages
- pandas>=1.3.0
- numpy>=1.21.0
- matplotlib>=3.4.0
- scipy>=1.7.0

---
## Instructions for Replication

Each script can be run individually:
```
# Example: Run a specific analysis
python3 code/productivity_definitions_analysis.py
```

Expected runtime per script: 10 seconds to 1 minute

All generated figures are saved to `results/` directory.

---
## Script Documentation

Each script contains a header with:
- FILE: Script filename
- PROJECT: Study title
- AUTHOR: Author name and contact
- DATA CREATED/MODIFIED: Version dates
- VERSIONS: Version number
- DESCRIPTION: What the script does
- DEPENDENCIES: Required packages
- INPUT: Required data files
- OUTPUT: Generated files
- USAGE: How to run the script

---
## Statistical Tests Used

The analysis strategy was designed to accommodate not only the ordinal nature of Likert-scale data but also the small, unequal group sizes and the exploratory objectives of the study. 

Descriptive statistics, including frequencies, percentages, means, medians, and standard deviations, were used to summarise responses and demographic characteristics.

For group comparisons, non-parametric tests were employed due to the ordinal nature of Likert-scale data. The Mann-Whitney U test was used to compare the two independent groups of managers and developers, while the Wilcoxon signed-rank test was applied for paired within-subject comparison. For comparisons involving three or more groups, e.g. work setup, the Kruskal-Wallis H test was used, followed by pairwise Mann-Whitney U tests where appropriate.

Pairwise associations between ordinal variables were assessed using Spearman's rank correlation coefficient (p). Effect sizes are reported as rank-biserial correlation (r) for Mann-Whitney U tests and as Spearman's ρ for correlations.

For binary outcomes, in multiple-choice questions, Fisher's exact test was used to compare proportions between groups.

All statistical tests were two-sided, and statistical significance was set at p=0.05.

**Significance Levels:**
- `***` - p<0.001
- `**`  - p<0.01
- `*`   - p<0.005
- `(ns)` - not significant

---
## Contact

For questions, issues, or clarification regarding this replication package, please contact:

**Matilde Faro Martins Castelo Pires**

Email: matildefaro.work@gmail.com

---
## Citation

If you use this replication package or its contents, please cite:

```
Pires, Matilde Faro (2026)
Measuring What Matters: Individual vs Team Productivity in Software Development
[Replication Package].
https://github.com/matifaro/individual-vs-team-productivity-replication/
```

---
## Version History
1.0 - 26/06/2026 - Initial release
