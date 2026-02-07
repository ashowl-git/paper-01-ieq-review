# Supplementary Materials: IEQ Systematic Review

**Effects of Indoor Environmental Quality on Occupant Health and Productivity: A Context-Dependent Systematic Review with Descriptive Effect-Size Analysis**

Ahn, S. H. & Lee, J. A. (2026)

Submitted to: Journal of the Architectural Institute of Korea (JAIK)

---

## Repository Structure

```
data/
  effect_sizes_67.csv          # 67 standardized effect sizes from 15 studies (Cohen's d)

scripts/
  generate_all_figures.py      # Figure generation (Python, matplotlib)
  effect_size_calculator.py    # Unified effect size calculator
  extract_effect_sizes.py      # Effect size extraction from database
  generate_tables.py           # Table generation (xlsx)

figures/
  figure_1_prisma.png          # PRISMA 2020 flow diagram
  figure_2_publication_trends.png  # Publication trends (2015-2025)
  figure_3_study_frequency_heatmap.png  # IEQ-outcome frequency mapping (762 studies)
  figure_4_forest_plot.png     # Forest plot by activity type
  figure_5_effect_size_distribution.png  # Effect size distribution by IEQ variable
  figure_6_subgroup_analysis.png  # Subgroup analysis by activity type
  figure_7_integration.png     # Integration of frequency mapping and effect-size analysis

supplementary_materials.md     # Full supplementary text (search strategy, classification criteria)
```

## Data Description

### Effect Sizes (data/effect_sizes_67.csv)

67 standardized effect sizes (Cohen's d) extracted from 15 empirical studies across 4 IEQ variables:
- ITQ (Indoor Thermal Quality): 6 studies
- IAQ (Indoor Air Quality): 4 studies
- ILQ (Indoor Lighting Quality): 3 studies
- ISQ (Indoor Sound Quality): 2 studies

Effect size coding: positive d = IEQ improvement leads to better health/productivity outcome.

### Search Strategy

- Databases: Semantic Scholar, OpenAlex, PubMed, arXiv
- Period: January 2015 - October 2025
- Initial results: 3,828 papers
- After screening: 762 empirical studies (frequency mapping), 15 studies (effect-size analysis)
- Full search strategy details: see `supplementary_materials.md`

## Requirements

```
python >= 3.10
matplotlib
numpy
pandas
openpyxl
```

## License

This supplementary material is provided for academic reproducibility purposes.
