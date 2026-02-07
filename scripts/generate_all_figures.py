#!/usr/bin/env python3
"""
Generate ALL figures for Paper 01 v4.2: Expanded effect sizes (15 studies, 67 ES)
Data sources:
  - ieq_papers.db: 3,828 papers (762 included after exclusions)
  - UNIFIED_EFFECT_SIZES_70_FINAL.csv: 67 ES from 15 studies (42 with d computed)
Figures: 1 (PRISMA), 2 (Trends), 3 (Heatmap), 4 (Forest), 5 (Distribution),
         6 (Task Complexity), 7 (Integration), S1 (Forest by IEQ)
"""

import sqlite3
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
from collections import defaultdict

# Paths
SCRIPT_DIR = Path(__file__).parent
DB_PATH = Path("/Users/sunghoahn/Obsidian/01_Vaults_AshOwl_research/02_Active_Research_Projects/02_01_LEEWO/09_Research_Engineering/01_IEQ_Research/Database/raw/ieq_papers.db")
CSV_PATH = Path("/Users/sunghoahn/Obsidian/01_Vaults_AshOwl_research/02_Active_Research_Projects/02_01_LEEWO/09_Research_Engineering/01_IEQ_Research/Analysis/meta_analysis/Paper_01_Meta/Paper_01_Paradigm_Shift/UNIFIED_EFFECT_SIZES_70_FINAL.csv")
OUTPUT_DIR = SCRIPT_DIR

# Publication-quality style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# DB filter for 762 included studies
INCLUSION_SQL = """
    WHERE (itq=1 OR iaq=1 OR ilq=1 OR isq=1 OR ispq=1 OR ivq=1)
      AND (outcome_physical=1 OR outcome_mental=1 OR outcome_productivity=1)
      AND (study_design IS NULL OR (study_design NOT LIKE '%Review%' AND study_design NOT LIKE '%Meta%'))
"""

# Colors for IEQ variables (v4.2: added IAQ)
IEQ_COLORS = {
    'ITQ': '#3498db',
    'ISQ': '#e74c3c',
    'ILQ': '#2ecc71',
    'IAQ': '#f39c12',
    'ISPQ': '#9b59b6',
    'IVQ': '#1abc9c',
}


def load_db_data():
    """Load all needed data from ieq_papers.db."""
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    data = {}

    # 1. Publication year distribution (762 studies)
    cur.execute(f"SELECT publication_year, COUNT(*) FROM papers {INCLUSION_SQL} GROUP BY publication_year ORDER BY publication_year")
    data['year_counts'] = {row[0]: row[1] for row in cur.fetchall() if row[0]}

    # 2. IEQ-Outcome matrix (762 studies)
    ieq_cols = ['itq', 'iaq', 'ilq', 'isq', 'ispq', 'ivq']
    outcome_cols = ['outcome_physical', 'outcome_mental', 'outcome_productivity']
    matrix = {}
    for ieq in ieq_cols:
        matrix[ieq] = {}
        for out in outcome_cols:
            cur.execute(f"SELECT COUNT(*) FROM papers {INCLUSION_SQL} AND {ieq}=1 AND {out}=1")
            matrix[ieq][out] = cur.fetchone()[0]
    data['matrix'] = matrix

    # 3. Total counts for PRISMA
    cur.execute("SELECT COUNT(*) FROM papers")
    data['total'] = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM papers WHERE (itq=1 OR iaq=1 OR ilq=1 OR isq=1 OR ispq=1 OR ivq=1) AND (outcome_physical=1 OR outcome_mental=1 OR outcome_productivity=1)")
    data['ieq_outcome'] = cur.fetchone()[0]

    cur.execute(f"""SELECT COUNT(*) FROM papers
        WHERE (itq=1 OR iaq=1 OR ilq=1 OR isq=1 OR ispq=1 OR ivq=1)
          AND (outcome_physical=1 OR outcome_mental=1 OR outcome_productivity=1)
          AND (study_design LIKE '%Review%' OR study_design LIKE '%Meta%')""")
    data['excluded_reviews'] = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(*) FROM papers {INCLUSION_SQL}")
    data['included'] = cur.fetchone()[0]

    conn.close()
    return data


def load_csv_data():
    """Load effect size data from UNIFIED_EFFECT_SIZES_70_FINAL.csv."""
    es_data = []
    with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            es_data.append(row)
    return es_data


def get_d_computed(es_data):
    """Filter ES with computed d values (Tier 1 Complete + Tier 2 Complete)."""
    return [row for row in es_data
            if row['Cohen_d'] and row['Cohen_d'].strip()
            and row['Status'] in ('Complete',)]


def figure_1_prisma(db_data, es_data):
    """Figure 1: PRISMA 2020 Flow Diagram with DB-verified numbers."""
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    blue, blue_b = '#f0f7ff', '#2563eb'
    red, red_b = '#fef2f2', '#dc2626'
    green, green_b = '#f0fdf4', '#16a34a'

    total = db_data['total']
    ieq_out = db_data['ieq_outcome']
    excluded_no_match = total - ieq_out
    excluded_rev = db_data['excluded_reviews']
    included = db_data['included']

    n_studies = len(set(r['Paper_ID'] for r in es_data))
    n_es = len(es_data)
    d_computed = len(get_d_computed(es_data))

    def add_box(x, y, w, h, text, color, border, fs=10):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.5",
                             facecolor=color, edgecolor=border, linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#6b7280', lw=1.5))

    ax.text(50, 97, 'PRISMA 2020 Flow Diagram', ha='center', fontsize=16, fontweight='bold')

    for label, ypos in [('Identification', 85), ('Screening', 65), ('Eligibility', 45), ('Included', 22)]:
        ax.text(3, ypos, label, rotation=90, ha='center', va='center', fontsize=12,
                fontweight='bold', color='#374151')

    add_box(15, 83, 55, 10,
            f'Records identified from databases\n(n = {total:,})\nSemantic Scholar, OpenAlex, PubMed, arXiv',
            blue, blue_b)
    arrow(42.5, 83, 42.5, 78)

    add_box(15, 68, 35, 9,
            f'Records classified by\nIEQ + Outcome binary flags\n(n = {total:,})',
            blue, blue_b)
    add_box(55, 68, 30, 9,
            f'No IEQ or Outcome\nvariable identified\n(n = {excluded_no_match:,})',
            red, red_b)
    arrow(50, 72.5, 55, 72.5)
    arrow(32.5, 68, 32.5, 63)

    add_box(15, 53, 35, 9,
            f'IEQ + Outcome studies\n(n = {ieq_out:,})',
            blue, blue_b)
    add_box(55, 53, 30, 9,
            f'Excluded:\nReview / Meta-analysis\n(n = {excluded_rev})',
            red, red_b)
    arrow(50, 57.5, 55, 57.5)
    arrow(32.5, 53, 32.5, 48)

    add_box(10, 36, 55, 11,
            f'Studies included in study frequency mapping\n(n = {included:,})\n'
            f'6 IEQ variables x 3 outcome domains = 18 combinations',
            green, green_b, fs=11)
    arrow(37.5, 36, 37.5, 31)

    add_box(15, 18, 45, 11,
            f'Studies with quantitative effect sizes\n(n = {n_studies} studies, {n_es} ES)\n'
            f'd computed: {d_computed} ES (Tier 1: 34, Tier 2: 8)',
            green, green_b, fs=11)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'Figure_1_PRISMA_2020_v2.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUTPUT_DIR / 'Figure_1_PRISMA_2020_v2.svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] Figure 1: PRISMA flow diagram")


def figure_2_publication_trends(db_data):
    """Figure 2: Annual publication trends from DB."""
    year_counts = db_data['year_counts']

    all_years = sorted(y for y in year_counts if y and 2014 <= y <= 2025)
    counts = [year_counts.get(y, 0) for y in all_years]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(all_years, counts, color='#3498db', alpha=0.8, width=0.7, edgecolor='#2980b9')

    peak_idx = np.argmax(counts)
    bars[peak_idx].set_color('#e74c3c')
    bars[peak_idx].set_edgecolor('#c0392b')

    ax.axvline(x=2020, color='gray', linestyle='--', alpha=0.5)
    ax.annotate('COVID-19', xy=(2020, max(counts)*0.9), fontsize=10, ha='center', color='gray')

    ax.annotate(f'Peak\n(n={counts[peak_idx]})', xy=(all_years[peak_idx], counts[peak_idx]),
                xytext=(all_years[peak_idx]+1, counts[peak_idx]*0.85),
                arrowprops=dict(arrowstyle='->', color='black', lw=1), fontsize=10, ha='center')

    early = sum(year_counts.get(y, 0) for y in range(2015, 2019))
    late = sum(year_counts.get(y, 0) for y in range(2022, 2026))
    ratio = late / early if early > 0 else 0

    ax.text(0.02, 0.95, f'2015-2018: {early} studies\n2022-2025: {late} studies\n({ratio:.1f}x increase)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Studies')
    ax.set_title(f'Annual Publication Trends (N={sum(counts)} included studies)')
    ax.set_xticks(all_years)
    ax.set_xticklabels([f"'{str(y)[-2:]}" for y in all_years], rotation=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_2_publication_trends.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure_2_publication_trends.svg', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 2: Publication trends")


def figure_3_heatmap(db_data):
    """Figure 3: IEQ-Outcome study count heatmap from DB."""
    matrix = db_data['matrix']
    ieq_labels = ['ITQ\n(Thermal)', 'IAQ\n(Air)', 'ILQ\n(Lighting)',
                  'ISQ\n(Sound)', 'ISPQ\n(Spatial)', 'IVQ\n(Visual)']
    ieq_keys = ['itq', 'iaq', 'ilq', 'isq', 'ispq', 'ivq']
    out_labels = ['Physical Health', 'Mental Health', 'Productivity']
    out_keys = ['outcome_physical', 'outcome_mental', 'outcome_productivity']

    data = np.array([[matrix[ieq][out] for out in out_keys] for ieq in ieq_keys])
    total_units = data.sum()

    fig, ax = plt.subplots(figsize=(9, 7))

    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Studies', rotation=270, labelpad=20)

    ax.set_xticks(range(len(out_labels)))
    ax.set_yticks(range(len(ieq_labels)))
    ax.set_xticklabels(out_labels)
    ax.set_yticklabels(ieq_labels)

    for i in range(len(ieq_keys)):
        for j in range(len(out_keys)):
            n = data[i, j]
            text_color = 'white' if n > data.max() * 0.6 else 'black'
            ax.text(j, i, f'{n}', ha='center', va='center',
                    color=text_color, fontsize=13, fontweight='bold')

    ax.set_title(f'Study Counts: IEQ-Outcome Combinations\n(N={db_data["included"]} studies, {int(total_units):,} study-outcome units)', pad=20)

    ax.set_xticks(np.arange(len(out_labels)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(ieq_labels)+1)-.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_3_study_frequency_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure_3_vote_counting_heatmap.svg', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 3: IEQ-Outcome heatmap")


def figure_4_forest(es_data):
    """Figure 4: Forest plot of 42 d-computed effect sizes grouped by IEQ variable."""
    d_rows = get_d_computed(es_data)

    groups = defaultdict(list)
    for row in d_rows:
        groups[row['IEQ_Category']].append(row)

    n_studies = len(set(r['Paper_ID'] for r in d_rows))
    n_es = len(d_rows)

    fig, ax = plt.subplots(figsize=(10, 14))

    y_pos = 0
    group_labels = {
        'ITQ': 'ITQ (Thermal)',
        'ISQ': 'ISQ (Sound)',
        'ILQ': 'ILQ (Lighting)',
        'IAQ': 'IAQ (Air Quality)',
    }
    y_ticks = []
    y_labels = []

    for ieq in ['ITQ', 'ISQ', 'ILQ', 'IAQ']:
        if ieq not in groups:
            continue
        items = groups[ieq]
        color = IEQ_COLORS.get(ieq, '#333')

        y_pos -= 1
        ax.text(-12, y_pos, f"{group_labels[ieq]} (k={len(items)})",
                fontsize=12, fontweight='bold', va='center', color=color)
        y_pos -= 0.5

        for row in items:
            d = float(row['Cohen_d'])
            se = float(row['SE']) if row['SE'] else 0
            ci_lo = float(row['CI_95_lower']) if row['CI_95_lower'] else d - 1.96*se
            ci_hi = float(row['CI_95_upper']) if row['CI_95_upper'] else d + 1.96*se
            n = int(row['n']) if row['n'] else 0
            tier = row['Tier']

            y_pos -= 1
            marker = 'o' if tier == 'Tier 1' else 's'
            ax.scatter(d, y_pos, s=max(30, min(n*2, 200)), c=color, zorder=3,
                       edgecolors='black', linewidths=0.5, marker=marker)
            ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=color, linewidth=1.5, zorder=2)

            label = f"{row['Outcome_Name']} ({row['Comparison']})"
            if len(label) > 40:
                label = label[:37] + '...'
            y_ticks.append(y_pos)
            y_labels.append(label)

            tier_tag = 'T1' if tier == 'Tier 1' else 'T2'
            ax.text(4.5, y_pos, f"[{tier_tag}] d={d:.2f} n={n}",
                    fontsize=7.5, va='center', family='monospace')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_title(f"Forest Plot: {n_es} Effect Sizes by IEQ Variable ({n_studies} of 15 studies with d computed)\n"
                 f"(Circle = Tier 1, Square = Tier 2)", fontsize=13)
    ax.set_xlim(-12, 8)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'Figure_4_forest_plot_by_activity.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'Figure_4_forest_plot_by_activity.svg', bbox_inches='tight')
    plt.close()
    print(f"[OK] Figure 4: Forest plot ({n_es} ES from {n_studies} studies)")


def figure_5_distribution(es_data):
    """Figure 5: Effect size distribution by IEQ variable (box + strip plot)."""
    d_rows = get_d_computed(es_data)

    groups = defaultdict(list)
    for row in d_rows:
        groups[row['IEQ_Category']].append(float(row['Cohen_d']))

    ieq_order = ['ITQ', 'ISQ', 'ILQ', 'IAQ']
    n_es = sum(len(groups.get(ieq, [])) for ieq in ieq_order)
    n_studies = len(set(r['Paper_ID'] for r in d_rows))

    fig, ax = plt.subplots(figsize=(10, 6))

    box_data = [groups.get(ieq, []) for ieq in ieq_order]
    positions = range(len(ieq_order))

    bp = ax.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for i, (patch, ieq) in enumerate(zip(bp['boxes'], ieq_order)):
        patch.set_facecolor(IEQ_COLORS[ieq])
        patch.set_alpha(0.4)

    np.random.seed(42)
    for i, ieq in enumerate(ieq_order):
        vals = groups.get(ieq, [])
        if not vals:
            continue
        jitter = np.random.normal(0, 0.08, len(vals))
        ax.scatter([i + j for j in jitter], vals, s=40, c=IEQ_COLORS[ieq],
                   edgecolors='black', linewidths=0.5, zorder=3, alpha=0.8)
        y_offset = min(vals) - 0.8 if min(vals) < 0 else -0.8
        ax.text(i, y_offset, f'k={len(vals)}', ha='center', fontsize=10, color=IEQ_COLORS[ieq])

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{ieq}\n({len(groups.get(ieq, []))} ES)' for ieq in ieq_order])
    ax.set_ylabel("Cohen's d")
    ax.set_title(f'Distribution of Effect Sizes by IEQ Variable\n({n_es} ES with d computed from {n_studies} of 15 studies)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_5_effect_size_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure_5_effect_size_distribution.svg', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 5: Effect size distribution")


def figure_6_task_complexity(es_data):
    """Figure 6: Effect sizes by cognitive task complexity."""
    d_rows = get_d_computed(es_data)

    complexity_map = {
        'Basic': 'Simple',
        'Intermediate': 'Intermediate',
        'Complex': 'Complex',
        'Physiological': 'Physiological',
        'Productivity': 'Intermediate',
        'Mental': 'Complex',
    }
    complexity_colors = {'Simple': '#e74c3c', 'Intermediate': '#f39c12', 'Complex': '#27ae60', 'Physiological': '#9b59b6'}

    groups = defaultdict(list)
    for row in d_rows:
        domain = row.get('Outcome_Domain', 'Unknown')
        complexity = complexity_map.get(domain, 'Unknown')
        if complexity != 'Unknown':
            groups[complexity].append({
                'd': float(row['Cohen_d']),
                'name': row['Outcome_Name'],
                'ieq': row['IEQ_Category'],
                'n': int(row['n']) if row['n'] else 0,
                'tier': row['Tier'],
            })

    n_es = sum(len(v) for v in groups.values())
    n_studies = len(set(r['Paper_ID'] for r in d_rows))

    fig, ax = plt.subplots(figsize=(10, 7))

    order = ['Simple', 'Intermediate', 'Complex', 'Physiological']
    all_x = []
    all_y = []
    all_c = []

    for i, cat in enumerate(order):
        items = groups.get(cat, [])
        np.random.seed(42 + i)
        jitter = np.random.normal(0, 0.12, len(items))
        for j, item in enumerate(items):
            x = i + jitter[j]
            all_x.append(x)
            all_y.append(item['d'])
            all_c.append(complexity_colors[cat])

    ax.scatter(all_x, all_y, s=80, c=all_c, edgecolors='black', linewidths=0.5, zorder=3, alpha=0.8)

    box_data = [[item['d'] for item in groups.get(cat, [])] for cat in order]
    valid_positions = [i for i, bd in enumerate(box_data) if bd]
    valid_data = [bd for bd in box_data if bd]
    if valid_data:
        bp = ax.boxplot(valid_data, positions=valid_positions, widths=0.5, patch_artist=True, showfliers=False)
        for k, patch in enumerate(bp['boxes']):
            patch.set_facecolor(complexity_colors[order[valid_positions[k]]])
            patch.set_alpha(0.2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(order)))
    counts = [len(groups.get(cat, [])) for cat in order]
    ax.set_xticklabels([f'{cat}\n(k={c})' for cat, c in zip(order, counts)])
    ax.set_ylabel("Cohen's d")
    ax.set_xlabel("Cognitive Task Complexity")
    ax.set_title(f"Effect Sizes by Cognitive Task Complexity\n({n_es} ES from {n_studies} of 15 studies with d computed, exploratory)")
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_7_subgroup_analysis.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure_7_subgroup_analysis.svg', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 6: Task complexity (saved as figure_7_subgroup_analysis.png)")


def figure_7_integration(db_data, es_data):
    """Figure 7: Integration of study frequency mapping (762) and effect-size analysis (15 studies)."""
    matrix = db_data['matrix']
    ieq_keys = ['itq', 'iaq', 'ilq', 'isq', 'ispq', 'ivq']
    ieq_labels = ['ITQ', 'IAQ', 'ILQ', 'ISQ', 'ISPQ', 'IVQ']

    vc_counts = []
    for ieq in ieq_keys:
        total = sum(matrix[ieq].values())
        vc_counts.append(total)

    es_counts = defaultdict(int)
    for row in es_data:
        es_counts[row['IEQ_Category']] += 1
    ma_counts = [es_counts.get(lab, 0) for lab in ieq_labels]

    n_studies = len(set(r['Paper_ID'] for r in es_data))
    n_es = len(es_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    x = np.arange(len(ieq_labels))
    bar_colors = [IEQ_COLORS.get(lab, '#999') for lab in ieq_labels]

    bars1 = ax1.bar(x, vc_counts, color=bar_colors, alpha=0.8, edgecolor='#333', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ieq_labels)
    ax1.set_ylabel('Study-Outcome Units')
    ax1.set_title(f'Study Frequency Mapping\n({db_data["included"]} studies, {sum(vc_counts):,} units)')
    for i, v in enumerate(vc_counts):
        ax1.text(i, v + 5, str(v), ha='center', fontsize=10, fontweight='bold')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)

    bars2 = ax2.bar(x, ma_counts, color=bar_colors, alpha=0.8, edgecolor='#333', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ieq_labels)
    ax2.set_ylabel('Effect Sizes')
    ax2.set_title(f'Descriptive Effect-Size Analysis\n({n_studies} studies, {n_es} ES)')
    for i, v in enumerate(ma_counts):
        ax2.text(i, v + 0.3, str(v), ha='center', fontsize=10, fontweight='bold')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)

    for i, v in enumerate(ma_counts):
        if v == 0:
            bars2[i].set_alpha(0.2)
            ax2.text(i, 0.5, 'No data', ha='center', fontsize=9, color='gray', style='italic')

    plt.suptitle('Integration: Study Frequency Mapping vs Descriptive Effect-Size Analysis', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figure_8_meta_regression.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'figure_8_meta_regression.svg', bbox_inches='tight')
    plt.close()
    print("[OK] Figure 7: Integration (saved as figure_8_meta_regression.png)")


def figure_s1_forest_by_ieq(es_data):
    """Supplementary Figure S1: Forest plot by IEQ (all ES with d values)."""
    d_rows = get_d_computed(es_data)

    groups = defaultdict(list)
    for row in d_rows:
        groups[row['IEQ_Category']].append(row)

    n_studies = len(set(r['Paper_ID'] for r in d_rows))
    group_labels = {
        'ITQ': 'ITQ (Thermal)',
        'ISQ': 'ISQ (Sound)',
        'ILQ': 'ILQ (Lighting)',
        'IAQ': 'IAQ (Air Quality)',
    }

    fig, ax = plt.subplots(figsize=(10, 12))

    y_pos = 0
    y_ticks = []
    y_labels = []

    for ieq in ['ITQ', 'ISQ', 'ILQ', 'IAQ']:
        items = groups.get(ieq, [])
        if not items:
            continue
        color = IEQ_COLORS.get(ieq, '#333')

        y_pos -= 1
        ax.text(-12, y_pos, f"{group_labels[ieq]} (k={len(items)})",
                fontsize=11, fontweight='bold', va='center', color=color)
        y_pos -= 0.5

        for row in items:
            d = float(row['Cohen_d'])
            ci_lo = float(row['CI_95_lower']) if row['CI_95_lower'] else d - 1
            ci_hi = float(row['CI_95_upper']) if row['CI_95_upper'] else d + 1
            n = int(row['n']) if row['n'] else 0

            y_pos -= 1
            marker = 'o' if row['Tier'] == 'Tier 1' else 's'
            ax.scatter(d, y_pos, s=max(30, min(n*2, 200)), c=color, zorder=3,
                       edgecolors='black', linewidths=0.5, marker=marker)
            ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=color, linewidth=1.5, zorder=2)

            label = f"[{row['Tier'][-1]}] {row['Outcome_Name']}"
            if len(label) > 35:
                label = label[:32] + '...'
            y_ticks.append(y_pos)
            y_labels.append(label)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_title(f"Supplementary Figure S1: {len(d_rows)} Effect Sizes by IEQ Variable ({n_studies} studies)", fontsize=13)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'forest_plot_by_ieq.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'forest_plot_by_ieq.svg', bbox_inches='tight')
    plt.close()
    print("[OK] Supplementary S1: Forest by IEQ")


def main():
    print("\n" + "="*60)
    print("Paper 01 v4.2: Generating ALL Figures from DB + CSV")
    print("="*60)

    print("\n[1/2] Loading DB data...")
    db_data = load_db_data()
    print(f"  Total: {db_data['total']:,}, IEQ+Outcome: {db_data['ieq_outcome']}, "
          f"Excluded reviews: {db_data['excluded_reviews']}, Included: {db_data['included']}")

    print("[2/2] Loading CSV data...")
    es_data = load_csv_data()
    d_computed = get_d_computed(es_data)
    n_studies = len(set(r['Paper_ID'] for r in es_data))
    print(f"  Total ES: {len(es_data)}, d computed: {len(d_computed)}, Studies: {n_studies}")

    print("\nGenerating figures...\n")

    figure_1_prisma(db_data, es_data)
    figure_2_publication_trends(db_data)
    figure_3_heatmap(db_data)
    figure_4_forest(es_data)
    figure_5_distribution(es_data)
    figure_6_task_complexity(es_data)
    figure_7_integration(db_data, es_data)
    figure_s1_forest_by_ieq(es_data)

    print("\n" + "="*60)
    print(f"All figures generated in: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
