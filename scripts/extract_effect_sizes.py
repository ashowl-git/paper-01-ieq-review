#!/usr/bin/env python3
"""
Effect Size Extraction Script

Purpose: Extract Cohen's d effect sizes from 690 meta-analysis eligible papers
Author: LEEWO IEQ Research Team
Date: 2025-10-23
Version: 1.0
"""

import sqlite3
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import argparse
import json

# Database path
DB_PATH = Path(__file__).parent / "raw" / "ieq_papers.db"

class StatisticsExtractor:
    """Extract statistical information from text using regex"""

    def __init__(self):
        # Regex patterns for various statistical formats
        self.patterns = {
            # Mean ± SD patterns
            'mean_sd_pm': r'(\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)',
            'mean_sd_plus_minus': r'(\d+\.?\d*)\s*\+/-\s*(\d+\.?\d*)',
            'mean_sd_parentheses': r'(\d+\.?\d*)\s*\((\d+\.?\d*)\)',

            # t-value patterns
            't_value': r't\s*\((\d+)\)\s*=\s*([-]?\d+\.?\d*)',
            't_value_alt': r't\s*=\s*([-]?\d+\.?\d*)',

            # F-value patterns
            'f_value': r'F\s*\((\d+),\s*(\d+)\)\s*=\s*(\d+\.?\d*)',
            'f_value_alt': r'F\s*=\s*(\d+\.?\d*)',

            # Correlation patterns
            'r_value': r'[^f]r\s*=\s*([-]?\d+\.?\d*)',
            'pearson_r': r'Pearson.*r\s*=\s*([-]?\d+\.?\d*)',

            # p-value patterns
            'p_value_less': r'p\s*<\s*(\d+\.?\d*)',
            'p_value_equals': r'p\s*=\s*(\d+\.?\d*)',

            # Sample size patterns
            'n_equals': r'[Nn]\s*=\s*(\d+)',
            'n_parentheses': r'\([Nn]\s*=\s*(\d+)\)',
        }

    def extract_mean_sd(self, text):
        """Extract mean ± SD pairs"""
        results = []

        # Try different formats
        for pattern_name in ['mean_sd_pm', 'mean_sd_plus_minus', 'mean_sd_parentheses']:
            matches = re.findall(self.patterns[pattern_name], text)
            if matches:
                for mean, sd in matches:
                    results.append({
                        'mean': float(mean),
                        'sd': float(sd),
                        'format': pattern_name
                    })

        return results if results else None

    def extract_t_value(self, text):
        """Extract t-value and degrees of freedom"""
        # Try with df
        match = re.search(self.patterns['t_value'], text, re.IGNORECASE)
        if match:
            return {
                'df': int(match.group(1)),
                't': float(match.group(2))
            }

        # Try without df
        match = re.search(self.patterns['t_value_alt'], text, re.IGNORECASE)
        if match:
            return {
                'df': None,
                't': float(match.group(1))
            }

        return None

    def extract_f_value(self, text):
        """Extract F-value and degrees of freedom"""
        match = re.search(self.patterns['f_value'], text, re.IGNORECASE)
        if match:
            return {
                'df1': int(match.group(1)),
                'df2': int(match.group(2)),
                'f': float(match.group(3))
            }

        return None

    def extract_r_value(self, text):
        """Extract correlation coefficient"""
        match = re.search(self.patterns['r_value'], text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        match = re.search(self.patterns['pearson_r'], text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None

    def extract_p_value(self, text):
        """Extract p-value"""
        match = re.search(self.patterns['p_value_less'], text, re.IGNORECASE)
        if match:
            return {'value': float(match.group(1)), 'type': 'less_than'}

        match = re.search(self.patterns['p_value_equals'], text, re.IGNORECASE)
        if match:
            return {'value': float(match.group(1)), 'type': 'equals'}

        return None

    def extract_sample_size(self, text):
        """Extract sample size"""
        matches = re.findall(self.patterns['n_equals'], text)
        if matches:
            return [int(n) for n in matches]

        matches = re.findall(self.patterns['n_parentheses'], text)
        if matches:
            return [int(n) for n in matches]

        return None

    def extract_all(self, text):
        """Extract all statistics from text"""
        return {
            'mean_sd': self.extract_mean_sd(text),
            't_value': self.extract_t_value(text),
            'f_value': self.extract_f_value(text),
            'r_value': self.extract_r_value(text),
            'p_value': self.extract_p_value(text),
            'sample_size': self.extract_sample_size(text)
        }


class EffectSizeCalculator:
    """Calculate Cohen's d from various statistics"""

    @staticmethod
    def from_means(mean1, sd1, n1, mean2, sd2, n2):
        """
        Calculate Cohen's d from two groups' means and SDs

        Returns: (cohens_d, se_d, variance, ci_lower, ci_upper)
        """
        # Pooled standard deviation
        sd_pooled = np.sqrt((sd1**2 + sd2**2) / 2)

        # Cohen's d
        d = (mean1 - mean2) / sd_pooled

        # Standard error
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

        # Variance
        variance = se**2

        # 95% CI
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se

        return d, se, variance, ci_lower, ci_upper

    @staticmethod
    def from_t_value(t, n1, n2):
        """
        Calculate Cohen's d from t-value

        Returns: (cohens_d, se_d, variance, ci_lower, ci_upper)
        """
        # Cohen's d from t-value
        d = t * np.sqrt((n1 + n2) / (n1 * n2))

        # Standard error
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

        # Variance
        variance = se**2

        # 95% CI
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se

        return d, se, variance, ci_lower, ci_upper

    @staticmethod
    def from_f_value(f, n1, n2):
        """
        Calculate Cohen's d from F-value (two-group comparison only)

        Returns: (cohens_d, se_d, variance, ci_lower, ci_upper)
        """
        # Cohen's d from F-value
        d = np.sqrt(f * (n1 + n2) / (n1 * n2))

        # Standard error
        se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

        # Variance
        variance = se**2

        # 95% CI
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se

        return d, se, variance, ci_lower, ci_upper

    @staticmethod
    def from_r_value(r, n):
        """
        Calculate Cohen's d from correlation coefficient

        Returns: (cohens_d, se_d, variance, ci_lower, ci_upper)
        """
        # Cohen's d from r
        d = 2 * r / np.sqrt(1 - r**2)

        # Standard error
        se = np.sqrt(4 / n * (1 + d**2 / 8))

        # Variance
        variance = se**2

        # 95% CI
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se

        return d, se, variance, ci_lower, ci_upper


class EffectSizeValidator:
    """Validate extracted effect sizes"""

    @staticmethod
    def validate(d, se, n_total):
        """
        Validate effect size and return warnings

        Returns: {'valid': bool, 'warnings': list, 'severity': str}
        """
        warnings = []
        severity = 'ok'

        # 1. Extremely large effect size
        if abs(d) > 5:
            warnings.append("CRITICAL: Extremely large effect size (|d| > 5)")
            severity = 'critical'
        elif abs(d) > 2:
            warnings.append("WARNING: Very large effect size (|d| > 2)")
            severity = max(severity, 'warning')

        # 2. Standard error check
        if se > 2:
            warnings.append("WARNING: Very large standard error (SE > 2)")
            severity = max(severity, 'warning')

        # 3. Negative effect (unexpected direction)
        if d < 0:
            warnings.append("INFO: Negative effect (unexpected direction)")
            severity = max(severity, 'info')

        # 4. Very small sample size
        if n_total < 10:
            warnings.append("WARNING: Very small sample size (n < 10)")
            severity = max(severity, 'warning')

        # 5. CI crosses zero (not significant)
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se
        if ci_lower < 0 < ci_upper:
            warnings.append("INFO: 95% CI crosses zero (not significant)")
            severity = max(severity, 'info')

        return {
            'valid': severity != 'critical',
            'warnings': warnings,
            'severity': severity
        }


def determine_tier(stats_dict, d_result):
    """
    Determine data quality tier

    Tier 1: Complete data (mean, SD, n)
    Tier 2: Partial data (t, F, r convertible)
    Tier 3: Minimal data (p-value only)
    """
    if d_result is not None:
        # Check if from complete data
        if stats_dict.get('mean_sd') and len(stats_dict['mean_sd']) >= 2:
            return 1
        # Check if from convertible statistics
        elif stats_dict.get('t_value') or stats_dict.get('f_value') or stats_dict.get('r_value'):
            return 2

    # Only p-value or direction
    if stats_dict.get('p_value'):
        return 3

    return None  # No usable statistics


def extract_effect_sizes_from_papers(mode='test', sample_size=100):
    """
    Extract effect sizes from papers

    Args:
        mode: 'test' or 'full'
        sample_size: number of papers for test mode
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Initialize extractors
    stats_extractor = StatisticsExtractor()
    effect_calc = EffectSizeCalculator()
    validator = EffectSizeValidator()

    # Load papers (meta-analysis eligible only)
    print("\n=== Loading Papers ===")
    if mode == 'test':
        print(f"TEST MODE: Processing {sample_size} papers")
        cursor.execute("""
            SELECT paper_id, title, study_design, sample_size,
                   itq, iaq, ilq, isq, ispq, ivq,
                   outcome_physical, outcome_mental, outcome_productivity,
                   population_elderly, population_children,
                   key_findings, methodology_summary, numerical_results
            FROM papers
            WHERE (itq + iaq + ilq + isq + ispq + ivq) > 0
              AND (outcome_physical + outcome_mental + outcome_productivity) > 0
            LIMIT ?
        """, (sample_size,))
    else:
        print("FULL MODE: Processing all eligible papers")
        cursor.execute("""
            SELECT paper_id, title, study_design, sample_size,
                   itq, iaq, ilq, isq, ispq, ivq,
                   outcome_physical, outcome_mental, outcome_productivity,
                   population_elderly, population_children,
                   key_findings, methodology_summary, numerical_results
            FROM papers
            WHERE (itq + iaq + ilq + isq + ispq + ivq) > 0
              AND (outcome_physical + outcome_mental + outcome_productivity) > 0
        """)

    papers = cursor.fetchall()
    print(f"Loaded {len(papers)} papers")

    # Process papers
    print("\n=== Extracting Effect Sizes ===")
    results = []
    stats = defaultdict(int)
    stats['total'] = len(papers)

    for idx, paper in enumerate(papers, 1):
        (paper_id, title, study_design, sample_size_db,
         itq, iaq, ilq, isq, ispq, ivq,
         outcome_physical, outcome_mental, outcome_productivity,
         population_elderly, population_children,
         key_findings, methodology, numerical_results) = paper

        # Combine text for extraction
        combined_text = ' '.join(filter(None, [key_findings, methodology, numerical_results]))

        # Extract statistics
        stats_dict = stats_extractor.extract_all(combined_text)

        # Try to calculate Cohen's d
        d_result = None
        tier = None

        # Attempt 1: From means and SDs
        if stats_dict['mean_sd'] and len(stats_dict['mean_sd']) >= 2:
            try:
                mean_sd_list = stats_dict['mean_sd']
                sample_sizes = stats_dict['sample_size'] or [sample_size_db//2, sample_size_db//2]

                if len(sample_sizes) >= 2:
                    d, se, var, ci_low, ci_up = effect_calc.from_means(
                        mean_sd_list[0]['mean'], mean_sd_list[0]['sd'], sample_sizes[0],
                        mean_sd_list[1]['mean'], mean_sd_list[1]['sd'], sample_sizes[1]
                    )
                    d_result = (d, se, var, ci_low, ci_up)
                    tier = 1
            except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
                pass  # 통계 계산 실패

        # Attempt 2: From t-value
        if d_result is None and stats_dict['t_value']:
            try:
                t_info = stats_dict['t_value']
                sample_sizes = stats_dict['sample_size'] or [sample_size_db//2, sample_size_db//2]

                if len(sample_sizes) >= 2:
                    d, se, var, ci_low, ci_up = effect_calc.from_t_value(
                        t_info['t'], sample_sizes[0], sample_sizes[1]
                    )
                    d_result = (d, se, var, ci_low, ci_up)
                    tier = 2
            except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
                pass  # 통계 계산 실패

        # Attempt 3: From F-value
        if d_result is None and stats_dict['f_value']:
            try:
                f_info = stats_dict['f_value']
                if f_info['df1'] == 1:  # Two-group comparison only
                    sample_sizes = stats_dict['sample_size'] or [sample_size_db//2, sample_size_db//2]

                    if len(sample_sizes) >= 2:
                        d, se, var, ci_low, ci_up = effect_calc.from_f_value(
                            f_info['f'], sample_sizes[0], sample_sizes[1]
                        )
                        d_result = (d, se, var, ci_low, ci_up)
                        tier = 2
            except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
                pass  # 통계 계산 실패

        # Attempt 4: From r-value
        if d_result is None and stats_dict['r_value']:
            try:
                r = stats_dict['r_value']
                n_total = sample_size_db or 100

                d, se, var, ci_low, ci_up = effect_calc.from_r_value(r, n_total)
                d_result = (d, se, var, ci_low, ci_up)
                tier = 2
            except (TypeError, ValueError, ZeroDivisionError):
                pass  # 통계 계산 실패

        # Determine tier if not yet set
        if tier is None:
            tier = determine_tier(stats_dict, d_result)

        # Validate if effect size calculated
        validation = None
        if d_result:
            d, se, var, ci_low, ci_up = d_result
            validation = validator.validate(d, se, sample_size_db or 100)

            if not validation['valid']:
                stats['invalid'] += 1

        # Store result
        result = {
            'paper_id': paper_id,
            'title': title[:100],
            'study_design': study_design,
            'sample_size': sample_size_db,
            'ieq_categories': {
                'itq': itq, 'iaq': iaq, 'ilq': ilq,
                'isq': isq, 'ispq': ispq, 'ivq': ivq
            },
            'outcomes': {
                'physical': outcome_physical,
                'mental': outcome_mental,
                'productivity': outcome_productivity
            },
            'population': {
                'elderly': population_elderly,
                'children': population_children
            },
            'statistics_extracted': stats_dict,
            'cohens_d': d_result[0] if d_result else None,
            'se_d': d_result[1] if d_result else None,
            'variance': d_result[2] if d_result else None,
            'ci_lower': d_result[3] if d_result else None,
            'ci_upper': d_result[4] if d_result else None,
            'tier': tier,
            'validation': validation
        }

        results.append(result)

        # Update statistics
        if tier:
            stats[f'tier_{tier}'] += 1
        else:
            stats['no_stats'] += 1

        # Progress
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(papers)} papers...")

    conn.close()

    # Print statistics
    print("\n" + "="*60)
    print("=== Effect Size Extraction Results ===")
    print("="*60)
    print(f"\nTotal papers processed: {stats['total']}")
    print(f"\nTier 1 (Complete data):     {stats.get('tier_1', 0):4d} ({stats.get('tier_1', 0)/stats['total']*100:5.1f}%)")
    print(f"Tier 2 (Partial data):      {stats.get('tier_2', 0):4d} ({stats.get('tier_2', 0)/stats['total']*100:5.1f}%)")
    print(f"Tier 3 (Minimal data):      {stats.get('tier_3', 0):4d} ({stats.get('tier_3', 0)/stats['total']*100:5.1f}%)")
    print(f"No statistics:              {stats.get('no_stats', 0):4d} ({stats.get('no_stats', 0)/stats['total']*100:5.1f}%)")
    print(f"\nTotal usable (Tier 1+2):    {stats.get('tier_1', 0) + stats.get('tier_2', 0):4d} ({(stats.get('tier_1', 0) + stats.get('tier_2', 0))/stats['total']*100:5.1f}%)")
    print(f"Invalid effect sizes:       {stats.get('invalid', 0):4d}")

    return results, stats


def main():
    parser = argparse.ArgumentParser(description='Extract effect sizes from IEQ papers')
    parser.add_argument('--mode', choices=['test', 'full'], default='test',
                        help='test: process sample, full: process all papers')
    parser.add_argument('--sample', type=int, default=100,
                        help='Sample size for test mode (default: 100)')
    parser.add_argument('--output', type=str, default='extracted_effect_sizes.json',
                        help='Output JSON file name')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Effect Size Extraction System")
    print("="*60)
    print(f"Database: {DB_PATH}")
    print(f"Mode: {args.mode.upper()}")
    if args.mode == 'test':
        print(f"Sample size: {args.sample}")
    print("="*60)

    # Extract effect sizes
    results, stats = extract_effect_sizes_from_papers(
        mode=args.mode,
        sample_size=args.sample
    )

    # Save results
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'mode': args.mode,
                'total_papers': stats['total'],
                'tier1': stats.get('tier_1', 0),
                'tier2': stats.get('tier_2', 0),
                'tier3': stats.get('tier_3', 0),
                'no_stats': stats.get('no_stats', 0)
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("="*60)


if __name__ == '__main__':
    main()
