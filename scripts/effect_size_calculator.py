#!/usr/bin/env python3
"""
70개 효과크기 Cohen's d 통일 계산
Tier 1-3 모두 통일된 Cohen's d + 95% CI

중립적 접근: 편견 없이 데이터가 보여주는 것을 계산
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 경로
BASE_DIR = Path(__file__).parent.parent
INPUT_CSV = BASE_DIR / "INTEGRATED_EFFECT_SIZES_70.csv"
OUTPUT_CSV = BASE_DIR / "UNIFIED_EFFECT_SIZES_70_FINAL.csv"

print("="*80)
print("70개 효과크기 Cohen's d 통일 계산")
print("="*80)
print(f"입력: {INPUT_CSV}")
print(f"출력: {OUTPUT_CSV}")
print()

# 데이터 로드 (숫자 컬럼 강제 변환)
df = pd.read_csv(INPUT_CSV)

# 숫자 컬럼 변환
numeric_cols = ['n', 'Mean_1', 'SD_1', 'Mean_2', 'SD_2', 'Cohen_d', 'SE']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"총 효과크기: {len(df)}개")
print(f"Tier 분포:")
print(df['Tier'].value_counts())
print()

# Cohen's d 계산 함수들
def calc_cohens_d_from_means(mean1, sd1, n1, mean2, sd2, n2):
    """Tier 1: Mean±SD에서 Cohen's d 계산"""
    pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_sd

    # SE 계산
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))

    # 95% CI
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    return d, se, ci_lower, ci_upper

def calc_cohens_d_from_b(b, se_b, n):
    """Tier 2: B coefficient → Cohen's d (근사)"""
    # 표준화: d ≈ B / SD_pooled (근사)
    # 여기서는 B를 그대로 사용 (이미 표준화된 경우)
    d = b
    se = se_b if se_b else np.sqrt(4/n)

    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    return d, se, ci_lower, ci_upper

def calc_cohens_d_from_r(r, n):
    """Tier 3: Correlation → Cohen's d"""
    d = (2 * r) / np.sqrt(1 - r**2)

    # SE 계산
    se = np.sqrt(4 / n)

    # 95% CI
    ci_lower = d - 1.96 * se
    ci_upper = d + 1.96 * se

    return d, se, ci_lower, ci_upper

# 각 행 처리
results = []

for idx, row in df.iterrows():
    result = {
        'ES_ID': row['ES_ID'],
        'Paper_ID': row['Paper_ID'],
        'IEQ_Category': row['IEQ_Category'],
        'Outcome_Domain': row['Outcome_Domain'],
        'Outcome_Name': row['Outcome_Name'],
        'Comparison': row['Comparison'],
        'n': row['n'],
        'Tier': row['Tier'],
        'Status': row['Status']
    }

    try:
        if row['Tier'] == 'Tier 1' and row['Status'] == 'Complete':
            # Mean±SD 직접 계산
            d, se, ci_l, ci_u = calc_cohens_d_from_means(
                row['Mean_1'], row['SD_1'], row['n'],
                row['Mean_2'], row['SD_2'], row['n']
            )
            result['Cohen_d'] = d
            result['SE'] = se
            result['CI_95_lower'] = ci_l
            result['CI_95_upper'] = ci_u
            result['Method'] = 'Mean_SD'

        elif row['Tier'] == 'Tier 2':
            # B coefficient 사용
            # 주의: 이미 Cohen_d가 있는 경우 사용
            if pd.notna(row.get('Cohen_d')):
                result['Cohen_d'] = row['Cohen_d']
                result['SE'] = row.get('SE', np.sqrt(4/row['n']))
                result['CI_95_lower'] = row['Cohen_d'] - 1.96 * result['SE']
                result['CI_95_upper'] = row['Cohen_d'] + 1.96 * result['SE']
            else:
                # B → d 변환 필요 (향후 구현)
                result['Cohen_d'] = np.nan
                result['SE'] = np.nan
                result['CI_95_lower'] = np.nan
                result['CI_95_upper'] = np.nan
            result['Method'] = 'B_coefficient'

        elif row['Tier'] == 'Tier 3':
            # Correlation → d
            # r값 추출 필요
            result['Cohen_d'] = np.nan
            result['SE'] = np.nan
            result['CI_95_lower'] = np.nan
            result['CI_95_upper'] = np.nan
            result['Method'] = 'Correlation'

        else:
            result['Cohen_d'] = np.nan
            result['SE'] = np.nan
            result['CI_95_lower'] = np.nan
            result['CI_95_upper'] = np.nan
            result['Method'] = 'Not_processed'

    except Exception as e:
        print(f"⚠ ES_{idx+1}: {e}")
        result['Cohen_d'] = np.nan
        result['SE'] = np.nan
        result['CI_95_lower'] = np.nan
        result['CI_95_upper'] = np.nan
        result['Method'] = 'Error'

    results.append(result)

# 결과 DataFrame
df_result = pd.DataFrame(results)

# 통계
print("="*80)
print("계산 결과")
print("="*80)
print(f"총 효과크기: {len(df_result)}")
print(f"\nMethod 분포:")
print(df_result['Method'].value_counts())
print(f"\n계산 완료: {df_result['Cohen_d'].notna().sum()}개")
print(f"미완료: {df_result['Cohen_d'].isna().sum()}개")

# 저장
df_result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n✓ 저장: {OUTPUT_CSV}")

# 요약 통계
calculated = df_result[df_result['Cohen_d'].notna()]
if len(calculated) > 0:
    print("\n" + "="*80)
    print("Cohen's d 요약 통계 (계산 완료분)")
    print("="*80)
    print(f"평균: {calculated['Cohen_d'].mean():.3f}")
    print(f"중앙값: {calculated['Cohen_d'].median():.3f}")
    print(f"범위: {calculated['Cohen_d'].min():.3f} ~ {calculated['Cohen_d'].max():.3f}")
    print(f"표준편차: {calculated['Cohen_d'].std():.3f}")

print("\n다음 단계: Random-Effects 메타분석")
