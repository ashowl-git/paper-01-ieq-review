# Supplementary Materials

## Effects of Indoor Environmental Quality on Health and Productivity: A Systematic Review

안승호, 이정아 (EAN Technology)

---

## S1. 검색 전략 (Search Strategy)

### S1.1 데이터베이스 및 API

검색에 사용된 데이터베이스와 API는 다음과 같다.

| Database/API | Coverage | Search period | Role |
|-------------|----------|---------------|------|
| Semantic Scholar API | Multidisciplinary | 2015-01 ~ 2025-10 | Primary search (bulk collection) |
| OpenAlex API | 240M+ works | 2015-01 ~ 2025-10 | Metadata enrichment (institution, funder, concept) |
| PubMed API (E-utilities) | Biomedical | 2015-01 ~ 2025-10 | Biomedical supplement |
| arXiv API | Preprints | 2015-01 ~ 2025-10 | Preprint supplement |
| CrossRef API | DOI registry | - | DOI validation and metadata verification |
| Unpaywall API | Open access | - | PDF URL collection |

### S1.2 검색 키워드

IEQ 변수 키워드와 결과 변수 키워드를 조합하여 검색하였다.

**IEQ 변수 키워드:**

| IEQ Variable | Search terms |
|-------------|-------------|
| ITQ (Thermal) | "indoor thermal quality", "thermal comfort", "indoor temperature", "heat stress indoor" |
| IAQ (Air) | "indoor air quality", "ventilation", "CO2 concentration indoor", "volatile organic compounds indoor" |
| ILQ (Lighting) | "indoor lighting", "daylighting", "illuminance", "color temperature lighting" |
| ISQ (Sound) | "indoor acoustic", "noise exposure indoor", "sound quality indoor", "office noise" |
| ISPQ (Spatial) | "spatial quality", "office layout", "workspace design", "room density" |
| IVQ (Visual) | "visual comfort", "window view", "biophilic design indoor", "visual environment" |

**결과 변수 키워드:**

| Outcome domain | Search terms |
|---------------|-------------|
| Physical health | "health", "sick building syndrome", "respiratory symptoms", "headache", "fatigue" |
| Mental health | "mental health", "well-being", "stress", "mood", "anxiety", "depression" |
| Productivity | "productivity", "performance", "cognitive function", "concentration", "work output" |

**취약계층 특화 키워드:**
- Elderly: "elderly", "older adult", "aging", "geriatric"
- Children: "children", "child", "pediatric", "school"

### S1.3 API 호출 파라미터 (Semantic Scholar 예시)

```
Endpoint: https://api.semanticscholar.org/graph/v1/paper/search
Parameters:
  query: "{IEQ_term} AND ({outcome_term})"
  year: "2015-2025"
  fields: "title,abstract,authors,year,venue,externalIds,citationCount"
  limit: 100
  offset: 0 (paginated)
Rate limit: 100 requests/second (authenticated)
```

### S1.4 검색 일자 및 결과

- 최초 검색일: 2025-10-15
- 최종 업데이트: 2025-10-31
- 총 수집: 3,828편
- API별 수집 분포: Semantic Scholar (주력, ~70%), OpenAlex (보완, ~20%), PubMed (~5%), arXiv (~5%)

---

## S2. Python 분류 스크립트 (Classification Rules)

### S2.1 바이너리 플래그 부여 규칙

각 논문의 제목(title), 초록(abstract), 키워드를 입력으로 하여, 다음의 키워드 매칭 규칙에 따라 6개 IEQ 변수 및 3개 결과 변수에 대한 바이너리 플래그(0/1)를 부여하였다.

**IEQ 변수 분류 키워드:**

```python
IEQ_KEYWORDS = {
    'itq': ['thermal comfort', 'indoor temperature', 'heat stress',
            'thermal environment', 'HVAC', 'air conditioning',
            'thermal sensation', 'operative temperature', 'PMV', 'PPD'],
    'iaq': ['indoor air quality', 'ventilation', 'CO2', 'carbon dioxide',
            'VOC', 'volatile organic', 'particulate matter', 'PM2.5',
            'air pollution indoor', 'formaldehyde'],
    'ilq': ['lighting', 'illuminance', 'daylighting', 'daylight',
            'color temperature', 'CCT', 'luminance', 'glare',
            'circadian', 'melanopic'],
    'isq': ['acoustic', 'noise', 'sound', 'speech intelligibility',
            'reverberation', 'sound masking', 'noise annoyance'],
    'ispq': ['spatial', 'office layout', 'open plan', 'workspace design',
             'room density', 'occupancy', 'floor area', 'ceiling height',
             'biophilic', 'green building'],
    'ivq': ['visual comfort', 'window view', 'visual environment',
            'view quality', 'prospect', 'refuge', 'nature view']
}
```

**결과 변수 분류 키워드:**

```python
OUTCOME_KEYWORDS = {
    'outcome_physical': ['health', 'sick building', 'respiratory',
                         'headache', 'fatigue', 'symptom', 'allergy',
                         'asthma', 'eye strain', 'musculoskeletal'],
    'outcome_mental': ['mental health', 'well-being', 'wellbeing',
                       'stress', 'mood', 'anxiety', 'depression',
                       'satisfaction', 'comfort perception', 'affect'],
    'outcome_productivity': ['productivity', 'performance', 'cognitive',
                             'concentration', 'work output', 'task performance',
                             'reaction time', 'attention', 'memory',
                             'creative', 'learning']
}
```

### S2.2 분류 정확도 검증

67편을 수동으로 검증한 결과:
- 정분류: 53편 (79.1%)
- 오분류: 14편 (20.9%) -- IEQ-건강/생산성 연구와 무관한 논문(의학, 컴퓨터 비전, 재료과학 등)
- 이 비율을 762편 전체에 적용하면 약 160편이 무관한 논문일 수 있음

### S2.3 연구 설계 분류

연구 설계는 초록의 키워드 매칭으로 자동 분류하였으며, 미분류(Unspecified) 203편(26.6%)이 존재한다.

---

## S3. PRISMA 2020 체크리스트 (완전판)

| Section | Item | Checklist item | Location |
|---------|------|---------------|----------|
| **Title** | 1 | Identify the report as a systematic review | Title |
| **Abstract** | 2 | Structured summary with background, objectives, methods, results, conclusions | Abstract |
| **Introduction** | | | |
| Rationale | 3 | Describe the rationale for the review | 1.1-1.2 |
| Objectives | 4 | Provide an explicit statement of objectives | 1.3 |
| **Methods** | | | |
| Eligibility criteria | 5 | Specify inclusion and exclusion criteria | 2.1 |
| Information sources | 6 | Describe all information sources searched | 2.1, S1.1 |
| Search strategy | 7 | Present full search strategy for at least one database | S1.2-S1.3 |
| Selection process | 8 | Describe the process of selecting studies | 2.1 |
| Data collection process | 9 | Describe methods of data extraction | 2.1, S2 |
| Data items | 10 | List all variables for which data were sought | 2.1-2.3 |
| Study risk of bias assessment | 11 | Describe methods for assessing risk of bias | 2.1, Table 3B |
| Effect measures | 12 | Specify effect measures used | 2.3, Eq.(1)-(4) |
| Synthesis methods | 13a | Describe processes for synthesizing results | 2.3-2.4 |
| | 13b | Describe any methods of handling data from multiple reports | 2.2-2.3 |
| | 13c | Describe any sensitivity or subgroup analyses | 2.4 |
| | 13d | Describe any methods used to assess certainty | 2.3, Table 3B |
| | 13e | Describe how studies were grouped for syntheses | 2.2, 2.4 |
| | 13f | Describe any methods required to prepare data for synthesis | 2.3 |
| Reporting bias assessment | 14 | Describe any methods for assessing risk of reporting bias | 2.4 |
| Certainty assessment | 15 | Describe any methods for assessing certainty of evidence | N/A (exploratory) |
| **Results** | | | |
| Study selection | 16a | Describe results of search and screening | 3.1, Figure 1 |
| | 16b | Cite studies that appear to meet criteria but were excluded | 2.1 |
| Study characteristics | 17 | Describe characteristics of included studies | Table 1 |
| Risk of bias in studies | 18 | Present assessment of risk of bias | Table 3B |
| Results of individual studies | 19 | Present results for all studies and syntheses | Table 3, Figures 4-6 |
| Results of syntheses | 20a | For each synthesis, summarize results | 3.3-3.5 |
| | 20b | Present results of all statistical syntheses | 3.3-3.4 |
| | 20c | Present results of all sensitivity analyses | N/A |
| | 20d | Present results of all subgroup analyses | 3.4 |
| Reporting biases | 21 | Present assessment of reporting bias | 2.4, 4.3 |
| Certainty of evidence | 22 | Present assessment of certainty of evidence | 4.3 |
| **Discussion** | | | |
| Discussion | 23a | Provide a general interpretation of results | 4.1 |
| | 23b | Discuss any limitations of evidence | 4.3 |
| | 23c | Discuss any limitations of the review process | 4.3 |
| | 23d | Discuss implications of results | 4.2 |
| **Other information** | | | |
| Registration and protocol | 24a | Protocol registration information | N/A (not registered) |
| | 24b | Where protocol can be accessed | N/A |
| | 24c | Amendments to protocol | N/A |
| Support | 25 | Sources of financial support | N/A |
| Competing interests | 26 | Declare competing interests | N/A |
| Availability of data | 27 | Availability of data, code, and other materials | S1-S4 |

Note: N/A items indicate areas not applicable or not completed in the current review. Protocol registration is recommended for future updates. The review was conducted as an exploratory systematic review without formal protocol registration.

---

## S4. 효과크기 추출 상세 (Effect Size Extraction Details)

### S4.1 효과크기 산출 방법

모든 효과크기는 Cohen's d로 표준화하였다. 산출 방법에 따라 3개 Tier로 분류하였다.

**Tier 1 Complete (34 ES, 8편):** 원 논문에서 직접 보고된 d값, 또는 평균과 표준편차로부터 식(1)에 의해 산출

```
d = (M_treatment - M_control) / SD_pooled
SD_pooled = sqrt(((n1-1)*SD1^2 + (n2-1)*SD2^2) / (n1+n2-2))
SE = sqrt((n1+n2)/(n1*n2) + d^2/(2*(n1+n2)))
```

**Tier 2 Complete (8 ES, 5편):** z-score DiD, eta-squared, F-statistic, 또는 percent change로부터 변환

- z-score Difference-in-Differences: d = z_DiD (표준화 차이)
- Eta-squared: d = 2*sqrt(eta2/(1-eta2))
- F-statistic: d = 2*sqrt(F/n)
- Percent change: d = percent_change / (CV * 100), CV = 0.5 assumed

**Tier 2 B coefficient (17 ES, 3편):** 회귀계수만 보고. d값 변환 불가 (원 논문의 SD 정보 부재)

**Tier 1 Unprocessed (3 ES, 1편):** 평균/SD 데이터 존재하나 미처리

**Tier 3 Correlation (5 ES, 1편):** 상관계수만 보고. d = 2r/sqrt(1-r^2) 변환 가능하나, 실험-대조군 비교가 아닌 연속변수 상관이므로 별도 분류

### S4.2 연구별 효과크기 추출 요약

전체 67개 효과크기의 상세 정보는 UNIFIED_EFFECT_SIZES_70_FINAL.csv에 수록되어 있다. 주요 필드: ES_ID, Paper_ID, IEQ_Category, Outcome_Domain, Outcome_Name, Comparison, n, Tier, Status, Cohen_d, SE, CI_95_lower, CI_95_upper, Method.

### S4.3 효과크기 부호 규약

- 양(+)의 방향: IEQ 개선 -> 건강/생산성 향상
- 음(-)의 방향: IEQ 악화 -> 건강/생산성 저하
- 원 연구에서 방향이 반대인 경우 부호를 반전하여 일관성 유지

---

작성일: 2026-02-06
본 보충자료는 PAPER_01_통합본_대한건축학회.md의 부속 자료임
