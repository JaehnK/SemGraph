# 실험 스크립트

AG News 데이터셋을 사용한 GRACE 모델 평가 실험

## 📁 구조

```
experiments/
├── README.md                          # 이 파일
├── rq1_single_vs_multi_embedding.py   # RQ1: 단일 vs 다중 임베딩 ✅
├── rq2_ablation_study.py              # RQ2: Ablation Study ✅
└── rq3_traditional_vs_ours.py         # RQ3: 전통적 방법 vs 제안 방법 ✅
```

## 🎯 연구 질문

### RQ1: 단일 vs 다중 임베딩
**질문**: 다중 임베딩(Word2Vec + DistilBERT)이 단일 임베딩보다 클러스터링 성능을 향상시키는가?

**비교 모델 (옵션 1: 단일 256d vs 다중 512d)**:
- W2V-KMeans: Word2Vec(256d) + K-means
- BERT-KMeans: DistilBERT(256d) + K-means
- Concat-KMeans: Word2Vec(256d) + DistilBERT(256d) = 512d (GraphMAE 없이) + K-means
- GRACE: Word2Vec(256d) + DistilBERT(256d) = 512d + GraphMAE + K-means

**주요 비교**:
1. **다중 임베딩 효과**: Concat-KMeans vs W2V-KMeans, BERT-KMeans
2. **GraphMAE 효과**: GRACE vs Concat-KMeans (모두 512d)

**설계 특징**:
- ✅ **정보 손실 없음**: 각 임베딩이 256d 공간에서 독립적으로 학습
- ✅ **PCA 제약 해결**: 각 임베딩 256d < 500 (vocab size)으로 안전
- ✅ **Multi-modal 표준**: 512d = 256d+256d 블록 구조
- ✅ **공정한 비교**: Concat vs GRACE 모두 512d

### RQ2: Ablation Study
**질문**: GraphMAE2의 하이퍼파라미터가 클러스터링 성능에 어떤 영향을 미치는가?

**실험**:
- 2.1 Mask Rate: [0.3, 0.5, 0.75, 0.9]
- 2.2 Training Epochs: [250, 500, 1000, 2000]
- 2.3 Embedding Dimension: [64, 128, 256]

### RQ3: 전통적 방법 vs 제안 방법
**질문**: GraphMAE2 기반 방법이 전통적 커뮤니티 탐지 알고리즘보다 의미적으로 일관된 클러스터를 생성하는가?

**비교 모델**:
- Louvain
- Leiden
- GRACE (최적 하이퍼파라미터)

---

## 🚀 실행 방법

### 환경 활성화
```bash
conda activate SENTIMENT
cd /path/to/SENTIMENT
```

### RQ1 실험 실행
```bash
python experiments/rq1_single_vs_multi_embedding.py
```

**예상 실행 시간**: 약 6-7시간 (GPU 사용 시)
- W2V-KMeans (256d): 5분 × 5회 = 25분
- BERT-KMeans (256d): 8분 × 5회 = 40분
- Concat-KMeans (256d+256d=512d): 10분 × 5회 = 50분
- GRACE (256d+256d=512d + GraphMAE): 60분 × 5회 = 300분 (5시간)

**출력**:
```
results/rq1_single_vs_multi/
├── raw_results_TIMESTAMP.json           # 전체 실험 결과
├── summary_statistics_TIMESTAMP.csv     # 평균±표준편차
├── statistical_tests_TIMESTAMP.json     # 통계 검정 결과
├── table1_rq1_TIMESTAMP.txt            # 논문용 Table 1
└── plots/
    ├── fig2_npmi_comparison_TIMESTAMP.png
    └── fig_heatmap_rq1_TIMESTAMP.png
```

### RQ2 실험 실행
```bash
python experiments/rq2_ablation_study.py
```

**예상 실행 시간**: 약 48-50시간 (GPU 사용 시)
- Mask Rate (4개 × 5회): ~20시간
- Epochs (4개 × 5회): ~20시간
- PCA Dim (3개 × 5회): ~10시간

**출력**:
```
results/rq2_ablation/
├── optimal_config.json              # 최적 하이퍼파라미터 (⚠️ RQ3에서 사용!)
├── mask_rate/
│   ├── raw_results_TIMESTAMP.json
│   ├── summary_TIMESTAMP.csv
│   ├── table_mask_rate_TIMESTAMP.txt
│   └── plots/
│       ├── npmi_mask_rate_TIMESTAMP.png
│       ├── multi_metric_mask_rate_TIMESTAMP.png
│       └── boxplot_mask_rate_TIMESTAMP.png
├── epochs/
│   └── ...
└── pca_dim/
    └── ...
```

### RQ3 실험 실행
```bash
python experiments/rq3_traditional_vs_ours.py
```

**⚠️ 중요**: RQ2를 먼저 실행하여 `optimal_config.json`을 생성하는 것을 권장합니다!

**예상 실행 시간**: 약 6시간 (GPU 사용 시)
- Louvain: 2분 × 5회 = 10분
- Leiden: 3분 × 5회 = 15분
- GRACE: 65분 × 5회 = 325분 (5.4시간)

**출력**:
```
results/rq3_traditional_vs_ours/
├── raw_results_TIMESTAMP.json
├── summary_TIMESTAMP.csv
├── statistical_tests_TIMESTAMP.json
├── table5_rq3_TIMESTAMP.txt        # 논문용 Table 5
└── plots/
    ├── rq3_comparison_TIMESTAMP.png
    └── rq3_heatmap_TIMESTAMP.png
```

---

## ⚙️ 실험 설정

### 공통 설정
- **데이터**: AG News (`data/ag_news.csv`)
- **문서 수**: 10,000개 (각 클래스 2,500개 × 4, 균형 잡힌 데이터셋)
- **Vocab size**: 500 단어
- **Random seeds**: [42, 123, 456, 789, 101] (n=5)
- **Edge weight threshold**: 5

### GraphMAE 설정
- **Epochs**: 500 (RQ2에서 최적화)
- **Learning rate**: 0.001
- **Mask rate**: 0.5 (RQ2에서 최적화)
- **Encoder/Decoder**: GAT

### 임베딩 설정
- **단일 임베딩**: 256d (W2V, BERT)
- **다중 임베딩**: 256d + 256d = 512d (Concat, GRACE)
- **Word2Vec**: Skip-gram, window=5, 100 epochs
- **DistilBERT**: distilbert-base-uncased, mean pooling

### 클러스터링
- **Method**: K-means
- **K 선택**: Elbow method (5-20 범위)

### 평가 지표
- NPMI (주 지표)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

---

## 📊 통계 검정

### Two-tailed t-test
각 모델 쌍에 대해 독립표본 t-검정 수행 (n=5)

### Bonferroni Correction (RQ1)
- α = 0.05
- 비교 수 = 3 (GRACE vs W2V-KMeans, BERT-KMeans, Concat-KMeans)
- α_corrected = 0.05 / 3 ≈ 0.0167

### Cohen's d (Effect Size)
효과 크기 해석:
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large

---

## 🔧 스크립트 커스터마이징

각 스크립트 상단의 `EXPERIMENT_CONFIG` 딕셔너리를 수정하여 설정 변경 가능:

```python
EXPERIMENT_CONFIG = {
    'num_documents': 10000,      # 문서 수 조정
    'top_n_words': 500,          # Vocab size
    'graphmae_epochs': 500,      # GraphMAE 학습 에폭
    'random_seeds': [42, 123, ...],  # 시드 변경
    # ...
}
```

---

## 📝 결과 해석

### Table 1 예시 (RQ1)
```
Model               NPMI            Silhouette      DBI             CHI
W2V-KMeans          0.380±0.030     0.420±0.040     1.850±0.120     245.6±18.3
BERT-KMeans         0.410±0.020     0.450±0.030     1.720±0.100     268.3±21.5
Concat-KMeans       0.480±0.030**   0.510±0.040**   1.580±0.110**   312.5±24.7**
GRACE               0.610±0.020***  0.630±0.030***  1.280±0.090***  398.5±28.2***

**: p < 0.01, ***: p < 0.0167 (Bonferroni corrected)

Note: 옵션 1 - 단일(256d) vs 다중(512d) 비교
      - 단일 임베딩: W2V(256d), BERT(256d)
      - 다중 임베딩: Concat(256d+256d=512d), GRACE(256d+256d=512d)
      - 각 임베딩은 독립적으로 256d 공간에서 학습 (n=500 > d=256 ✓)
      - 512d는 두 개의 독립적인 256d 블록 구조
      - Multi-modal learning의 표준 접근 방식
```

### 주요 확인 사항
1. **정보 손실 없음**: 각 임베딩이 256d 공간에서 독립적으로 학습
2. **Concat-KMeans > Single**: 다중 임베딩의 효과 입증
3. **GRACE > Concat-KMeans**: GraphMAE의 기여도 입증 (모두 512d)
4. **통계적 유의성**: p < 0.0167 (Bonferroni corrected)
5. **효과 크기**: Cohen's d ≥ 0.5 (medium 이상)

---

## 🐛 문제 해결

### GPU 메모리 부족
```python
# EXPERIMENT_CONFIG 수정
'num_documents': 5000,  # 문서 수 감소
'top_n_words': 300,     # Vocab size 감소
```

### 실행 시간 단축 (테스트용)
```python
'random_seeds': [42],   # 1회만 실행
'graphmae_epochs': 100, # 에폭 감소
```

### CUDA 에러
```bash
# CPU만 사용
export CUDA_VISIBLE_DEVICES=""
python experiments/rq1_single_vs_multi_embedding.py
```

---

## 📚 참고

- 실험 프로토콜: `docs/experiment_protocol(1).md`
- GRACE Pipeline: `core/services/GRACE/GRACEPipeline.py`
- 설정 파일: `core/services/GRACE/GRACEConfig.py`
