# 실험 스크립트

AG News 데이터셋을 사용한 GRACE 모델 평가 실험

## 📁 구조

```
experiments/
├── README.md                          # 이 파일
├── rq1_single_vs_multi_embedding.py   # RQ1: 단일 vs 다중 임베딩
├── rq2_ablation_study.py              # RQ2: Ablation Study (예정)
└── rq3_traditional_vs_ours.py         # RQ3: 전통적 방법 vs 제안 방법 (예정)
```

## 🎯 연구 질문

### RQ1: 단일 vs 다중 임베딩
**질문**: 다중 임베딩(Word2Vec + DistilBERT)이 단일 임베딩보다 클러스터링 성능을 향상시키는가?

**비교 모델**:
- W2V-KMeans: Word2Vec만 + K-means
- BERT-KMeans: DistilBERT만 + K-means
- Concat-KMeans: Word2Vec + DistilBERT (GraphMAE 없이) + K-means
- Ours: Word2Vec + DistilBERT + GraphMAE + K-means

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
- Ours (최적 하이퍼파라미터)

---

## 🚀 실행 방법

### 환경 활성화
```bash
conda activate SENTIMENT
cd /home/jaehun/lab/SENTIMENT
```

### RQ1 실험 실행
```bash
python experiments/rq1_single_vs_multi_embedding.py
```

**예상 실행 시간**: 약 7-8시간 (GPU 사용 시)
- W2V-KMeans: 5분 × 5회 = 25분
- BERT-KMeans: 8분 × 5회 = 40분
- Concat-KMeans: 10분 × 5회 = 50분
- Ours: 65분 × 5회 = 325분 (5.4시간)

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

### RQ2 실험 실행 (예정)
```bash
python experiments/rq2_ablation_study.py
```

### RQ3 실험 실행 (예정)
```bash
python experiments/rq3_traditional_vs_ours.py
```

---

## ⚙️ 실험 설정

### 공통 설정
- **데이터**: AG News (`data/ag_news.csv`)
- **문서 수**: 10,000개
- **Vocab size**: 500 단어
- **Random seeds**: [42, 123, 456, 789, 101] (n=5)
- **Edge weight threshold**: 5

### GraphMAE 설정
- **Epochs**: 500 (RQ2에서 최적화)
- **Learning rate**: 0.001
- **Mask rate**: 0.5 (RQ2에서 최적화)
- **Encoder/Decoder**: GAT

### 임베딩 설정
- **PCA dimension**: 256 (각 임베딩)
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

### Bonferroni Correction
- α = 0.05
- 비교 수 = 3 (Ours vs W2V, BERT, Concat)
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
Ours                0.610±0.020***  0.630±0.030***  1.280±0.090***  398.5±28.2***

**: p < 0.01, ***: p < 0.017 (Bonferroni corrected)
```

### 주요 확인 사항
1. **Ours > Concat-KMeans**: GraphMAE의 기여도 입증
2. **Concat-KMeans > Single**: 다중 임베딩의 효과 입증
3. **통계적 유의성**: p < 0.017 (Bonferroni corrected)
4. **효과 크기**: Cohen's d ≥ 0.5 (medium 이상)

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
