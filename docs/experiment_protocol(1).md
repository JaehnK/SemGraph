# 실험 프로토콜 - AG News 데이터셋

## 📋 목차
1. [연구 질문](#연구-질문)
2. [데이터셋 설정](#데이터셋-설정)
3. [비교 모델 구성](#비교-모델-구성)
4. [실험별 세부 설계](#실험별-세부-설계)
5. [평가 지표](#평가-지표)
6. [실험 실행 프로토콜](#실험-실행-프로토콜)
7. [결과 보고 형식](#결과-보고-형식)

---

## 🎯 연구 질문

### RQ1. 단일 vs 다중 임베딩 (Multi-modal Embedding Effect)
**질문**: 다중 임베딩(Word2Vec + DistilBERT)이 단일 임베딩보다 클러스터링 성능을 향상시키는가?

**가설**:
- H1: 다중 임베딩이 단일 임베딩보다 NPMI가 높다
- H2: Word2Vec과 DistilBERT는 상호보완적이다

---

### RQ2. 모델 구성 요소의 기여도 (Component-wise Ablation)
**질문**: GraphMAE2의 하이퍼파라미터가 클러스터링 성능에 어떤 영향을 미치는가?

**가설**:
- H3: 적절한 mask rate (0.5)가 최적 성능을 낸다
- H4: 500 epochs에서 수렴한다
- H5: 임베딩 차원 256이 최적이다

---

### RQ3. 전통적 방법 vs 제안 방법 (Overall Comparison)
**질문**: GraphMAE2 기반 방법이 전통적 커뮤니티 탐지 알고리즘보다 의미적으로 일관된 클러스터를 생성하는가?

**가설**:
- H6: 제안 모델의 NPMI가 Louvain/Leiden보다 높다
- H7: Louvain/Leiden의 모듈성이 제안 모델보다 높다 (예상됨)
- H8: 제안 모델의 Silhouette Score가 더 높다

---

## 📊 데이터셋 설정

### AG News 데이터셋

```python
data/ag_news.csv text칼럼 활용
```


### 어휘 구성

```python
# 상위 빈도 500 단어 선택
- Vocab size: 500 단어
```

### 공출현 그래프 구축

```python
# Co-occurrence graph parameters
- Min co-occurrence: 5 (최소 5회 공출현, edge_weight_threshold 하이퍼 파라미터 활용)
- Edge weight: 공출현 빈도
```

---

## 🔬 비교 모델 구성

| 모델명 | 입력 그래프 | 노드 임베딩 | 클러스터링 방법 | 목적 |
|--------|------------|------------|----------------|------|
| **Louvain** | 공출현 그래프 | 없음 (구조만) | Louvain | 전통적 구조 기반 |
| **Leiden** | 공출현 그래프 | 없음 (구조만) | Leiden | 전통적 구조 기반 (개선) |
| **GNN-W2V** | 공출현 그래프 | Word2Vec만 | GraphMAE2 + K-means | GNN + 단일 임베딩 |
| **GNN-BERT** | 공출현 그래프 | DistilBERT만 | GraphMAE2 + K-means | GNN + 단일 임베딩 |
| **제안 모델 (Ours)** | 공출현 그래프 | Word2Vec + DistilBERT | GraphMAE2 + K-means | 최종 제안 방법 |

---

## 🧪 실험별 세부 설계

### 실험 1: RQ1 - 단일 vs 다중 임베딩

#### 비교군
1. **W2V-KMeans**
   - Word2Vec 임베딩만 사용
   - PCA로 256차원 축소
   - K-means 클러스터링

2. **BERT-KMeans**
   - DistilBERT 임베딩만 사용
   - PCA로 256차원 축소
   - K-means 클러스터링

#### 실험군
3. **Concat-KMeans**
   - Word2Vec (256차원) + DistilBERT (256차원)
   - Concatenation → 512차원
   - K-means 클러스터링

4. **제안 모델 (Ours)**
   - Word2Vec + DistilBERT + GraphMAE2
   - 최종 비교

#### 평가 지표
- NPMI (주 지표)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

#### 통계 검정
- Two-tailed t-test (n=5)
- Bonferroni correction (3 comparisons)
- Cohen's d (effect size)

---

### 실험 2: RQ2 - Ablation Study

#### 2.1 Mask Rate 실험

```python
mask_rates = [0.3, 0.5, 0.75, 0.9]

# 고정 파라미터
- epochs: 500
- PCA dim: 256 (each embedding) # 실험 1 이후 최적의 값 선택
- learning_rate: 0.001
```

**평가**:
- NPMI (주 지표)
- Silhouette Score
- Training time
- Reconstruction loss
---

#### 2.2 Training Epochs 실험

```python
epochs_list = [250, 500, 1000, 2000]

# 고정 파라미터
- mask_rate: 0.5 # (2.1에서 선택된 최적값)
- PCA dim: 256
- learning_rate: 0.001
```

**평가**:
- NPMI
- Silhouette Score
- Training time
- 수렴 분석

---

#### 2.3 Embedding Dimension 실험

```python
pca_dims = [64, 128, 256]  # 각 임베딩의 차원
# Final dimension = pca_dim × 2 (concat)

# 고정 파라미터
- mask_rate: 0.5
- epochs: 500
- learning_rate: 0.001
```

**평가**:
- NPMI
- Silhouette Score
- Davies-Bouldin Index
---

#### 2.4 최적 설정 결정

Ablation Study 결과를 바탕으로 최적 하이퍼파라미터 결정:
```python
OPTIMAL_CONFIG = {
    'mask_rate': 0.5,  # (예상)
    'epochs': 500,     # (예상)
    'pca_dim': 256,    # (예상)
    'final_dim': 512,  # 256 × 2
}
```

**이후 RQ3의 모든 실험은 이 설정 사용**

---

### 실험 3: RQ3 - 전통적 방법 vs 제안 방법

#### 비교군
1. **Louvain**
   - 공출현 그래프에 직접 적용
   - Resolution parameter: 1.0

2. **Leiden**
   - 공출현 그래프에 직접 적용
   - Resolution parameter: 1.0

#### 실험군
4. **제안 모델 (Ours)**
   - 최적 하이퍼파라미터 사용
   - GraphMAE2 + Word2Vec + DistilBERT

#### 평가 지표 (모든 지표 사용)


**임베딩 공간 품질** (제안 모델에 유리):
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

**의미적 일관성** (핵심 지표):
- NPMI

#### 통계 검정
- Two-tailed t-test (Ours vs Leiden)
- Bonferroni correction
- Cohen's d

---

## 🔧 실험 실행 프로토콜

### 공통 설정

```python
COMMON_CONFIG = {
    'random_seeds': [42, 123, 456, 789, 101],
    'n_runs': 5,
}
```

---

### Word2Vec 설정

```python
WORD2VEC_CONFIG = {
    'vector_size': 300,  # 원본 차원
    'window': 5,
    'min_count': 5,
    'workers': 4,
    'sg': 1,  # Skip-gram
    'epochs': 100,
}
```

---

### DistilBERT 설정

```python
DISTILBERT_CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 32,
    'pooling': 'mean',  # Mean pooling over tokens
}
```

---

### PCA 설정

```python
PCA_CONFIG = {
    'n_components': 256,  # 각 임베딩
    'whiten': True,
    'random_state': 42,
}
```

---

### GraphMAE2 설정

```python
GRAPHMAE2_CONFIG = {
    'mask_rate': 0.5,  # Ablation 후 결정
    'encoder_hidden_dim': 256,
    'decoder_hidden_dim': 256,
    'encoder_num_layers': 2,
    'decoder_num_layers': 1,
    'encoder_activation': 'relu',
    'decoder_activation': 'relu',
    'encoder_dropout': 0.2,
    'decoder_dropout': 0.2,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 500,  # Ablation 후 결정
    'batch_size': None,  # Full batch
}
```

---

### K-means 설정

```python
KMEANS_CONFIG = {
    'method': 'elbow',
    'k_range': (5, 20),
    'n_init': 10,
    'max_iter': 300,
    'random_state': None,  # 각 run마다 다름
}

# Elbow method로 최적 k 선택
# Silhouette score도 함께 고려
```

---

### Louvain/Leiden 설정

```python
COMMUNITY_CONFIG = {
    'resolution': 1.0,
    'random_state': None,  # 각 run마다 다름
}
```

---

### 실험 실행 순서

```python
for seed in random_seeds:
    # 1. 데이터 준비
    set_all_seeds(seed)
    train, val, test = split_data(dataset, seed=seed)
    
    # 2. 공출현 그래프 구축
    G = build_cooccurrence_graph(train, vocab, window_size=5)
    
    # 3. 임베딩 생성
    word2vec_emb = train_word2vec(train, vocab)
    distilbert_emb = extract_distilbert(vocab)
    
    # 4. PCA 적용
    word2vec_pca = apply_pca(word2vec_emb, n_components=256)
    distilbert_pca = apply_pca(distilbert_emb, n_components=256)
    
    # 5. 모델별 실행
    for model_name in models:
        if model_name == 'Louvain':
            clusters = louvain(G)
        elif model_name == 'Leiden':
            clusters = leiden(G)
        elif model_name == 'W2V-KMeans':
            clusters = kmeans(word2vec_pca)
        elif model_name == 'BERT-KMeans':
            clusters = kmeans(distilbert_pca)
        elif model_name == 'Concat-KMeans':
            concat = np.concatenate([word2vec_pca, distilbert_pca], axis=1)
            clusters = kmeans(concat)
        elif model_name == 'GNN-W2V':
            gnn_emb = graphmae2(G, word2vec_pca)
            clusters = kmeans(gnn_emb)
        elif model_name == 'GNN-BERT':
            gnn_emb = graphmae2(G, distilbert_pca)
            clusters = kmeans(gnn_emb)
        elif model_name == 'Ours':
            concat = np.concatenate([word2vec_pca, distilbert_pca], axis=1)
            gnn_emb = graphmae2(G, concat)
            clusters = kmeans(gnn_emb)
        
        # 6. 평가
        results[model_name][seed] = evaluate(clusters, G, embeddings, corpus)
    
    # 7. 결과 저장
    save_results(results, seed)
```

---

## 📊 결과 보고 형식

### Table 1: RQ1 - Single vs Multi-modal Embeddings (Mean ± Std, n=5)

```
| Model          | NPMI ↑         | Silhouette ↑   | DBI ↓          | CHI ↑          | NMI ↑          | ARI ↑          |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| W2V-KMeans     | 0.38 ± 0.03    | 0.42 ± 0.04    | 1.85 ± 0.12    | 245.6 ± 18.3   | 0.35 ± 0.02    | 0.28 ± 0.03    |
| BERT-KMeans    | 0.41 ± 0.02    | 0.45 ± 0.03    | 1.72 ± 0.10    | 268.3 ± 21.5   | 0.38 ± 0.03    | 0.31 ± 0.02    |
| Concat-KMeans  | 0.48 ± 0.03**  | 0.51 ± 0.04**  | 1.58 ± 0.11**  | 312.5 ± 24.7** | 0.45 ± 0.02**  | 0.39 ± 0.03**  |
| Ours           | 0.61 ± 0.02*** | 0.63 ± 0.03*** | 1.28 ± 0.09*** | 398.5 ± 28.2***| 0.58 ± 0.02*** | 0.52 ± 0.03*** |

**: p < 0.01 (vs BERT-KMeans, best single embedding)
***: p < 0.001 (vs Concat-KMeans, best non-GNN baseline)
```

---

### Table 2: RQ2.1 - Effect of Mask Rate (Mean ± Std, n=5)

```
| Mask Rate | NPMI ↑         | Silhouette ↑   | Training Time  | Recon. Loss    |
|-----------|----------------|----------------|----------------|----------------|
| 0.3       | 0.58 ± 0.03    | 0.60 ± 0.04    | 45 ± 3 min     | 0.152 ± 0.008  |
| 0.5       | 0.61 ± 0.02*   | 0.63 ± 0.03*   | 52 ± 4 min     | 0.148 ± 0.006* |
| 0.75      | 0.59 ± 0.04    | 0.61 ± 0.05    | 68 ± 5 min     | 0.165 ± 0.009  |

*: Best performance
```

---

### Table 3: RQ2.2 - Effect of Training Epochs (Mean ± Std, n=5)

```
| Epochs | NPMI ↑         | Silhouette ↑   | Training Time  |
|--------|----------------|----------------|----------------|
| 250    | 0.57 ± 0.03    | 0.59 ± 0.04    | 26 ± 2 min     |
| 500    | 0.61 ± 0.02*   | 0.63 ± 0.03*   | 52 ± 4 min     |
| 1000   | 0.61 ± 0.02    | 0.63 ± 0.03    | 104 ± 6 min    |

*: Best performance (converged)
```

---

### Table 4: RQ2.3 - Effect of Embedding Dimension (Mean ± Std, n=5)

```
| PCA Dim (each) | Final Dim | NPMI ↑         | Silhouette ↑   | DBI ↓          |
|----------------|-----------|----------------|----------------|----------------|
| 128            | 256       | 0.58 ± 0.03    | 0.60 ± 0.04    | 1.42 ± 0.10    |
| 256            | 512       | 0.61 ± 0.02*   | 0.63 ± 0.03*   | 1.28 ± 0.09*   |
| 512            | 1024      | 0.60 ± 0.02    | 0.62 ± 0.03    | 1.31 ± 0.11    |

*: Best performance
```

---

### Table 5: RQ3 - Overall Performance Comparison (Mean ± Std, n=5)

```
| Model          | Modularity ↑   | NPMI ↑         | Silhouette ↑   | DBI ↓          | CHI ↑          | NMI ↑          | ARI ↑          |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Louvain        | 0.45 ± 0.02*   | 0.32 ± 0.04    | -              | -              | -              | 0.28 ± 0.03    | 0.22 ± 0.04    |
| Leiden         | 0.47 ± 0.02*   | 0.34 ± 0.03    | -              | -              | -              | 0.31 ± 0.02    | 0.25 ± 0.03    |
| Concat-KMeans  | -              | 0.48 ± 0.03    | 0.51 ± 0.04    | 1.58 ± 0.11    | 312.5 ± 24.7   | 0.45 ± 0.02    | 0.39 ± 0.03    |
| **Ours**       | 0.41 ± 0.03    | **0.61 ± 0.02***| **0.63 ± 0.03***| **1.28 ± 0.09***| **398.5 ± 28.2***| **0.58 ± 0.02***| **0.52 ± 0.03***|

*: Best in category
**: p < 0.001 (vs Leiden for structure-based, vs Concat-KMeans for embedding-based)
```

---

### 통계 검정 세부사항

```python
# Bonferroni correction
alpha = 0.05
n_comparisons = 3  # Ours vs (Leiden, Concat-KMeans, GNN-BERT)
alpha_corrected = alpha / n_comparisons  # 0.0167

# Effect size (Cohen's d)
def cohens_d(group1, group2):
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std

# 해석:
# |d| < 0.2: negligible
# 0.2 ≤ |d| < 0.5: small
# 0.5 ≤ |d| < 0.8: medium
# |d| ≥ 0.8: large
```

---

## 📈 시각화 계획

### Figure 1: Cluster Visualization (t-SNE)
- 4개 subplot: Louvain / Leiden / Ours
- 각 subplot: 500개 단어를 2D 투영, 색상으로 클러스터 구분
- 같은 t-SNE 투영 사용 (공정한 비교)

### Figure 2: NPMI Comparison (Bar Chart with Error Bars)
- X축: 모든 모델
- Y축: NPMI
- 에러바: 표준편차 (n=5)

### Figure 3: Ablation Study - Mask Rate (Box Plot)
- X축: Mask rate (0.3, 0.5, 0.75)
- Y축: NPMI
- Box plot으로 분포 표시

### Figure 4: Ablation Study - Training Epochs (Line Plot)
- X축: Epochs (0~1000)
- Y축: NPMI (validation)
- 수렴 과정 시각화

### Figure 5: Word Clouds for Top Clusters
- 3x2 grid (상위 6개 클러스터)
- 각 클러스터의 상위 30개 단어
- 제안 모델의 결과

### Figure 6: Performance Heatmap
- Rows: 모델
- Columns: 평가 지표 (normalized)
- 색상: 성능 (높을수록 진함)

---

## ✅ 실험 체크리스트

### 실험 전 준비
- [ ] 모든 시각화 함수 구현 완료
- [ ] 각 통계 준비(기술 통계 + 통계 검정, 이건 core디렉토리가 아닌 실행함수단에서 처리하도록)

### RQ1 실험
- [ ] W2V-KMeans 실행 (5회)
- [ ] BERT-KMeans 실행 (5회)
- [ ] Concat-KMeans 실행 (5회)
- [ ] 제안 모델 실행 (5회)
- [ ] 모든 평가 지표 계산
- [ ] 통계적 검정 수행
- [ ] Table 1 작성
- [ ] Figure 1, 2 생성

### RQ2 실험 (Ablation Study)
- [ ] 2.1 Mask rate 실험 (3 × 5 = 15회)
- [ ] 2.2 Epochs 실험 (3 × 5 = 15회)
- [ ] 2.3 Embedding dim 실험 (3 × 5 = 15회)
- [ ] 최적 설정 결정 및 문서화
- [ ] Table 2, 3, 4 작성
- [ ] Figure 3, 4 생성

### RQ3 실험
- [ ] Louvain 실행 (5회)
- [ ] Leiden 실행 (5회)
- [ ] 제안 모델 실행 (최적 설정, 5회)
- [ ] 모든 평가 지표 계산
- [ ] 통계적 검정 수행
- [ ] Effect size 계산
- [ ] Table 5 작성
- [ ] Figure 5, 6 생성

### 결과 분석 및 문서화
- [ ] 모든 표 완성
- [ ] 모든 그림 생성
- [ ] 통계 검정 결과 정리
- [ ] Qualitative analysis (클러스터 예시)
- [ ] 결과 해석 및 논의 작성
- [ ] 코드 및 데이터 아카이빙

---

## 🚨 중요 참고사항

### 1. 공정한 비교
- 모든 모델에 동일한 공출현 그래프 사용
- 동일한 어휘(500 단어) 사용
- 동일한 데이터 분할 사용 (각 seed별)
- 동일한 평가 지표 사용

### 2. 재현성
- 모든 설정을 config 파일에 저장
- 각 실행의 난수 시드 기록
- 중간 결과물 저장 (임베딩, 그래프 등)
- 최종 클러스터 결과 저장

### 3. 모듈성 지표 해석
- Louvain/Leiden이 모듈성에서 높은 것은 당연함 (직접 최적화)
- 제안 모델이 모듈성에서 낮더라도 NPMI가 높으면 성공
- 합리적인 모듈성 유지만 확인 (0.3 이상)

### 4. NPMI가 핵심
- 단어 클러스터링의 최종 목표는 의미적 일관성
- NPMI에서 제안 모델이 우수해야 함
- 다른 지표는 보조적

### 5. 계산 비용 관리
```python
# 예상 실행 시간 (1 run)
Louvain: 2분
Leiden: 3분
W2V-KMeans: 5분
BERT-KMeans: 8분
Concat-KMeans: 10분
GNN-W2V: 55분
GNN-BERT: 60분
Ours: 65분

# 총 실험 시간 (5 runs)
RQ1: (5 + 8 + 10 + 65) × 5 = 440분 (7.3시간)
RQ2: 65 × 3 × 3 × 5 = 2,925분 (48.8시간)  # Ablation
RQ3: (2 + 3 + 65) × 5 = 350분 (5.8시간)

# 총: 약 62시간
# 병렬 처리 가능하면 훨씬 단축
```

### 6. 디버깅 팁
- 소규모 실험으로 먼저 검증 (vocab=100, 1 run)
- 각 단계별로 중간 결과 저장
- 로그 자세히 기록
- Sanity check (클러스터 수, 임베딩 차원 등)

---

## 📝 예상 결과 시나리오

### 시나리오 1: 이상적 (예상)
```
NPMI: Ours > GNN-BERT > GNN-W2V > Concat-KM > BERT-KM > W2V-KM
Modularity: Leiden > Louvain > Ours
NMI: Ours > Concat-KM > BERT-KM > W2V-KM > Leiden > Louvain
```
**해석**: 제안 방법이 구조와 의미를 모두 잘 포착

### 시나리오 2: GNN 효과 미미
```
NPMI: Concat-KM ≈ Ours
```
**해석**: GNN이 추가 이득을 제공하지 못함 → 방법론 재검토

### 시나리오 3: 다중 임베딩 효과 미미
```
NPMI: GNN-BERT ≈ Ours
```
**해석**: Word2Vec 추가가 도움 안됨 → 임베딩 결합 방법 재고

---

## 🎓 논문 작성 시 권장사항

### Results 섹션 구조
```
4. Results
  4.1 Dataset and Experimental Setup
  4.2 RQ1: Effect of Multi-modal Embeddings
  4.3 RQ2: Ablation Study on Model Components
    4.3.1 Effect of Mask Rate
    4.3.2 Effect of Training Epochs
    4.3.3 Effect of Embedding Dimension
    4.3.4 Optimal Configuration
  4.4 RQ3: Comparison with Traditional Methods
  4.5 Qualitative Analysis
  4.6 Key Findings
```

### 주요 메시지
1. **Multi-modal embeddings are essential** (RQ1)
2. **Optimal hyperparameters matter** (RQ2)
3. **Proposed method outperforms baselines** (RQ3)
4. **Trade-off: quality vs speed** (현실적 제약 인정)

---

**끝. 이 프로토콜을 따라 체계적으로 실험을 진행하세요!**