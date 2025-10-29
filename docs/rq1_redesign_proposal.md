# RQ1 재설계 제안: 완전한 Ablation Study

## 📊 현재 문제점

### 1. **NPMI 하락 설명 불가**
```
BERT-KMeans:  0.170 ± 0.029  (GraphMAE 없음)
GRACE:        0.135 ± 0.008  (GraphMAE 있음)

왜 낮아지는가?
- GraphMAE 때문인가?
- Multi-modal 융합 때문인가?
- 클러스터 수 차이 때문인가?

→ 현재 실험으로는 원인 분리 불가능!
```

### 2. **GraphMAE 효과 vs Multi-modal 효과 분리 불가**
```
현재 비교:
BERT-KMeans (no GraphMAE) vs GRACE (GraphMAE + Multi-modal)
           └────────────────────────┬─────────────────────────┘
                        2가지 변수가 동시에 변경됨!

→ 어떤 것이 성능 향상의 주된 원인인지 불명확
```

### 3. **불완전한 Ablation Study**
```
현재 4가지 조건만 테스트:
✓ W2V + no GraphMAE
✓ BERT + no GraphMAE
✓ W2V+BERT + no GraphMAE (Concat)
✓ W2V+BERT + GraphMAE (GRACE)

누락:
✗ W2V + GraphMAE
✗ BERT + GraphMAE

→ 단일 모달에서 GraphMAE 효과를 측정 불가
```

---

## 🎯 제안: RQ1 실험 재설계

### **새로운 실험 설계 (2×3 Factorial Design)**

| 모달리티 \ GraphMAE | No GraphMAE | GraphMAE |
|---------------------|-------------|----------|
| **W2V only**        | 1. W2V-KMeans | 3. W2V-GraphMAE ⭐ |
| **BERT only**       | 2. BERT-KMeans | 4. BERT-GraphMAE ⭐ |
| **W2V+BERT**        | 5. Concat-KMeans | 6. GRACE |

⭐ = 추가할 실험

---

## 📋 상세 실험 조건

### **1. W2V-KMeans** (기존)
```python
config = {
    'embedding_method': 'w2v',
    'use_graphmae': False,
    'pca_dim': 256,
    'clustering': 'kmeans'
}
```
- **목적**: W2V 베이스라인
- **차원**: 256d

### **2. BERT-KMeans** (기존)
```python
config = {
    'embedding_method': 'bert',
    'use_graphmae': False,
    'pca_dim': 256,
    'clustering': 'kmeans'
}
```
- **목적**: BERT 베이스라인
- **차원**: 256d

### **3. W2V-GraphMAE** ⭐ (신규)
```python
config = {
    'embedding_method': 'w2v',
    'use_graphmae': True,
    'graphmae_epochs': 500,
    'encoder_type': 'gat',
    'decoder_type': 'gat',
    'pca_dim': 256,
    'clustering': 'kmeans'
}
```
- **목적**: W2V에서 GraphMAE 효과 측정
- **차원**: 256d → GraphMAE → 256d
- **비교**: 1 vs 3 → GraphMAE가 W2V를 얼마나 개선하는가?

### **4. BERT-GraphMAE** ⭐ (신규)
```python
config = {
    'embedding_method': 'bert',
    'use_graphmae': True,
    'graphmae_epochs': 500,
    'encoder_type': 'gat',
    'decoder_type': 'gat',
    'pca_dim': 256,
    'clustering': 'kmeans'
}
```
- **목적**: BERT에서 GraphMAE 효과 측정
- **차원**: 256d → GraphMAE → 256d
- **비교**: 2 vs 4 → GraphMAE가 BERT를 얼마나 개선하는가?

### **5. Concat-KMeans** (기존)
```python
config = {
    'embedding_method': 'concat',  # W2V(256d) + BERT(256d)
    'use_graphmae': False,
    'pca_dim': 256,  # 각각
    'clustering': 'kmeans'
}
```
- **목적**: 단순 연결 베이스라인
- **차원**: 512d (256+256)

### **6. GRACE** (기존)
```python
config = {
    'embedding_method': 'concat',  # W2V(256d) + BERT(256d)
    'use_graphmae': True,
    'graphmae_epochs': 500,
    'encoder_type': 'gat',
    'decoder_type': 'gat',
    'pca_dim': 256,  # 각각
    'clustering': 'kmeans'
}
```
- **목적**: 최종 모델
- **차원**: 512d → GraphMAE → 256d

---

## 🔬 분석 가능한 Research Questions

### **RQ1-1: GraphMAE의 효과는?**
```
비교 1: W2V-KMeans (1) vs W2V-GraphMAE (3)
비교 2: BERT-KMeans (2) vs BERT-GraphMAE (4)
비교 3: Concat-KMeans (5) vs GRACE (6)

예측:
- GraphMAE가 모든 경우에서 성능 향상
- 향상 정도: W2V > BERT > Concat (W2V가 GraphMAE로부터 가장 큰 도움)
```

### **RQ1-2: Multi-modal 융합의 효과는?**
```
비교 1: BERT-GraphMAE (4) vs GRACE (6)
         └─ 단일 BERT + GraphMAE
                              └─ Multi-modal + GraphMAE

비교 2: W2V-GraphMAE (3) + BERT-GraphMAE (4) vs GRACE (6)
         └─ 두 단일 모델의 앙상블?
                                           └─ 통합 모델

예측:
- GRACE > BERT-GraphMAE (multi-modal이 추가 향상 제공)
- 향상 정도가 작다면: GraphMAE가 주요 기여자
- 향상 정도가 크다면: Multi-modal 융합이 중요
```

### **RQ1-3: 단순 연결의 무용함**
```
비교: BERT-KMeans (2) vs Concat-KMeans (5)

예측:
- 성능 차이 없음 (이미 확인됨)
- W2V 정보가 완전히 무시됨
```

### **RQ1-4: NPMI 하락 원인 규명**
```
비교 분석:
Scenario A: GraphMAE가 NPMI를 낮춤
  2 (BERT-KMeans): 높은 NPMI
  4 (BERT-GraphMAE): 낮은 NPMI
  → GraphMAE가 원인

Scenario B: Multi-modal이 NPMI를 낮춤
  4 (BERT-GraphMAE): 높은 NPMI
  6 (GRACE): 낮은 NPMI
  → Multi-modal 융합이 원인

Scenario C: 클러스터 수 차이
  클러스터 수와 NPMI의 상관관계 분석
```

---

## 📊 예상 결과 패턴

### **시나리오 1: GraphMAE가 주된 기여자**
```
Silhouette Score 예측:
1. W2V-KMeans:      0.00
2. BERT-KMeans:    -0.03
3. W2V-GraphMAE:    0.20  ← 큰 향상
4. BERT-GraphMAE:   0.25  ← 큰 향상
5. Concat-KMeans:  -0.03
6. GRACE:           0.31  ← 추가 향상 (작음)

해석:
- GraphMAE가 단일 모달에서도 큰 향상
- Multi-modal은 추가적이지만 작은 향상
```

### **시나리오 2: Multi-modal이 중요**
```
Silhouette Score 예측:
1. W2V-KMeans:      0.00
2. BERT-KMeans:    -0.03
3. W2V-GraphMAE:    0.10  ← 작은 향상
4. BERT-GraphMAE:   0.12  ← 작은 향상
5. Concat-KMeans:  -0.03
6. GRACE:           0.31  ← 큰 향상

해석:
- GraphMAE 단독으로는 제한적 향상
- Multi-modal + GraphMAE의 시너지가 핵심
```

### **시나리오 3: NPMI Trade-off**
```
NPMI 예측:
1. W2V-KMeans:      0.05
2. BERT-KMeans:     0.17
3. W2V-GraphMAE:    0.12  ← 향상
4. BERT-GraphMAE:   0.18  ← 유지 또는 약간 향상
5. Concat-KMeans:   0.17
6. GRACE:           0.14  ← 약간 하락

해석:
- GraphMAE는 클러스터 분리를 최적화
- 의미적 일관성(NPMI)을 약간 희생
- 하지만 전반적 품질(Silhouette)은 크게 향상
```

---

## 🛠️ 구현 계획

### **Phase 1: 코드 수정**
```python
# experiments/rq1_single_vs_multi_embedding.py

def run_single_experiment(config, seed):
    """단일 실험 실행"""

    # 1. 데이터 로드
    graph = load_graph(config)

    # 2. 임베딩 생성
    if config['embedding_method'] == 'w2v':
        embeddings = get_w2v_embeddings(graph, dim=config['pca_dim'])
    elif config['embedding_method'] == 'bert':
        embeddings = get_bert_embeddings(graph, dim=config['pca_dim'])
    elif config['embedding_method'] == 'concat':
        w2v_emb = get_w2v_embeddings(graph, dim=config['pca_dim'])
        bert_emb = get_bert_embeddings(graph, dim=config['pca_dim'])
        embeddings = np.hstack([w2v_emb, bert_emb])

    # 3. GraphMAE (선택적)
    if config['use_graphmae']:
        embeddings = apply_graphmae(
            graph,
            embeddings,
            epochs=config['graphmae_epochs'],
            encoder=config['encoder_type'],
            decoder=config['decoder_type']
        )

    # 4. 클러스터링
    clusters = clustering(embeddings, config)

    # 5. 평가
    metrics = evaluate(clusters, config)

    return metrics

# 실험 조건 정의
experiments = [
    {'name': 'W2V-KMeans', 'embedding': 'w2v', 'graphmae': False},
    {'name': 'BERT-KMeans', 'embedding': 'bert', 'graphmae': False},
    {'name': 'W2V-GraphMAE', 'embedding': 'w2v', 'graphmae': True},      # 신규
    {'name': 'BERT-GraphMAE', 'embedding': 'bert', 'graphmae': True},    # 신규
    {'name': 'Concat-KMeans', 'embedding': 'concat', 'graphmae': False},
    {'name': 'GRACE', 'embedding': 'concat', 'graphmae': True},
]
```

### **Phase 2: 실험 실행**
```bash
# 전체 실험 실행 (6 models × 5 seeds = 30 runs)
python experiments/rq1_single_vs_multi_embedding.py --full-ablation

# 예상 소요 시간:
# - W2V/BERT + no GraphMAE: ~5분/run × 10 runs = 50분
# - W2V/BERT + GraphMAE: ~20분/run × 10 runs = 200분
# - Concat + no GraphMAE: ~10분/run × 5 runs = 50분
# - GRACE: ~30분/run × 5 runs = 150분
# 총: ~450분 (7.5시간)
```

### **Phase 3: 분석 및 시각화**
```python
# 1. 기본 통계 테이블
# 2. Ablation 분석 테이블
# 3. Heatmap (6 models × 4 metrics)
# 4. Bar plots with error bars
# 5. NPMI 하락 원인 분석
```

---

## 📝 논문에 포함할 내용

### **Table 1: Complete Ablation Study Results**
```
Model               Modality    GraphMAE   Silhouette    DB Index    CH Index    NPMI
------------------------------------------------------------------------------------
W2V-KMeans         Single       ✗          0.00±0.00    7.04±0.60   2.08±0.23   0.05±0.01
BERT-KMeans        Single       ✗         -0.03±0.01    4.75±0.42   1.47±0.01   0.17±0.03
W2V-GraphMAE       Single       ✓          ?.??±?.??    ?.??±?.??   ?.??±?.??   ?.??±?.??
BERT-GraphMAE      Single       ✓          ?.??±?.??    ?.??±?.??   ?.??±?.??   ?.??±?.??
Concat-KMeans      Multi        ✗         -0.03±0.01    4.75±0.42   1.47±0.01   0.17±0.03
GRACE              Multi        ✓          0.31±0.04    1.46±0.11  177.34±41.3  0.13±0.01
```

### **Table 2: Effect Decomposition**
```
Comparison                                  ΔSilhouette    Interpretation
--------------------------------------------------------------------------------
GraphMAE Effect on W2V (3 vs 1)            +0.XX          GraphMAE improves W2V
GraphMAE Effect on BERT (4 vs 2)           +0.XX          GraphMAE improves BERT
Multi-modal Effect (6 vs 4)                +0.XX          Multi-modal adds value
Concat Failure (5 vs 2)                    +0.00          Naive concat useless
```

---

## ⚠️ 주의사항

### 1. **계산 비용**
- 단일 모달 GraphMAE는 512d 아닌 256d만 처리
- 따라서 GRACE보다 빠를 것 (약 1/2 시간)

### 2. **공정한 비교를 위한 설정 통일**
```python
# 모든 실험에서 동일하게:
- vocab_size = 500
- pca_dim = 256 (단일), 256+256 (다중)
- graphmae_epochs = 500
- encoder/decoder = 'gat'
- random_seeds = [42, 123, 456, 789, 101]
```

### 3. **예상되는 도전**
- W2V-GraphMAE가 BERT-GraphMAE보다 낮을 수 있음
  → 이는 자연스러운 결과 (W2V 자체가 약함)
- BERT-GraphMAE가 GRACE와 비슷하면?
  → Multi-modal의 가치가 제한적임을 시사
  → 하지만 여전히 GRACE가 최고라면 논문에 문제없음

---

## 🎯 결론

### **왜 이 실험이 필요한가?**

1. **NPMI 하락 원인 규명**
   - 논문 리뷰어가 질문할 가능성 높음
   - 명확한 ablation study로 선제적 대응

2. **GraphMAE vs Multi-modal 기여도 분리**
   - 더 강력한 주장 가능
   - "GRACE는 두 가지 혁신을 통합" vs "GRACE는 단순히 GraphMAE 적용"

3. **완전한 스토리 구성**
   - 단일 모달: W2V < BERT
   - GraphMAE 추가: 둘 다 향상
   - Multi-modal: 추가 향상
   - 단순 연결: 실패
   - GRACE: 최고

### **시작할까요?**

제가 지금 바로:
1. ✅ 실험 코드 수정 (W2V-GraphMAE, BERT-GraphMAE 추가)
2. ✅ 실험 실행 스크립트 작성
3. ✅ 분석 코드 업데이트

진행할까요?
