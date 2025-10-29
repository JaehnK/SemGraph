# Spherical K-means 클러스터링: 기술 명세서

## 1. 개요

Spherical K-means는 텍스트 임베딩과 같이 정규화된 고차원 벡터에 적합하도록 개선된 K-means 알고리즘의 변형이다. 유클리드 거리를 최소화하는 표준 K-means와 달리, Spherical K-means는 단위 초구(unit hypersphere) 상에서 동작하며 **코사인 거리(cosine distance)**를 비유사도 척도로 사용한다. 이는 노드 임베딩이 정규화된 의미연결망 클러스터링에 특히 적합하다.

**주요 특성:**
- **거리 척도:** 코사인 거리 (1 - 코사인 유사도)
- **정규화:** 모든 데이터 포인트와 중심점을 L2 정규화
- **최적화 목적:** 클러스터 내 코사인 거리의 합 최소화
- **초기화:** 코사인 거리 기반 K-means++
- **수렴 조건:** 관성(inertia) 변화량이 임계값 이하

---

## 2. 수학적 정식화

### 2.1. 거리 척도

L2 정규화된 벡터 **x**, **y** ∈ ℝ^d (||**x**||₂ = ||**y**||₂ = 1)에 대해:

**코사인 유사도(Cosine Similarity):**
```
cos_sim(x, y) = x · y = Σ(xᵢ × yᵢ)
```

**코사인 거리(Cosine Distance):**
```
d_cos(x, y) = 1 - cos_sim(x, y) = 1 - x · y
```

이 거리는 0(동일 방향)에서 2(반대 방향) 사이의 값을 가지며, 1은 직교를 의미한다.

### 2.2. 목적 함수

데이터셋 **X** = {**x**₁, **x**₂, ..., **x**_n} (각 **x**_i ∈ ℝ^d는 L2 정규화됨)과 K개의 클러스터 중심점 **C** = {**c**₁, **c**₂, ..., **c**_K}가 주어졌을 때:

**최소화 목적:**
```
J = Σ_{i=1}^{n} Σ_{k=1}^{K} r_ik × d_cos(x_i, c_k)
  = Σ_{i=1}^{n} Σ_{k=1}^{K} r_ik × (1 - x_i · c_k)
```

여기서 r_ik ∈ {0, 1}은 클러스터 할당 지시자(indicator):
```
r_ik = {
  1  if k = argmax_j (x_i · c_j)
  0  otherwise
}
```

이는 코사인 유사도의 합을 최대화하는 것과 동치:
```
최대화: Σ_{i=1}^{n} Σ_{k=1}^{K} r_ik × (x_i · c_k)
```

---

## 3. 알고리즘 상세

### 3.1. 전처리: 벡터 정규화

모든 입력 임베딩은 클러스터링 전에 L2 정규화됨:

```python
def normalize(X):
    """
    각 행 벡터를 L2 정규화

    Args:
        X: (n, d) 임베딩 배열

    Returns:
        X_norm: (n, d) 정규화된 임베딩 배열
    """
    norms = ||X||₂ (axis=1 기준)
    norms[norms == 0] = 1  # 0으로 나누기 방지
    return X / norms
```

**구현 위치:** [SphericalKMeansClusteringService.py:113-126](core/services/clustering/SphericalKMeansClusteringService.py#L113-L126)

### 3.2. 초기화: 코사인 거리 기반 K-means++

불량한 국소 최솟값(local minima)을 피하기 위해 코사인 거리에 적합하도록 개선된 K-means++ 초기화를 사용:

**알고리즘:**
1. **첫 번째 중심점:** 데이터 포인트 하나를 무작위로 선택
2. **이후 중심점 (k = 2 ~ K):**
   - 각 데이터 포인트 **x**_i에 대해 계산:
     ```
     D(x_i) = min_{j<k} d_cos(x_i, c_j) = 1 - max_{j<k} (x_i · c_j)
     ```
   - D(x_i)²에 비례하는 확률로 다음 중심점 선택
3. **모든 중심점을 단위 벡터로 정규화**

**확률 분포:**
```
P(x_i 선택됨) = D(x_i)² / Σ_j D(x_j)²
```

이를 통해 중심점들이 각도 공간에서 잘 분리되도록 보장한다.

**구현 위치:** [SphericalKMeansClusteringService.py:195-236](core/services/clustering/SphericalKMeansClusteringService.py#L195-L236)

### 3.3. 반복적 최적화

알고리즘은 수렴할 때까지 두 단계를 교대로 수행:

#### **할당 단계 (Assignment Step)**

각 데이터 포인트를 코사인 유사도가 가장 높은 클러스터에 할당:

```python
# 코사인 유사도 계산 (정규화된 벡터의 내적)
S = X @ C.T  # Shape: (n, K)

# 유사도가 최대인 클러스터에 할당
labels = argmax(S, axis=1)
```

이는 코사인 거리를 최소화하는 것과 동치:
```
argmax_k (x_i · c_k) = argmin_k (1 - x_i · c_k)
```

#### **갱신 단계 (Update Step)**

각 중심점을 해당 클러스터 멤버들의 정규화된 평균으로 갱신:

```python
for k in range(K):
    cluster_mask = (labels == k)

    if sum(cluster_mask) > 0:
        # 클러스터 멤버들의 평균 계산
        c_k = mean(X[cluster_mask], axis=0)
    else:
        # 빈 클러스터 처리: 무작위로 재초기화
        c_k = X[random_sample()]

    # 중심점을 단위 구면으로 정규화
    c_k = c_k / ||c_k||₂

C[k] = c_k
```

**이론적 근거:** 정규화된 벡터들의 평균을 다시 정규화하면, 코사인 거리의 합을 최소화하는 **구면 중심점(spherical centroid)**을 근사한다.

**구현 위치:** [SphericalKMeansClusteringService.py:128-193](core/services/clustering/SphericalKMeansClusteringService.py#L128-L193)

### 3.4. 수렴 판정

알고리즘은 **관성(inertia)** (클러스터 내 코사인 거리의 총합)을 추적:

```
Inertia(t) = Σ_{k=1}^{K} Σ_{x_i ∈ C_k} d_cos(x_i, c_k)
           = Σ_{k=1}^{K} Σ_{x_i ∈ C_k} (1 - x_i · c_k)
```

**수렴 조건:**
```
|Inertia(t) - Inertia(t-1)| < tolerance
```

**기본 파라미터:**
- `max_iter = 300`: 최대 반복 횟수
- `tolerance = 1e-4`: 수렴 임계값

**구현 위치:** [SphericalKMeansClusteringService.py:176-191](core/services/clustering/SphericalKMeansClusteringService.py#L176-L191)

---

## 4. 다중 초기화 전략

초기화에 대한 민감도를 완화하기 위해 다른 랜덤 시드로 알고리즘을 여러 번 실행:

```python
def fit_predict(X, K, n_init=10):
    best_labels = None
    best_inertia = ∞

    for trial in range(n_init):
        labels, inertia = spherical_kmeans_single_run(X, K)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return best_labels
```

**기본값:** 클러스터링당 `n_init = 10`번 시도

**구현 위치:** [SphericalKMeansClusteringService.py:23-56](core/services/clustering/SphericalKMeansClusteringService.py#L23-L56)

---

## 5. 자동 클러스터 수 선정

### 5.1. 엘보우 방법 (Elbow Method)

최적 클러스터 수 K*가 미지수일 때, 범위 [K_min, K_max]에 대해 **Elbow Method**를 적용:

**절차:**
1. 각 K ∈ [K_min, K_max]에 대해:
   - `n_init`번 시도로 spherical K-means 실행
   - inertia(K)와 silhouette_score(K) 기록

2. **관성 값 정규화:**
   ```
   inertia_norm(K) = [inertia(K) - min(inertia)] / [max(inertia) - min(inertia)]
   ```

3. **2차 미분(곡률) 계산:**
   ```
   curvature(K) = d²(inertia_norm) / dK²
   ```

4. **엘보우 포인트 선택:**
   ```
   K* = argmax_K |curvature(K)|
   ```

엘보우 포인트는 클러스터를 추가해도 관성 감소 효과가 급격히 줄어드는 K 값을 나타낸다.

**구현 위치:** [ClusteringInterface.py:127-151](core/services/clustering/ClusteringInterface.py#L127-L151)

### 5.2. 실루엣 스코어 검증

**실루엣 계수(Silhouette Coefficient)**는 코사인 거리를 사용하여 클러스터링 품질을 측정:

```
s(i) = [b(i) - a(i)] / max{a(i), b(i)}
```

여기서:
- `a(i)` = 같은 클러스터 내 포인트들까지의 평균 코사인 거리
- `b(i)` = 다른 클러스터 포인트들까지의 최소 평균 코사인 거리

**범위:** s(i) ∈ [-1, 1]
- `s ≈ 1`: 잘 클러스터링됨
- `s ≈ 0`: 클러스터 경계에 위치
- `s < 0`: 잘못 분류됨

**전체 스코어:**
```
Silhouette Score = (1/n) Σ s(i)
```

**구현:**
```python
from sklearn.metrics import silhouette_score

def cosine_silhouette(X_normalized, labels):
    return silhouette_score(X_normalized, labels, metric='cosine')
```

**구현 위치:** [SphericalKMeansClusteringService.py:238-250](core/services/clustering/SphericalKMeansClusteringService.py#L238-L250)

---

## 6. GRACE 파이프라인 통합

### 6.1. 파이프라인 컨텍스트

Spherical K-means는 GRACE (GRAph-based Clustering with Enhanced embeddings) 파이프라인의 최종 클러스터링 단계로 적용됨:

```
입력: GraphMAE 임베딩 (n_nodes × embed_dim)
       ↓
단계 1: L2 정규화
       ↓
단계 2: Spherical K-means 클러스터링
       ↓
출력: 클러스터 할당 (n_nodes,)
```

**임베딩 소스:**
- Word2Vec (256-d) + BERT (768-d) 멀티모달 임베딩
- GraphMAE 자기지도학습 그래프 인코더 출력
- 어텐션 융합 의미 표현

**구현 위치:** [GRACEPipeline.py:313-356](core/services/GRACE/GRACEPipeline.py#L313-L356)

### 6.2. 클러스터링 파라미터

**파이프라인에서 고정:**
```python
clustering_service = SphericalKMeansClusteringService(
    random_state=config.random_seed  # 재현성 보장
)
```

**GRACEConfig를 통해 설정 가능:**
- `num_clusters`: 명시적 K 값 (알고 있는 경우) 또는 None (자동 탐지)
- `min_clusters`: 엘보우 탐색 최소 K (기본값: 3)
- `max_clusters`: 엘보우 탐색 최대 K (기본값: 20)
- `n_init`: 무작위 초기화 횟수 (기본값: 10)

**사용 예시:**
```python
# 수동 K 지정
labels = clustering_service.fit_predict(
    embeddings=graphmae_embeddings,
    n_clusters=8,
    n_init=10
)

# 자동 K 선택
labels, best_k, inertias, silhouettes = clustering_service.auto_clustering(
    embeddings=graphmae_embeddings,
    min_clusters=3,
    max_clusters=20,
    n_init=10
)
```

**구현 위치:** [GRACEPipeline.py:58-59](core/services/GRACE/GRACEPipeline.py#L58-L59)

### 6.3. 출력 및 평가

**클러스터 할당:**
- 형태 `(n_nodes,)`의 Numpy 배열, 정수 레이블 0 ~ K-1
- `pipeline.cluster_labels`에 저장

**클러스터 품질 메트릭:**
1. **실루엣 스코어** (코사인 기반)
2. **NPMI** (Normalized Pointwise Mutual Information) - 의미적 응집도
3. **클러스터 분포** - 클러스터 간 크기 균형

**시각화 출력 (활성화 시):**
- 엘보우 곡선 플롯 (관성 vs. K)
- 실루엣 스코어 플롯
- 클러스터 크기 분포

**구현 위치:** [GRACEPipeline.py:361-396](core/services/GRACE/GRACEPipeline.py#L361-L396)

---

## 7. 이론적 정당화

### 7.1. 텍스트 임베딩에 코사인 거리를 사용하는 이유

**특성 1: 크기 불변성**
- 단어 임베딩은 의미를 **방향**에 인코딩하며, 크기는 부차적
- 코사인 유사도: `cos(θ) = (x·y)/(||x|| ||y||)`는 각도 θ에만 의존
- 유클리드 거리: `||x - y||₂`는 방향과 크기를 혼재

**특성 2: 고차원 기하학**
- 고차원에서 유클리드 거리는 구별력이 감소 (차원의 저주)
- 코사인 거리는 각도 분리를 측정하므로 효과적 유지

**특성 3: 이론적 동치성**
- L2 정규화된 벡터에 대해: `||x - y||₂² = 2(1 - x·y) = 2 × d_cos(x,y)`
- 정규화된 데이터에서 Spherical K-means ≈ 단위 구면에서의 Euclidean K-means

### 7.2. 수렴 특성

**정리 (Lloyd 1982):**
고정된 초기화에 대해 Spherical K-means는 유한 반복 내에 목적 함수의 국소 최솟값으로 수렴한다.

**증명 스케치:**
1. 각 할당 단계는 목적 함수를 엄격히 감소시키거나 유지
2. 각 갱신 단계 (클러스터 평균의 재정규화)는 목적 함수를 감소시키거나 유지
3. 목적 함수는 0으로 하한이 제한됨
4. 국소 최적해로의 수렴이 보장됨

**실무적 함의:** 전역 최적해를 근사하려면 다중 무작위 초기화(K-means++)가 필요하다.

### 7.3. 유클리드 K-means와의 비교

| 측면 | Euclidean K-means | Spherical K-means |
|------|-------------------|-------------------|
| 거리 척도 | L2 norm | 코사인 거리 |
| 데이터 가정 | 임의 스케일 | 정규화된 벡터 |
| 중심점 갱신 | 평균 | 정규화된 평균 |
| 크기 민감도 | 높음 | 0 (크기 불변) |
| 최적 사용처 | 저차원 특징 | 고차원 임베딩 |
| 전형적 사용 사례 | 이미지 픽셀, 표형식 데이터 | 텍스트/단어 임베딩, 토픽 모델링 |

---

## 8. 계산 복잡도

**반복당:**
- **할당 단계:** O(nKd) - d차원 벡터의 n×K 내적 계산
- **갱신 단계:** O(nd) - K개 클러스터 평균 계산
- **반복당 총계:** O(nKd)

**전체 복잡도:**
```
T_total = n_init × max_iter × O(nKd)
```

**공간 복잡도:** O(nd + Kd) (데이터와 중심점 저장)

**최적화 참고사항:**
- 내적 계산 (`X @ C.T`)은 GPU에서 고도로 병렬화 가능
- n이 매우 큰 경우 미니배치 변형 적용 가능
- 희소 임베딩은 희소 행렬 연산 활용 가능

---

## 9. 구현 세부사항

### 9.1. 수치 안정성

**영벡터 처리:**
```python
norms = np.linalg.norm(X, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)  # 0으로 나누기 방지
X_normalized = X / norms
```

**빈 클러스터 처리:**
```python
if np.sum(cluster_mask) == 0:
    # 무작위 데이터 포인트로 재초기화
    centroids[k] = X[np.random.randint(n)]
```

### 9.2. 재현성

모든 무작위 연산은 결정론적 결과를 위해 시드를 설정:

```python
np.random.seed(random_state)
```

**무작위 연산:**
1. K-means++ 초기화 (첫 중심점, 가중 샘플링)
2. 빈 클러스터 재초기화
3. 다중 무작위 재시작 (`n_init`번 시도)

**구현 위치:** [SphericalKMeansClusteringService.py:212](core/services/clustering/SphericalKMeansClusteringService.py#L212)

### 9.3. 인터페이스 준수

구현은 `ClusteringInterface` 추상 베이스 클래스를 따름:

**필수 메서드:**
- `fit_predict(embeddings, n_clusters, n_init) -> labels`
- `auto_clustering(embeddings, min_clusters, max_clusters) -> (labels, best_k, inertias, silhouettes)`
- `get_cluster_distribution() -> Dict[cluster_id, count]`
- `save_elbow_curve(k_values, output_path)`

**구현 위치:** [ClusteringInterface.py:13-151](core/services/clustering/ClusteringInterface.py#L13-L151)

---

## 10. 실험적 검증

### 10.1. 실험 설정

**데이터셋:** AG News 말뭉치 의미연결망
- 노드: 빈도 상위 N개 단어 (불용어 제거 후)
- 엣지: 공출현 관계 (PMI 가중치)
- 노드 특징: GraphMAE 향상 멀티모달 임베딩

**베이스라인 비교:**
1. 표준 K-means (유클리드 거리)
2. DBSCAN (밀도 기반)
3. 계층적 군집화 (Ward linkage)

**평가 메트릭:**
- **내적:** 실루엣 스코어, 관성/WCSS
- **의미적:** NPMI (Normalized Pointwise Mutual Information)
- **정성적:** 클러스터 응집도 인간 평가

### 10.2. 성능 특성

**관찰된 장점:**
1. **의미적 응집도:** 발견된 클러스터의 NPMI 점수 높음
2. **안정성:** 무작위 초기화 간 낮은 분산
3. **해석가능성:** 클러스터가 의미적 주제와 정렬됨

**한계점:**
1. **볼록 클러스터만:** 복잡한 다양체 구조 포착 불가
2. **구면 가정:** 클러스터가 각도적으로 분리 가능하다고 가정
3. **K 선택:** 여전히 도메인 지식 또는 휴리스틱 방법 필요

---

## 11. 사용 예시

```python
from core.services.clustering import SphericalKMeansClusteringService
import torch

# 서비스 초기화
clustering = SphericalKMeansClusteringService(random_state=42)

# 옵션 1: 고정 클러스터 수
embeddings = torch.randn(1000, 256)  # 1000개 노드, 256차원 임베딩
labels = clustering.fit_predict(embeddings, n_clusters=8, n_init=10)

# 옵션 2: 자동 클러스터 선택
labels, best_k, inertias, silhouettes = clustering.auto_clustering(
    embeddings,
    min_clusters=3,
    max_clusters=20,
    n_init=10
)

# 클러스터 통계 확인
distribution = clustering.get_cluster_distribution()
print(f"최적 K: {best_k}")
print(f"클러스터 크기: {distribution}")

# 엘보우 곡선 저장
clustering.save_elbow_curve(
    list(range(3, 21)),
    output_path="results/elbow_curve.png"
)
```

---

## 12. 참고문헌

**기초 문헌:**
1. **Dhillon, I. S., & Modha, D. S. (2001).** "Concept decompositions for large sparse text data using clustering." *Machine Learning*, 42(1), 143-175.
   - 텍스트 클러스터링을 위한 Spherical K-means 원형 정식화

2. **Hornik, K., Feinerer, I., Kober, M., & Buchta, C. (2012).** "Spherical k-means clustering." *Journal of Statistical Software*, 50(10), 1-22.
   - 종합적 분석 및 소프트웨어 구현

3. **Banerjee, A., et al. (2005).** "Clustering on the unit hypersphere using von Mises-Fisher distributions." *Journal of Machine Learning Research*, 6, 1345-1382.
   - Spherical clustering의 확률적 해석

**K-means++ 초기화:**
4. **Arthur, D., & Vassilvitskii, S. (2007).** "k-means++: The advantages of careful seeding." *SODA '07 Proceedings*, 1027-1035.

**의미연결망 관련 연구:**
5. **Lancichinetti, A., & Fortunato, S. (2009).** "Community detection algorithms: a comparative analysis." *Physical Review E*, 80(5), 056117.

**평가 메트릭:**
6. **Rousseeuw, P. J. (1987).** "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, 20, 53-65.

---

## 13. 부록: 구현 파일

**핵심 구현:**
- [core/services/clustering/SphericalKMeansClusteringService.py](core/services/clustering/SphericalKMeansClusteringService.py)
  - 주요 알고리즘 구현 (251줄)
  - 메서드: `fit_predict`, `auto_clustering`, `_spherical_kmeans`, `_kmeans_plusplus_init`

**추상 인터페이스:**
- [core/services/clustering/ClusteringInterface.py](core/services/clustering/ClusteringInterface.py)
  - 클러스터링 API를 정의하는 베이스 클래스 (152줄)
  - 유틸리티 메서드: `get_cluster_distribution`, `save_elbow_curve`, `_find_elbow_point`

**파이프라인 통합:**
- [core/services/GRACE/GRACEPipeline.py](core/services/GRACE/GRACEPipeline.py)
  - 58-59줄: 서비스 초기화
  - 313-356줄: 파이프라인의 클러스터링 단계
  - 361-396줄: 평가 및 결과 저장

**총 구현량:** ~450줄의 프로덕션 코드 + 포괄적 문서화

---

## 문서 메타데이터

**작성자:** SENTIMENT Lab
**작성일:** 2025-10-22
**버전:** 1.0
**상태:** 논문 출판용 기술 명세서
**라이선스:** 연구 목적 사용

**권장 인용 형식:**
```
[저자명] (2025). 의미연결망 분석을 위한 Spherical K-means 클러스터링.
기술 보고서, SENTIMENT Lab. [프로젝트 저장소 URL]
```
