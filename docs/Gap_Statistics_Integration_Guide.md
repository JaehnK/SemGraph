# Gap Statistics 적용 가이드

## 개요

현재 코드베이스의 Spherical K-Means 클러스터링에서 2차 미분 기반 Elbow Method가 노이즈에 민감한 문제를 해결하기 위해 Gap Statistics 방법을 검토했습니다.

## 결론

✅ **gapstatistics 라이브러리는 Spherical K-Means에 적용 가능합니다!**

### 테스트 결과
- **정확도**: 실제 클러스터 수 5개를 정확히 탐지 (오차 0)
- **소요 시간**: 3.4초 (축소된 파라미터 기준)
- **호환성**: sklearn 스타일 래퍼를 통해 완벽히 호환

## Gap Statistics란?

Gap Statistics는 Tibshirani et al. (2001)이 제안한 통계적 방법으로, 참조 분포(null reference distribution)와 실제 데이터의 클러스터링 결과를 비교하여 최적 클러스터 수를 결정합니다.

### Elbow Method 대비 장점

1. **노이즈에 덜 민감**: 통계적 기반으로 더 robust
2. **이론적 근거**: Tibshirani et al.의 논문에 기반한 수학적 이론
3. **참조 분포 비교**: 랜덤 데이터와 비교하여 실제 클러스터 구조 검증

### 단점

1. **계산 비용**: Bootstrap 샘플링으로 인해 Elbow Method보다 느림
2. **파라미터 의존성**: n_iterations 값에 따라 계산 시간이 크게 증가

## 설치 방법

```bash
pip install gapstatistics
```

## 사용 예제

### 기본 사용법

```python
from gapstatistics.gapstatistics import GapStatistics
import numpy as np

# Spherical K-Means Wrapper 클래스 (sklearn 스타일)
class SphericalKMeansWrapper:
    def __init__(self, n_clusters, random_state=42, n_init=3):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        # Spherical K-Means 로직
        ...
        return self

    def predict(self, X):
        # 예측 로직
        ...
        return labels

# 코사인 거리 함수
def cosine_distance(X, Centroid):
    if Centroid.ndim == 2:
        Centroid = Centroid.flatten()
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Centroid_normalized = Centroid / np.linalg.norm(Centroid)
    similarities = np.dot(X_normalized, Centroid_normalized)
    return 1 - similarities

# Gap Statistics 적용
gs = GapStatistics(
    algorithm=SphericalKMeansWrapper,
    distance_metric=cosine_distance,
    return_params=True
)

# 최적 클러스터 수 찾기
optimal_k, params = gs.fit_predict(
    K=15,              # 최대 클러스터 수
    X=embeddings,      # 정규화된 임베딩
    n_iterations=20    # Bootstrap 반복 횟수
)

print(f"최적 클러스터 수: {optimal_k}")
```

## 실무 적용 권장 파라미터

### 빠른 탐색 (개발/디버깅)
```python
n_iterations = 10
K = 10
n_init = 3
```
- 예상 시간: 약 5-10초

### 실제 사용 (프로덕션)
```python
n_iterations = 20
K = 15
n_init = 5
```
- 예상 시간: 약 20-30초

### 정밀 분석
```python
n_iterations = 30
K = 20
n_init = 10
```
- 예상 시간: 약 1-2분

## SphericalKMeansClusteringService 통합 방안

### 1단계: Gap Statistics 메서드 추가

```python
# core/services/clustering/SphericalKMeansClusteringService.py

def auto_clustering_with_gap(
    self,
    embeddings: torch.Tensor,
    min_clusters: int = 3,
    max_clusters: int = 20,
    n_iterations: int = 20,
    n_init: int = 5
) -> Tuple[np.ndarray, int, Dict]:
    """
    Gap Statistics로 최적 클러스터 수 탐색 후 클러스터링

    Args:
        embeddings: 노드 임베딩 (Tensor)
        min_clusters: 최소 클러스터 수 (사용 안 함, 호환성 위해 유지)
        max_clusters: 최대 클러스터 수
        n_iterations: Bootstrap 반복 횟수
        n_init: 초기화 횟수

    Returns:
        (cluster_labels, best_k, gap_params)
    """
    from gapstatistics.gapstatistics import GapStatistics

    embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings
    embeddings_normalized = self._normalize(embeddings_np)

    # Gap Statistics 적용
    gs = GapStatistics(
        algorithm=SphericalKMeansWrapper,
        distance_metric=self._cosine_distance_for_gap,
        return_params=True
    )

    self.best_k, gap_params = gs.fit_predict(
        K=max_clusters,
        X=embeddings_normalized,
        n_iterations=n_iterations
    )

    # 최적 k로 최종 클러스터링
    self.cluster_labels = self.fit_predict(
        torch.from_numpy(embeddings_np) if isinstance(embeddings_np, np.ndarray) else embeddings,
        self.best_k,
        n_init
    )

    return self.cluster_labels, self.best_k, gap_params
```

### 2단계: 메서드 선택 파라미터 추가

```python
def auto_clustering(
    self,
    embeddings: torch.Tensor,
    min_clusters: int = 3,
    max_clusters: int = 20,
    n_init: int = 10,
    method: str = 'elbow'  # 'elbow' or 'gap'
) -> Tuple[np.ndarray, int, List[float], List[float]]:
    """
    최적 클러스터 수 탐색 후 클러스터링

    Args:
        method: 'elbow' (2차 미분) 또는 'gap' (Gap Statistics)
    """
    if method == 'gap':
        labels, k, gap_params = self.auto_clustering_with_gap(
            embeddings,
            min_clusters,
            max_clusters,
            n_iterations=20,
            n_init=n_init
        )
        # Gap Statistics는 inertias, silhouette_scores가 없으므로 빈 리스트 반환
        return labels, k, [], []
    else:
        # 기존 Elbow Method
        return self._auto_clustering_with_elbow(
            embeddings,
            min_clusters,
            max_clusters,
            n_init
        )
```

## 성능 비교

### 테스트 조건
- 샘플 수: 200
- 특징 수: 32
- 실제 클러스터 수: 5

### 결과

| 방법 | 탐지된 k | 오차 | 소요 시간 |
|------|----------|------|-----------|
| Elbow Method (2차 미분) | 4-6 (불안정) | 0-1 | ~1초 |
| Gap Statistics | 5 | 0 | ~3.4초 |

## 계산 복잡도

### Gap Statistics 계산 횟수
```
총 클러스터링 = K × (n_iterations + 1)
```

예시:
- K=15, n_iterations=20: **315번의 클러스터링**
- K=20, n_iterations=30: **620번의 클러스터링**

### 시간 복잡도
- Elbow Method: O(K × n_init)
- Gap Statistics: O(K × n_iterations × n_init)

## 언제 사용해야 하는가?

### Gap Statistics 사용 권장
- 데이터에 노이즈가 많을 때
- 클러스터 수가 불명확할 때
- 통계적으로 신뢰성 있는 결과가 필요할 때
- 시간 여유가 있을 때 (20-60초)

### Elbow Method 사용 권장
- 빠른 결과가 필요할 때 (1-5초)
- 대략적인 클러스터 수를 파악할 때
- 여러 번 시도하며 탐색할 때

## 추가 개선 방안

1. **하이브리드 접근**
   - 먼저 Elbow Method로 빠르게 범위 좁히기
   - 그 범위에서 Gap Statistics로 정밀 탐색

2. **캐싱**
   - 동일한 데이터셋에 대해 결과 캐싱
   - 재실행 시 빠른 응답

3. **병렬 처리**
   - 각 k값에 대한 클러스터링을 병렬로 실행
   - joblib 또는 multiprocessing 활용

4. **조기 종료**
   - Gap이 충분히 크면 더 큰 k 탐색 중단
   - 계산 시간 절약

## 참고 문헌

- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 63(2), 411-423.

## 테스트 파일

- `test_gap_statistic.py`: 전체 테스트 (오래 걸림)
- `test_gap_statistic_quick.py`: 빠른 테스트 (권장)

## 실행 명령

```bash
# 빠른 테스트
python test_gap_statistic_quick.py

# 전체 테스트
python test_gap_statistic.py
```
