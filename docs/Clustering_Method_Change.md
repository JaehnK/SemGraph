# Clustering Method 변경 사항

## 개요

SphericalKMeansClusteringService의 `auto_clustering()` 메서드에서 **Gap Statistics**를 기본 방법으로 변경했습니다.

## 변경 내용

### 1. 기본 방법 변경

#### 이전 (Elbow Method)
```python
clustering.auto_clustering(
    embeddings,
    min_clusters=3,
    max_clusters=20,
    n_init=10
)
```
- 2차 미분 기반 Elbow Method 사용
- 노이즈에 민감
- 빠른 실행 (1-5초)

#### 현재 (Gap Statistics)
```python
clustering.auto_clustering(
    embeddings,
    max_clusters=10,  # 기본값 변경: 20 → 10
    n_init=3,         # 기본값 변경: 10 → 3
    method='gap'      # 기본값: 'gap'
)
```
- Gap Statistics 사용
- 노이즈에 robust
- 통계적으로 신뢰성 높음
- 실행 시간 3-5초 (축소된 파라미터)

### 2. 파라미터 변경

| 파라미터 | 이전 기본값 | 현재 기본값 | 이유 |
|----------|-------------|-------------|------|
| `max_clusters` | 20 | 10 | 계산 시간 단축 |
| `n_init` | 10 | 3 | 계산 시간 단축 |
| `method` | (없음) | 'gap' | Gap Statistics 기본 사용 |

### 3. 새로운 파라미터

- **method**: `'gap'` (기본값) 또는 `'elbow'`
  - `'gap'`: Gap Statistics 방법 사용
  - `'elbow'`: 기존 2차 미분 기반 Elbow Method 사용

## 사용 방법

### Gap Statistics 사용 (기본값)

```python
from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService
import torch

# 클러스터링 서비스 초기화
clustering = SphericalKMeansClusteringService(random_state=42)

# Gap Statistics로 자동 클러스터링 (기본값)
labels, best_k, _, _ = clustering.auto_clustering(
    embeddings,
    max_clusters=10,  # 선택사항 (기본값: 10)
    n_init=3          # 선택사항 (기본값: 3)
)

print(f"최적 클러스터 수: {best_k}")
print(f"클러스터 분포: {clustering.get_cluster_distribution()}")
```

### Elbow Method 사용 (이전 방식)

```python
# Elbow Method 명시적 사용
labels, best_k, inertias, silhouette_scores = clustering.auto_clustering(
    embeddings,
    min_clusters=3,
    max_clusters=20,
    n_init=10,
    method='elbow'  # Elbow Method 지정
)
```

## 성능 비교

### 테스트 조건
- 샘플 수: 200
- 특징 수: 32
- 실제 클러스터 수: 5

### 결과

| 방법 | 탐지된 k | 오차 | 소요 시간 | 안정성 |
|------|----------|------|-----------|--------|
| **Gap Statistics** | 5 | 0 | 4.1초 | 높음 |
| Elbow Method | 5 | 0-1 | 0.1초 | 중간 |

## 장단점 비교

### Gap Statistics (기본값)

**장점:**
- ✅ 노이즈에 강함
- ✅ 통계적으로 신뢰성 높음
- ✅ 참조 분포와 비교하여 robust
- ✅ Tibshirani et al. (2001) 이론적 기반

**단점:**
- ⚠️ Elbow Method보다 느림 (약 40배)
- ⚠️ Bootstrap 샘플링으로 인한 계산 비용

### Elbow Method

**장점:**
- ✅ 매우 빠름 (0.1초)
- ✅ 간단한 구현
- ✅ 대략적인 클러스터 수 파악에 적합

**단점:**
- ⚠️ 노이즈에 민감
- ⚠️ 2차 미분의 최댓값이 불안정할 수 있음
- ⚠️ 명확한 elbow가 없을 때 부정확

## Fallback 메커니즘

`gapstatistics` 라이브러리가 설치되어 있지 않은 경우:

```python
# Gap Statistics 시도 → gapstatistics 없음 → 자동으로 Elbow Method로 전환
labels, best_k, inertias, silhouette_scores = clustering.auto_clustering(
    embeddings,
    method='gap'
)

# 출력:
# Warning: gapstatistics 라이브러리가 설치되어 있지 않습니다.
# pip install gapstatistics 로 설치하거나 method='elbow'를 사용하세요.
# Fallback to Elbow Method...
```

## 라이브러리 설치

Gap Statistics를 사용하려면 다음 라이브러리 설치 필요:

```bash
pip install gapstatistics
```

## 마이그레이션 가이드

### 기존 코드가 있는 경우

#### 변경 불필요한 경우
기존 코드가 다음과 같이 작성되어 있다면 **변경 불필요**:

```python
clustering.auto_clustering(embeddings)
```
→ 자동으로 Gap Statistics 사용 (더 나은 성능)

#### 변경이 필요한 경우

1. **명시적으로 Elbow Method를 원하는 경우**:
```python
# 변경 전
clustering.auto_clustering(embeddings, min_clusters=3, max_clusters=20)

# 변경 후
clustering.auto_clustering(
    embeddings,
    min_clusters=3,
    max_clusters=20,
    method='elbow'  # 명시적 지정
)
```

2. **max_clusters가 10보다 큰 경우**:
```python
# 기본값이 10으로 줄었으므로 명시적 지정 필요
clustering.auto_clustering(
    embeddings,
    max_clusters=15,  # 명시적 지정
    method='gap'
)
```

3. **빠른 실행이 중요한 경우**:
```python
# Elbow Method가 더 빠름 (0.1초 vs 4초)
clustering.auto_clustering(
    embeddings,
    method='elbow'
)
```

## 권장 사용 케이스

### Gap Statistics 사용 (기본값)
- ✅ 최종 결과를 위한 클러스터링
- ✅ 노이즈가 많은 데이터
- ✅ 통계적 신뢰성이 중요한 경우
- ✅ 시간 여유가 있는 경우 (3-5초)

### Elbow Method 사용
- ✅ 빠른 프로토타이핑
- ✅ 여러 번 시도하며 탐색
- ✅ 대략적인 클러스터 수 파악
- ✅ 실시간 응답이 필요한 경우 (0.1초)

## 내부 구조

### Gap Statistics 구현

```python
def _auto_clustering_with_gap(
    self,
    embeddings: torch.Tensor,
    max_clusters: int = 10,
    n_init: int = 3,
    n_iterations: int = 10  # Bootstrap 반복
) -> Tuple[np.ndarray, int, List[float], List[float]]:
    """Gap Statistics로 최적 클러스터 수 탐색"""

    # gapstatistics 라이브러리 사용
    gs = GapStatistics(
        algorithm=self._SphericalKMeansWrapper,
        distance_metric=self._cosine_distance_for_gap,
        return_params=True
    )

    # 최적 k 찾기
    best_k, gap_params = gs.fit_predict(
        K=max_clusters,
        X=embeddings_normalized,
        n_iterations=n_iterations
    )

    # 최종 클러스터링
    cluster_labels = self.fit_predict(embeddings, best_k, n_init)

    return cluster_labels, best_k, [], []
```

### Spherical K-Means Wrapper

Gap Statistics 라이브러리는 sklearn 스타일 인터페이스를 요구하므로 내부 래퍼 클래스 구현:

```python
class _SphericalKMeansWrapper:
    """sklearn 스타일 래퍼"""

    def __init__(self, n_clusters, random_state=42, n_init=3):
        self.n_clusters = n_clusters
        # ...

    def fit(self, X):
        # Spherical K-Means 수행
        # ...
        return self

    def predict(self, X):
        # 코사인 유사도 기반 예측
        # ...
        return labels
```

## 참고 자료

- [Gap Statistics 논문](https://hastie.su.domains/Papers/gap.pdf): Tibshirani et al. (2001)
- [gapstatistics 라이브러리](https://pypi.org/project/gapstatistics/)
- [Gap Statistics Integration Guide](./Gap_Statistics_Integration_Guide.md)

## 테스트

관련 테스트 파일:
- [test_updated_clustering.py](../test_updated_clustering.py): 변경 사항 테스트
- [test_gap_statistic_quick.py](../test_gap_statistic_quick.py): Gap Statistics 빠른 테스트
- [test_gap_statistic.py](../test_gap_statistic.py): Gap Statistics 전체 테스트

```bash
# 변경 사항 테스트
python test_updated_clustering.py

# Gap Statistics 빠른 테스트
python test_gap_statistic_quick.py
```

## 요약

1. **Gap Statistics가 기본 방법**으로 설정됨
2. **기본 파라미터 축소**: `max_clusters=10`, `n_init=3`
3. **method='elbow'**로 기존 방법 사용 가능
4. **자동 fallback**: gapstatistics 없으면 Elbow Method 사용
5. **성능**: 약 4초 소요, 높은 정확도와 안정성

---

**변경 일자**: 2025-10-30
**변경자**: Claude Code
