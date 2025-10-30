# Gap Statistics with min_clusters 제약

## 문제 상황

Gap Statistics를 기본 클러스터링 방법으로 적용했을 때 **k=1을 탐지하는 문제**가 발생했습니다.

```
Gap Statistics 탐지: k=1
최적 클러스터 수: 1
클러스터별 단어 수: {0: 500}
```

이는 명백히 잘못된 결과입니다.

## 원인 분석

Gap Statistics가 k=1을 선택한 이유:
1. **실제 데이터의 특성**: 임베딩이 매우 균일하게 분포
2. **코사인 거리 특성**: 정규화된 벡터에서 모든 점이 유사하게 보임
3. **통계적 기준**: Gap 값이 k=1에서 최대가 됨

디버깅 결과:
- Gap Statistics: k=1 탐지
- Elbow Method: k=4 탐지
- k=5로 직접 클러스터링: Silhouette score 0.54 (양호)

## 해결 방법

**min_clusters 제약**을 추가하여 k=1 문제를 방지합니다.

### 수정 사항

#### 1. SphericalKMeansClusteringService.py

```python
def _auto_clustering_with_gap(
    self,
    embeddings: torch.Tensor,
    min_clusters: int = 3,  # 추가!
    max_clusters: int = 10,
    n_init: int = 3,
    n_iterations: int = 10
):
    # Gap Statistics 실행
    self.best_k, gap_params = gs.fit_predict(
        K=max_clusters,
        X=embeddings_normalized,
        n_iterations=n_iterations
    )

    # 강건성을 위한 제약: min_clusters 미만이면 강제 설정
    if self.best_k < min_clusters:
        print(f"Warning: Gap Statistics가 k={self.best_k}를 제안했지만, ")
        print(f"         min_clusters={min_clusters} 제약으로 인해 k={min_clusters}로 설정합니다.")
        self.best_k = min_clusters

    # 클러스터링 수행
    self.cluster_labels = self.fit_predict(embeddings, self.best_k, n_init)
    return self.cluster_labels, self.best_k, [], []
```

#### 2. GRACEPipeline.py

```python
# Gap Statistics로 최적 클러스터 수 탐색 (min_clusters 제약 적용)
self._log(f"  Gap Statistics로 최적 클러스터 수 탐색 중 ({self.config.min_clusters}-{self.config.max_clusters})...")

self.cluster_labels, best_k, inertias, silhouette_scores = \
    self.clustering_service.auto_clustering(
        self.graphmae_embeddings,
        min_clusters=self.config.min_clusters,  # 전달!
        max_clusters=self.config.max_clusters
    )

self._log(f"  Gap Statistics 탐지: k={best_k}")
self._log(f"  최적 클러스터 수: {best_k} (min_clusters={self.config.min_clusters} 제약 적용)")
```

#### 3. main.py

```python
# 클러스터링
clustering_method='kmeans',
num_clusters=None,  # Gap Statistics로 자동 결정 (min_clusters 제약 적용)
min_clusters=3,  # 강건성을 위한 최소값 제약
max_clusters=10,
```

## 테스트 결과

### 합성 데이터 (centers=5)
```
Gap Statistics (min_clusters=3)
  탐지된 k: 5
  ✓ 제약 준수: k=5 >= 3
  클러스터 분포: {0: 100, 1: 100, 2: 100, 3: 100, 4: 100}
```

### 실제 임베딩 데이터
```
Gap Statistics (min_clusters=3)
  Warning: Gap Statistics가 k=1를 제안했지만,
           min_clusters=3 제약으로 인해 k=3로 설정합니다.
  탐지된 k: 3
  ✓ 제약 준수: k=3 >= 3 (이전 k=1 문제 해결!)
  클러스터 분포: {0: 239, 1: 127, 2: 134}
  Silhouette score: 0.5027
```

## 작동 방식

```
1. Gap Statistics 실행
   ↓
2. 최적 k 탐지 (예: k=1)
   ↓
3. 제약 확인: k < min_clusters?
   ↓ (Yes)
4. k = min_clusters로 강제 설정
   ↓
5. 경고 메시지 출력
   ↓
6. 클러스터링 수행 (k=3)
```

## 장점

1. **강건성(Robustness)**: k=1 같은 부적절한 결과 방지
2. **유연성**: Gap Statistics의 통계적 장점 유지
3. **안전성**: 최소 클러스터 수 보장
4. **투명성**: 제약 적용 시 경고 메시지 출력

## 단점

1. **Gap Statistics 무시 가능**: 원래 제안(k=1)을 강제로 변경
2. **최적값 아닐 수 있음**: 통계적 기준과 다른 결과

## 대안 비교

| 방법 | k=1 방지 | 통계적 근거 | 실행 시간 |
|------|----------|-------------|-----------|
| **Gap + 제약 (채택)** | ✅ | ⭐⭐⭐⭐ | 10-20초 |
| Elbow Method | ✅ | ⭐⭐⭐ | 0.1초 |
| Gap (제약 없음) | ❌ | ⭐⭐⭐⭐⭐ | 10-20초 |
| 고정 k | ✅ | ❌ | 즉시 |

## 권장 설정

### 기본 설정 (현재)
```python
min_clusters = 3
max_clusters = 10
method = 'gap'
```

### 빠른 프로토타이핑
```python
min_clusters = 3
max_clusters = 10
method = 'elbow'
```

### 정밀 분석
```python
min_clusters = 3
max_clusters = 15
n_iterations = 20  # 더 많은 bootstrap
```

## 사용 방법

### main.py 실행 (자동 적용)
```bash
# 그냥 실행 - Gap Statistics + min_clusters=3 자동 적용!
python main.py --mode train

# 다른 min_clusters는 코드 수정 필요
# main.py의 create_default_config()에서
min_clusters=5  # 변경
```

### 직접 사용
```python
from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService

clustering = SphericalKMeansClusteringService(random_state=42)

labels, best_k, _, _ = clustering.auto_clustering(
    embeddings,
    min_clusters=3,  # k < 3 방지
    max_clusters=10,
    method='gap'
)

print(f"탐지된 k: {best_k}")  # 최소 3 이상 보장
```

## 경고 메시지 해석

```
Warning: Gap Statistics가 k=1를 제안했지만,
         min_clusters=3 제약으로 인해 k=3로 설정합니다.
```

**의미**:
- Gap Statistics는 통계적으로 k=1이 최적이라고 판단
- 하지만 실용적으로 k=1은 부적절하므로 k=3으로 강제 설정
- 데이터가 매우 균일하거나 클러스터 구조가 약함을 시사

**대응**:
1. **무시**: 대부분의 경우 k=3으로 충분
2. **데이터 확인**: 임베딩 품질이 낮을 수 있음
3. **Elbow Method 시도**: 다른 관점에서 확인

## 검증

테스트 스크립트:
```bash
# 제약 작동 확인
python test_gap_with_constraint.py

# 실제 파이프라인 테스트
python main.py --mode train --max-docs 1000
```

예상 출력:
```
Gap Statistics로 최적 클러스터 수 탐색 중 (3-10)...
Gap Statistics 탐지: k=3
최적 클러스터 수: 3 (min_clusters=3 제약 적용)
클러스터별 단어 수: {0: 200, 1: 150, 2: 150}
```

## 요약

✅ **Gap Statistics + min_clusters 제약 = 강건한 클러스터링**

- Gap Statistics의 통계적 장점 유지
- k=1 같은 부적절한 결과 방지
- 최소 클러스터 수 보장
- main.py 실행 시 자동 적용

**강건성(Robustness)**이 필요한 경우 최적의 선택입니다!

---

**작성 일자**: 2025-10-30
**테스트 완료**: ✅
**상태**: Production Ready
