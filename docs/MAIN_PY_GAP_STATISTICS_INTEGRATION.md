# main.py Gap Statistics 통합 완료

## 요약

✅ **main.py를 실행하면 Gap Statistics가 자동으로 적용됩니다!**

## 변경 사항

### 1. SphericalKMeansClusteringService.py
- `auto_clustering()` 메서드의 기본 방법을 Gap Statistics로 변경
- 기본 파라미터 최적화:
  - `method='gap'` (기본값)
  - `max_clusters=10` (20 → 10)
  - `n_init=3` (10 → 3)
  - `n_iterations=10` (내부, Gap Statistics용)

### 2. GRACEPipeline.py
- `auto_clustering()` 호출 시 불필요한 파라미터 제거
- 로그 메시지 수정: "Elbow Method" → "Gap Statistics"
- Gap Statistics는 elbow curve를 생성하지 않으므로 시각화 비활성화

### 3. main.py
- `max_clusters` 기본값: 20 → 10
- 주석 업데이트: "Elbow method" → "Gap Statistics"

## 사용 방법

### 기본 실행 (Gap Statistics 자동 적용)

```bash
# 기본 실행 - Gap Statistics 자동 사용
python main.py --mode train

# 문서 수 제한
python main.py --mode train --max-docs 1000

# 출력 디렉토리 지정
python main.py --mode train --output results/my_experiment

# GPU 사용
python main.py --mode train --device cuda
```

### 설정 확인

main.py 실행 시 다음과 같은 로그를 확인할 수 있습니다:

```
Gap Statistics로 최적 클러스터 수 탐색 중 (max: 10)...
Gap Statistics 탐지: k=5
최적 클러스터 수: 5
```

## 실행 흐름

```
main.py
  ↓
create_default_config()
  ↓ max_clusters=10 (기본값)
  ↓
GRACEPipeline(config)
  ↓
pipeline.run()
  ↓
clustering_service.auto_clustering(
    embeddings,
    max_clusters=10
)  # method='gap' (기본값), n_init=3 (기본값)
  ↓
_auto_clustering_with_gap()
  ↓
GapStatistics.fit_predict(
    K=10,
    n_iterations=10
)
  ↓
최적 클러스터 수 반환
```

## 성능 예상

### 소규모 데이터 (1,000 문서)
- 단어 수: ~500
- Gap Statistics 시간: **3-5초**
- 전체 파이프라인 시간: ~30초

### 중규모 데이터 (10,000 문서)
- 단어 수: ~500
- Gap Statistics 시간: **3-5초** (단어 수에 의존)
- 전체 파이프라인 시간: ~2분

### 대규모 데이터 (100,000 문서)
- 단어 수: ~500
- Gap Statistics 시간: **3-5초**
- 전체 파이프라인 시간: ~10분

**주의**: Gap Statistics 시간은 주로 `단어 수(top_n_words)`와 `max_clusters`에 의존하며, 문서 수와는 직접적인 관련이 적습니다.

## 파라미터 조정

### 더 빠른 실행 (프로토타이핑)

Elbow Method로 변경:
```python
# GRACEPipeline.py에서 수동 수정 필요
clustering_service.auto_clustering(
    embeddings,
    max_clusters=10,
    method='elbow'  # 명시적 지정
)
```

또는 설정에서 클러스터 수를 직접 지정:
```python
config.num_clusters = 5  # 자동 탐색 비활성화
```

### 더 정밀한 탐색

`max_clusters`를 증가:
```bash
# main.py에서 직접 수정 필요
# main.py의 create_default_config()에서
max_clusters=15  # 10 → 15
```

## Fallback 메커니즘

`gapstatistics` 라이브러리가 설치되어 있지 않으면 자동으로 Elbow Method로 전환됩니다:

```
Warning: gapstatistics 라이브러리가 설치되어 있지 않습니다.
pip install gapstatistics 로 설치하거나 method='elbow'를 사용하세요.
Fallback to Elbow Method...
```

## 라이브러리 설치 확인

```bash
# gapstatistics 설치 확인
python -c "import gapstatistics; print('설치됨')"

# 없으면 설치
pip install gapstatistics
```

## 이전 버전으로 돌아가기

Elbow Method를 기본값으로 사용하고 싶다면:

### SphericalKMeansClusteringService.py 수정

```python
def auto_clustering(
    self,
    embeddings: torch.Tensor,
    min_clusters: int = 3,
    max_clusters: int = 20,  # 10 → 20
    n_init: int = 10,        # 3 → 10
    method: str = 'elbow'    # 'gap' → 'elbow'
):
```

### GRACEPipeline.py 수정

```python
self.cluster_labels, best_k, inertias, silhouette_scores = \
    self.clustering_service.auto_clustering(
        self.graphmae_embeddings,
        min_clusters=self.config.min_clusters,  # 추가
        max_clusters=self.config.max_clusters,
        n_init=10,                                # 추가
        method='elbow'                             # 추가
    )
```

## 검증 테스트

### 단위 테스트

```bash
# Gap Statistics 기능 테스트
python test_gap_statistic_quick.py

# SphericalKMeansClusteringService 테스트
python test_updated_clustering.py

# 통합 테스트 (GRACEPipeline)
python test_main_integration.py
```

### 실제 실행 테스트

```bash
# 작은 데이터로 빠른 테스트
python main.py --mode train --max-docs 100 --epochs 10 --output results/test_gap

# 실행 로그에서 다음을 확인:
# "Gap Statistics로 최적 클러스터 수 탐색 중..."
```

## 문제 해결

### Gap Statistics가 너무 느린 경우

1. `max_clusters` 줄이기 (main.py):
```python
max_clusters=5  # 10 → 5
```

2. `top_n_words` 줄이기 (main.py):
```python
top_n_words=300  # 500 → 300
```

3. Elbow Method로 변경 (GRACEPipeline.py):
```python
method='elbow'
```

### gapstatistics 설치 오류

```bash
# 현재 환경 확인
conda list | grep gapstatistics

# 재설치
pip uninstall gapstatistics
pip install gapstatistics
```

### 결과가 이상한 경우

1. 랜덤 시드 고정 확인:
```bash
python main.py --random-seed 42
```

2. 로그 확인:
```bash
python main.py --verbose
```

3. Elbow Method와 비교:
```python
# GRACEPipeline.py에서 method='elbow'로 실행 후 비교
```

## 관련 문서

- [Gap_Statistics_Integration_Guide.md](./Gap_Statistics_Integration_Guide.md): Gap Statistics 상세 가이드
- [Clustering_Method_Change.md](./Clustering_Method_Change.md): 변경 사항 상세 문서

## 커밋 메시지 (참고)

```
feat: Apply Gap Statistics as default clustering method

- Change SphericalKMeansClusteringService default to 'gap'
- Optimize default parameters: max_clusters=10, n_init=3
- Update GRACEPipeline to use Gap Statistics automatically
- Add fallback to Elbow Method when gapstatistics not installed
- Update main.py configuration defaults
```

## 요약

| 항목 | 값 |
|------|------|
| **기본 방법** | Gap Statistics |
| **max_clusters** | 10 |
| **n_init** | 3 |
| **n_iterations** | 10 |
| **예상 시간** | 3-5초 (클러스터링 단계) |
| **Fallback** | Elbow Method (라이브러리 없을 시) |
| **변경 필요 여부** | ❌ 없음 (자동 적용) |

---

**최종 업데이트**: 2025-10-30
**작성자**: Claude Code
