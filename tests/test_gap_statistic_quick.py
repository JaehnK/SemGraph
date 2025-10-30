#!/usr/bin/env python3
"""
Gap Statistics 빠른 테스트 버전 (파라미터 축소)
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService


class SphericalKMeansWrapper:
    """Spherical K-Means Wrapper for gapstatistics"""

    def __init__(self, n_clusters, random_state=42, n_init=3):  # n_init 축소
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self._service = None

    def fit(self, X):
        import torch
        self._service = SphericalKMeansClusteringService(random_state=self.random_state)
        X_tensor = torch.from_numpy(X).float()
        self.labels_ = self._service.fit_predict(X_tensor, n_clusters=self.n_clusters, n_init=self.n_init)

        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_mask = (self.labels_ == k)
            if np.sum(cluster_mask) > 0:
                cluster_mean = np.mean(X_normalized[cluster_mask], axis=0)
                self.cluster_centers_[k] = cluster_mean / np.linalg.norm(cluster_mean)

        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        similarities = np.dot(X_normalized, self.cluster_centers_.T)
        return np.argmax(similarities, axis=1)


def cosine_distance(X: np.ndarray, Centroid: np.ndarray) -> np.ndarray:
    """코사인 거리 계산"""
    if Centroid.ndim == 2:
        Centroid = Centroid.flatten()
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Centroid_normalized = Centroid / np.linalg.norm(Centroid)
    similarities = np.dot(X_normalized, Centroid_normalized)
    distances = 1 - similarities
    return distances


def quick_test():
    """빠른 테스트"""
    try:
        from gapstatistics.gapstatistics import GapStatistics
    except ImportError:
        print("✗ gapstatistics 라이브러리를 먼저 설치해주세요.")
        return

    print("=" * 70)
    print("Gap Statistics 빠른 테스트 (축소된 파라미터)")
    print("=" * 70)

    # 작은 데이터셋
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(
        n_samples=200,  # 500 -> 200
        n_features=32,  # 64 -> 32
        centers=5,      # 10 -> 5
        cluster_std=1.5,
        random_state=42
    )
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    print(f"\n데이터 정보:")
    print(f"  샘플 수: {X_normalized.shape[0]}")
    print(f"  특징 수: {X_normalized.shape[1]}")
    print(f"  실제 클러스터 수: 5")

    print(f"\nGap Statistic 계산 중...")
    print(f"  클러스터 수 범위: 2 ~ 10 (축소)")
    print(f"  반복 횟수: 10 (축소)")

    import time
    start_time = time.time()

    try:
        gs = GapStatistics(
            algorithm=SphericalKMeansWrapper,
            distance_metric=cosine_distance,
            return_params=True
        )

        # 축소된 파라미터
        n_clusters, params = gs.fit_predict(
            K=10,              # 20 -> 10
            X=X_normalized,
            n_iterations=10    # 30 -> 10
        )

        elapsed = time.time() - start_time

        print(f"\n결과:")
        print(f"  소요 시간: {elapsed:.1f}초")
        print(f"  Gap Statistic이 제안한 최적 클러스터 수: {n_clusters}")
        print(f"  실제 클러스터 수: 5")
        print(f"  차이: {abs(n_clusters - 5)}")

        # Gap 값 출력
        if 'gap' in params:
            print(f"\nGap 값:")
            for k, gap_val in enumerate(params['gap'], start=1):
                print(f"  k={k}: {gap_val:.4f}")

        return True

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def estimate_full_test_time():
    """전체 테스트 예상 시간 계산"""
    print("\n" + "=" * 70)
    print("전체 테스트 예상 시간")
    print("=" * 70)

    # 한 번의 클러스터링 예상 시간
    import time
    import torch
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=500, n_features=64, centers=10, random_state=42)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_tensor = torch.from_numpy(X_normalized).float()

    clustering = SphericalKMeansClusteringService(random_state=42)

    start = time.time()
    clustering.fit_predict(X_tensor, n_clusters=10, n_init=3)
    single_time = time.time() - start

    print(f"단일 클러스터링 시간 (n_init=3): {single_time:.2f}초")

    # 전체 예상 시간
    k_range = 20  # 1~20
    n_iterations = 30  # bootstrap
    total_clusterings = k_range * (n_iterations + 1)  # +1 for actual data

    estimated_total = single_time * total_clusterings

    print(f"\n예상 계산:")
    print(f"  k 범위: 1-{k_range}")
    print(f"  Bootstrap 반복: {n_iterations}")
    print(f"  총 클러스터링 횟수: {total_clusterings}")
    print(f"  예상 소요 시간: {estimated_total/60:.1f}분 ({estimated_total:.0f}초)")

    print(f"\n권장 사항:")
    print(f"  - 빠른 테스트: n_iterations=10, K=10 → 약 {estimated_total/60/3:.1f}분")
    print(f"  - 실제 사용: n_iterations=20, K=15 → 약 {estimated_total/60*0.6:.1f}분")


if __name__ == "__main__":
    # 빠른 테스트 실행
    success = quick_test()

    if success:
        print("\n✓ Gap Statistics가 Spherical K-Means에 성공적으로 적용되었습니다!")

        # 시간 예측
        estimate_full_test_time()

        print("\n" + "=" * 70)
        print("결론")
        print("=" * 70)
        print("✓ gapstatistics 라이브러리는 코드베이스에 적용 가능합니다.")
        print("✓ 다만 계산 비용이 크므로 파라미터 조정이 필요합니다.")
        print()
        print("실무 적용 시 권장 파라미터:")
        print("  - n_iterations: 10-20 (기본 30은 너무 많음)")
        print("  - K (최대 클러스터): 10-15")
        print("  - n_init (Spherical K-Means): 3-5")
        print("=" * 70)
