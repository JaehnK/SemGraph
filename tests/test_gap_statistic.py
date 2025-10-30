#!/usr/bin/env python3
"""
Gap Statistics를 Spherical K-Means에 적용 가능한지 테스트
"""

import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService


def test_gap_stat_installation():
    """gapstatistics 라이브러리 설치 여부 확인"""
    try:
        import gapstatistics
        print("✓ gapstatistics 라이브러리가 설치되어 있습니다.")
        print(f"  버전: {gapstatistics.__version__ if hasattr(gapstatistics, '__version__') else 'N/A'}")
        return True
    except ImportError:
        print("✗ gapstatistics 라이브러리가 설치되어 있지 않습니다.")
        print("  설치 명령: pip install gapstatistics")
        return False


class SphericalKMeansWrapper:
    """
    gapstatistics 라이브러리와 호환되도록
    Spherical K-Means를 래핑한 클래스

    gapstatistics는 sklearn 스타일의 인터페이스를 요구합니다:
    - __init__(n_clusters)
    - fit(X) 메서드
    - predict(X) 메서드
    - cluster_centers_ 속성
    """

    def __init__(self, n_clusters, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_centers_ = None
        self.labels_ = None
        self._service = None

    def fit(self, X):
        """클러스터링 학습"""
        import torch

        self._service = SphericalKMeansClusteringService(random_state=self.random_state)
        X_tensor = torch.from_numpy(X).float()

        # 클러스터링 수행
        self.labels_ = self._service.fit_predict(X_tensor, n_clusters=self.n_clusters, n_init=self.n_init)

        # 클러스터 중심 계산 (정규화된 평균)
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_mask = (self.labels_ == k)
            if np.sum(cluster_mask) > 0:
                cluster_mean = np.mean(X_normalized[cluster_mask], axis=0)
                # 중심점 정규화
                self.cluster_centers_[k] = cluster_mean / np.linalg.norm(cluster_mean)

        return self

    def predict(self, X):
        """새로운 데이터에 대한 클러스터 예측"""
        if self.cluster_centers_ is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")

        # 정규화
        X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

        # 코사인 유사도 기반 예측
        similarities = np.dot(X_normalized, self.cluster_centers_.T)
        return np.argmax(similarities, axis=1)


def cosine_distance(X: np.ndarray, Centroid: np.ndarray) -> np.ndarray:
    """
    코사인 거리 계산 함수
    gapstatistics의 distance_metric으로 사용

    Args:
        X: 데이터 포인트들 (n_samples, n_features)
        Centroid: 중심점 (1, n_features) 또는 (n_features,)

    Returns:
        코사인 거리 배열 (n_samples,)
    """
    # Centroid 차원 조정
    if Centroid.ndim == 2:
        Centroid = Centroid.flatten()

    # 정규화
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Centroid_normalized = Centroid / np.linalg.norm(Centroid)

    # 코사인 유사도
    similarities = np.dot(X_normalized, Centroid_normalized)

    # 코사인 거리 = 1 - 코사인 유사도
    distances = 1 - similarities

    return distances


def test_gap_statistic_with_spherical_kmeans():
    """Gap Statistics를 Spherical K-Means에 적용"""

    try:
        from gapstatistics.gapstatistics import GapStatistics
    except ImportError:
        print("\n✗ gapstatistics 라이브러리를 먼저 설치해주세요.")
        return

    print("\n" + "=" * 70)
    print("Gap Statistics with Spherical K-Means 테스트")
    print("=" * 70)

    # 테스트 데이터 생성 (실제 임베딩과 유사한 형태)
    np.random.seed(42)

    # 10개의 클러스터를 가진 500개 샘플 생성
    from sklearn.datasets import make_blobs
    X, true_labels = make_blobs(
        n_samples=500,
        n_features=64,
        centers=10,
        cluster_std=1.5,
        random_state=42
    )

    # L2 정규화 (Spherical K-Means를 위해)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    print(f"\n데이터 정보:")
    print(f"  샘플 수: {X_normalized.shape[0]}")
    print(f"  특징 수: {X_normalized.shape[1]}")
    print(f"  실제 클러스터 수: 10")

    # Gap Statistic 적용
    print(f"\nGap Statistic 계산 중...")
    print(f"  클러스터 수 범위: 2 ~ 20")
    print(f"  반복 횟수: 30")

    try:
        gs = GapStatistics(
            algorithm=SphericalKMeansWrapper,
            distance_metric=cosine_distance,
            return_params=True
        )

        # 최적 클러스터 수 찾기
        n_clusters, params = gs.fit_predict(K=20, X=X_normalized, n_iterations=30)

        print(f"\n결과:")
        print(f"  Gap Statistic이 제안한 최적 클러스터 수: {n_clusters}")
        print(f"  실제 클러스터 수: 10")
        print(f"  차이: {abs(n_clusters - 10)}")

        # 시각화
        try:
            gs.plot()
            print(f"\n그래프가 표시되었습니다.")

        except Exception as e:
            print(f"\n경고: 시각화 중 오류 발생: {e}")

        return True, n_clusters, params

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def compare_elbow_vs_gap():
    """기존 Elbow Method와 Gap Statistic 비교"""

    try:
        from gapstatistics.gapstatistics import GapStatistics
    except ImportError:
        print("\n✗ gapstatistics 라이브러리를 먼저 설치해주세요.")
        return

    print("\n" + "=" * 70)
    print("Elbow Method vs Gap Statistic 비교")
    print("=" * 70)

    # 테스트 데이터 생성
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, n_features=64, centers=10, random_state=42)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)

    import torch
    X_tensor = torch.from_numpy(X_normalized).float()

    # 1. 기존 Elbow Method
    print("\n1. Elbow Method (2차 미분) 실행...")
    clustering_service = SphericalKMeansClusteringService(random_state=42)
    _, best_k_elbow, inertias, silhouette_scores = clustering_service.auto_clustering(
        X_tensor,
        min_clusters=3,
        max_clusters=20,
        n_init=10
    )
    print(f"   최적 클러스터 수: {best_k_elbow}")

    # 2. Gap Statistic
    print("\n2. Gap Statistic 실행...")
    gs = GapStatistics(
        algorithm=SphericalKMeansWrapper,
        distance_metric=cosine_distance,
        return_params=True
    )
    best_k_gap, params = gs.fit_predict(K=20, X=X_normalized, n_iterations=30)
    print(f"   최적 클러스터 수: {best_k_gap}")

    # 비교 결과
    print("\n" + "=" * 70)
    print("비교 결과:")
    print("=" * 70)
    print(f"  실제 클러스터 수: 10")
    print(f"  Elbow Method: {best_k_elbow} (오차: {abs(best_k_elbow - 10)})")
    print(f"  Gap Statistic: {best_k_gap} (오차: {abs(best_k_gap - 10)})")
    print()

    # 시각화
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Elbow Method
        ax1 = axes[0]
        k_values = list(range(3, 21))
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(x=best_k_elbow, color='r', linestyle='--', linewidth=2,
                   label=f'Elbow Point (k={best_k_elbow})')
        ax1.axvline(x=10, color='g', linestyle='--', linewidth=2, alpha=0.5,
                   label='True k=10')
        ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax1.set_ylabel('Inertia', fontsize=12)
        ax1.set_title('Elbow Method (2nd Derivative)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gap Statistic
        ax2 = axes[1]
        gap_values = params['gap']
        k_values_gap = list(range(1, len(gap_values) + 1))
        ax2.plot(k_values_gap, gap_values, 'mo-', linewidth=2, markersize=8)
        ax2.axvline(x=best_k_gap, color='r', linestyle='--', linewidth=2,
                   label=f'Optimal k={best_k_gap}')
        ax2.axvline(x=10, color='g', linestyle='--', linewidth=2, alpha=0.5,
                   label='True k=10')
        ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax2.set_ylabel('Gap Statistic', fontsize=12)
        ax2.set_title('Gap Statistic Method', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('elbow_vs_gap_comparison.png', dpi=300)
        print(f"그래프 저장: elbow_vs_gap_comparison.png")
        plt.close()

    except Exception as e:
        print(f"경고: 시각화 중 오류 발생: {e}")


def main():
    """메인 함수"""
    print("=" * 70)
    print("Gap Statistics 적용 가능성 테스트")
    print("=" * 70)

    # 1. 라이브러리 설치 확인
    if not test_gap_stat_installation():
        print("\n설치 후 다시 실행해주세요.")
        return

    # 2. Gap Statistic 테스트
    result = test_gap_statistic_with_spherical_kmeans()

    if result and result[0]:  # success
        # 3. 비교 테스트
        compare_elbow_vs_gap()

        print("\n" + "=" * 70)
        print("결론:")
        print("=" * 70)
        print("✓ gapstatistics 라이브러리는 Spherical K-Means에 적용 가능합니다.")
        print("✓ sklearn 스타일의 클래스 래퍼를 제공하면 됩니다.")
        print("✓ 정규화된 벡터(normalized vectors)에도 문제없이 작동합니다.")
        print("✓ 코사인 거리(cosine distance)를 커스텀 메트릭으로 사용 가능합니다.")
        print()
        print("장점:")
        print("  - 2차 미분보다 노이즈에 덜 민감함")
        print("  - 통계적으로 더 robust한 방법")
        print("  - 참조 분포(null reference distribution)와의 비교를 통해 최적 k 결정")
        print("  - Tibshirani et al.의 이론적 기반")
        print()
        print("적용 방법:")
        print("  1. SphericalKMeansClusteringService에 Gap Statistic 옵션 추가")
        print("  2. auto_clustering() 메서드에 method 파라미터 추가")
        print("     (method='elbow' 또는 method='gap')")
        print("  3. GapStatistics 클래스를 통합하여 사용")
        print("=" * 70)


if __name__ == "__main__":
    main()
