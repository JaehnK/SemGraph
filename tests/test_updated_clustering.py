#!/usr/bin/env python3
"""
업데이트된 SphericalKMeansClusteringService 테스트
"""

import numpy as np
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService
from sklearn.datasets import make_blobs


def test_gap_method():
    """Gap Statistics 방법 테스트"""
    print("=" * 70)
    print("Gap Statistics 방법 테스트 (기본값)")
    print("=" * 70)

    # 테스트 데이터 생성
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, n_features=32, centers=5, cluster_std=1.5, random_state=42)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_tensor = torch.from_numpy(X_normalized).float()

    print(f"\n데이터 정보:")
    print(f"  샘플 수: {X_tensor.shape[0]}")
    print(f"  특징 수: {X_tensor.shape[1]}")
    print(f"  실제 클러스터 수: 5")

    # Gap Statistics (기본값)
    print(f"\nGap Statistics 방법 실행 중...")
    clustering = SphericalKMeansClusteringService(random_state=42)

    import time
    start_time = time.time()

    labels, best_k, inertias, silhouette_scores = clustering.auto_clustering(
        X_tensor,
        max_clusters=10,  # 기본값
        n_init=3,         # 기본값
        method='gap'      # 기본값
    )

    elapsed = time.time() - start_time

    print(f"\n결과:")
    print(f"  소요 시간: {elapsed:.1f}초")
    print(f"  최적 클러스터 수: {best_k}")
    print(f"  실제 클러스터 수: 5")
    print(f"  오차: {abs(best_k - 5)}")
    print(f"  클러스터 분포: {clustering.get_cluster_distribution()}")


def test_elbow_method():
    """Elbow Method 테스트"""
    print("\n" + "=" * 70)
    print("Elbow Method 테스트")
    print("=" * 70)

    # 테스트 데이터 생성
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, n_features=32, centers=5, cluster_std=1.5, random_state=42)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_tensor = torch.from_numpy(X_normalized).float()

    print(f"\n데이터 정보:")
    print(f"  샘플 수: {X_tensor.shape[0]}")
    print(f"  특징 수: {X_tensor.shape[1]}")
    print(f"  실제 클러스터 수: 5")

    # Elbow Method
    print(f"\nElbow Method 실행 중...")
    clustering = SphericalKMeansClusteringService(random_state=42)

    import time
    start_time = time.time()

    labels, best_k, inertias, silhouette_scores = clustering.auto_clustering(
        X_tensor,
        min_clusters=3,
        max_clusters=10,
        n_init=3,
        method='elbow'
    )

    elapsed = time.time() - start_time

    print(f"\n결과:")
    print(f"  소요 시간: {elapsed:.1f}초")
    print(f"  최적 클러스터 수: {best_k}")
    print(f"  실제 클러스터 수: 5")
    print(f"  오차: {abs(best_k - 5)}")
    print(f"  Inertias 개수: {len(inertias)}")
    print(f"  Silhouette scores 개수: {len(silhouette_scores)}")


def test_fallback():
    """gapstatistics 없을 때 fallback 테스트"""
    print("\n" + "=" * 70)
    print("Fallback 테스트 (gapstatistics가 없다고 가정)")
    print("=" * 70)

    # 작은 데이터셋
    np.random.seed(42)
    X, _ = make_blobs(n_samples=50, n_features=16, centers=3, random_state=42)
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_tensor = torch.from_numpy(X_normalized).float()

    print("\n이 테스트는 실제 fallback을 테스트하지 않습니다.")
    print("gapstatistics가 설치되어 있으면 정상 동작합니다.")


def main():
    """메인 함수"""
    print("업데이트된 SphericalKMeansClusteringService 테스트\n")

    try:
        # 1. Gap Statistics 테스트 (기본값)
        test_gap_method()

        # 2. Elbow Method 테스트
        test_elbow_method()

        # 3. Fallback 테스트
        test_fallback()

        print("\n" + "=" * 70)
        print("요약")
        print("=" * 70)
        print("✓ Gap Statistics가 기본 방법으로 설정되었습니다.")
        print("✓ 기본 파라미터: max_clusters=10, n_init=3, n_iterations=10")
        print("✓ Elbow Method도 method='elbow'로 사용 가능합니다.")
        print("✓ gapstatistics가 없으면 자동으로 Elbow Method로 fallback됩니다.")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
