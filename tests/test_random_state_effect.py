#!/usr/bin/env python3
"""
random_state가 클러스터 수(k)에 미치는 영향 테스트
"""

import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService


def test_random_state_effect_on_k():
    """random_state가 k값에 영향을 주는지 테스트"""
    print("=" * 70)
    print("random_state가 클러스터 수(k)에 미치는 영향 테스트")
    print("=" * 70)

    # 실제 임베딩 로드
    results_dir = Path("results/grace_gcn_edge_weight")
    embedding_files = sorted(results_dir.glob("embeddings_*.pt"))

    if not embedding_files:
        print("임베딩 파일을 찾을 수 없습니다.")
        return

    latest_embedding = embedding_files[-1]
    print(f"\n임베딩 파일: {latest_embedding.name}")

    data = torch.load(latest_embedding)
    if isinstance(data, dict):
        embeddings = data.get('graphmae_embeddings', list(data.values())[0])
    else:
        embeddings = data

    print(f"Shape: {embeddings.shape}\n")

    # 여러 random_state로 테스트
    random_states = [42, 0, 123, 456, 789, 999, 2024, 2025]
    results = {}

    print("다양한 random_state로 Kneedle 실행:")
    print("-" * 70)
    print(f"{'Seed':<10} {'k':<5} {'Inertias (3~10)':<50}")
    print("-" * 70)

    for seed in random_states:
        clustering = SphericalKMeansClusteringService(random_state=seed)

        labels, k, inertias, sil_scores = clustering.auto_clustering(
            embeddings,
            min_clusters=3,
            max_clusters=10,
            method='kneedle'
        )

        cluster_dist = clustering.get_cluster_distribution()
        sorted_dist = sorted(cluster_dist.values())

        # Inertias를 짧게 표시
        inertias_str = ', '.join([f'{v:.1f}' for v in inertias[:4]]) + ', ...'

        print(f"{seed:<10} {k:<5} {inertias_str:<50}")

        results[seed] = {
            'k': k,
            'inertias': inertias,
            'dist': sorted_dist,
            'silhouette': sil_scores
        }

    print("-" * 70)

    # 분석
    print("\n분석:")
    print("=" * 70)

    k_values = [r['k'] for r in results.values()]
    unique_k = set(k_values)

    print(f"탐지된 k 값들: {sorted(unique_k)}")
    print(f"k 값의 개수: {len(unique_k)}")
    print(f"가장 많이 나온 k: {max(set(k_values), key=k_values.count)} (빈도: {k_values.count(max(set(k_values), key=k_values.count))}/{len(k_values)})")

    if len(unique_k) == 1:
        print(f"\n✅ random_state와 무관하게 항상 k={k_values[0]}를 탐지합니다.")
        print("   → Kneedle이 매우 안정적입니다!")
    else:
        print(f"\n⚠️  random_state에 따라 k가 변동됩니다: {sorted(unique_k)}")
        print(f"   범위: {min(k_values)} ~ {max(k_values)}")
        print(f"   표준편차: {np.std(k_values):.2f}")

    # 클러스터 분포 비교
    print(f"\n클러스터 분포 패턴:")
    print("-" * 70)
    print(f"{'Seed':<10} {'k':<5} {'분포 (정렬됨)':<50}")
    print("-" * 70)

    for seed, result in results.items():
        dist_str = str(result['dist'])
        print(f"{seed:<10} {result['k']:<5} {dist_str:<50}")

    print("-" * 70)

    # 분포 유사도 확인
    print(f"\n분포 유사도 분석:")
    distributions = [r['dist'] for r in results.values()]

    # 가장 큰 클러스터 크기 비교
    largest_clusters = [max(d) for d in distributions]
    print(f"  가장 큰 클러스터 크기: {min(largest_clusters)} ~ {max(largest_clusters)}")
    print(f"  평균: {np.mean(largest_clusters):.1f}, 표준편차: {np.std(largest_clusters):.1f}")

    # Inertia 곡선 비교
    print(f"\nInertia 곡선 유사도:")
    inertia_curves = [r['inertias'] for r in results.values()]

    # k=3일 때 inertia 비교
    inertias_at_k3 = [curve[0] for curve in inertia_curves]
    print(f"  k=3일 때 inertia: {min(inertias_at_k3):.1f} ~ {max(inertias_at_k3):.1f}")
    print(f"  차이: {max(inertias_at_k3) - min(inertias_at_k3):.1f} ({(max(inertias_at_k3) - min(inertias_at_k3)) / min(inertias_at_k3) * 100:.1f}%)")

    # 결론
    print("\n" + "=" * 70)
    print("결론")
    print("=" * 70)

    if len(unique_k) == 1:
        print("✅ random_state는 k값에 영향을 주지 않습니다.")
        print("   - Kneedle이 일관되게 같은 knee point를 탐지")
        print("   - Inertia 곡선이 random_state와 무관하게 안정적")
        print("   - 클러스터 분포 패턴도 유사")
    else:
        print("⚠️  random_state가 k값에 영향을 줍니다.")
        print("   원인:")
        print("   1. K-means 초기화 위치가 달라져서 local minima가 다름")
        print("   2. Inertia 곡선이 약간씩 달라짐")
        print("   3. Knee point가 모호한 구간에 있음")
        print()
        print("   해결 방안:")
        print("   1. n_init을 증가 (더 많은 초기화 시도)")
        print("   2. random_state를 고정 (재현성)")
        print("   3. 여러 random_state로 실행 후 다수결")

    print("\n클러스터 순서:")
    print("  - K-means는 클러스터에 임의로 번호 할당")
    print("  - 순서는 의미 없음 (0번이 첫 번째가 아님)")
    print("  - 분포 패턴만 중요 (큰 클러스터 vs 작은 클러스터)")

    print("=" * 70)


def test_inertia_curve_stability():
    """Inertia 곡선의 안정성 테스트"""
    print("\n\n" + "=" * 70)
    print("Inertia 곡선 안정성 상세 분석")
    print("=" * 70)

    # 실제 임베딩 로드
    results_dir = Path("results/grace_gcn_edge_weight")
    embedding_files = sorted(results_dir.glob("embeddings_*.pt"))

    if not embedding_files:
        return

    data = torch.load(embedding_files[-1])
    if isinstance(data, dict):
        embeddings = data.get('graphmae_embeddings', list(data.values())[0])
    else:
        embeddings = data

    print("\n5개의 다른 random_state로 inertia 곡선 비교:")

    seeds = [42, 100, 200, 300, 400]
    all_inertias = []

    for seed in seeds:
        clustering = SphericalKMeansClusteringService(random_state=seed)
        _, _, inertias, _ = clustering.auto_clustering(
            embeddings, min_clusters=3, max_clusters=10, method='kneedle'
        )
        all_inertias.append(inertias)

    print(f"\n{'k':<5}", end='')
    for seed in seeds:
        print(f"{'seed=' + str(seed):<12}", end='')
    print(f"{'Std':>10}")
    print("-" * 70)

    for i, k in enumerate(range(3, 11)):
        print(f"{k:<5}", end='')
        values = [inertias[i] for inertias in all_inertias]
        for v in values:
            print(f"{v:<12.2f}", end='')
        print(f"{np.std(values):>10.2f}")

    print()
    print("해석:")
    print("  - Std(표준편차)가 작으면 random_state와 무관하게 안정적")
    print("  - Std가 크면 random_state가 영향을 줌")


if __name__ == "__main__":
    test_random_state_effect_on_k()
    test_inertia_curve_stability()
