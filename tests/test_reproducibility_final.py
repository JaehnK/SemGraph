#!/usr/bin/env python3
"""
졸업논문 재현성 최종 테스트
n_init=10 + random_seed 고정으로 완벽한 재현성 검증
"""

import numpy as np
import torch
from pathlib import Path
import sys
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService


def test_reproducibility_10_runs():
    """10번 실행하여 완벽한 재현성 검증"""
    print("=" * 70)
    print("졸업논문 재현성 최종 테스트 (n_init=10)")
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

    print(f"Shape: {embeddings.shape}")

    # 10번 실행
    results = []

    print("\n" + "=" * 70)
    print("10번 반복 실행 (random_seed=42, n_init=10)")
    print("=" * 70)
    print(f"{'Run':<5} {'k':<5} {'Silhouette':<12} {'시간(초)':<10} {'클러스터 분포':<50}")
    print("-" * 70)

    for run in range(1, 11):
        clustering = SphericalKMeansClusteringService(random_state=42)

        start = time.time()
        labels, k, inertias, sil_scores = clustering.auto_clustering(
            embeddings,
            min_clusters=3,
            max_clusters=10,
            method='kneedle'
        )
        elapsed = time.time() - start

        cluster_dist = clustering.get_cluster_distribution()
        sorted_dist = sorted(cluster_dist.values())

        # Silhouette score 추출 (k-3 인덱스)
        sil = sil_scores[k-3] if sil_scores and len(sil_scores) > k-3 else 0.0

        dist_str = str(sorted_dist)
        print(f"{run:<5} {k:<5} {sil:<12.4f} {elapsed:<10.2f} {dist_str:<50}")

        results.append({
            'run': run,
            'k': k,
            'silhouette': sil,
            'dist': sorted_dist,
            'labels': labels.copy(),
            'elapsed': elapsed
        })

    print("-" * 70)

    # 분석
    print("\n" + "=" * 70)
    print("재현성 분석")
    print("=" * 70)

    k_values = [r['k'] for r in results]
    unique_k = set(k_values)

    print(f"\n✓ 탐지된 k 값들: {sorted(unique_k)}")
    print(f"✓ k 값의 종류: {len(unique_k)}개")

    if len(unique_k) == 1:
        print(f"\n✅ 완벽한 재현성! 10번 모두 k={k_values[0]}를 탐지했습니다.")
        print("   → 졸업논문에 사용 가능합니다!")
    else:
        print(f"\n⚠️  k가 변동됩니다: {sorted(unique_k)}")
        for k_val in sorted(unique_k):
            count = k_values.count(k_val)
            print(f"   k={k_val}: {count}번 ({count/10*100:.0f}%)")

    # Silhouette 일관성
    sil_values = [r['silhouette'] for r in results]
    print(f"\nSilhouette Score 일관성:")
    print(f"  평균: {np.mean(sil_values):.4f}")
    print(f"  표준편차: {np.std(sil_values):.6f}")
    print(f"  범위: {min(sil_values):.4f} ~ {max(sil_values):.4f}")

    if np.std(sil_values) < 0.0001:
        print(f"  ✅ Silhouette가 완벽히 일치합니다!")
    else:
        print(f"  ⚠️  Silhouette에 약간 차이가 있습니다.")

    # 클러스터 분포 일관성
    print(f"\n클러스터 분포 패턴:")
    print(f"  첫 번째 실행: {results[0]['dist']}")
    print(f"  마지막 실행:  {results[-1]['dist']}")

    all_same_dist = all(r['dist'] == results[0]['dist'] for r in results)
    if all_same_dist:
        print(f"  ✅ 10번 모두 동일한 분포 패턴!")
    else:
        print(f"  ⚠️  분포 패턴이 약간 다릅니다.")

    # 라벨 일치 확인
    print(f"\n라벨 일치 확인:")
    first_labels = results[0]['labels']
    all_same_labels = all(np.array_equal(r['labels'], first_labels) for r in results)

    if all_same_labels:
        print(f"  ✅ 10번 모두 완벽히 동일한 클러스터 할당!")
    else:
        print(f"  ⚠️  라벨이 약간 다릅니다.")
        # 일치율 계산
        match_counts = [np.sum(r['labels'] == first_labels) / len(first_labels) * 100
                       for r in results]
        print(f"  평균 일치율: {np.mean(match_counts):.2f}%")

    # 실행 시간
    elapsed_times = [r['elapsed'] for r in results]
    print(f"\n실행 시간:")
    print(f"  평균: {np.mean(elapsed_times):.2f}초")
    print(f"  범위: {min(elapsed_times):.2f} ~ {max(elapsed_times):.2f}초")

    # 최종 결론
    print("\n" + "=" * 70)
    print("최종 결론")
    print("=" * 70)

    if len(unique_k) == 1 and np.std(sil_values) < 0.0001 and all_same_labels:
        print("✅ 완벽한 재현성이 보장됩니다!")
        print("   - k 값 일치: ✅")
        print("   - Silhouette 일치: ✅")
        print("   - 클러스터 할당 일치: ✅")
        print("\n졸업논문에 다음과 같이 명시할 수 있습니다:")
        print("━" * 70)
        print("\"실험의 재현성을 보장하기 위해 모든 난수 생성기에 random_seed=42를")
        print("설정하였으며, 클러스터링의 안정성을 위해 n_init=10으로 설정하였다.")
        print(f"그 결과, 동일한 조건에서 10회 반복 실행 시 항상 k={k_values[0]}개의")
        print("클러스터를 탐지하였으며, 클러스터 할당도 완벽히 일치하였다.\"")
        print("━" * 70)
    else:
        print("⚠️  아직 완벽한 재현성이 보장되지 않습니다.")
        if len(unique_k) > 1:
            print(f"   - k 값이 변동: {sorted(unique_k)}")
        if np.std(sil_values) >= 0.0001:
            print(f"   - Silhouette에 미세한 차이")
        if not all_same_labels:
            print(f"   - 클러스터 할당이 다름")
        print("\n추가 조치가 필요합니다:")
        print("   1. PyTorch DataLoader의 worker_init_fn 확인")
        print("   2. DGL 그래프 생성 시 random_seed 확인")
        print("   3. n_init를 더 늘리기 (10 → 20)")

    print("=" * 70)


def test_different_seeds():
    """다른 random_seed로도 재현성 확인"""
    print("\n\n" + "=" * 70)
    print("다른 random_seed로 재현성 검증")
    print("=" * 70)

    results_dir = Path("results/grace_gcn_edge_weight")
    embedding_files = sorted(results_dir.glob("embeddings_*.pt"))

    if not embedding_files:
        return

    data = torch.load(embedding_files[-1])
    if isinstance(data, dict):
        embeddings = data.get('graphmae_embeddings', list(data.values())[0])
    else:
        embeddings = data

    seeds = [42, 100, 123, 999, 2024]

    print(f"\n{'Seed':<10} {'k':<5} {'Silhouette':<12}")
    print("-" * 30)

    results_by_seed = {}

    for seed in seeds:
        # 각 seed로 2번씩 실행
        k_list = []
        sil_list = []

        for _ in range(2):
            clustering = SphericalKMeansClusteringService(random_state=seed)
            labels, k, inertias, sil_scores = clustering.auto_clustering(
                embeddings, min_clusters=3, max_clusters=10, method='kneedle'
            )
            sil = sil_scores[k-3] if sil_scores and len(sil_scores) > k-3 else 0.0
            k_list.append(k)
            sil_list.append(sil)

        # 일치 확인
        consistent = (k_list[0] == k_list[1])
        symbol = "✅" if consistent else "⚠️"

        print(f"{seed:<10} {k_list[0]:<5} {sil_list[0]:<12.4f} {symbol}")

        results_by_seed[seed] = {
            'k': k_list,
            'consistent': consistent
        }

    print("-" * 30)

    all_consistent = all(r['consistent'] for r in results_by_seed.values())

    if all_consistent:
        print("\n✅ 모든 random_seed에서 재현성이 보장됩니다!")
        print("   각 seed로 2번 실행 시 항상 동일한 k를 탐지")
    else:
        print("\n⚠️  일부 seed에서 재현성이 보장되지 않습니다.")


if __name__ == "__main__":
    test_reproducibility_10_runs()
    test_different_seeds()

    print("\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)
    print("n_init=10 + random_seed 고정으로 재현성을 검증했습니다.")
    print("졸업논문 작성 시 이 결과를 Method 섹션에 명시하세요.")
    print("=" * 70)
