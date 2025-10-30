#!/usr/bin/env python3
"""
Spherical K-means의 안정성 테스트
"""

import sys
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).parent / "core"))

import numpy as np
import torch
from services.clustering import SphericalKMeansClusteringService

# 테스트 데이터 생성
np.random.seed(42)
embeddings = torch.randn(500, 64)  # 500개 단어, 64차원

# 서로 다른 시드로 테스트
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50]

print("=" * 80)
print("Spherical K-Means + Elbow Method 안정성 테스트")
print("=" * 80)
print(f"데이터: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")
print(f"클러스터 범위: 3 ~ 20")
print(f"n_init: 10 (각 k마다 10번 초기화)")
print("=" * 80)

results = {}
inertias_all = {}

for seed in seeds:
    print(f"\n[Seed {seed}]")

    service = SphericalKMeansClusteringService(random_state=seed)

    # auto_clustering 실행
    cluster_labels, best_k, inertias, silhouette_scores = service.auto_clustering(
        embeddings,
        min_clusters=3,
        max_clusters=20,
        n_init=10
    )

    results[seed] = {
        'best_k': best_k,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'n_clusters': len(np.unique(cluster_labels))
    }

    inertias_all[seed] = inertias

    print(f"  최적 k: {best_k}")
    print(f"  실제 클러스터 수: {results[seed]['n_clusters']}")
    print(f"  Silhouette: {silhouette_scores[best_k - 3]:.4f}")
    print(f"  Inertia: {inertias[best_k - 3]:.4f}")

print("\n" + "=" * 80)
print("결과 요약")
print("=" * 80)

best_ks = [r['best_k'] for r in results.values()]
print(f"최적 클러스터 수 범위: {min(best_ks)} ~ {max(best_ks)}")
print(f"평균: {np.mean(best_ks):.2f}")
print(f"표준편차: {np.std(best_ks):.2f}")
print(f"분포: {dict(zip(*np.unique(best_ks, return_counts=True)))}")

# Inertia 값의 변동성 분석
print("\n" + "=" * 80)
print("Inertia 값의 변동성 분석")
print("=" * 80)

k_range = range(3, 21)
for k in k_range:
    inertia_values = [inertias_all[seed][k-3] for seed in seeds]
    std_dev = np.std(inertia_values)
    cv = std_dev / np.mean(inertia_values) * 100  # Coefficient of Variation
    print(f"k={k:2d}: mean={np.mean(inertia_values):8.4f}, std={std_dev:8.4f}, CV={cv:5.2f}%")

# 2차 미분 값 분석
print("\n" + "=" * 80)
print("2차 미분 값 분석 (Elbow Detection)")
print("=" * 80)

for seed in seeds[:3]:  # 처음 3개만
    inertias = results[seed]['inertias']
    inertias_norm = np.array(inertias)
    inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())
    second_derivative = np.diff(inertias_norm, 2)

    print(f"\nSeed {seed}:")
    for i, val in enumerate(second_derivative):
        k = list(k_range)[i+1]
        marker = " <-- MAX" if i == np.argmax(second_derivative) else ""
        print(f"  k={k:2d}: 2nd_deriv={val:8.6f}{marker}")

print("\n" + "=" * 80)
print("결론")
print("=" * 80)

if np.std(best_ks) > 2:
    print("⚠️  WARNING: Elbow Method가 매우 불안정합니다!")
    print("   -> 랜덤 시드만으로 클러스터 개수가 크게 달라집니다.")
    print("   -> 개선 방안:")
    print("      1. Silhouette Score를 함께 고려")
    print("      2. Gap Statistic 사용")
    print("      3. 더 안정적인 Elbow 알고리즘 (Kneedle)")
elif np.std(best_ks) > 0.5:
    print("⚠️  주의: Elbow Method에 약간의 불안정성이 있습니다.")
else:
    print("✅ Elbow Method가 안정적입니다.")

print(f"\n표준편차: {np.std(best_ks):.2f}")
