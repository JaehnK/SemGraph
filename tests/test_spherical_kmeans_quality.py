#!/usr/bin/env python3
"""
Spherical K-means 알고리즘 자체의 품질 테스트
고정된 k 값에서 시드에 따라 결과가 일관되는지 확인
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 테스트 데이터 생성
np.random.seed(42)
embeddings = torch.randn(500, 64)

print("=" * 80)
print("Spherical K-means 알고리즘 품질 테스트")
print("=" * 80)
print(f"데이터: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")
print("테스트: 고정된 k=10으로 여러 시드에서 실행")
print("=" * 80)

# 고정된 k 값으로 테스트
k = 10
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50]

results = {}
all_labels = []
inertias = []
silhouettes = []

print(f"\n[K={k} 고정, n_init=10]")
print("-" * 80)

for seed in seeds:
    service = SphericalKMeansClusteringService(random_state=seed)

    # 클러스터링 수행
    labels = service.fit_predict(embeddings, n_clusters=k, n_init=10)

    # Inertia 계산 (수동)
    embeddings_np = embeddings.numpy()
    embeddings_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    centroids = []
    inertia = 0.0
    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.sum(mask) > 0:
            cluster_mean = np.mean(embeddings_norm[mask], axis=0)
            centroid = cluster_mean / np.linalg.norm(cluster_mean)
            centroids.append(centroid)

            # 코사인 거리 계산
            similarities = np.dot(embeddings_norm[mask], centroid)
            distances = 1 - similarities
            inertia += np.sum(distances)

    # Silhouette score
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(embeddings_norm, labels, metric='cosine')

    results[seed] = {
        'labels': labels,
        'inertia': inertia,
        'silhouette': silhouette,
        'n_unique': len(np.unique(labels))
    }

    all_labels.append(labels)
    inertias.append(inertia)
    silhouettes.append(silhouette)

    print(f"Seed {seed}: inertia={inertia:8.4f}, silhouette={silhouette:.4f}, unique_clusters={len(np.unique(labels))}")

print("\n" + "=" * 80)
print("통계 분석")
print("=" * 80)

# Inertia 안정성
print(f"\nInertia:")
print(f"  평균: {np.mean(inertias):.4f}")
print(f"  표준편차: {np.std(inertias):.4f}")
print(f"  변동계수(CV): {np.std(inertias)/np.mean(inertias)*100:.2f}%")
print(f"  범위: {np.min(inertias):.4f} ~ {np.max(inertias):.4f}")

# Silhouette 안정성
print(f"\nSilhouette Score:")
print(f"  평균: {np.mean(silhouettes):.4f}")
print(f"  표준편차: {np.std(silhouettes):.4f}")
print(f"  범위: {np.min(silhouettes):.4f} ~ {np.max(silhouettes):.4f}")

# 클러스터링 결과 일관성 (첫 번째 결과를 기준으로)
print(f"\n클러스터링 결과 일관성 (Seed 42 기준):")
reference_labels = all_labels[0]

ari_scores = []
nmi_scores = []

for i, labels in enumerate(all_labels[1:], start=1):
    ari = adjusted_rand_score(reference_labels, labels)
    nmi = normalized_mutual_info_score(reference_labels, labels)
    ari_scores.append(ari)
    nmi_scores.append(nmi)
    print(f"  Seed {seeds[i]} vs Seed 42: ARI={ari:.4f}, NMI={nmi:.4f}")

print(f"\n평균 ARI: {np.mean(ari_scores):.4f} (1.0 = 완전 일치)")
print(f"평균 NMI: {np.mean(nmi_scores):.4f} (1.0 = 완전 일치)")

# 결론
print("\n" + "=" * 80)
print("결론")
print("=" * 80)

cv_threshold = 1.0  # 1% 이하면 안정적
ari_threshold = 0.9  # 0.9 이상이면 매우 유사

if np.std(inertias)/np.mean(inertias)*100 < cv_threshold:
    print("✅ Inertia: 매우 안정적 (CV < 1%)")
else:
    print(f"⚠️  Inertia: 다소 불안정 (CV = {np.std(inertias)/np.mean(inertias)*100:.2f}%)")

if np.mean(ari_scores) > ari_threshold:
    print("✅ 클러스터링 결과: 거의 동일 (평균 ARI > 0.9)")
elif np.mean(ari_scores) > 0.7:
    print("⚠️  클러스터링 결과: 유사하지만 약간 다름 (평균 ARI > 0.7)")
else:
    print("❌ 클러스터링 결과: 상당히 다름 (평균 ARI < 0.7)")

print("\n종합 평가:")
if np.std(inertias)/np.mean(inertias)*100 < cv_threshold and np.mean(ari_scores) > ari_threshold:
    print("✅ Spherical K-means 알고리즘: 문제 없음!")
    print("   동일한 k에서 시드에 관계없이 일관된 결과를 생성합니다.")
elif np.mean(ari_scores) > 0.7:
    print("⚠️  Spherical K-means 알고리즘: 대체로 안정적")
    print("   미세한 차이는 있지만 전체적으로 유사한 클러스터링을 생성합니다.")
    print("   이는 여러 지역 최적해(local optima)가 존재하기 때문이며 정상적입니다.")
else:
    print("❌ Spherical K-means 알고리즘: 불안정!")
    print("   시드에 따라 매우 다른 클러스터링 결과를 생성합니다.")

# 클러스터 크기 분포 분석
print("\n" + "=" * 80)
print("클러스터 크기 분포 분석")
print("=" * 80)

for i, seed in enumerate(seeds[:3]):  # 처음 3개만
    labels = all_labels[i]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nSeed {seed}:")
    cluster_sizes = sorted(counts, reverse=True)
    print(f"  클러스터 크기: {cluster_sizes}")
    print(f"  최대/최소 비율: {max(counts)/min(counts):.2f}")
    print(f"  표준편차: {np.std(counts):.2f}")
