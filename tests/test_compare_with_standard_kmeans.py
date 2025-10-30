#!/usr/bin/env python3
"""
Spherical K-means vs 표준 K-means 비교
둘 다 같은 정도의 불안정성을 보이는지 확인
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 테스트 데이터 생성
np.random.seed(42)
embeddings = torch.randn(500, 64)
embeddings_np = embeddings.numpy()

print("=" * 80)
print("Spherical K-means vs 표준 K-means 안정성 비교")
print("=" * 80)

k = 10
seeds = [42, 43, 44, 45, 46]

# 1. Spherical K-means
print("\n[1] Spherical K-means (코사인 거리)")
print("-" * 80)

spherical_labels = []
for seed in seeds:
    service = SphericalKMeansClusteringService(random_state=seed)
    labels = service.fit_predict(embeddings, n_clusters=k, n_init=10)
    spherical_labels.append(labels)

spherical_ari = []
for i in range(1, len(spherical_labels)):
    ari = adjusted_rand_score(spherical_labels[0], spherical_labels[i])
    spherical_ari.append(ari)
    print(f"Seed {seeds[i]} vs Seed {seeds[0]}: ARI={ari:.4f}")

print(f"평균 ARI: {np.mean(spherical_ari):.4f}")

# 2. 표준 K-means (Euclidean 거리)
print("\n[2] 표준 K-means (유클리드 거리)")
print("-" * 80)

standard_labels = []
for seed in seeds:
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings_np)
    standard_labels.append(labels)

standard_ari = []
for i in range(1, len(standard_labels)):
    ari = adjusted_rand_score(standard_labels[0], standard_labels[i])
    standard_ari.append(ari)
    print(f"Seed {seeds[i]} vs Seed {seeds[0]}: ARI={ari:.4f}")

print(f"평균 ARI: {np.mean(standard_ari):.4f}")

# 3. 표준 K-means (정규화된 데이터)
print("\n[3] 표준 K-means (L2 정규화된 데이터)")
print("-" * 80)

embeddings_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

normalized_labels = []
for seed in seeds:
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings_norm)
    normalized_labels.append(labels)

normalized_ari = []
for i in range(1, len(normalized_labels)):
    ari = adjusted_rand_score(normalized_labels[0], normalized_labels[i])
    normalized_ari.append(ari)
    print(f"Seed {seeds[i]} vs Seed {seeds[0]}: ARI={ari:.4f}")

print(f"평균 ARI: {np.mean(normalized_ari):.4f}")

# 비교
print("\n" + "=" * 80)
print("비교 결과")
print("=" * 80)

print(f"\nSpherical K-means 평균 ARI: {np.mean(spherical_ari):.4f}")
print(f"표준 K-means 평균 ARI:       {np.mean(standard_ari):.4f}")
print(f"정규화 K-means 평균 ARI:     {np.mean(normalized_ari):.4f}")

# 데이터 구조 분석
print("\n" + "=" * 80)
print("데이터 구조 분석 (낮은 ARI의 원인)")
print("=" * 80)

# Silhouette score로 클러스터 품질 확인
from sklearn.metrics import silhouette_score

sil_spherical = silhouette_score(embeddings_norm, spherical_labels[0], metric='cosine')
sil_standard = silhouette_score(embeddings_np, standard_labels[0], metric='euclidean')
sil_normalized = silhouette_score(embeddings_norm, normalized_labels[0], metric='euclidean')

print(f"\nSilhouette Score (클러스터 분리도):")
print(f"  Spherical K-means: {sil_spherical:.4f}")
print(f"  표준 K-means:      {sil_standard:.4f}")
print(f"  정규화 K-means:    {sil_normalized:.4f}")

print("\n해석:")
if sil_spherical < 0.2:
    print("⚠️  낮은 Silhouette Score는 데이터에 명확한 클러스터 구조가 없음을 의미합니다.")
    print("   이런 경우 여러 지역 최적해(local optima)가 존재하며,")
    print("   시드에 따라 다른 해를 찾는 것이 정상입니다.")
    print("\n이는 알고리즘의 문제가 아니라 **데이터의 특성**입니다:")
    print("  - 랜덤 생성된 데이터 → 자연스러운 클러스터 없음")
    print("  - 고차원 데이터 → '차원의 저주'로 거리가 비슷해짐")
    print("  - 텍스트 임베딩 → 연속적인 의미 공간, 경계가 모호함")
else:
    print("✅ 클러스터 구조가 명확합니다.")

# 실제 데이터로 재테스트
print("\n" + "=" * 80)
print("실제 클러스터 구조가 있는 데이터로 테스트")
print("=" * 80)

from sklearn.datasets import make_blobs
X_blobs, y_true = make_blobs(n_samples=500, n_features=64, centers=10,
                              cluster_std=3.0, random_state=42)
X_blobs_norm = X_blobs / np.linalg.norm(X_blobs, axis=1, keepdims=True)

print("\n데이터: 10개의 명확한 가우시안 클러스터")

# Spherical K-means on blobs
blob_labels_spherical = []
for seed in seeds:
    service = SphericalKMeansClusteringService(random_state=seed)
    labels = service.fit_predict(torch.from_numpy(X_blobs), n_clusters=10, n_init=10)
    blob_labels_spherical.append(labels)

blob_ari_spherical = []
for i in range(1, len(blob_labels_spherical)):
    ari = adjusted_rand_score(blob_labels_spherical[0], blob_labels_spherical[i])
    blob_ari_spherical.append(ari)

print(f"\nSpherical K-means ARI (blobs): {np.mean(blob_ari_spherical):.4f}")
print(f"Silhouette: {silhouette_score(X_blobs_norm, blob_labels_spherical[0], metric='cosine'):.4f}")

print("\n" + "=" * 80)
print("최종 결론")
print("=" * 80)

if np.mean(blob_ari_spherical) > 0.9:
    print("✅ Spherical K-means 알고리즘: 문제 없음!")
    print("   명확한 클러스터 구조가 있을 때는 안정적으로 작동합니다.")
    print("\n⚠️  이전 테스트의 낮은 ARI는:")
    print("   - 랜덤 데이터에 클러스터 구조가 없어서")
    print("   - 여러 동등하게 좋은 해(equally good solutions)가 존재")
    print("   - 이는 정상적인 현상입니다!")
else:
    print("⚠️  Spherical K-means에 구조적 문제가 있을 수 있습니다.")

print(f"\n참고:")
print(f"  랜덤 데이터 ARI: {np.mean(spherical_ari):.4f} (낮음)")
print(f"  실제 클러스터 ARI: {np.mean(blob_ari_spherical):.4f} (높음)")
print(f"\n차이: {np.mean(blob_ari_spherical) - np.mean(spherical_ari):.4f}")
