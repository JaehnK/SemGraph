#!/usr/bin/env python3
"""
Elbow Method의 민감도 테스트
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 예시 데이터 생성 (실제로는 500개의 단어 임베딩)
X, _ = make_blobs(n_samples=500, n_features=64, centers=10, random_state=42)

# 서로 다른 시드로 K-means 실행
def compute_inertias(X, k_range, seed):
    from sklearn.cluster import KMeans
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    return inertias

def find_elbow(k_values, inertias):
    """현재 구현과 동일한 로직"""
    inertias_norm = np.array(inertias)
    inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())
    second_derivative = np.diff(inertias_norm, 2)
    elbow_idx = np.argmax(second_derivative) + 1
    return k_values[elbow_idx]

# 테스트
k_range = list(range(3, 21))
seeds = [42, 43, 44, 45, 46]

print("=" * 60)
print("Elbow Method 민감도 테스트")
print("=" * 60)

results = {}
for seed in seeds:
    inertias = compute_inertias(X, k_range, seed)
    best_k = find_elbow(k_range, inertias)
    results[seed] = (inertias, best_k)
    print(f"Seed {seed}: 최적 클러스터 수 = {best_k}")

print("\n클러스터 개수 범위:", min(r[1] for r in results.values()), "~", max(r[1] for r in results.values()))
print("표준편차:", np.std([r[1] for r in results.values()]))

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# 상단: Inertia 곡선
ax1 = axes[0]
for seed, (inertias, best_k) in results.items():
    ax1.plot(k_range, inertias, 'o-', label=f'Seed {seed} (k={best_k})', alpha=0.7)
    ax1.axvline(best_k, linestyle='--', alpha=0.3)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Inertia Curves for Different Seeds')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 하단: 2차 미분
ax2 = axes[1]
for seed, (inertias, best_k) in results.items():
    inertias_norm = np.array(inertias)
    inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())
    second_derivative = np.diff(inertias_norm, 2)
    k_range_2nd = k_range[1:-1]
    ax2.plot(k_range_2nd, second_derivative, 'o-', label=f'Seed {seed}', alpha=0.7)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Second Derivative')
ax2.set_title('Second Derivative (Elbow Detection)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('elbow_sensitivity_test.png', dpi=300)
print(f"\n그래프 저장: elbow_sensitivity_test.png")
