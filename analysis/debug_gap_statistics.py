#!/usr/bin/env python3
"""
Gap Statistics 디버깅 - 왜 k=1이 나오는지 확인
"""

import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 가장 최근 임베딩 파일 로드
candidate_dirs = [
    Path("artifacts/runs/results/grace_gcn_edge_weight"),
    Path("results/grace_gcn_edge_weight"),
]
results_dir = next((p for p in candidate_dirs if p.exists()), candidate_dirs[0])
embedding_files = sorted(results_dir.glob("embeddings_*.pt"))

if embedding_files:
    latest_embedding = embedding_files[-1]
    print(f"임베딩 파일 로드: {latest_embedding}")

    data = torch.load(latest_embedding)

    # dict인 경우 embeddings 키 찾기
    if isinstance(data, dict):
        print(f"\n저장된 키: {list(data.keys())}")
        if 'embeddings' in data:
            embeddings = data['embeddings']
        elif 'graphmae_embeddings' in data:
            embeddings = data['graphmae_embeddings']
        else:
            embeddings = list(data.values())[0]
    else:
        embeddings = data

    print(f"\n임베딩 정보:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")

    # 정규화 확인
    norms = torch.norm(embeddings, dim=1)
    print(f"\nL2 Norm 통계:")
    print(f"  Min: {norms.min().item():.4f}")
    print(f"  Max: {norms.max().item():.4f}")
    print(f"  Mean: {norms.mean().item():.4f}")

    # 이미 정규화되어 있는지 확인
    if torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        print("  ✓ 임베딩이 이미 정규화되어 있습니다.")
    else:
        print("  ✗ 임베딩이 정규화되어 있지 않습니다.")

    # Gap Statistics로 테스트
    print("\n" + "=" * 70)
    print("Gap Statistics 테스트")
    print("=" * 70)

    from gapstatistics.gapstatistics import GapStatistics
    from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService

    embeddings_np = embeddings.numpy()

    # 정규화
    embeddings_normalized = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    print(f"\n테스트 1: K=10, n_iterations=10")
    gs1 = GapStatistics(
        algorithm=SphericalKMeansClusteringService._SphericalKMeansWrapper,
        distance_metric=SphericalKMeansClusteringService._cosine_distance_for_gap,
        return_params=True
    )
    n_clusters1, params1 = gs1.fit_predict(K=10, X=embeddings_normalized, n_iterations=10)
    print(f"  탐지된 k: {n_clusters1}")
    if 'gap' in params1:
        gap_values = params1['gap']
        print(f"  Gap 값 (k=1~{len(gap_values)}): {[f'{v:.4f}' for v in gap_values]}")

    print(f"\n테스트 2: K=15, n_iterations=20 (더 정밀)")
    gs2 = GapStatistics(
        algorithm=SphericalKMeansClusteringService._SphericalKMeansWrapper,
        distance_metric=SphericalKMeansClusteringService._cosine_distance_for_gap,
        return_params=True
    )
    n_clusters2, params2 = gs2.fit_predict(K=15, X=embeddings_normalized, n_iterations=20)
    print(f"  탐지된 k: {n_clusters2}")
    if 'gap' in params2:
        gap_values = params2['gap']
        print(f"  Gap 값 (k=1~{len(gap_values)}): {[f'{v:.4f}' for v in gap_values[:10]]}")

    # Elbow Method와 비교
    print(f"\n테스트 3: Elbow Method와 비교")
    clustering = SphericalKMeansClusteringService(random_state=42)
    embeddings_tensor = torch.from_numpy(embeddings_normalized).float()

    labels_elbow, best_k_elbow, inertias, silhouette_scores = clustering.auto_clustering(
        embeddings_tensor,
        min_clusters=3,
        max_clusters=15,
        n_init=3,
        method='elbow'
    )
    print(f"  Elbow Method 탐지: k={best_k_elbow}")
    print(f"  Inertias: {[f'{v:.4f}' for v in inertias[:5]]}...")

    # 직접 클러스터링 테스트
    print(f"\n테스트 4: 직접 k=5로 클러스터링")
    labels_k5 = clustering.fit_predict(embeddings_tensor, n_clusters=5, n_init=10)
    cluster_dist = np.bincount(labels_k5)
    print(f"  클러스터 분포: {cluster_dist}")

    from sklearn.metrics import silhouette_score
    if len(np.unique(labels_k5)) > 1:
        sil_score = silhouette_score(embeddings_normalized, labels_k5, metric='cosine')
        print(f"  Silhouette score: {sil_score:.4f}")

    # 분석
    print("\n" + "=" * 70)
    print("분석")
    print("=" * 70)

    if n_clusters1 == 1:
        print("⚠️  Gap Statistics가 k=1을 선택한 이유:")
        print("  1. 데이터가 매우 균일하게 분포되어 있을 수 있음")
        print("  2. 임베딩의 분산이 너무 작을 수 있음")
        print("  3. 코사인 거리에서 모든 점이 유사할 수 있음")
        print()
        print("해결 방안:")
        print("  1. Elbow Method 사용 (더 안정적일 수 있음)")
        print("  2. min_clusters를 Gap Statistics에도 적용")
        print("  3. n_iterations 증가")
        print("  4. 데이터 전처리 개선 (임베딩 품질 향상)")

    print(f"\n권장사항:")
    if best_k_elbow > 1:
        print(f"  → Elbow Method 결과(k={best_k_elbow})를 사용하는 것을 권장합니다.")
    else:
        print(f"  → 데이터 또는 임베딩에 문제가 있을 수 있습니다.")

else:
    print("임베딩 파일을 찾을 수 없습니다.")
    print("먼저 pipelines/main.py를 실행하여 임베딩을 생성하세요.")
