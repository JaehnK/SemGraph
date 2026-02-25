#!/usr/bin/env python3
"""
여러 클러스터링 방법 비교 - 어떤 방법이 가장 좋은지 확인
"""

import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.services.clustering.SphericalKMeansClusteringService import SphericalKMeansClusteringService
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def load_latest_embeddings():
    """최신 임베딩 로드"""
    candidate_dirs = [
        Path("artifacts/runs/results/grace_gcn_edge_weight"),
        Path("results/grace_gcn_edge_weight"),
    ]
    results_dir = next((p for p in candidate_dirs if p.exists()), candidate_dirs[0])
    embedding_files = sorted(results_dir.glob("embeddings_*.pt"))

    if not embedding_files:
        print("임베딩 파일을 찾을 수 없습니다.")
        return None

    latest_embedding = embedding_files[-1]
    print(f"임베딩 파일: {latest_embedding.name}")

    data = torch.load(latest_embedding)
    if isinstance(data, dict):
        embeddings = data.get('graphmae_embeddings', list(data.values())[0])
    else:
        embeddings = data

    return embeddings


def evaluate_clustering(embeddings_np, labels, method_name):
    """클러스터링 품질 평가"""
    embeddings_normalized = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    n_clusters = len(np.unique(labels))
    cluster_dist = np.bincount(labels)

    print(f"\n{method_name}:")
    print(f"  클러스터 수: {n_clusters}")
    print(f"  클러스터 분포: {dict(enumerate(cluster_dist))}")

    if n_clusters > 1:
        sil = silhouette_score(embeddings_normalized, labels, metric='cosine')
        db = davies_bouldin_score(embeddings_normalized, labels)
        ch = calinski_harabasz_score(embeddings_normalized, labels)

        print(f"  Silhouette: {sil:.4f} (높을수록 좋음, -1~1)")
        print(f"  Davies-Bouldin: {db:.4f} (낮을수록 좋음)")
        print(f"  Calinski-Harabasz: {ch:.4f} (높을수록 좋음)")

        # 클러스터 크기 균형
        balance = cluster_dist.std() / cluster_dist.mean()
        print(f"  균형도: {balance:.4f} (낮을수록 균등)")

        return {
            'k': n_clusters,
            'silhouette': sil,
            'davies_bouldin': db,
            'calinski_harabasz': ch,
            'balance': balance
        }
    else:
        print(f"  ⚠️  단일 클러스터 - 평가 불가")
        return None


def main():
    print("=" * 70)
    print("클러스터링 방법 비교")
    print("=" * 70)

    embeddings = load_latest_embeddings()
    if embeddings is None:
        return

    print(f"\n임베딩 정보:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")

    embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

    results = {}

    # 1. Gap Statistics (min_clusters=3 제약)
    print(f"\n{'=' * 70}")
    print("1. Gap Statistics (min_clusters=3)")
    print('=' * 70)

    clustering1 = SphericalKMeansClusteringService(random_state=42)
    import time
    start = time.time()

    labels1, k1, _, _ = clustering1.auto_clustering(
        embeddings,
        min_clusters=3,
        max_clusters=10,
        n_init=3,
        method='gap'
    )

    elapsed1 = time.time() - start
    print(f"  소요 시간: {elapsed1:.1f}초")

    results['gap'] = evaluate_clustering(embeddings_np, labels1, "Gap Statistics")

    # 2. Elbow Method
    print(f"\n{'=' * 70}")
    print("2. Elbow Method (2차 미분)")
    print('=' * 70)

    clustering2 = SphericalKMeansClusteringService(random_state=42)
    start = time.time()

    labels2, k2, inertias, silhouette_scores = clustering2.auto_clustering(
        embeddings,
        min_clusters=3,
        max_clusters=10,
        n_init=3,
        method='elbow'
    )

    elapsed2 = time.time() - start
    print(f"  소요 시간: {elapsed2:.1f}초")

    results['elbow'] = evaluate_clustering(embeddings_np, labels2, "Elbow Method")

    # 3. Silhouette Score 최대화
    print(f"\n{'=' * 70}")
    print("3. Silhouette Score 최대화")
    print('=' * 70)

    best_k = 3
    best_sil = -1
    best_labels = None

    print("  k별 Silhouette Score:")
    for k in range(3, 11):
        clustering_temp = SphericalKMeansClusteringService(random_state=42)
        labels_temp = clustering_temp.fit_predict(embeddings, n_clusters=k, n_init=3)

        embeddings_normalized = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        sil = silhouette_score(embeddings_normalized, labels_temp, metric='cosine')

        print(f"    k={k}: {sil:.4f}")

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels_temp

    print(f"\n  최적 k: {best_k} (Silhouette={best_sil:.4f})")
    results['silhouette'] = evaluate_clustering(embeddings_np, best_labels, "Silhouette 최대화")

    # 4. 고정 k (전문가 판단)
    print(f"\n{'=' * 70}")
    print("4. 고정 k=5 (전문가 판단)")
    print('=' * 70)

    clustering4 = SphericalKMeansClusteringService(random_state=42)
    labels4 = clustering4.fit_predict(embeddings, n_clusters=5, n_init=10)

    results['fixed_5'] = evaluate_clustering(embeddings_np, labels4, "고정 k=5")

    # 결과 비교
    print(f"\n{'=' * 70}")
    print("종합 비교")
    print('=' * 70)

    print(f"\n{'방법':<20} {'k':>3} {'Silhouette':>12} {'Davies-B':>10} {'Calinski-H':>12} {'균형도':>8}")
    print("-" * 70)

    for method_name, method_label in [
        ('gap', 'Gap Statistics'),
        ('elbow', 'Elbow Method'),
        ('silhouette', 'Silhouette 최대'),
        ('fixed_5', '고정 k=5')
    ]:
        if method_name in results and results[method_name]:
            r = results[method_name]
            print(f"{method_label:<20} {r['k']:>3} {r['silhouette']:>12.4f} {r['davies_bouldin']:>10.4f} "
                  f"{r['calinski_harabasz']:>12.1f} {r['balance']:>8.4f}")

    # 권장사항
    print(f"\n{'=' * 70}")
    print("권장사항")
    print('=' * 70)

    # Silhouette 기준 최고
    best_method_sil = max(
        [(k, v) for k, v in results.items() if v],
        key=lambda x: x[1]['silhouette']
    )

    print(f"\n1. Silhouette 기준 최고: {best_method_sil[0].upper()}")
    print(f"   - Silhouette: {best_method_sil[1]['silhouette']:.4f}")
    print(f"   - k={best_method_sil[1]['k']}")

    # Davies-Bouldin 기준 최고 (낮을수록)
    best_method_db = min(
        [(k, v) for k, v in results.items() if v],
        key=lambda x: x[1]['davies_bouldin']
    )

    print(f"\n2. Davies-Bouldin 기준 최고: {best_method_db[0].upper()}")
    print(f"   - Davies-Bouldin: {best_method_db[1]['davies_bouldin']:.4f}")
    print(f"   - k={best_method_db[1]['k']}")

    print(f"\n결론:")

    if results['gap'] and results['gap']['k'] == 3:
        print("  ⚠️  Gap Statistics가 계속 k=1을 제안하여 k=3으로 강제 조정됨")
        print("  → Gap Statistics는 이 데이터에 적합하지 않음")
        print()

    if results['silhouette']['silhouette'] > 0.5:
        print("  ✓ Silhouette Score 최대화 방법 권장")
        print(f"    최적 k={results['silhouette']['k']}")
    elif results['elbow']:
        print("  ✓ Elbow Method 권장")
        print(f"    최적 k={results['elbow']['k']}")
    else:
        print("  ✓ 고정 k=5 권장 (안정적)")

    print("=" * 70)


if __name__ == "__main__":
    main()
