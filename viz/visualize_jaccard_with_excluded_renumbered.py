#!/usr/bin/env python3
"""
SemGraph C2 포함 Jaccard 히트맵 생성
단, SemGraph는 제외된 클러스터 없이 0-3으로 재번호링
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# ====================================================================
# 재넘버링 매핑
# ====================================================================

LOUVAIN_RENUMBERING = {
    0: 1,  # Sports → C1
    1: 0,  # Politics → C0
    2: 2,  # Tech → C2
    3: 3,  # Finance → C3
}

LEIDEN_RENUMBERING = {
    0: 0,  # Politics → C0
    1: 1,  # Sports → C1
    2: 2,  # Tech → C2
    3: 3,  # Finance → C3
}

# SemGraph: 원본 번호 그대로 유지 (C0~C4)
SEMGRAPH_RENUMBERING_WITH_EXCLUDED = {
    0: 0,  # Politics → C0
    1: 1,  # Biz/Sports → C1
    2: 2,  # General News → C2 (Excluded)
    3: 3,  # Tech/Finance → C3
    4: 4,  # Pure Sports → C4
}


def load_matching_data(json_path: str) -> Dict:
    """매칭 데이터 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)


def remap_similarity_matrix(
    matrix: List[List[float]],
    cluster1_ids: List[int],
    cluster2_ids: List[int],
    cluster1_mapping: Dict[int, int],
    cluster2_mapping: Dict[int, int]
) -> tuple:
    """유사도 매트릭스를 재넘버링"""

    # 새로운 클러스터 ID 목록
    new_c1_ids = []
    new_c2_ids = []

    # 원본 → 새 번호 매핑된 인덱스
    c1_idx_map = {}
    c2_idx_map = {}

    for i, old_id in enumerate(cluster1_ids):
        new_id = cluster1_mapping.get(old_id)
        if new_id is not None:
            new_c1_ids.append(new_id)
            c1_idx_map[i] = len(new_c1_ids) - 1

    for j, old_id in enumerate(cluster2_ids):
        new_id = cluster2_mapping.get(old_id)
        if new_id is not None:  # -1도 포함
            new_c2_ids.append(new_id)
            c2_idx_map[j] = len(new_c2_ids) - 1

    # 새 매트릭스 생성
    new_matrix = np.zeros((len(new_c1_ids), len(new_c2_ids)))

    for old_i, new_i in c1_idx_map.items():
        for old_j, new_j in c2_idx_map.items():
            new_matrix[new_i, new_j] = matrix[old_i][old_j]

    # 정렬 (클러스터 ID 순서대로)
    c1_sort_idx = np.argsort(new_c1_ids)
    c2_sort_idx = np.argsort(new_c2_ids)

    sorted_c1_ids = [new_c1_ids[i] for i in c1_sort_idx]
    sorted_c2_ids = [new_c2_ids[i] for i in c2_sort_idx]
    sorted_matrix = new_matrix[c1_sort_idx][:, c2_sort_idx]

    return sorted_matrix, sorted_c1_ids, sorted_c2_ids


def create_heatmap_with_excluded(
    matching_data: Dict,
    output_path: str
):
    """전체 Jaccard 히트맵 생성 (SemGraph 0-3 재번호링)"""

    print("\n[히트맵 생성: SemGraph 0-3 재번호링]")

    # Louvain 데이터 재매핑
    louvain_matrix, louvain_c1_ids, louvain_c2_ids = remap_similarity_matrix(
        matching_data['Louvain_vs_Ours']['similarity_matrix'],
        matching_data['Louvain_vs_Ours']['cluster1_ids'],
        matching_data['Louvain_vs_Ours']['cluster2_ids'],
        LOUVAIN_RENUMBERING,
        SEMGRAPH_RENUMBERING_WITH_EXCLUDED
    )

    # Leiden 데이터 재매핑
    leiden_matrix, leiden_c1_ids, leiden_c2_ids = remap_similarity_matrix(
        matching_data['Leiden_vs_Ours']['similarity_matrix'],
        matching_data['Leiden_vs_Ours']['cluster1_ids'],
        matching_data['Leiden_vs_Ours']['cluster2_ids'],
        LEIDEN_RENUMBERING,
        SEMGRAPH_RENUMBERING_WITH_EXCLUDED
    )

    print(f"  Louvain: {len(louvain_c1_ids)} x {len(louvain_c2_ids)} 매트릭스")
    print(f"  SemGraph IDs: {louvain_c2_ids}")

    # 서브플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # === Louvain vs SemGraph ===
    ax1 = axes[0]

    sns.heatmap(
        louvain_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=0.4,
        ax=ax1,
        cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
        linewidths=1,
        linecolor='white',
        square=True,
        annot_kws={'fontsize': 14, 'fontweight': 'bold'}
    )

    ax1.set_title('Louvain - SemGraph',
                  fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel('SemGraph', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Louvain', fontsize=14, fontweight='bold')

    # 축 레이블: C2는 "C2 (Excluded)"로 표시
    x_labels = []
    for c in louvain_c2_ids:
        if c == 2:
            x_labels.append("C2\n(Excluded)")
        else:
            x_labels.append(f"C{c}")

    ax1.set_xticklabels(x_labels, rotation=0, fontsize=13, fontweight='bold')
    ax1.set_yticklabels([f"C{c}" for c in louvain_c1_ids],
                        rotation=0, fontsize=13, fontweight='bold')

    # C2 열 강조
    if 2 in louvain_c2_ids:
        c2_idx = list(louvain_c2_ids).index(2)
        ax1.add_patch(plt.Rectangle(
            (c2_idx, 0), 1, len(louvain_c1_ids),
            fill=True, facecolor='gray', alpha=0.1, edgecolor='red', linewidth=3
        ))

    # === Leiden vs SemGraph ===
    ax2 = axes[1]

    sns.heatmap(
        leiden_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=0.4,
        ax=ax2,
        cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
        linewidths=1,
        linecolor='white',
        square=True,
        annot_kws={'fontsize': 14, 'fontweight': 'bold'}
    )

    ax2.set_title('Leiden - SemGraph',
                  fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('SemGraph', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Leiden', fontsize=14, fontweight='bold')

    # 축 레이블: C2는 "C2 (Excluded)"로 표시
    x_labels = []
    for c in leiden_c2_ids:
        if c == 2:
            x_labels.append("C2\n(Excluded)")
        else:
            x_labels.append(f"C{c}")

    ax2.set_xticklabels(x_labels, rotation=0, fontsize=13, fontweight='bold')
    ax2.set_yticklabels([f"C{c}" for c in leiden_c1_ids],
                        rotation=0, fontsize=13, fontweight='bold')

    # C2 열 강조
    if 2 in leiden_c2_ids:
        c2_idx = list(leiden_c2_ids).index(2)
        ax2.add_patch(plt.Rectangle(
            (c2_idx, 0), 1, len(leiden_c1_ids),
            fill=True, facecolor='gray', alpha=0.1, edgecolor='red', linewidth=3
        ))

    plt.tight_layout()

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 히트맵 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""

    print("="*70)
    print("Jaccard 히트맵 생성 (SemGraph 0-3 재번호링)")
    print("="*70)

    # 데이터 로드
    print("\n[1] 매칭 데이터 로드")
    matching_data = load_matching_data(
        "results/rq3_integrated_old/cluster_similarity/cluster_matching.json"
    )

    # 시각화 생성
    print("\n[2] 시각화 생성")
    create_heatmap_with_excluded(
        matching_data,
        "jaccard_full_heatmap_with_excluded.png"
    )

    print("\n" + "="*70)
    print("✅ 시각화 생성 완료!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - jaccard_full_heatmap_with_excluded.png")


if __name__ == "__main__":
    main()
