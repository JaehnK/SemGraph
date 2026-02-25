#!/usr/bin/env python3
"""
재넘버링된 클러스터 ID로 Jaccard 시각화 생성

재넘버링 규칙:
- Louvain: {0: 1, 1: 0, 2: 3, 3: 2} (원본 → 새 번호)
- Leiden: {0: 0, 1: 1, 2: 2, 3: 3} (변경 없음)
- SemGraph: {0: 0, 1: 4, 2: 제외, 3: 2, 4: 1} (원본 → 새 번호)

새 번호 체계:
- C0: Politics & International
- C1: Sports
- C2: Technology & Business
- C3: Economy & Finance
- C4: Pure Sports (SemGraph only)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# ====================================================================
# 재넘버링 매핑
# ====================================================================

LOUVAIN_RENUMBERING = {
    0: 1,  # ap/world/game → C1 (Sports)
    1: 0,  # iraq/president → C0 (Politics)
    2: 3,  # reuters/oil → C3 (Finance)
    3: 2,  # microsoft/company → C2 (Technology)
}

LEIDEN_RENUMBERING = {
    0: 0,  # iraq/president → C0 (Politics)
    1: 1,  # ap/world/game → C1 (Sports)
    2: 2,  # microsoft/company → C2 (Technology)
    3: 3,  # reuters/oil → C3 (Finance)
}

SEMGRAPH_RENUMBERING = {
    0: 0,  # minister/baghdad → C0 (Politics)
    1: 4,  # company/game → C4 (Pure Sports) - 제외된 클러스터
    2: None,  # reuters/oil → 제외 (General News)
    3: 2,  # microsoft/service → C2 (Technology)
    4: 1,  # team/victory → C1 (Sports)
}

# 주제 정보
TOPIC_INFO = {
    0: {'label': 'Politics &\nInternational', 'short': 'Politics'},
    1: {'label': 'Sports', 'short': 'Sports'},
    2: {'label': 'Technology &\nBusiness', 'short': 'Tech'},
    3: {'label': 'Economy &\nFinance', 'short': 'Finance'},
    4: {'label': 'Pure Sports\n(SemGraph only)', 'short': 'Pure Sports'},
}

# 주제별 색상
TOPIC_COLORS = {
    0: '#e74c3c',  # Politics - Red
    1: '#3498db',  # Sports - Blue
    2: '#f39c12',  # Tech - Orange
    3: '#2ecc71',  # Finance - Green
    4: '#9b59b6',  # Pure Sports - Purple
}


def load_matching_data(json_path: str) -> Dict:
    """매칭 데이터 로드"""
    with open(json_path, 'r') as f:
        return json.load(f)


def remap_matches(
    matches: List[Dict],
    cluster1_mapping: Dict[int, int],
    cluster2_mapping: Dict[int, int]
) -> List[Dict]:
    """매칭 결과를 재넘버링"""
    remapped = []

    for match in matches:
        old_c1 = match['cluster1_id']
        old_c2 = match['cluster2_id']

        new_c1 = cluster1_mapping.get(old_c1)
        new_c2 = cluster2_mapping.get(old_c2)

        # 제외된 클러스터는 스킵 (SemGraph C2 원본)
        if new_c1 is None or new_c2 is None:
            continue

        remapped.append({
            'cluster1_id': new_c1,
            'cluster2_id': new_c2,
            'jaccard': match['jaccard'],
            'cluster1_size': match['cluster1_size'],
            'cluster2_size': match['cluster2_size'],
            'intersection_size': match['intersection_size']
        })

    return remapped


def create_jaccard_heatmap(
    louvain_matches: List[Dict],
    leiden_matches: List[Dict],
    output_path: str
):
    """재넘버링된 Jaccard 히트맵 생성"""

    print("\n[히트맵 생성]")

    # 매칭 데이터를 딕셔너리로 변환
    louvain_dict = {(m['cluster1_id'], m['cluster2_id']): m['jaccard'] for m in louvain_matches}
    leiden_dict = {(m['cluster1_id'], m['cluster2_id']): m['jaccard'] for m in leiden_matches}

    # 전통적 방법: C0, C1, C2, C3 (4개)
    # SemGraph: C0, C1, C2, C4 (4개, C3 제외)
    traditional_clusters = [0, 1, 2, 3]
    semgraph_clusters = sorted(set([m['cluster2_id'] for m in louvain_matches]))

    print(f"  전통적 방법 클러스터: {traditional_clusters}")
    print(f"  SemGraph 클러스터: {semgraph_clusters}")

    # 서브플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # === Louvain vs SemGraph ===
    ax1 = axes[0]

    # 매트릭스 생성
    louvain_matrix = np.zeros((len(traditional_clusters), len(semgraph_clusters)))
    for i, trad_c in enumerate(traditional_clusters):
        for j, sem_c in enumerate(semgraph_clusters):
            louvain_matrix[i, j] = louvain_dict.get((trad_c, sem_c), 0.0)

    # 히트맵
    sns.heatmap(
        louvain_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=0.4,
        ax=ax1,
        cbar_kws={'label': 'Jaccard Similarity'},
        linewidths=2,
        linecolor='white'
    )

    # 매칭 강조 (대각선 아님!)
    for match in louvain_matches:
        trad_idx = traditional_clusters.index(match['cluster1_id'])
        sem_idx = semgraph_clusters.index(match['cluster2_id'])
        ax1.add_patch(plt.Rectangle(
            (sem_idx, trad_idx), 1, 1,
            fill=False, edgecolor='blue', linewidth=4
        ))

    ax1.set_title('Louvain vs SemGraph\n(Hungarian Matched Only)',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('SemGraph Clusters', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Louvain Clusters', fontsize=13, fontweight='bold')

    # 레이블
    ax1.set_xticklabels([f"C{c}\n{TOPIC_INFO[c]['short']}" for c in semgraph_clusters],
                        rotation=0, fontsize=11)
    ax1.set_yticklabels([f"C{c} {TOPIC_INFO[c]['short']}" for c in traditional_clusters],
                        rotation=0, fontsize=11)

    # === Leiden vs SemGraph ===
    ax2 = axes[1]

    # 매트릭스 생성
    leiden_matrix = np.zeros((len(traditional_clusters), len(semgraph_clusters)))
    for i, trad_c in enumerate(traditional_clusters):
        for j, sem_c in enumerate(semgraph_clusters):
            leiden_matrix[i, j] = leiden_dict.get((trad_c, sem_c), 0.0)

    # 히트맵
    sns.heatmap(
        leiden_matrix,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=0.4,
        ax=ax2,
        cbar_kws={'label': 'Jaccard Similarity'},
        linewidths=2,
        linecolor='white'
    )

    # 매칭 강조
    for match in leiden_matches:
        trad_idx = traditional_clusters.index(match['cluster1_id'])
        sem_idx = semgraph_clusters.index(match['cluster2_id'])
        ax2.add_patch(plt.Rectangle(
            (sem_idx, trad_idx), 1, 1,
            fill=False, edgecolor='blue', linewidth=4
        ))

    ax2.set_title('Leiden vs SemGraph\n(Hungarian Matched Only)',
                  fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('SemGraph Clusters', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Leiden Clusters', fontsize=13, fontweight='bold')

    # 레이블
    ax2.set_xticklabels([f"C{c}\n{TOPIC_INFO[c]['short']}" for c in semgraph_clusters],
                        rotation=0, fontsize=11)
    ax2.set_yticklabels([f"C{c} {TOPIC_INFO[c]['short']}" for c in traditional_clusters],
                        rotation=0, fontsize=11)

    # 레이아웃 조정
    plt.tight_layout()

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 히트맵 저장: {output_path}")
    plt.close()


def create_flow_diagram(
    louvain_matches: List[Dict],
    leiden_matches: List[Dict],
    output_path: str
):
    """재넘버링된 Flow 다이어그램 생성"""

    print("\n[Flow 다이어그램 생성]")

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # === Louvain vs SemGraph ===
    ax1 = axes[0]

    # 전통적 방법과 SemGraph 클러스터
    traditional_clusters = [0, 1, 2, 3]
    semgraph_clusters = sorted(set([m['cluster2_id'] for m in louvain_matches]))

    # Y 위치 계산 (주제별 정렬)
    y_spacing = 1.0
    trad_y_positions = {c: i * y_spacing for i, c in enumerate(traditional_clusters)}
    sem_y_positions = {c: i * y_spacing for i, c in enumerate(semgraph_clusters)}

    # 전통적 방법 노드 (왼쪽)
    for cluster_id in traditional_clusters:
        y = trad_y_positions[cluster_id]
        circle = plt.Circle((0.2, y), 0.08,
                           color=TOPIC_COLORS[cluster_id],
                           ec='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(0.05, y, f"C{cluster_id}\n{TOPIC_INFO[cluster_id]['short']}",
                ha='right', va='center', fontsize=11, fontweight='bold')

    # SemGraph 노드 (오른쪽)
    for cluster_id in semgraph_clusters:
        y = sem_y_positions[cluster_id]
        circle = plt.Circle((0.8, y), 0.08,
                           color=TOPIC_COLORS[cluster_id],
                           ec='black', linewidth=2, zorder=3)
        ax1.add_patch(circle)
        ax1.text(0.95, y, f"C{cluster_id}\n{TOPIC_INFO[cluster_id]['short']}",
                ha='left', va='center', fontsize=11, fontweight='bold')

    # 매칭 화살표
    for match in louvain_matches:
        trad_c = match['cluster1_id']
        sem_c = match['cluster2_id']
        jaccard = match['jaccard']

        y1 = trad_y_positions[trad_c]
        y2 = sem_y_positions[sem_c]

        # 화살표 (두께는 Jaccard에 비례)
        linewidth = 2 + jaccard * 10
        ax1.annotate('',
                    xy=(0.72, y2), xytext=(0.28, y1),
                    arrowprops=dict(
                        arrowstyle='->',
                        lw=linewidth,
                        color=TOPIC_COLORS[trad_c],
                        alpha=0.6
                    ))

        # Jaccard 값 표시
        mid_x, mid_y = 0.5, (y1 + y2) / 2
        ax1.text(mid_x, mid_y, f'{jaccard:.3f}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.8))

    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.5, len(traditional_clusters) * y_spacing - 0.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Louvain vs SemGraph (Hungarian Matched)',
                 fontsize=14, fontweight='bold', pad=20)

    # Louvain, SemGraph 레이블
    ax1.text(0.2, len(traditional_clusters) * y_spacing + 0.2, 'Louvain',
            ha='center', fontsize=13, fontweight='bold')
    ax1.text(0.8, len(semgraph_clusters) * y_spacing + 0.2, 'SemGraph',
            ha='center', fontsize=13, fontweight='bold')

    # === Leiden vs SemGraph ===
    ax2 = axes[1]

    # Leiden 노드 (왼쪽)
    for cluster_id in traditional_clusters:
        y = trad_y_positions[cluster_id]
        circle = plt.Circle((0.2, y), 0.08,
                           color=TOPIC_COLORS[cluster_id],
                           ec='black', linewidth=2, zorder=3)
        ax2.add_patch(circle)
        ax2.text(0.05, y, f"C{cluster_id}\n{TOPIC_INFO[cluster_id]['short']}",
                ha='right', va='center', fontsize=11, fontweight='bold')

    # SemGraph 노드 (오른쪽)
    for cluster_id in semgraph_clusters:
        y = sem_y_positions[cluster_id]
        circle = plt.Circle((0.8, y), 0.08,
                           color=TOPIC_COLORS[cluster_id],
                           ec='black', linewidth=2, zorder=3)
        ax2.add_patch(circle)
        ax2.text(0.95, y, f"C{cluster_id}\n{TOPIC_INFO[cluster_id]['short']}",
                ha='left', va='center', fontsize=11, fontweight='bold')

    # 매칭 화살표
    for match in leiden_matches:
        trad_c = match['cluster1_id']
        sem_c = match['cluster2_id']
        jaccard = match['jaccard']

        y1 = trad_y_positions[trad_c]
        y2 = sem_y_positions[sem_c]

        linewidth = 2 + jaccard * 10
        ax2.annotate('',
                    xy=(0.72, y2), xytext=(0.28, y1),
                    arrowprops=dict(
                        arrowstyle='->',
                        lw=linewidth,
                        color=TOPIC_COLORS[trad_c],
                        alpha=0.6
                    ))

        mid_x, mid_y = 0.5, (y1 + y2) / 2
        ax2.text(mid_x, mid_y, f'{jaccard:.3f}',
                ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.8))

    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.5, len(traditional_clusters) * y_spacing - 0.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Leiden vs SemGraph (Hungarian Matched)',
                 fontsize=14, fontweight='bold', pad=20)

    # Leiden, SemGraph 레이블
    ax2.text(0.2, len(traditional_clusters) * y_spacing + 0.2, 'Leiden',
            ha='center', fontsize=13, fontweight='bold')
    ax2.text(0.8, len(semgraph_clusters) * y_spacing + 0.2, 'SemGraph',
            ha='center', fontsize=13, fontweight='bold')

    # 레이아웃 조정
    plt.tight_layout()

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Flow 다이어그램 저장: {output_path}")
    plt.close()


def main():
    """메인 실행 함수"""

    print("="*70)
    print("재넘버링된 Jaccard 시각화 생성")
    print("="*70)

    # 데이터 로드
    print("\n[1] 매칭 데이터 로드")
    matching_data = load_matching_data(
        "results/rq3_integrated_old/cluster_similarity/cluster_matching.json"
    )

    # 재넘버링
    print("\n[2] 클러스터 재넘버링")

    louvain_matches = remap_matches(
        matching_data['Louvain_vs_Ours']['matches'],
        LOUVAIN_RENUMBERING,
        SEMGRAPH_RENUMBERING
    )
    print(f"  ✓ Louvain: {len(louvain_matches)} 매칭")

    leiden_matches = remap_matches(
        matching_data['Leiden_vs_Ours']['matches'],
        LEIDEN_RENUMBERING,
        SEMGRAPH_RENUMBERING
    )
    print(f"  ✓ Leiden: {len(leiden_matches)} 매칭")

    # 시각화 생성
    print("\n[3] 시각화 생성")

    create_jaccard_heatmap(
        louvain_matches,
        leiden_matches,
        "jaccard_heatmap_renumbered.png"
    )

    create_flow_diagram(
        louvain_matches,
        leiden_matches,
        "jaccard_flow_renumbered.png"
    )

    print("\n" + "="*70)
    print("✅ 모든 시각화 생성 완료!")
    print("="*70)
    print("\n생성된 파일:")
    print("  - jaccard_heatmap_renumbered.png")
    print("  - jaccard_flow_renumbered.png")


if __name__ == "__main__":
    main()
