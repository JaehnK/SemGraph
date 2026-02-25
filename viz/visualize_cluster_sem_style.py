#!/usr/bin/env python3
"""
구조방정식 모델(SEM) 스타일 클러스터 흐름 다이어그램

얇은 화살표와 간결한 박스로 학술적 느낌의 클러스터 흐름을 시각화
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ====================================================================
# JSON → LaTeX 클러스터 ID 매핑
# ====================================================================

LOUVAIN_MAPPING = {
    0: 2,  # ap/world/game → C2 (스포츠/일반)
    1: 0,  # iraq/president → C0 (정치/뉴스)
    2: 4,  # reuters/oil → C4 (금융 보도)
    3: 3,  # microsoft/company → C3 (기술 비즈니스)
}

LEIDEN_MAPPING = {
    0: 0,  # iraq/president → C0 (정치/사건)
    1: 2,  # ap/world/game → C2 (스포츠/일반)
    2: 3,  # microsoft/company → C3 (기술 비즈니스)
    3: 4,  # reuters/oil → C4 (경제/시장)
}

OURS_MAPPING = {
    0: 0,  # 중동/정치 → C0
    1: 1,  # 스포츠 비즈니스 → C1
    2: 2,  # 일반 뉴스 → C2
    3: 3,  # 기술/금융 → C3
    4: 4,  # 스포츠 → C4
}


# ====================================================================
# 클러스터 정보
# ====================================================================

CLUSTER_INFO = {
    'Louvain': {
        0: {'label': 'Political', 'size': 170},
        2: {'label': 'Sports', 'size': 122},
        3: {'label': 'Technology', 'size': 101},
        4: {'label': 'Finance', 'size': 107},
    },
    'Leiden': {
        0: {'label': 'Political', 'size': 165},
        2: {'label': 'Sports', 'size': 119},
        3: {'label': 'Technology', 'size': 109},
        4: {'label': 'Finance', 'size': 107},
    },
    'SemGraph': {
        0: {'label': 'Political', 'size': 60},
        1: {'label': 'Sports Biz', 'size': 109},
        2: {'label': 'General', 'size': 156},
        3: {'label': 'Tech/Finance', 'size': 129},
        4: {'label': 'Sports', 'size': 46},
    }
}


def load_and_remap_clusters(
    json_path: str,
    mapping: Dict[int, int]
) -> Dict[int, List[str]]:
    """JSON 파일을 로드하고 LaTeX 클러스터 ID로 재매핑"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    remapped = {}
    for json_id, words in data.items():
        json_id = int(json_id)
        latex_id = mapping[json_id]
        remapped[latex_id] = words

    return remapped


def calculate_word_flow(
    source_clusters: Dict[int, List[str]],
    target_clusters: Dict[int, List[str]]
) -> List[Tuple[int, int, int]]:
    """두 클러스터링 간의 단어 흐름 계산"""
    flows = []

    for src_id, src_words in source_clusters.items():
        src_set = set(src_words)

        for tgt_id, tgt_words in target_clusters.items():
            tgt_set = set(tgt_words)
            shared = src_set & tgt_set

            if len(shared) > 0:
                flows.append((src_id, tgt_id, len(shared)))

    return flows


def draw_cluster_box(ax, x, y, cluster_id, label, size, width=1.2, height=0.6):
    """클러스터 박스 그리기"""

    # 박스
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.05",
        edgecolor='#2C3E50',
        facecolor='white',
        linewidth=1.5,
        zorder=10
    )
    ax.add_patch(box)

    # 클러스터 ID (위)
    ax.text(x, y + 0.12, f'C{cluster_id}',
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='#2C3E50', zorder=11)

    # 레이블 (중간)
    ax.text(x, y - 0.02, label,
            ha='center', va='center',
            fontsize=9,
            color='#34495E', zorder=11)

    # 크기 (아래)
    ax.text(x, y - 0.16, f'n={size}',
            ha='center', va='center',
            fontsize=8, style='italic',
            color='#7F8C8D', zorder=11)


def draw_arrow(ax, x1, y1, x2, y2, flow_count,
               curve=0, min_flow=5, max_flow=170):
    """
    화살표 그리기 (얇고 깔끔하게)

    Args:
        curve: 곡선 정도 (0=직선, 양수=위로, 음수=아래로)
    """

    # 흐름이 너무 작으면 그리지 않음
    if flow_count < min_flow:
        return

    # 화살표 두께 (매우 얇게)
    linewidth = 0.5 + 1.5 * (flow_count / max_flow)

    # 화살표 색상 (흐름이 클수록 진하게)
    alpha = 0.3 + 0.5 * (flow_count / max_flow)
    color = '#34495E'

    if curve != 0:
        # 곡선 화살표
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            connectionstyle=f"arc3,rad={curve}",
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=5
        )
    else:
        # 직선 화살표
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            zorder=5
        )

    ax.add_patch(arrow)

    # 흐름 숫자 (화살표 경로 위에 정확히 배치)
    if curve != 0:
        # 곡선의 중간점을 베지어 곡선으로 계산
        # arc3의 rad 파라미터를 고려한 실제 곡선 경로
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # 곡선의 정점 계산 (rad 값에 비례)
        dx = x2 - x1
        dy = y2 - y1
        # 수직 방향으로 오프셋 추가
        offset = curve * 1.5  # 곡선 강도에 비례
        mid_y += offset

    else:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

    # 숫자 배경 (흰색 박스)
    ax.text(mid_x, mid_y, str(flow_count),
            ha='center', va='center',
            fontsize=8, fontweight='normal',
            color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.2',
                     facecolor='white',
                     edgecolor='none',
                     alpha=0.9),
            zorder=15)


def create_sem_style_diagram(
    louvain_clusters: Dict[int, List[str]],
    leiden_clusters: Dict[int, List[str]],
    ours_clusters: Dict[int, List[str]],
    output_path: str,
    min_flow: int = 5
):
    """구조방정식 모델 스타일 다이어그램 생성"""

    print(f"\n[1/3] 단어 흐름 계산")

    # Louvain → Leiden 흐름
    flow_l2l = calculate_word_flow(louvain_clusters, leiden_clusters)
    print(f"  Louvain → Leiden: {len(flow_l2l)} flows")

    # Leiden → SemGraph 흐름
    flow_l2s = calculate_word_flow(leiden_clusters, ours_clusters)
    print(f"  Leiden → SemGraph: {len(flow_l2s)} flows")

    # ====================================================================
    # Figure 설정
    # ====================================================================

    print(f"\n[2/3] 다이어그램 생성")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 10)
    ax.axis('off')

    # ====================================================================
    # 노드 위치 정의
    # ====================================================================

    # x 좌표: 3개 열
    x_louvain = 1.5
    x_leiden = 5.0
    x_semgraph = 8.5

    # y 좌표: 클러스터별
    y_positions = {
        0: 8.0,   # C0: Political
        2: 6.0,   # C2: Sports
        3: 4.0,   # C3: Technology
        4: 2.0,   # C4: Finance
        1: 5.0,   # C1: Sports Biz (SemGraph only)
    }

    # ====================================================================
    # 헤더 (방법 이름)
    # ====================================================================

    ax.text(x_louvain, 9.2, 'Louvain',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#2C3E50')

    ax.text(x_leiden, 9.2, 'Leiden',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#2C3E50')

    ax.text(x_semgraph, 9.2, 'SemGraph',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#2C3E50')

    # ====================================================================
    # Louvain 클러스터 박스
    # ====================================================================

    print(f"  Louvain 클러스터 그리기")
    for cluster_id in sorted(CLUSTER_INFO['Louvain'].keys()):
        info = CLUSTER_INFO['Louvain'][cluster_id]
        y = y_positions[cluster_id]
        draw_cluster_box(ax, x_louvain, y, cluster_id,
                        info['label'], info['size'])

    # ====================================================================
    # Leiden 클러스터 박스
    # ====================================================================

    print(f"  Leiden 클러스터 그리기")
    for cluster_id in sorted(CLUSTER_INFO['Leiden'].keys()):
        info = CLUSTER_INFO['Leiden'][cluster_id]
        y = y_positions[cluster_id]
        draw_cluster_box(ax, x_leiden, y, cluster_id,
                        info['label'], info['size'])

    # ====================================================================
    # SemGraph 클러스터 박스
    # ====================================================================

    print(f"  SemGraph 클러스터 그리기")
    for cluster_id in sorted(CLUSTER_INFO['SemGraph'].keys()):
        info = CLUSTER_INFO['SemGraph'][cluster_id]
        y = y_positions[cluster_id]
        draw_cluster_box(ax, x_semgraph, y, cluster_id,
                        info['label'], info['size'])

    # ====================================================================
    # Louvain → Leiden 화살표
    # ====================================================================

    print(f"  Louvain → Leiden 화살표 그리기")
    for src_id, tgt_id, count in flow_l2l:
        if count >= min_flow:
            y1 = y_positions[src_id]
            y2 = y_positions[tgt_id]

            draw_arrow(ax, x_louvain + 0.6, y1, x_leiden - 0.6, y2,
                      count, curve=0, min_flow=min_flow)

    # ====================================================================
    # Leiden → SemGraph 화살표
    # ====================================================================

    print(f"  Leiden → SemGraph 화살표 그리기")
    for src_id, tgt_id, count in flow_l2s:
        if count >= min_flow:
            y1 = y_positions[src_id]
            y2 = y_positions[tgt_id]

            draw_arrow(ax, x_leiden + 0.6, y1, x_semgraph - 0.6, y2,
                      count, curve=0, min_flow=min_flow)

    # ====================================================================
    # 제목
    # ====================================================================

    fig.suptitle('Cluster Evolution: Louvain → Leiden → SemGraph',
                 fontsize=18, fontweight='bold', y=0.97,
                 color='#2C3E50')

    # ====================================================================
    # 저장
    # ====================================================================

    print(f"\n[3/3] 저장 중...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  ✓ 저장 완료: {output_path}")

    plt.close()

    return output_path


def main():
    """메인 실행 함수"""

    print("="*70)
    print("구조방정식 스타일 클러스터 흐름 다이어그램")
    print("="*70)

    # ====================================================================
    # 클러스터 데이터 로드
    # ====================================================================

    print("\n[단계 1] 클러스터 데이터 로드")

    base_path = Path("results/rq3_integrated/qualitative")

    louvain_clusters = load_and_remap_clusters(
        base_path / "Louvain" / "cluster_words.json",
        LOUVAIN_MAPPING
    )
    print(f"  ✓ Louvain: {len(louvain_clusters)} 클러스터")

    leiden_clusters = load_and_remap_clusters(
        base_path / "Leiden" / "cluster_words.json",
        LEIDEN_MAPPING
    )
    print(f"  ✓ Leiden: {len(leiden_clusters)} 클러스터")

    ours_clusters = load_and_remap_clusters(
        base_path / "Ours" / "cluster_words.json",
        OURS_MAPPING
    )
    print(f"  ✓ SemGraph: {len(ours_clusters)} 클러스터")

    # ====================================================================
    # 다이어그램 생성
    # ====================================================================

    print("\n[단계 2] 다이어그램 생성")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/rq3_integrated/visualizations/cluster_sem_style_{timestamp}.png"

    create_sem_style_diagram(
        louvain_clusters=louvain_clusters,
        leiden_clusters=leiden_clusters,
        ours_clusters=ours_clusters,
        output_path=output_path,
        min_flow=5  # 최소 5개 이상의 공유 단어
    )

    print("\n" + "="*70)
    print("✅ 모든 작업 완료!")
    print("="*70)
    print(f"  저장 위치: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
