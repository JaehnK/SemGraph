#!/usr/bin/env python3
"""
Alluvial Diagram for Cluster Flow Visualization

세로 방향의 리본 스타일로 클러스터 간 단어 흐름을 시각화
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath


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


# 클러스터 색상 (일관성 유지)
CLUSTER_COLORS = {
    0: '#E74C3C',  # Political - Red
    1: '#3498DB',  # Sports Biz - Blue
    2: '#2ECC71',  # Sports/General - Green
    3: '#F39C12',  # Technology - Orange
    4: '#9B59B6',  # Finance/Sports - Purple
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


def draw_ribbon(ax, x1, y1_start, y1_end, x2, y2_start, y2_end, color, alpha=0.5):
    """
    리본(곡선 사각형) 그리기

    Args:
        x1, x2: 시작/끝 x 좌표
        y1_start, y1_end: 시작 지점의 y 범위
        y2_start, y2_end: 끝 지점의 y 범위
        color: 리본 색상
        alpha: 투명도
    """

    # 베지어 곡선을 위한 제어점
    ctrl_x = (x1 + x2) / 2

    # 상단 곡선 경로
    top_points = []
    for t in np.linspace(0, 1, 50):
        # 3차 베지어 곡선
        x = (1-t)**3 * x1 + 3*(1-t)**2*t * ctrl_x + 3*(1-t)*t**2 * ctrl_x + t**3 * x2
        y = (1-t)**3 * y1_start + 3*(1-t)**2*t * y1_start + 3*(1-t)*t**2 * y2_start + t**3 * y2_start
        top_points.append([x, y])

    # 하단 곡선 경로 (역순)
    bottom_points = []
    for t in np.linspace(1, 0, 50):
        x = (1-t)**3 * x1 + 3*(1-t)**2*t * ctrl_x + 3*(1-t)*t**2 * ctrl_x + t**3 * x2
        y = (1-t)**3 * y1_end + 3*(1-t)**2*t * y1_end + 3*(1-t)*t**2 * y2_end + t**3 * y2_end
        bottom_points.append([x, y])

    # 폴리곤 생성
    polygon_points = top_points + bottom_points
    polygon = Polygon(polygon_points, facecolor=color, edgecolor='none',
                     alpha=alpha, zorder=1)
    ax.add_patch(polygon)


def draw_cluster_bar(ax, x, y_start, y_end, cluster_id, label, size, method_name):
    """
    클러스터를 세로 막대로 그리기

    Args:
        x: x 좌표
        y_start, y_end: 막대의 시작/끝 y 좌표
        cluster_id: 클러스터 ID
        label: 클러스터 레이블
        size: 클러스터 크기
        method_name: 방법 이름
    """

    height = y_end - y_start

    # 막대 그리기
    rect = patches.Rectangle(
        (x - 0.3, y_start), 0.6, height,
        linewidth=1.5,
        edgecolor='white',
        facecolor=CLUSTER_COLORS[cluster_id],
        zorder=10
    )
    ax.add_patch(rect)

    # 레이블 (막대 왼쪽)
    ax.text(x - 0.4, (y_start + y_end) / 2, f'C{cluster_id}',
            ha='right', va='center',
            fontsize=10, fontweight='bold',
            color='#2C3E50')

    # 크기 (막대 오른쪽)
    ax.text(x + 0.4, (y_start + y_end) / 2, f'n={size}',
            ha='left', va='center',
            fontsize=9,
            color='#7F8C8D')


def create_alluvial_diagram(
    louvain_clusters: Dict[int, List[str]],
    leiden_clusters: Dict[int, List[str]],
    ours_clusters: Dict[int, List[str]],
    output_path: str,
    min_flow: int = 5
):
    """Alluvial Diagram 생성"""

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

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-2, 55)
    ax.axis('off')

    # ====================================================================
    # 각 방법의 클러스터를 세로로 쌓기
    # ====================================================================

    # x 좌표
    x_louvain = 2.0
    x_leiden = 5.5
    x_semgraph = 9.0

    # 총 단어 수 (정규화 기준)
    total_words = 500

    # ====================================================================
    # Louvain 클러스터 위치 계산
    # ====================================================================

    print(f"  Louvain 클러스터 배치 계산")
    louvain_positions = {}
    y_current = 0

    # 클러스터 순서: C0, C2, C3, C4 (C1 없음)
    for cluster_id in [0, 2, 3, 4]:
        size = CLUSTER_INFO['Louvain'][cluster_id]['size']
        height = (size / total_words) * 40  # 전체 높이 40으로 스케일

        louvain_positions[cluster_id] = {
            'y_start': y_current,
            'y_end': y_current + height,
            'size': size
        }
        y_current += height + 1.0  # 간격 증가

    # ====================================================================
    # Leiden 클러스터 위치 계산
    # ====================================================================

    print(f"  Leiden 클러스터 배치 계산")
    leiden_positions = {}
    y_current = 0

    # 클러스터 순서: C0, C2, C3, C4 (C1 없음)
    for cluster_id in [0, 2, 3, 4]:
        size = CLUSTER_INFO['Leiden'][cluster_id]['size']
        height = (size / total_words) * 40

        leiden_positions[cluster_id] = {
            'y_start': y_current,
            'y_end': y_current + height,
            'size': size
        }
        y_current += height + 1.0  # 간격 증가

    # ====================================================================
    # SemGraph 클러스터 위치 계산 (간격을 더 넓게)
    # ====================================================================

    print(f"  SemGraph 클러스터 배치 계산")
    semgraph_positions = {}
    y_current = 0

    for cluster_id in [0, 1, 2, 3, 4]:
        size = CLUSTER_INFO['SemGraph'][cluster_id]['size']
        height = (size / total_words) * 40

        semgraph_positions[cluster_id] = {
            'y_start': y_current,
            'y_end': y_current + height,
            'size': size
        }
        y_current += height + 1.5  # 간격을 0.5 → 1.5로 증가

    # ====================================================================
    # Louvain → Leiden 리본 그리기
    # ====================================================================

    print(f"  Louvain → Leiden 리본 그리기")

    # 각 소스 클러스터에서 출발하는 흐름 추적
    for src_id in [0, 2, 3, 4]:
        src_flows = [(tgt_id, count) for s, tgt_id, count in flow_l2l
                     if s == src_id and count >= min_flow]

        if not src_flows:
            continue

        # 소스의 y 범위
        src_y_start = louvain_positions[src_id]['y_start']
        src_y_end = louvain_positions[src_id]['y_end']
        src_height = src_y_end - src_y_start

        # 흐름을 크기순으로 정렬
        src_flows_sorted = sorted(src_flows, key=lambda x: x[1], reverse=True)

        # 각 흐름에 대해 리본 그리기
        y_offset = 0
        for tgt_id, count in src_flows_sorted:
            # 리본 높이 계산
            ribbon_height = (count / louvain_positions[src_id]['size']) * src_height

            # 타겟의 y 위치 찾기
            tgt_y_start = leiden_positions[tgt_id]['y_start']
            tgt_y_end = leiden_positions[tgt_id]['y_end']
            tgt_height = tgt_y_end - tgt_y_start

            # 타겟에서 이 소스로부터 받는 비율 계산
            tgt_ribbon_height = (count / leiden_positions[tgt_id]['size']) * tgt_height

            # 리본 그리기
            draw_ribbon(
                ax,
                x1=x_louvain + 0.3,
                y1_start=src_y_start + y_offset,
                y1_end=src_y_start + y_offset + ribbon_height,
                x2=x_leiden - 0.3,
                y2_start=tgt_y_start,
                y2_end=tgt_y_start + tgt_ribbon_height,
                color=CLUSTER_COLORS[src_id],
                alpha=0.4
            )

            y_offset += ribbon_height

    # ====================================================================
    # Leiden → SemGraph 리본 그리기
    # ====================================================================

    print(f"  Leiden → SemGraph 리본 그리기")

    for src_id in [0, 2, 3, 4]:
        src_flows = [(tgt_id, count) for s, tgt_id, count in flow_l2s
                     if s == src_id and count >= min_flow]

        if not src_flows:
            continue

        src_y_start = leiden_positions[src_id]['y_start']
        src_y_end = leiden_positions[src_id]['y_end']
        src_height = src_y_end - src_y_start

        src_flows_sorted = sorted(src_flows, key=lambda x: x[1], reverse=True)

        y_offset = 0
        for tgt_id, count in src_flows_sorted:
            ribbon_height = (count / leiden_positions[src_id]['size']) * src_height

            tgt_y_start = semgraph_positions[tgt_id]['y_start']
            tgt_y_end = semgraph_positions[tgt_id]['y_end']
            tgt_height = tgt_y_end - tgt_y_start

            tgt_ribbon_height = (count / semgraph_positions[tgt_id]['size']) * tgt_height

            draw_ribbon(
                ax,
                x1=x_leiden + 0.3,
                y1_start=src_y_start + y_offset,
                y1_end=src_y_start + y_offset + ribbon_height,
                x2=x_semgraph - 0.3,
                y2_start=tgt_y_start,
                y2_end=tgt_y_start + tgt_ribbon_height,
                color=CLUSTER_COLORS[src_id],
                alpha=0.4
            )

            y_offset += ribbon_height

    # ====================================================================
    # 클러스터 막대 그리기
    # ====================================================================

    print(f"  클러스터 막대 그리기")

    # Louvain
    for cluster_id in [0, 2, 3, 4]:
        pos = louvain_positions[cluster_id]
        info = CLUSTER_INFO['Louvain'][cluster_id]
        draw_cluster_bar(ax, x_louvain, pos['y_start'], pos['y_end'],
                        cluster_id, info['label'], info['size'], 'Louvain')

    # Leiden
    for cluster_id in [0, 2, 3, 4]:
        pos = leiden_positions[cluster_id]
        info = CLUSTER_INFO['Leiden'][cluster_id]
        draw_cluster_bar(ax, x_leiden, pos['y_start'], pos['y_end'],
                        cluster_id, info['label'], info['size'], 'Leiden')

    # SemGraph
    for cluster_id in [0, 1, 2, 3, 4]:
        pos = semgraph_positions[cluster_id]
        info = CLUSTER_INFO['SemGraph'][cluster_id]
        draw_cluster_bar(ax, x_semgraph, pos['y_start'], pos['y_end'],
                        cluster_id, info['label'], info['size'], 'SemGraph')

    # ====================================================================
    # 헤더 (방법 이름)
    # ====================================================================

    header_y = 50
    ax.text(x_louvain, header_y, 'Louvain',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#2C3E50')

    ax.text(x_leiden, header_y, 'Leiden',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#2C3E50')

    ax.text(x_semgraph, header_y, 'SemGraph',
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='#2C3E50')

    # ====================================================================
    # 제목
    # ====================================================================

    fig.suptitle('Cluster Evolution: Alluvial Diagram',
                 fontsize=18, fontweight='bold', y=0.96,
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
    print("Alluvial Diagram 생성")
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

    print("\n[단계 2] Alluvial 다이어그램 생성")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/rq3_integrated/visualizations/cluster_alluvial_{timestamp}.png"

    create_alluvial_diagram(
        louvain_clusters=louvain_clusters,
        leiden_clusters=leiden_clusters,
        ours_clusters=ours_clusters,
        output_path=output_path,
        min_flow=5
    )

    print("\n" + "="*70)
    print("✅ 모든 작업 완료!")
    print("="*70)
    print(f"  저장 위치: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
