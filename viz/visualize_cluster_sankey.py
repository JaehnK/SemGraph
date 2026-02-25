#!/usr/bin/env python3
"""
클러스터 흐름 Sankey 다이어그램 시각화

세 가지 클러스터링 방법(Louvain → Leiden → SemGraph)의
단어 흐름을 Sankey 다이어그램으로 시각화합니다.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import plotly.graph_objects as go
from collections import defaultdict


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
        0: {'size': 170, 'label': 'C0: 정치/뉴스'},
        2: {'size': 122, 'label': 'C2: 스포츠/일반'},
        3: {'size': 101, 'label': 'C3: 기술 비즈니스'},
        4: {'size': 107, 'label': 'C4: 금융 보도'},
    },
    'Leiden': {
        0: {'size': 165, 'label': 'C0: 정치/사건'},
        2: {'size': 119, 'label': 'C2: 스포츠/일반'},
        3: {'size': 109, 'label': 'C3: 기술 비즈니스'},
        4: {'size': 107, 'label': 'C4: 경제/시장'},
    },
    'SemGraph': {
        0: {'size': 60, 'label': 'C0: 중동/정치'},
        1: {'size': 109, 'label': 'C1: 스포츠 비즈니스'},
        2: {'size': 156, 'label': 'C2: 일반 뉴스'},
        3: {'size': 129, 'label': 'C3: 기술/금융'},
        4: {'size': 46, 'label': 'C4: 스포츠'},
    }
}


def load_and_remap_clusters(
    json_path: str,
    mapping: Dict[int, int]
) -> Dict[int, List[str]]:
    """
    JSON 파일을 로드하고 LaTeX 클러스터 ID로 재매핑

    Args:
        json_path: cluster_words.json 경로
        mapping: JSON ID → LaTeX ID 매핑

    Returns:
        LaTeX ID를 키로 하는 클러스터 딕셔너리
    """
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
) -> List[Tuple[int, int, int, List[str]]]:
    """
    두 클러스터링 간의 단어 흐름 계산

    Args:
        source_clusters: 소스 클러스터 {cluster_id: [words]}
        target_clusters: 타겟 클러스터 {cluster_id: [words]}

    Returns:
        List of (source_id, target_id, flow_count, shared_words)
    """
    flows = []

    for src_id, src_words in source_clusters.items():
        src_set = set(src_words)

        for tgt_id, tgt_words in target_clusters.items():
            tgt_set = set(tgt_words)
            shared = src_set & tgt_set

            if len(shared) > 0:
                flows.append((src_id, tgt_id, len(shared), sorted(list(shared))))

    return flows


def create_sankey_diagram(
    louvain_clusters: Dict[int, List[str]],
    leiden_clusters: Dict[int, List[str]],
    ours_clusters: Dict[int, List[str]],
    output_path: str,
    min_flow: int = 5
):
    """
    Sankey 다이어그램 생성

    Args:
        louvain_clusters: Louvain 클러스터 (LaTeX ID)
        leiden_clusters: Leiden 클러스터 (LaTeX ID)
        ours_clusters: SemGraph 클러스터 (LaTeX ID)
        output_path: 출력 HTML 파일 경로
        min_flow: 최소 흐름 개수 (이보다 작은 flow는 제외)
    """

    print(f"\n[1/3] 단어 흐름 계산")

    # Louvain → Leiden 흐름
    flow_l2l = calculate_word_flow(louvain_clusters, leiden_clusters)
    print(f"  Louvain → Leiden: {len(flow_l2l)} flows")

    # Leiden → SemGraph 흐름
    flow_l2s = calculate_word_flow(leiden_clusters, ours_clusters)
    print(f"  Leiden → SemGraph: {len(flow_l2s)} flows")

    # ====================================================================
    # Sankey 노드 구성
    # ====================================================================

    print(f"\n[2/3] Sankey 노드 구성")

    # 노드 ID 할당
    # Louvain: 0-3 (C0, C2, C3, C4)
    # Leiden: 4-7 (C0, C2, C3, C4)
    # SemGraph: 8-12 (C0, C1, C2, C3, C4)

    node_labels = []
    node_colors = []

    # 클러스터 색상
    cluster_colors = {
        0: '#FF6B6B',  # C0: 정치 - 빨강
        1: '#4ECDC4',  # C1: 스포츠 비즈니스 - 청록
        2: '#95E1D3',  # C2: 스포츠/일반 - 민트
        3: '#F38181',  # C3: 기술 - 주황
        4: '#AA96DA',  # C4: 금융 - 보라
    }

    # Louvain 노드 (0-3)
    louvain_node_map = {}  # latex_id → node_index
    for i, cluster_id in enumerate(sorted(CLUSTER_INFO['Louvain'].keys())):
        info = CLUSTER_INFO['Louvain'][cluster_id]
        node_labels.append(f"Louvain<br>{info['label']}<br>({info['size']} words)")
        node_colors.append(cluster_colors[cluster_id])
        louvain_node_map[cluster_id] = i

    # Leiden 노드 (4-7)
    leiden_node_map = {}
    for i, cluster_id in enumerate(sorted(CLUSTER_INFO['Leiden'].keys())):
        info = CLUSTER_INFO['Leiden'][cluster_id]
        node_labels.append(f"Leiden<br>{info['label']}<br>({info['size']} words)")
        node_colors.append(cluster_colors[cluster_id])
        leiden_node_map[cluster_id] = i + 4

    # SemGraph 노드 (8-12)
    ours_node_map = {}
    for i, cluster_id in enumerate(sorted(CLUSTER_INFO['SemGraph'].keys())):
        info = CLUSTER_INFO['SemGraph'][cluster_id]
        node_labels.append(f"SemGraph<br>{info['label']}<br>({info['size']} words)")
        node_colors.append(cluster_colors[cluster_id])
        ours_node_map[cluster_id] = i + 8

    print(f"  총 노드 수: {len(node_labels)}")
    print(f"    Louvain: {len(louvain_node_map)} 클러스터")
    print(f"    Leiden: {len(leiden_node_map)} 클러스터")
    print(f"    SemGraph: {len(ours_node_map)} 클러스터")

    # ====================================================================
    # Sankey 링크 구성
    # ====================================================================

    print(f"\n[3/3] Sankey 링크 구성 (최소 흐름: {min_flow})")

    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []
    link_labels = []

    # Louvain → Leiden
    for src_id, tgt_id, count, shared_words in flow_l2l:
        if count >= min_flow:
            src_node = louvain_node_map[src_id]
            tgt_node = leiden_node_map[tgt_id]
            link_sources.append(src_node)
            link_targets.append(tgt_node)
            link_values.append(count)
            link_colors.append(f"rgba(150, 150, 150, 0.3)")

            # 호버 텍스트
            sample_words = ', '.join(shared_words[:10])
            if len(shared_words) > 10:
                sample_words += f"... (+{len(shared_words)-10} more)"
            link_labels.append(
                f"Louvain C{src_id} → Leiden C{tgt_id}<br>"
                f"{count} words<br>"
                f"Examples: {sample_words}"
            )

    print(f"  Louvain → Leiden: {len([v for v in link_values])} links (before filter)")

    # Leiden → SemGraph
    offset = len(link_sources)
    for src_id, tgt_id, count, shared_words in flow_l2s:
        if count >= min_flow:
            src_node = leiden_node_map[src_id]
            tgt_node = ours_node_map[tgt_id]
            link_sources.append(src_node)
            link_targets.append(tgt_node)
            link_values.append(count)
            link_colors.append(f"rgba(150, 150, 150, 0.3)")

            # 호버 텍스트
            sample_words = ', '.join(shared_words[:10])
            if len(shared_words) > 10:
                sample_words += f"... (+{len(shared_words)-10} more)"
            link_labels.append(
                f"Leiden C{src_id} → SemGraph C{tgt_id}<br>"
                f"{count} words<br>"
                f"Examples: {sample_words}"
            )

    print(f"  Leiden → SemGraph: {len(link_sources) - offset} links (before filter)")
    print(f"  총 링크 수: {len(link_sources)}")

    # ====================================================================
    # Sankey 다이어그램 생성
    # ====================================================================

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="white", width=2),
            label=node_labels,
            color=node_colors,
            customdata=[label.replace('<br>', '\n') for label in node_labels],
            hovertemplate='%{customdata}<extra></extra>',
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color=link_colors,
            customdata=link_labels,
            hovertemplate='%{customdata}<extra></extra>',
        ),
        arrangement='snap',  # 노드 정렬
        orientation='h',  # 수평 방향
    )])

    fig.update_layout(
        title={
            'text': "Cluster Word Flow: Louvain → Leiden → SemGraph",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font=dict(size=12, family="Arial"),
        height=800,
        width=1400,
        margin=dict(l=20, r=20, t=80, b=20),
    )

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    print(f"\n✅ Sankey 다이어그램 생성 완료!")
    print(f"  저장 위치: {output_path}")
    print(f"  브라우저에서 열어보세요: file://{Path(output_path).resolve()}")

    return output_path


def print_flow_statistics(
    louvain_clusters: Dict[int, List[str]],
    leiden_clusters: Dict[int, List[str]],
    ours_clusters: Dict[int, List[str]]
):
    """흐름 통계 출력"""

    print("\n" + "="*70)
    print("단어 흐름 통계")
    print("="*70)

    # Louvain → Leiden
    print("\n[Louvain → Leiden]")
    flows = calculate_word_flow(louvain_clusters, leiden_clusters)
    flows_sorted = sorted(flows, key=lambda x: x[2], reverse=True)

    for src_id, tgt_id, count, shared_words in flows_sorted[:10]:
        print(f"  C{src_id} → C{tgt_id}: {count:3d} words (예: {', '.join(shared_words[:5])})")

    # Leiden → SemGraph
    print("\n[Leiden → SemGraph]")
    flows = calculate_word_flow(leiden_clusters, ours_clusters)
    flows_sorted = sorted(flows, key=lambda x: x[2], reverse=True)

    for src_id, tgt_id, count, shared_words in flows_sorted[:10]:
        print(f"  C{src_id} → C{tgt_id}: {count:3d} words (예: {', '.join(shared_words[:5])})")

    print("="*70)


def main():
    """메인 실행 함수"""

    print("="*70)
    print("클러스터 Sankey 다이어그램 생성")
    print("="*70)

    # ====================================================================
    # 클러스터 데이터 로드 및 재매핑
    # ====================================================================

    print("\n[단계 1] 클러스터 데이터 로드 및 재매핑")

    base_path = Path("results/rq3_integrated/qualitative")

    louvain_clusters = load_and_remap_clusters(
        base_path / "Louvain" / "cluster_words.json",
        LOUVAIN_MAPPING
    )
    print(f"  ✓ Louvain: {len(louvain_clusters)} 클러스터 (C{sorted(louvain_clusters.keys())})")

    leiden_clusters = load_and_remap_clusters(
        base_path / "Leiden" / "cluster_words.json",
        LEIDEN_MAPPING
    )
    print(f"  ✓ Leiden: {len(leiden_clusters)} 클러스터 (C{sorted(leiden_clusters.keys())})")

    ours_clusters = load_and_remap_clusters(
        base_path / "Ours" / "cluster_words.json",
        OURS_MAPPING
    )
    print(f"  ✓ SemGraph: {len(ours_clusters)} 클러스터 (C{sorted(ours_clusters.keys())})")

    # ====================================================================
    # 흐름 통계 출력
    # ====================================================================

    print_flow_statistics(louvain_clusters, leiden_clusters, ours_clusters)

    # ====================================================================
    # Sankey 다이어그램 생성
    # ====================================================================

    print("\n[단계 2] Sankey 다이어그램 생성")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/rq3_integrated/visualizations/cluster_sankey_{timestamp}.html"

    create_sankey_diagram(
        louvain_clusters=louvain_clusters,
        leiden_clusters=leiden_clusters,
        ours_clusters=ours_clusters,
        output_path=output_path,
        min_flow=5  # 최소 5개 이상의 공유 단어가 있는 흐름만 표시
    )

    print("\n" + "="*70)
    print("✅ 모든 작업 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
