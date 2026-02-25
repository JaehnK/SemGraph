#!/usr/bin/env python3
"""
개선된 Sankey 다이어그램

노드 위치를 수동으로 지정하여 가독성을 극대화
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import plotly.graph_objects as go


# ====================================================================
# JSON → 통일된 주제별 클러스터 ID 매핑
# ====================================================================
# 통일된 주제:
# C0: Politics & International (정치/국제)
# C1: Sports (스포츠)
# C2: Technology & Business (기술/비즈니스)
# C3: Economy & Finance (경제/금융)
# C4: General News (일반 뉴스 - Ours only)

LOUVAIN_MAPPING = {
    1: 0,  # iraq/president → C0 (Politics, 170)
    0: 1,  # ap/world/game → C1 (Sports, 122)
    3: 2,  # microsoft/company → C2 (Technology, 101)
    2: 3,  # reuters/oil → C3 (Finance, 107)
}

LEIDEN_MAPPING = {
    0: 0,  # iraq/president → C0 (Politics, 165)
    1: 1,  # ap/world/game → C1 (Sports, 119)
    2: 2,  # microsoft/company → C2 (Technology, 109)
    3: 3,  # reuters/oil → C3 (Finance, 107)
}

OURS_MAPPING = {
    0: 0,  # minister/baghdad → C0 (Politics, 60)
    4: 1,  # team/victory → C1 (Sports, 46)
    3: 2,  # microsoft/service → C2 (Technology, 129)
    2: 3,  # reuters/oil → C3 (Finance, 156)
    1: 4,  # company/game → C4 (General News, 109)
}


# ====================================================================
# 클러스터 정보
# ====================================================================

CLUSTER_INFO = {
    'Louvain': {
        0: {'label': 'Politics & International', 'size': 170},
        1: {'label': 'Sports', 'size': 122},
        2: {'label': 'Technology & Business', 'size': 101},
        3: {'label': 'Economy & Finance', 'size': 107},
    },
    'Leiden': {
        0: {'label': 'Politics & International', 'size': 165},
        1: {'label': 'Sports', 'size': 119},
        2: {'label': 'Technology & Business', 'size': 109},
        3: {'label': 'Economy & Finance', 'size': 107},
    },
    'SemGraph': {
        0: {'label': 'Politics & International', 'size': 60},
        1: {'label': 'Sports', 'size': 46},
        2: {'label': 'Technology & Business', 'size': 129},
        3: {'label': 'Economy & Finance', 'size': 156},
        4: {'label': 'General News', 'size': 109},
    }
}


# 클러스터 색상 (주제별 통일)
CLUSTER_COLORS = {
    0: 'rgba(231, 76, 60, 0.8)',   # C0: Politics - Red
    1: 'rgba(52, 152, 219, 0.8)',  # C1: Sports - Blue
    2: 'rgba(243, 156, 18, 0.8)',  # C2: Technology - Orange
    3: 'rgba(46, 204, 113, 0.8)',  # C3: Economy - Green
    4: 'rgba(155, 89, 182, 0.8)',  # C4: General News - Purple
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
) -> List[Tuple[int, int, int, List[str]]]:
    """두 클러스터링 간의 단어 흐름 계산"""
    flows = []

    for src_id, src_words in source_clusters.items():
        src_set = set(src_words)

        for tgt_id, tgt_words in target_clusters.items():
            tgt_set = set(tgt_words)
            shared = src_set & tgt_set

            if len(shared) > 0:
                flows.append((src_id, tgt_id, len(shared), sorted(list(shared))))

    return flows


def create_improved_sankey(
    louvain_clusters: Dict[int, List[str]],
    leiden_clusters: Dict[int, List[str]],
    ours_clusters: Dict[int, List[str]],
    output_path: str,
    min_flow: int = 5
):
    """개선된 Sankey 다이어그램 생성"""

    print(f"\n[1/3] 단어 흐름 계산")

    # Louvain → Leiden 흐름
    flow_l2l = calculate_word_flow(louvain_clusters, leiden_clusters)
    print(f"  Louvain → Leiden: {len(flow_l2l)} flows")

    # Leiden → SemGraph 흐름
    flow_l2s = calculate_word_flow(leiden_clusters, ours_clusters)
    print(f"  Leiden → SemGraph: {len(flow_l2s)} flows")

    # ====================================================================
    # 노드 정의 (수동 배치)
    # ====================================================================

    print(f"\n[2/3] Sankey 노드 구성")

    # 노드 인덱스 할당
    # Louvain: 0-3 (C0, C2, C3, C4)
    # Leiden: 4-7 (C0, C2, C3, C4)
    # SemGraph: 8-12 (C0, C1, C2, C3, C4)

    node_labels = []
    node_colors = []
    node_x = []  # x 좌표 (0~1)
    node_y = []  # y 좌표 (0~1)

    # 주제별 수평 정렬을 위한 y 좌표 (같은 주제는 같은 높이)
    topic_y_positions = {
        0: 0.85,  # C0: Politics
        1: 0.65,  # C1: Sports
        2: 0.45,  # C2: Technology
        3: 0.25,  # C3: Economy
        4: 0.05,  # C4: General News (Ours only)
    }

    # Louvain 노드 (왼쪽) - C0, C1, C2, C3
    louvain_node_map = {}
    for i, cluster_id in enumerate([0, 1, 2, 3]):
        info = CLUSTER_INFO['Louvain'][cluster_id]
        node_labels.append(
            f"<b style='font-size:18px'>Louvain C{cluster_id}</b><br>"
            f"<span style='font-size:15px'>{info['label']}</span><br>"
            f"<span style='font-size:13px'>({info['size']} words)</span>"
        )
        node_colors.append(CLUSTER_COLORS[cluster_id])
        node_x.append(0.01)
        node_y.append(topic_y_positions[cluster_id])
        louvain_node_map[cluster_id] = i

    # Leiden 노드 (중간) - C0, C1, C2, C3
    leiden_node_map = {}
    for i, cluster_id in enumerate([0, 1, 2, 3]):
        info = CLUSTER_INFO['Leiden'][cluster_id]
        node_labels.append(
            f"<b style='font-size:18px'>Leiden C{cluster_id}</b><br>"
            f"<span style='font-size:15px'>{info['label']}</span><br>"
            f"<span style='font-size:13px'>({info['size']} words)</span>"
        )
        node_colors.append(CLUSTER_COLORS[cluster_id])
        node_x.append(0.5)
        node_y.append(topic_y_positions[cluster_id])
        leiden_node_map[cluster_id] = i + 4

    # SemGraph 노드 (오른쪽) - C0, C1, C2, C3, C4 (5개)
    ours_node_map = {}
    for i, cluster_id in enumerate([0, 1, 2, 3, 4]):
        info = CLUSTER_INFO['SemGraph'][cluster_id]
        node_labels.append(
            f"<b style='font-size:18px'>SemGraph C{cluster_id}</b><br>"
            f"<span style='font-size:15px'>{info['label']}</span><br>"
            f"<span style='font-size:13px'>({info['size']} words)</span>"
        )
        node_colors.append(CLUSTER_COLORS[cluster_id])
        node_x.append(0.99)
        node_y.append(topic_y_positions[cluster_id])
        ours_node_map[cluster_id] = i + 8

    print(f"  총 노드 수: {len(node_labels)}")

    # ====================================================================
    # 링크 정의
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

            # 링크 색상 (소스 클러스터 색상 기반, 투명도 추가)
            base_color = CLUSTER_COLORS[src_id]
            link_colors.append(base_color.replace('0.8', '0.3'))

            # 호버 텍스트
            sample_words = ', '.join(shared_words[:10])
            if len(shared_words) > 10:
                sample_words += f" ... (+{len(shared_words)-10})"
            link_labels.append(
                f"<b>Louvain C{src_id} → Leiden C{tgt_id}</b><br>"
                f"Shared words: {count}<br>"
                f"Examples: {sample_words}"
            )

    # Leiden → SemGraph
    for src_id, tgt_id, count, shared_words in flow_l2s:
        if count >= min_flow:
            src_node = leiden_node_map[src_id]
            tgt_node = ours_node_map[tgt_id]
            link_sources.append(src_node)
            link_targets.append(tgt_node)
            link_values.append(count)

            base_color = CLUSTER_COLORS[src_id]
            link_colors.append(base_color.replace('0.8', '0.3'))

            sample_words = ', '.join(shared_words[:10])
            if len(shared_words) > 10:
                sample_words += f" ... (+{len(shared_words)-10})"
            link_labels.append(
                f"<b>Leiden C{src_id} → SemGraph C{tgt_id}</b><br>"
                f"Shared words: {count}<br>"
                f"Examples: {sample_words}"
            )

    print(f"  총 링크 수: {len(link_sources)}")

    # ====================================================================
    # Sankey 다이어그램 생성
    # ====================================================================

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=40,  # 노드 간 간격 증가
            thickness=30,  # 노드 두께 증가
            line=dict(color="white", width=3),  # 테두리 두께 증가
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            customdata=node_labels,
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
    )])

    fig.update_layout(
        title={
            'text': "<b>Cluster Evolution: Louvain → Leiden → SemGraph</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2C3E50'}  # 제목 크기 증가
        },
        font=dict(size=14, family="Arial"),  # 기본 폰트 크기 증가
        height=1000,  # 높이 증가
        width=1800,   # 너비 증가
        margin=dict(l=20, r=20, t=100, b=40),  # 여백 증가
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    print(f"\n✅ Sankey 다이어그램 생성 완료!")
    print(f"  저장 위치: {output_path}")
    print(f"  브라우저에서 열어보세요: file://{Path(output_path).resolve()}")

    # PNG로도 저장 (kaleido 필요)
    try:
        png_path = output_path.replace('.html', '.png')
        fig.write_image(png_path, width=1600, height=900, scale=2)
        print(f"  PNG 저장: {png_path}")
    except Exception as e:
        print(f"  PNG 저장 실패 (kaleido 미설치): {e}")

    return output_path


def main():
    """메인 실행 함수"""

    print("="*70)
    print("개선된 Sankey 다이어그램 생성")
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
    print(f"  ✓ Louvain: {len(louvain_clusters)} 클러스터 (C0-C3)")

    leiden_clusters = load_and_remap_clusters(
        base_path / "Leiden" / "cluster_words.json",
        LEIDEN_MAPPING
    )
    print(f"  ✓ Leiden: {len(leiden_clusters)} 클러스터 (C0-C3)")

    ours_clusters = load_and_remap_clusters(
        base_path / "Ours" / "cluster_words.json",
        OURS_MAPPING
    )
    print(f"  ✓ SemGraph: {len(ours_clusters)} 클러스터 (C0-C4)")

    # ====================================================================
    # 다이어그램 생성
    # ====================================================================

    print("\n[단계 2] Sankey 다이어그램 생성")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/rq3_integrated/visualizations/cluster_sankey_improved_{timestamp}.html"

    create_improved_sankey(
        louvain_clusters=louvain_clusters,
        leiden_clusters=leiden_clusters,
        ours_clusters=ours_clusters,
        output_path=output_path,
        min_flow=5
    )

    print("\n" + "="*70)
    print("✅ 모든 작업 완료!")
    print("="*70)


if __name__ == "__main__":
    main()
