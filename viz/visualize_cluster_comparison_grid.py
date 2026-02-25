#!/usr/bin/env python3
"""
클러스터 비교 그리드 시각화

세 가지 방법(Louvain, Leiden, SemGraph)의 대표 클러스터를
네트워크 형태로 3x3 그리드에 시각화합니다.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.services.Document.DocumentService import DocumentService
from core.services.Graph.GraphService import GraphService


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
        0: {'label': 'C0: 정치/뉴스', 'size': 170},
        2: {'label': 'C2: 스포츠/일반', 'size': 122},
        3: {'label': 'C3: 기술 비즈니스', 'size': 101},
        4: {'label': 'C4: 금융 보도', 'size': 107},
    },
    'Leiden': {
        0: {'label': 'C0: 정치/사건', 'size': 165},
        2: {'label': 'C2: 스포츠/일반', 'size': 119},
        3: {'label': 'C3: 기술 비즈니스', 'size': 109},
        4: {'label': 'C4: 경제/시장', 'size': 107},
    },
    'SemGraph': {
        0: {'label': 'C0: 중동/정치', 'size': 60},
        1: {'label': 'C1: 스포츠 비즈니스', 'size': 109},
        2: {'label': 'C2: 일반 뉴스', 'size': 156},
        3: {'label': 'C3: 기술/금융', 'size': 129},
        4: {'label': 'C4: 스포츠', 'size': 46},
    }
}


# 각 방법에서 비교할 대표 클러스터 선택
SELECTED_CLUSTERS = {
    'Louvain': [0, 2, 3],   # C0(정치), C2(스포츠), C3(기술)
    'Leiden': [0, 2, 3],    # C0(정치), C2(스포츠), C3(기술)
    'SemGraph': [0, 4, 3],  # C0(중동정치), C4(스포츠), C3(기술)
}


# 클러스터 색상
CLUSTER_COLORS = {
    0: '#FF6B6B',  # C0: 정치 - 빨강
    1: '#4ECDC4',  # C1: 스포츠 비즈니스 - 청록
    2: '#95E1D3',  # C2: 스포츠/일반 - 민트
    3: '#F38181',  # C3: 기술 - 주황/핑크
    4: '#AA96DA',  # C4: 금융/스포츠 - 보라
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


def build_word_graph(
    csv_path: str,
    text_column: str,
    num_documents: int,
    top_n_words: int,
    edge_weight_threshold: int = 5
):
    """데이터에서 WordGraph 생성"""

    print(f"  데이터 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    documents = df[text_column].dropna().head(num_documents).tolist()
    print(f"  문서 수: {len(documents)}")

    # DocumentService로 전처리
    doc_service = DocumentService()
    doc_service.create_sentence_list(documents=documents)
    print(f"  전처리 완료")

    # GraphService로 공출현 그래프 생성
    graph_service = GraphService(doc_service)
    word_graph = graph_service.build_complete_graph(
        top_n=top_n_words,
        exclude_stopwords=True,
        max_length=-1
    )
    print(f"  그래프 생성: {word_graph.num_nodes} nodes, {word_graph.num_edges} edges")

    # 엣지 필터링
    if edge_weight_threshold > 0:
        word_graph.filter_edges_by_weight(edge_weight_threshold)
        print(f"  엣지 필터링 (>={edge_weight_threshold}): {word_graph.num_edges} edges")

    return word_graph


def create_cluster_subgraph(
    word_graph,
    cluster_words: List[str],
    top_k: int = 15,
    min_edge_weight: float = 5.0
) -> nx.Graph:
    """
    클러스터 내 단어들의 서브그래프 생성

    Args:
        word_graph: WordGraph 객체
        cluster_words: 클러스터에 속한 단어 리스트
        top_k: 상위 k개 단어만 선택
        min_edge_weight: 최소 엣지 가중치

    Returns:
        NetworkX 그래프
    """
    # WordGraph를 NetworkX로 변환
    G_full = word_graph.export_to_networkx(include_weights=True)

    # 단어 -> 인덱스 매핑
    word_to_idx = {word.content: idx for idx, word in enumerate(word_graph.words)}

    # 클러스터 단어의 인덱스
    cluster_indices = [word_to_idx[w] for w in cluster_words if w in word_to_idx]

    # 빈도순으로 정렬하여 상위 k개 선택
    word_freq = [(idx, word_graph.words[idx].freq) for idx in cluster_indices]
    word_freq_sorted = sorted(word_freq, key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in word_freq_sorted[:top_k]]

    # 서브그래프 생성
    subG = G_full.subgraph(top_indices).copy()

    # 최소 가중치 필터링
    edges_to_remove = [(u, v) for u, v, d in subG.edges(data=True)
                       if d.get('weight', 0) < min_edge_weight]
    subG.remove_edges_from(edges_to_remove)

    # 노드에 단어 레이블 추가
    nx.set_node_attributes(subG, {
        idx: word_graph.words[idx].content
        for idx in top_indices
    }, 'label')

    # 노드에 빈도 추가
    nx.set_node_attributes(subG, {
        idx: word_graph.words[idx].freq
        for idx in top_indices
    }, 'freq')

    return subG


def draw_cluster_network(
    ax,
    subG: nx.Graph,
    cluster_id: int,
    cluster_size: int,
    title: str,
    top_labels: int = 5
):
    """
    단일 클러스터의 네트워크 시각화

    Args:
        ax: matplotlib axes
        subG: NetworkX 서브그래프
        cluster_id: 클러스터 ID (색상 결정용)
        cluster_size: 전체 클러스터 크기
        title: 서브플롯 제목
        top_labels: 상위 몇 개 노드에만 레이블 표시
    """

    if subG.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                fontsize=12, color='gray')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')
        return

    # 레이아웃 계산
    if subG.number_of_edges() > 0:
        pos = nx.spring_layout(subG, k=0.9, iterations=50, seed=42, weight='weight')
    else:
        pos = nx.circular_layout(subG)

    # 중심성 계산 (빈도 기반)
    node_freq = nx.get_node_attributes(subG, 'freq')
    max_freq = max(node_freq.values()) if node_freq else 1

    # 노드 크기 (빈도 기반)
    node_sizes = [300 + 700 * (node_freq.get(node, 1) / max_freq)
                  for node in subG.nodes()]

    # 엣지 그리기
    if subG.number_of_edges() > 0:
        edge_weights = [subG[u][v]['weight'] for u, v in subG.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + 3 * (w / max_weight) for w in edge_weights]

        nx.draw_networkx_edges(
            subG, pos, ax=ax,
            width=edge_widths,
            alpha=0.3,
            edge_color='gray'
        )

    # 노드 그리기
    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        node_size=node_sizes,
        node_color=CLUSTER_COLORS[cluster_id],
        alpha=0.8,
        linewidths=2,
        edgecolors='white'
    )

    # 레이블 (상위 k개만)
    node_labels = nx.get_node_attributes(subG, 'label')
    freq_sorted = sorted(node_freq.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in freq_sorted[:top_labels]]

    labels_to_draw = {node: node_labels[node] for node in top_nodes
                      if node in node_labels}

    nx.draw_networkx_labels(
        subG, pos, labels_to_draw, ax=ax,
        font_size=9,
        font_weight='bold',
        font_color='#333333'
    )

    # 제목
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # 클러스터 크기 표시
    ax.text(0.5, -0.12, f"n={cluster_size} words",
            transform=ax.transAxes, ha='center',
            fontsize=9, style='italic', color='#666666')

    ax.axis('off')
    ax.set_aspect('equal')


def create_cluster_comparison_grid(
    louvain_clusters: Dict[int, List[str]],
    leiden_clusters: Dict[int, List[str]],
    semgraph_clusters: Dict[int, List[str]],
    word_graph,
    output_path: str,
    top_k: int = 15,
    min_edge_weight: float = 5.0
):
    """
    3x4 그리드 비교 시각화 생성

    행: 클러스터 비교 (정치, 스포츠, 기술)
    열: 방법 (Louvain, Leiden, SemGraph)
    """

    print(f"\n[1/2] 서브그래프 생성")

    # Figure 생성
    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                  top=0.95, bottom=0.05, left=0.05, right=0.95)

    # ====================================================================
    # 헤더 (방법 이름)
    # ====================================================================

    methods = ['Louvain', 'Leiden', 'SemGraph']
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, method,
                fontsize=22, fontweight='bold', ha='center', va='center',
                color='#2C3E50')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # ====================================================================
    # 각 클러스터 비교 (3개 행)
    # ====================================================================

    cluster_data = [
        ('Louvain', louvain_clusters, SELECTED_CLUSTERS['Louvain']),
        ('Leiden', leiden_clusters, SELECTED_CLUSTERS['Leiden']),
        ('SemGraph', semgraph_clusters, SELECTED_CLUSTERS['SemGraph']),
    ]

    # 행별로 비교할 주제 (정치, 스포츠, 기술)
    comparison_themes = ['Political', 'Sports', 'Technology']

    for row_idx, theme in enumerate(comparison_themes):
        print(f"\n  Row {row_idx + 1}: {theme} Clusters")

        for col_idx, (method_name, clusters, selected_ids) in enumerate(cluster_data):
            cluster_id = selected_ids[row_idx]
            cluster_words = clusters[cluster_id]

            print(f"    {method_name} C{cluster_id}: {len(cluster_words)} words")

            # 서브그래프 생성
            subG = create_cluster_subgraph(
                word_graph=word_graph,
                cluster_words=cluster_words,
                top_k=top_k,
                min_edge_weight=min_edge_weight
            )

            print(f"      → Subgraph: {subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges")

            # 시각화
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])

            cluster_info = CLUSTER_INFO[method_name][cluster_id]
            title = cluster_info['label']

            draw_cluster_network(
                ax=ax,
                subG=subG,
                cluster_id=cluster_id,
                cluster_size=len(cluster_words),
                title=title,
                top_labels=5
            )

    # ====================================================================
    # 전체 제목
    # ====================================================================

    fig.suptitle(
        'Cluster Comparison: Network Visualization',
        fontsize=24, fontweight='bold', y=0.98,
        color='#2C3E50'
    )

    # ====================================================================
    # 저장
    # ====================================================================

    print(f"\n[2/2] 저장 중...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ 저장 완료: {output_path}")

    plt.close()

    return output_path


def main():
    """메인 실행 함수"""

    print("="*70)
    print("클러스터 비교 그리드 시각화")
    print("="*70)

    # ====================================================================
    # 설정
    # ====================================================================

    CSV_PATH = "data/ag_news.csv"
    TEXT_COLUMN = "text"
    NUM_DOCUMENTS = 10000
    TOP_N_WORDS = 500
    EDGE_WEIGHT_THRESHOLD = 5

    TOP_K_NODES = 15  # 각 클러스터에서 표시할 노드 수
    MIN_EDGE_WEIGHT = 5.0  # 서브그래프 최소 엣지 가중치

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/rq3_integrated/visualizations/cluster_comparison_grid_{timestamp}.png"

    # ====================================================================
    # 1. 클러스터 데이터 로드
    # ====================================================================

    print("\n[단계 1/3] 클러스터 데이터 로드")

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

    semgraph_clusters = load_and_remap_clusters(
        base_path / "Ours" / "cluster_words.json",
        OURS_MAPPING
    )
    print(f"  ✓ SemGraph: {len(semgraph_clusters)} 클러스터")

    # ====================================================================
    # 2. WordGraph 생성
    # ====================================================================

    print("\n[단계 2/3] WordGraph 생성")

    word_graph = build_word_graph(
        csv_path=CSV_PATH,
        text_column=TEXT_COLUMN,
        num_documents=NUM_DOCUMENTS,
        top_n_words=TOP_N_WORDS,
        edge_weight_threshold=EDGE_WEIGHT_THRESHOLD
    )

    # ====================================================================
    # 3. 그리드 시각화 생성
    # ====================================================================

    print("\n[단계 3/3] 그리드 시각화 생성")

    result_path = create_cluster_comparison_grid(
        louvain_clusters=louvain_clusters,
        leiden_clusters=leiden_clusters,
        semgraph_clusters=semgraph_clusters,
        word_graph=word_graph,
        output_path=output_path,
        top_k=TOP_K_NODES,
        min_edge_weight=MIN_EDGE_WEIGHT
    )

    print("\n" + "="*70)
    print("✅ 모든 작업 완료!")
    print("="*70)
    print(f"  저장 위치: {result_path}")
    print("="*70)


if __name__ == "__main__":
    main()
