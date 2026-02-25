#!/usr/bin/env python3
"""
클러스터 정보를 기반으로 의미연결망 시각화

cluster_words.json 파일에서 클러스터 정보를 읽어서
WordGraph를 생성하고 Fruchterman-Reingold 레이아웃으로 시각화합니다.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.services.Document.DocumentService import DocumentService
from core.services.Graph.GraphService import GraphService
import matplotlib.pyplot as plt
import networkx as nx


def load_cluster_info(cluster_json_path: str) -> Dict[str, List[str]]:
    """클러스터 정보 로드"""
    with open(cluster_json_path, 'r') as f:
        cluster_data = json.load(f)
    return cluster_data


def build_word_graph_from_data(
    csv_path: str,
    text_column: str,
    num_documents: int,
    top_n_words: int,
    edge_weight_threshold: int = 5
):
    """데이터에서 WordGraph 생성"""

    # 문서 로드
    df = pd.read_csv(csv_path)
    documents = df[text_column].dropna().head(num_documents).tolist()

    # DocumentService로 전처리
    doc_service = DocumentService()
    doc_service.create_sentence_list(documents=documents)

    # GraphService로 공출현 그래프 생성
    graph_service = GraphService(doc_service)
    word_graph = graph_service.build_complete_graph(
        top_n=top_n_words,
        exclude_stopwords=True,
        max_length=-1
    )

    # 엣지 필터링
    if edge_weight_threshold > 0:
        word_graph.filter_edges_by_weight(edge_weight_threshold)

    return word_graph, doc_service


def create_cluster_labels_from_json(
    word_graph,
    cluster_data: Dict[str, List[str]]
) -> np.ndarray:
    """
    cluster_words.json의 정보를 바탕으로 클러스터 레이블 생성

    Args:
        word_graph: WordGraph 객체
        cluster_data: {cluster_id: [word1, word2, ...]} 형태의 딕셔너리

    Returns:
        cluster_labels: (num_nodes,) 형태의 클러스터 레이블
    """
    # WordGraph의 단어 리스트
    vocab = [word.content for word in word_graph.words]

    # 단어 -> 인덱스 매핑
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # 클러스터 레이블 초기화 (-1: 미할당)
    cluster_labels = np.full(len(vocab), -1, dtype=np.int32)

    # JSON에서 각 클러스터의 단어를 읽어서 레이블 할당
    for cluster_id, words in cluster_data.items():
        cluster_id = int(cluster_id)
        for word in words:
            if word in word_to_idx:
                idx = word_to_idx[word]
                cluster_labels[idx] = cluster_id

    # 미할당된 노드 확인
    unassigned_count = np.sum(cluster_labels == -1)
    if unassigned_count > 0:
        print(f"⚠️  Warning: {unassigned_count} nodes were not assigned to any cluster")
        # 미할당 노드를 별도 클러스터로 처리
        max_cluster_id = max([int(cid) for cid in cluster_data.keys()])
        cluster_labels[cluster_labels == -1] = max_cluster_id + 1

    return cluster_labels


def visualize_network(
    word_graph,
    cluster_labels: np.ndarray,
    output_path: str,
    title: str = "Semantic Network with Clusters",
    top_k: int = 500,
    min_weight: float = 5.0,
    figsize: tuple = (16, 12),
    dpi: int = 300
):
    """GraphService 기반 빠른 네트워크 시각화 (클러스터 색상 포함)"""

    # NetworkX 그래프로 변환
    G = word_graph.export_to_networkx(include_weights=True)

    # 상위 k개 노드 선택
    selected_nodes = list(range(min(top_k, len(word_graph.words))))
    node_labels = {i: word_graph.words[i].content for i in selected_nodes}

    # 서브그래프 생성
    G_sub = G.subgraph(selected_nodes).copy()

    # 최소 가중치 필터링
    if min_weight is not None:
        edges_to_remove = [(u, v) for u, v, d in G_sub.edges(data=True)
                            if d.get('weight', 0) < min_weight]
        G_sub.remove_edges_from(edges_to_remove)

    # 시각화
    plt.figure(figsize=figsize)

    # 레이아웃 계산 - Kamada-Kawai layout (균일한 엣지 길이와 클러스터 구조)
    print(f"  레이아웃 계산 중... (노드: {G_sub.number_of_nodes()}, 엣지: {G_sub.number_of_edges()})")

    # Kamada-Kawai layout - 스트레스 최소화 기반
    # Spring보다 깔끔하고 대칭적이며, Spectral보다 희소 그래프에 안정적
    pos = nx.kamada_kawai_layout(
        G_sub,
        weight='weight'  # 엣지 가중치 반영
    )

    # 클러스터별 색상
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    node_colors = [colors[int(cluster_labels[i]) % len(colors)] for i in selected_nodes]

    # 노드 크기 (작게 조정하여 겹침 방지)
    node_sizes = [np.log1p(word_graph.words[i].freq) * 50 for i in selected_nodes]

    # 엣지 먼저 그리기 (뒤쪽 레이어)
    if G_sub.edges():
        edge_weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = np.array(edge_weights) / max_weight * 1.5
        nx.draw_networkx_edges(G_sub, pos, width=edge_widths, alpha=0.4, edge_color='#444444')

    # 노드 그리기 (동심원 스타일)
    nx.draw_networkx_nodes(G_sub, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          node_shape='o',
                          alpha=0.85,
                          linewidths=1.5,
                          edgecolors='white')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  저장 완료: {output_path}")
    plt.close()

    return output_path


def visualize_single_model(
    model_name: str,
    cluster_json_path: str,
    output_path: str,
    csv_path: str = "data/ag_news.csv",
    text_column: str = "text",
    num_documents: int = 10000,
    top_n_words: int = 500,
    edge_weight_threshold: int = 5,
    top_k: int = 500,
    min_weight: float = 5.0,
    figsize: tuple = (16, 12),
    dpi: int = 300
):
    """단일 모델의 시각화 실행"""

    print(f"\n{'='*70}")
    print(f"{model_name} 시각화")
    print(f"{'='*70}")

    # 1. 클러스터 정보 로드
    print(f"\n[1/4] 클러스터 정보 로드: {cluster_json_path}")
    cluster_data = load_cluster_info(cluster_json_path)
    print(f"  - 클러스터 개수: {len(cluster_data)}")
    for cluster_id, words in cluster_data.items():
        print(f"  - Cluster {cluster_id}: {len(words)} words")

    # 2. WordGraph 생성
    print(f"\n[2/4] WordGraph 생성")
    print(f"  - 데이터: {csv_path}")
    print(f"  - 문서 개수: {num_documents}")
    print(f"  - 상위 단어 개수: {top_n_words}")
    print(f"  - 엣지 가중치 임계값: {edge_weight_threshold}")

    word_graph, doc_service = build_word_graph_from_data(
        csv_path=csv_path,
        text_column=text_column,
        num_documents=num_documents,
        top_n_words=top_n_words,
        edge_weight_threshold=edge_weight_threshold
    )

    print(f"  - 생성된 그래프: {word_graph.num_nodes} nodes, {word_graph.num_edges} edges")

    # 3. 클러스터 레이블 생성
    print(f"\n[3/4] 클러스터 레이블 생성")
    cluster_labels = create_cluster_labels_from_json(word_graph, cluster_data)
    print(f"  - 레이블 분포:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    Cluster {label}: {count} nodes")

    # 4. 네트워크 시각화
    print(f"\n[4/4] 네트워크 시각화")
    print(f"  - 출력 경로: {output_path}")
    print(f"  - 레이아웃: Kamada-Kawai Layout (균일한 엣지 길이)")
    print(f"  - 최소 엣지 가중치: {min_weight}")

    # 출력 디렉토리 생성
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    network_path = visualize_network(
        word_graph=word_graph,
        cluster_labels=cluster_labels,
        output_path=output_path,
        title=f"Semantic Network with {model_name} Clusters",
        top_k=top_k,
        min_weight=min_weight,
        figsize=figsize,
        dpi=dpi
    )

    print(f"\n✅ {model_name} 시각화 완료!")
    print(f"  - 저장 위치: {network_path}")

    return network_path


def main():
    """메인 실행 함수 - 모든 모델 시각화"""

    # ====================================================================
    # 공통 설정
    # ====================================================================

    CSV_PATH = "data/ag_news.csv"
    TEXT_COLUMN = "text"
    NUM_DOCUMENTS = 10000
    TOP_N_WORDS = 500
    EDGE_WEIGHT_THRESHOLD = 5
    TOP_K = 500
    MIN_WEIGHT = 5.0
    FIGSIZE = (16, 12)
    DPI = 300

    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ====================================================================
    # 각 모델별 시각화
    # ====================================================================

    models = [
        {
            'name': 'Ours',
            'cluster_json': 'results/rq3_integrated/qualitative/Ours/cluster_words.json',
            'output': f'results/rq3_integrated/visualizations/semantic_network_ours_{timestamp}.png'
        },
        {
            'name': 'Louvain',
            'cluster_json': 'results/rq3_integrated/qualitative/Louvain/cluster_words.json',
            'output': f'results/rq3_integrated/visualizations/semantic_network_louvain_{timestamp}.png'
        },
        {
            'name': 'Leiden',
            'cluster_json': 'results/rq3_integrated/qualitative/Leiden/cluster_words.json',
            'output': f'results/rq3_integrated/visualizations/semantic_network_leiden_{timestamp}.png'
        }
    ]

    print("=" * 70)
    print("의미연결망 시각화 (전체 모델)")
    print("=" * 70)
    print(f"모델 수: {len(models)}")

    results = []
    for model in models:
        result = visualize_single_model(
            model_name=model['name'],
            cluster_json_path=model['cluster_json'],
            output_path=model['output'],
            csv_path=CSV_PATH,
            text_column=TEXT_COLUMN,
            num_documents=NUM_DOCUMENTS,
            top_n_words=TOP_N_WORDS,
            edge_weight_threshold=EDGE_WEIGHT_THRESHOLD,
            top_k=TOP_K,
            min_weight=MIN_WEIGHT,
            figsize=FIGSIZE,
            dpi=DPI
        )
        results.append(result)

    print("\n" + "=" * 70)
    print("✅ 전체 시각화 완료!")
    print("=" * 70)
    for i, result in enumerate(results):
        print(f"  [{i+1}] {result}")
    print("=" * 70)


if __name__ == "__main__":
    main()
