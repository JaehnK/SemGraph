#!/usr/bin/env python3
"""
RQ3 질적 분석: 클러스터 품질 비교 (단어 리스트 + 워드 클라우드)

목적:
1. Louvain, Leiden, Ours 각 방법의 클러스터별 단어 리스트 추출
2. 워드 클라우드 시각화 생성
3. 클러스터별 NPMI, 크기, 대표 단어 분석
4. 질적 비교를 위한 자료 생성

출력:
- 각 모델별 클러스터 단어 리스트 (JSON, TXT)
- 워드 클라우드 이미지 (클러스터별)
- 클러스터 품질 요약 테이블
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.services.GRACE import GRACEConfig, GRACEPipeline
from core.services.GRACE.TraditionalGraphClusteringService import TraditionalGraphClusteringService
from core.services.Metric import MetricsService
from core.services.Document.DocumentService import DocumentService
from core.services.Graph.GraphService import GraphService
from entities import WordGraph


# ============================================================
# 실험 설정
# ============================================================

EXPERIMENT_CONFIG = {
    # 데이터
    'csv_path': 'data/ag_news.csv',
    'num_documents': 10000,
    'text_column': 'text',

    # 그래프
    'top_n_words': 500,
    'exclude_stopwords': True,
    'edge_weight_threshold': 5,

    # 임베딩
    'embedding_method': 'attention',

    # 최적 하이퍼파라미터
    'optimal_mask_rate': 0.3,
    'optimal_epochs': 1000,
    'optimal_pca_dim': 256,

    # GraphMAE
    'encoder_type': 'gat',
    'decoder_type': 'gat',

    # 클러스터링
    'clustering_method': 'kmeans',
    'min_clusters': 5,
    'max_clusters': 20,

    # 재현성 (단일 시드로 대표 결과 생성)
    'random_seed': 42,

    # 전통적 방법 설정
    'louvain_resolution': 1.0,
    'leiden_resolution': 1.0,

    # 시각화
    'top_words_per_cluster': 20,  # 클러스터당 상위 단어 개수
    'wordcloud_words': 50,  # 워드클라우드에 표시할 단어 개수

    # 출력
    'output_dir': 'results/rq3_qualitative_analysis',
    'verbose': False,
}


# ============================================================
# 유틸리티 함수
# ============================================================

def set_all_seeds(seed: int):
    """모든 랜덤 시드 고정"""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_optimal_config() -> Dict[str, Any]:
    """RQ2 결과에서 최적 설정 로드"""
    optimal_config_path = Path("results/rq2_ablation/optimal_config.json")

    if optimal_config_path.exists():
        with open(optimal_config_path, 'r') as f:
            return json.load(f)
    else:
        return {
            'mask_rate': EXPERIMENT_CONFIG['optimal_mask_rate'],
            'epochs': EXPERIMENT_CONFIG['optimal_epochs'],
            'pca_dim': EXPERIMENT_CONFIG['optimal_pca_dim'],
        }


# ============================================================
# 그래프 구축
# ============================================================

def build_word_graph() -> Tuple[WordGraph, DocumentService, List[str]]:
    """단어 그래프 구축 및 단어 리스트 반환"""

    print(f"공출현 그래프 구축 중...")

    set_all_seeds(EXPERIMENT_CONFIG['random_seed'])

    # 문서 로드
    df = pd.read_csv(EXPERIMENT_CONFIG['csv_path'])
    documents = df[EXPERIMENT_CONFIG['text_column']].dropna().head(
        EXPERIMENT_CONFIG['num_documents']
    ).tolist()

    # DocumentService로 전처리
    doc_service = DocumentService()
    doc_service.create_sentence_list(documents=documents)

    # GraphService로 공출현 그래프 생성
    graph_service = GraphService(doc_service)
    word_graph = graph_service.build_complete_graph(
        top_n=EXPERIMENT_CONFIG['top_n_words'],
        exclude_stopwords=EXPERIMENT_CONFIG['exclude_stopwords'],
        max_length=-1
    )

    # 엣지 필터링
    if EXPERIMENT_CONFIG['edge_weight_threshold'] > 0:
        word_graph.filter_edges_by_weight(EXPERIMENT_CONFIG['edge_weight_threshold'])

    # 단어 리스트 추출 (Word 객체 -> 문자열)
    vocab = [word.content for word in word_graph.words]

    print(f"  → Nodes: {word_graph.num_nodes}, Edges: {word_graph.num_edges}")
    print(f"  → Vocabulary size: {len(vocab)}")

    return word_graph, doc_service, vocab


# ============================================================
# 클러스터링 실행 및 단어 추출
# ============================================================

def run_louvain_clustering(word_graph: WordGraph, vocab: List[str]) -> Dict[str, Any]:
    """Louvain 클러스터링 실행"""
    print("\nLouvain 클러스터링 실행 중...")

    service = TraditionalGraphClusteringService(random_state=EXPERIMENT_CONFIG['random_seed'])
    cluster_labels, metrics = service.louvain_clustering(
        word_graph,
        resolution=EXPERIMENT_CONFIG['louvain_resolution']
    )

    # NPMI 계산
    metrics_service = MetricsService()
    npmi = metrics_service.calculate_metrics(
        embeddings=np.zeros((word_graph.num_nodes, 1)),
        labels=cluster_labels,
        metric_names=['npmi'],
        word_graph=word_graph,
        total_docs=EXPERIMENT_CONFIG['num_documents']
    )['npmi']

    print(f"  → Clusters: {metrics['num_clusters']}, Modularity: {metrics['modularity']:.4f}, NPMI: {npmi:.4f}")

    return {
        'model_name': 'Louvain',
        'cluster_labels': cluster_labels,
        'num_clusters': metrics['num_clusters'],
        'modularity': metrics['modularity'],
        'npmi': npmi,
        'vocab': vocab,
    }


def run_leiden_clustering(word_graph: WordGraph, vocab: List[str]) -> Dict[str, Any]:
    """Leiden 클러스터링 실행"""
    print("\nLeiden 클러스터링 실행 중...")

    service = TraditionalGraphClusteringService(random_state=EXPERIMENT_CONFIG['random_seed'])
    cluster_labels, metrics = service.leiden_clustering(
        word_graph,
        resolution=EXPERIMENT_CONFIG['leiden_resolution']
    )

    # NPMI 계산
    metrics_service = MetricsService()
    npmi = metrics_service.calculate_metrics(
        embeddings=np.zeros((word_graph.num_nodes, 1)),
        labels=cluster_labels,
        metric_names=['npmi'],
        word_graph=word_graph,
        total_docs=EXPERIMENT_CONFIG['num_documents']
    )['npmi']

    print(f"  → Clusters: {metrics['num_clusters']}, Modularity: {metrics['modularity']:.4f}, NPMI: {npmi:.4f}")

    return {
        'model_name': 'Leiden',
        'cluster_labels': cluster_labels,
        'num_clusters': metrics['num_clusters'],
        'modularity': metrics['modularity'],
        'npmi': npmi,
        'vocab': vocab,
    }


def run_ours_clustering(optimal_config: Dict[str, Any], vocab: List[str]) -> Dict[str, Any]:
    """Ours (GRACE) 클러스터링 실행"""
    print("\nOurs (GRACE) 클러스터링 실행 중...")

    set_all_seeds(EXPERIMENT_CONFIG['random_seed'])

    pca_dim = optimal_config['pca_dim']
    grace_config = GRACEConfig(
        csv_path=EXPERIMENT_CONFIG['csv_path'],
        num_documents=EXPERIMENT_CONFIG['num_documents'],
        text_column=EXPERIMENT_CONFIG['text_column'],
        top_n_words=EXPERIMENT_CONFIG['top_n_words'],
        exclude_stopwords=EXPERIMENT_CONFIG['exclude_stopwords'],
        edge_weight_threshold=EXPERIMENT_CONFIG['edge_weight_threshold'],
        edge_top_k=-1,
        embedding_method=EXPERIMENT_CONFIG['embedding_method'],
        embed_size=pca_dim * 2,
        w2v_dim=pca_dim,
        bert_dim=pca_dim,
        graphmae_epochs=optimal_config['epochs'],
        graphmae_lr=0.001,
        mask_rate=optimal_config['mask_rate'],
        encoder_type=EXPERIMENT_CONFIG['encoder_type'],
        decoder_type=EXPERIMENT_CONFIG['decoder_type'],
        clustering_method=EXPERIMENT_CONFIG['clustering_method'],
        num_clusters=None,
        min_clusters=EXPERIMENT_CONFIG['min_clusters'],
        max_clusters=EXPERIMENT_CONFIG['max_clusters'],
        eval_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],
        random_seed=EXPERIMENT_CONFIG['random_seed'],
        save_results=False,
        output_dir=EXPERIMENT_CONFIG['output_dir'],
        save_graph_viz=False,
        save_embeddings=False,
        verbose=EXPERIMENT_CONFIG['verbose'],
    )

    pipeline = GRACEPipeline(grace_config)
    results = pipeline.run()

    # Modularity 계산
    import networkx as nx
    import community.community_louvain as community_louvain
    nx_graph = pipeline.word_graph.export_to_networkx(include_weights=True)
    partition = {i: int(pipeline.cluster_labels[i]) for i in range(len(pipeline.cluster_labels))}
    modularity = community_louvain.modularity(partition, nx_graph)

    print(f"  → Clusters: {results['num_clusters']}, Modularity: {modularity:.4f}, "
          f"NPMI: {results['metrics'].get('npmi', 0):.4f}")

    return {
        'model_name': 'Ours',
        'cluster_labels': pipeline.cluster_labels,
        'num_clusters': results['num_clusters'],
        'modularity': modularity,
        'npmi': results['metrics'].get('npmi', 0),
        'vocab': vocab,
        'embeddings': pipeline.final_embeddings,  # 임베딩 추가
    }


# ============================================================
# 클러스터 분석
# ============================================================

def extract_cluster_words(result: Dict[str, Any]) -> Dict[int, List[str]]:
    """각 클러스터별 단어 리스트 추출"""

    cluster_labels = result['cluster_labels']
    vocab = result['vocab']

    clusters = defaultdict(list)
    for word_idx, cluster_id in enumerate(cluster_labels):
        word = vocab[word_idx]
        clusters[int(cluster_id)].append(word)

    return dict(clusters)


def calculate_cluster_npmi(
    word_graph: WordGraph,
    cluster_words: List[str],
    total_docs: int
) -> float:
    """클러스터 내 단어들의 평균 NPMI 계산"""

    if len(cluster_words) < 2:
        return 0.0

    # 클러스터 내 모든 단어 쌍의 NPMI 계산
    npmi_values = []
    vocab_to_idx = {word.content: idx for idx, word in enumerate(word_graph.words)}

    for i, word1 in enumerate(cluster_words):
        for word2 in cluster_words[i+1:]:
            # 단어 쌍의 엣지 가중치 가져오기
            edge_attr = word_graph.get_edge_between_words(word1, word2)

            if edge_attr is not None:
                co_occur = float(edge_attr[0]) if len(edge_attr) > 0 else 0

                if co_occur > 0:
                    # 개별 단어의 총 빈도 (이웃 노드들과의 엣지 합)
                    idx1 = vocab_to_idx.get(word1)
                    idx2 = vocab_to_idx.get(word2)

                    if idx1 is not None and idx2 is not None:
                        neighbors1 = word_graph.get_node_neighbors(idx1)
                        neighbors2 = word_graph.get_node_neighbors(idx2)

                        freq1 = 0
                        freq2 = 0

                        for neighbor in neighbors1:
                            neighbor_word = word_graph.words[neighbor].content
                            edge = word_graph.get_edge_between_words(word1, neighbor_word)
                            if edge is not None:
                                freq1 += float(edge[0])

                        for neighbor in neighbors2:
                            neighbor_word = word_graph.words[neighbor].content
                            edge = word_graph.get_edge_between_words(word2, neighbor_word)
                            if edge is not None:
                                freq2 += float(edge[0])

                        # NPMI 계산
                        if freq1 > 0 and freq2 > 0:
                            pmi = np.log((co_occur * total_docs) / (freq1 * freq2))
                            npmi = pmi / (-np.log(co_occur / total_docs))
                            npmi_values.append(npmi)

    return np.mean(npmi_values) if npmi_values else 0.0


def analyze_clusters(
    result: Dict[str, Any],
    word_graph: WordGraph,
    cluster_words: Dict[int, List[str]]
) -> List[Dict[str, Any]]:
    """클러스터별 상세 분석"""

    analysis = []

    for cluster_id in sorted(cluster_words.keys()):
        words = cluster_words[cluster_id]

        # 클러스터 NPMI 계산
        cluster_npmi = calculate_cluster_npmi(
            word_graph, words, EXPERIMENT_CONFIG['num_documents']
        )

        # 상위 단어 선택 (빈도 기반)
        top_words = words[:EXPERIMENT_CONFIG['top_words_per_cluster']]

        analysis.append({
            'cluster_id': cluster_id,
            'size': len(words),
            'npmi': cluster_npmi,
            'top_words': top_words,
            'all_words': words,
        })

    return analysis


# ============================================================
# 차원 축소 시각화 (t-SNE, UMAP)
# ============================================================

def create_tsne_visualization(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    vocab: List[str],
    title: str,
    output_path: Path
):
    """t-SNE를 사용한 클러스터 시각화"""
    from sklearn.manifold import TSNE

    print(f"  t-SNE 계산 중... (n={len(embeddings)})")

    # t-SNE 적용
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 시각화
    plt.figure(figsize=(12, 8))

    # 클러스터별 색상
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label} (n={mask.sum()})',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ t-SNE 저장: {output_path}")


def create_umap_visualization(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    vocab: List[str],
    title: str,
    output_path: Path
):
    """UMAP을 사용한 클러스터 시각화"""
    try:
        import umap
    except ImportError:
        print("  ⚠️  umap-learn 패키지가 없습니다. pip install umap-learn 실행 필요")
        return

    print(f"  UMAP 계산 중... (n={len(embeddings)})")

    # UMAP 적용
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(embeddings)

    # 시각화
    plt.figure(figsize=(12, 8))

    # 클러스터별 색상
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label} (n={mask.sum()})',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ UMAP 저장: {output_path}")


# ============================================================
# 그래프 시각화 (NetworkX)
# ============================================================

def create_graph_visualization(
    word_graph: WordGraph,
    cluster_labels: np.ndarray,
    vocab: List[str],
    title: str,
    output_path: Path,
    layout: str = 'spring'
):
    """클러스터별 색상이 다른 그래프 시각화"""
    import networkx as nx

    print(f"  그래프 시각화 생성 중... (layout={layout})")

    # NetworkX 그래프로 변환
    G = word_graph.export_to_networkx(include_weights=True)

    # 레이아웃 계산
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # 시각화
    plt.figure(figsize=(16, 12))

    # 클러스터별 색상
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    node_colors = [colors[label] for label in cluster_labels]

    # 노드 그리기
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=100,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5
    )

    # 엣지 그리기 (가중치에 따라 투명도 조절)
    edges = G.edges()
    weights = [G[u][v].get('weight', 1) for u, v in edges]
    max_weight = max(weights) if weights else 1

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,
        width=[w/max_weight for w in weights],
        edge_color='gray'
    )

    # 라벨 추가 (노드가 너무 많으면 생략)
    if len(vocab) <= 100:
        labels = {i: vocab[i] for i in range(len(vocab))}
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=6,
            font_color='black'
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    ✓ 그래프 저장: {output_path}")


# ============================================================
# 워드 클라우드 생성
# ============================================================

def create_wordcloud(
    cluster_words: List[str],
    title: str,
    output_path: Path
):
    """워드 클라우드 생성"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️  wordcloud 패키지가 없습니다. pip install wordcloud 실행 필요")
        return

    # 단어 빈도 (동일 가중치)
    word_freq = {word: 1 for word in cluster_words[:EXPERIMENT_CONFIG['wordcloud_words']]}

    if not word_freq:
        print(f"⚠️  {title}: 단어가 없어 워드클라우드를 생성할 수 없습니다.")
        return

    # 워드 클라우드 생성
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10,
    ).generate_from_frequencies(word_freq)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_wordclouds(
    model_name: str,
    cluster_analysis: List[Dict[str, Any]],
    output_dir: Path
):
    """모든 클러스터의 워드 클라우드 생성"""

    wordcloud_dir = output_dir / model_name / "wordclouds"
    wordcloud_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{model_name} 워드 클라우드 생성 중...")

    for cluster_info in cluster_analysis:
        cluster_id = cluster_info['cluster_id']
        words = cluster_info['all_words']

        title = f"{model_name} - Cluster {cluster_id} (n={len(words)}, NPMI={cluster_info['npmi']:.3f})"
        output_path = wordcloud_dir / f"cluster_{cluster_id}.png"

        create_wordcloud(words, title, output_path)

    print(f"  ✓ 워드 클라우드 저장: {wordcloud_dir}")


# ============================================================
# 결과 저장
# ============================================================

def save_cluster_words(
    model_name: str,
    cluster_words: Dict[int, List[str]],
    cluster_analysis: List[Dict[str, Any]],
    output_dir: Path
):
    """클러스터 단어 리스트 저장"""

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON 형식 (전체 단어)
    json_path = model_dir / "cluster_words.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_words, f, indent=2, ensure_ascii=False)

    # 2. TXT 형식 (가독성)
    txt_path = model_dir / "cluster_words.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{model_name} - Cluster Words\n")
        f.write("="*100 + "\n\n")

        for cluster_info in cluster_analysis:
            cluster_id = cluster_info['cluster_id']
            size = cluster_info['size']
            npmi = cluster_info['npmi']
            top_words = cluster_info['top_words']

            f.write(f"Cluster {cluster_id} (Size: {size}, NPMI: {npmi:.4f})\n")
            f.write("-"*100 + "\n")
            f.write(f"Top {len(top_words)} words:\n")
            f.write(", ".join(top_words))
            f.write("\n\n")

    # 3. CSV 형식 (클러스터 요약)
    summary_data = []
    for cluster_info in cluster_analysis:
        summary_data.append({
            'cluster_id': cluster_info['cluster_id'],
            'size': cluster_info['size'],
            'npmi': cluster_info['npmi'],
            'top_10_words': ', '.join(cluster_info['top_words'][:10]),
        })

    summary_df = pd.DataFrame(summary_data)
    csv_path = model_dir / "cluster_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"\n{model_name} 결과 저장:")
    print(f"  ✓ JSON: {json_path}")
    print(f"  ✓ TXT:  {txt_path}")
    print(f"  ✓ CSV:  {csv_path}")


def create_comparison_table(
    all_results: Dict[str, Any],
    output_dir: Path
):
    """모델 간 비교 테이블 생성"""

    comparison_path = output_dir / "model_comparison.txt"

    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write("RQ3 Qualitative Analysis - Model Comparison\n")
        f.write("="*100 + "\n\n")

        # 전체 요약
        f.write(f"{'Model':<15} {'Num Clusters':>15} {'Modularity':>15} {'NPMI':>15} {'Avg Cluster Size':>20}\n")
        f.write("-"*100 + "\n")

        for model_name in ['Louvain', 'Leiden', 'Ours']:
            result = all_results[model_name]
            cluster_analysis = result['cluster_analysis']

            avg_size = np.mean([c['size'] for c in cluster_analysis])

            f.write(f"{model_name:<15} "
                   f"{result['num_clusters']:>15} "
                   f"{result['modularity']:>15.4f} "
                   f"{result['npmi']:>15.4f} "
                   f"{avg_size:>20.2f}\n")

        f.write("\n" + "="*100 + "\n")

        # 클러스터별 상세
        for model_name in ['Louvain', 'Leiden', 'Ours']:
            f.write(f"\n\n{model_name} - Cluster Details\n")
            f.write("="*100 + "\n")

            cluster_analysis = all_results[model_name]['cluster_analysis']

            for cluster_info in cluster_analysis:
                f.write(f"\nCluster {cluster_info['cluster_id']}: ")
                f.write(f"Size={cluster_info['size']}, NPMI={cluster_info['npmi']:.4f}\n")
                f.write(f"  Top words: {', '.join(cluster_info['top_words'][:15])}\n")

    print(f"\n✓ 비교 테이블 저장: {comparison_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """RQ3 질적 분석 메인 함수"""

    print("="*100)
    print("RQ3 질적 분석: 클러스터 단어 추출 및 시각화")
    print("="*100)

    # 최적 설정 로드
    optimal_config = load_optimal_config()

    # 그래프 구축
    word_graph, doc_service, vocab = build_word_graph()

    # 출력 디렉토리
    output_dir = Path(EXPERIMENT_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 각 모델 실행
    all_results = {}

    # 1. Louvain
    louvain_result = run_louvain_clustering(word_graph, vocab)
    louvain_cluster_words = extract_cluster_words(louvain_result)
    louvain_analysis = analyze_clusters(louvain_result, word_graph, louvain_cluster_words)
    louvain_result['cluster_analysis'] = louvain_analysis

    save_cluster_words('Louvain', louvain_cluster_words, louvain_analysis, output_dir)
    generate_all_wordclouds('Louvain', louvain_analysis, output_dir)

    all_results['Louvain'] = louvain_result

    # 2. Leiden
    leiden_result = run_leiden_clustering(word_graph, vocab)
    leiden_cluster_words = extract_cluster_words(leiden_result)
    leiden_analysis = analyze_clusters(leiden_result, word_graph, leiden_cluster_words)
    leiden_result['cluster_analysis'] = leiden_analysis

    save_cluster_words('Leiden', leiden_cluster_words, leiden_analysis, output_dir)
    generate_all_wordclouds('Leiden', leiden_analysis, output_dir)

    all_results['Leiden'] = leiden_result

    # 3. Ours
    ours_result = run_ours_clustering(optimal_config, vocab)
    ours_cluster_words = extract_cluster_words(ours_result)
    ours_analysis = analyze_clusters(ours_result, word_graph, ours_cluster_words)
    ours_result['cluster_analysis'] = ours_analysis

    save_cluster_words('Ours', ours_cluster_words, ours_analysis, output_dir)
    generate_all_wordclouds('Ours', ours_analysis, output_dir)

    all_results['Ours'] = ours_result

    # 비교 테이블 생성
    create_comparison_table(all_results, output_dir)

    # ============================================================
    # 시각화 생성
    # ============================================================

    print("\n" + "="*100)
    print("시각화 생성 중...")
    print("="*100)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Ours 방법만 임베딩이 있으므로 t-SNE/UMAP 생성
    if 'embeddings' in ours_result:
        print("\nOurs - 차원 축소 시각화:")

        # t-SNE
        tsne_path = viz_dir / "ours_tsne.png"
        create_tsne_visualization(
            ours_result['embeddings'],
            ours_result['cluster_labels'],
            vocab,
            f"Ours - t-SNE Visualization (NPMI={ours_result['npmi']:.4f})",
            tsne_path
        )

        # UMAP
        umap_path = viz_dir / "ours_umap.png"
        create_umap_visualization(
            ours_result['embeddings'],
            ours_result['cluster_labels'],
            vocab,
            f"Ours - UMAP Visualization (NPMI={ours_result['npmi']:.4f})",
            umap_path
        )

    # 모든 방법의 그래프 시각화
    print("\n그래프 시각화 (클러스터별 색상):")

    for model_name in ['Louvain', 'Leiden', 'Ours']:
        result = all_results[model_name]

        graph_path = viz_dir / f"{model_name.lower()}_graph.png"
        create_graph_visualization(
            word_graph,
            result['cluster_labels'],
            vocab,
            f"{model_name} - Graph Visualization (Modularity={result['modularity']:.4f}, NPMI={result['npmi']:.4f})",
            graph_path,
            layout='spring'
        )

    print("\n" + "="*100)
    print("✅ RQ3 질적 분석 완료!")
    print("="*100)
    print(f"결과 디렉토리: {output_dir.absolute()}")
    print(f"\n생성된 파일:")
    print(f"  - 각 모델별 디렉토리 (Louvain, Leiden, Ours)")
    print(f"    - cluster_words.json: 클러스터별 전체 단어 리스트")
    print(f"    - cluster_words.txt:  클러스터별 단어 리스트 (가독성)")
    print(f"    - cluster_summary.csv: 클러스터 요약 통계")
    print(f"    - wordclouds/: 클러스터별 워드 클라우드 이미지")
    print(f"  - model_comparison.txt: 모델 간 비교 테이블")


if __name__ == "__main__":
    main()
