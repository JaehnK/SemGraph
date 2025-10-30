#!/usr/bin/env python3
"""
RQ3 통합 분석: 전통적 방법 vs 제안 방법 (정량 + 질적 비교)

연구 질문: GraphMAE2 기반 방법이 전통적 커뮤니티 탐지 알고리즘보다
           의미적으로 일관된 클러스터를 생성하는가?

비교 모델:
1. Louvain: 전통적 modularity 최적화
2. Leiden: Louvain 개선판
3. Ours: GraphMAE2 + Word2Vec + DistilBERT

실험 구성:
- Phase 1: 30회 반복 실험 (정량적 비교)
  * NPMI, Modularity 등 성능 지표
  * ARI, NMI 클러스터링 유사도
  * 통계 검정 (t-test, Cohen's d)
  * Median ± IQR 통계량

- Phase 2: 대표 시드 질적 분석
  * 클러스터별 단어 리스트
  * 클러스터 내부 NPMI
  * Hungarian Algorithm 최적 매칭
  * Top-K 단어 Jaccard 유사도
  * 워드 클라우드 생성

출력:
- statistical/: 통계 검정 결과, 테이블
- qualitative/: 클러스터 단어 리스트, 워드클라우드
- cluster_similarity/: ARI, NMI, 매칭 결과
- visualizations/: 박스플롯, 히트맵, 네트워크
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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

    # 재현성 (30회 반복)
    'random_seeds': [42, 123, 456, 789, 101, 202, 303, 404, 505, 606,
                     707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616,
                     1717, 1818, 1919, 2020, 2121, 2222, 2323, 2424, 2525, 2626],

    # 평가
    'eval_metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

    # 전통적 방법 설정
    'louvain_resolution': 1.0,
    'leiden_resolution': 1.0,

    # 질적 분석 설정
    'top_words_per_cluster': 50,
    'wordcloud_words': 50,
    'topk_jaccard_values': [10, 20, 50],

    # 출력
    'output_dir': 'results/rq3_integrated',
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

def build_shared_word_graph(random_seed: int) -> Tuple[WordGraph, DocumentService, List[str]]:
    """모든 모델이 공유할 공출현 그래프 구축"""

    set_all_seeds(random_seed)

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

    # 단어 리스트 추출
    vocab = [word.content for word in word_graph.words]

    return word_graph, doc_service, vocab


# ============================================================
# 클러스터링 실행
# ============================================================

def run_louvain_experiment(
    word_graph: WordGraph,
    vocab: List[str],
    random_seed: int
) -> Dict[str, Any]:
    """Louvain 커뮤니티 탐지"""

    service = TraditionalGraphClusteringService(random_state=random_seed)
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

    metrics['npmi'] = npmi

    return {
        'model_name': 'Louvain',
        'random_seed': random_seed,
        'cluster_labels': cluster_labels,
        'num_clusters': metrics['num_clusters'],
        'modularity': metrics['modularity'],
        'npmi': npmi,
        'metrics': metrics,
        'vocab': vocab,
    }


def run_leiden_experiment(
    word_graph: WordGraph,
    vocab: List[str],
    random_seed: int
) -> Dict[str, Any]:
    """Leiden 커뮤니티 탐지"""

    service = TraditionalGraphClusteringService(random_state=random_seed)
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

    metrics['npmi'] = npmi

    return {
        'model_name': 'Leiden',
        'random_seed': random_seed,
        'cluster_labels': cluster_labels,
        'num_clusters': metrics['num_clusters'],
        'modularity': metrics['modularity'],
        'npmi': npmi,
        'metrics': metrics,
        'vocab': vocab,
    }


def run_ours_experiment(
    random_seed: int,
    optimal_config: Dict[str, Any],
    vocab: List[str]
) -> Dict[str, Any]:
    """제안 방법 (GRACE) 실행"""

    set_all_seeds(random_seed)

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
        eval_metrics=EXPERIMENT_CONFIG['eval_metrics'],
        random_seed=random_seed,
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

    return {
        'model_name': 'Ours',
        'random_seed': random_seed,
        'cluster_labels': pipeline.cluster_labels,
        'num_clusters': results['num_clusters'],
        'modularity': modularity,
        'npmi': results['metrics'].get('npmi', 0),
        'metrics': {
            **results['metrics'],
            'modularity': modularity,
        },
        'vocab': vocab,
        'embeddings': pipeline.final_embeddings,
    }


# ============================================================
# Phase 1: 정량적 비교 (30회 반복)
# ============================================================

def run_quantitative_experiments(optimal_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """30회 반복 실험 실행"""

    print("\n" + "="*80)
    print("Phase 1: 정량적 비교 (30회 반복 실험)")
    print("="*80)

    all_results = {
        'Louvain': [],
        'Leiden': [],
        'Ours': [],
    }

    for i, seed in enumerate(EXPERIMENT_CONFIG['random_seeds'], 1):
        print(f"\n[{i}/{len(EXPERIMENT_CONFIG['random_seeds'])}] Seed {seed}")

        # 공통 그래프 구축
        word_graph, doc_service, vocab = build_shared_word_graph(seed)
        print(f"  Graph: {word_graph.num_nodes} nodes, {word_graph.num_edges} edges")

        # Louvain
        louvain_result = run_louvain_experiment(word_graph, vocab, seed)
        all_results['Louvain'].append(louvain_result)
        print(f"  Louvain: Clusters={louvain_result['num_clusters']}, "
              f"NPMI={louvain_result['npmi']:.4f}, Modularity={louvain_result['modularity']:.4f}")

        # Leiden
        leiden_result = run_leiden_experiment(word_graph, vocab, seed)
        all_results['Leiden'].append(leiden_result)
        print(f"  Leiden:  Clusters={leiden_result['num_clusters']}, "
              f"NPMI={leiden_result['npmi']:.4f}, Modularity={leiden_result['modularity']:.4f}")

        # Ours
        ours_result = run_ours_experiment(seed, optimal_config, vocab)
        all_results['Ours'].append(ours_result)
        print(f"  Ours:    Clusters={ours_result['num_clusters']}, "
              f"NPMI={ours_result['npmi']:.4f}, Modularity={ours_result['modularity']:.4f}")

    return all_results


# ============================================================
# 통계 분석 (Median, IQR)
# ============================================================

def calculate_robust_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Median, IQR, Range 계산"""

    common_metrics = ['modularity', 'npmi']
    embedding_metrics = EXPERIMENT_CONFIG['eval_metrics']
    all_metrics = list(set(common_metrics + embedding_metrics))

    metrics_dict = {metric: [] for metric in all_metrics}

    for result in results:
        for metric in all_metrics:
            if metric in result.get('metrics', {}):
                value = result['metrics'][metric]
            elif metric in result:
                value = result[metric]
            else:
                value = np.nan

            metrics_dict[metric].append(value)

    stats = {}
    for metric, values in metrics_dict.items():
        valid_values = [v for v in values if not np.isnan(v)]

        if len(valid_values) > 0:
            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)
            iqr = q3 - q1

            stats[metric] = {
                'median': np.median(valid_values),
                'iqr': iqr,
                'q1': q1,
                'q3': q3,
                'min': np.min(valid_values),
                'max': np.max(valid_values),
                'mean': np.mean(valid_values),
                'std': np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0,
                'values': valid_values,
            }
        else:
            stats[metric] = {
                'median': np.nan,
                'iqr': np.nan,
                'q1': np.nan,
                'q3': np.nan,
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'values': [],
            }

    return stats


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d 효과 크기 계산"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


def interpret_cohens_d(d: float) -> str:
    """Cohen's d 효과 크기 해석"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def perform_statistical_tests(
    all_statistics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Any]:
    """통계적 검정 (Ours vs Leiden, Ours vs Louvain)"""

    alpha = 0.05
    n_models = 2  # Leiden, Louvain
    common_metrics = ['modularity', 'npmi']
    n_metrics = len(common_metrics)
    n_comparisons = n_models * n_metrics
    alpha_corrected = alpha / n_comparisons

    test_results = {
        'alpha': alpha,
        'n_comparisons': n_comparisons,
        'alpha_corrected': alpha_corrected,
        'metrics': {}
    }

    for metric in common_metrics:
        test_results['metrics'][metric] = {}

        ours_values = all_statistics['Ours'][metric]['values']

        for other_model in ['Louvain', 'Leiden']:
            other_values = all_statistics[other_model][metric]['values']

            # t-test
            t_stat, p_value = stats.ttest_ind(ours_values, other_values)

            # Cohen's d
            effect_size = cohens_d(ours_values, other_values)

            # 유의성 판단
            if p_value < alpha_corrected:
                significance = '***'
            elif p_value < 0.01:
                significance = '**'
            elif p_value < 0.05:
                significance = '*'
            else:
                significance = 'ns'

            # Ours가 더 나은지 판단
            ours_median = all_statistics['Ours'][metric]['median']
            other_median = all_statistics[other_model][metric]['median']

            if metric in ['modularity', 'npmi', 'silhouette', 'calinski_harabasz']:
                ours_is_better = ours_median > other_median
            elif metric in ['davies_bouldin']:
                ours_is_better = ours_median < other_median
            else:
                ours_is_better = None

            test_results['metrics'][metric][other_model] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'alpha_corrected': float(alpha_corrected),
                'cohens_d': float(effect_size),
                'effect_size_interpretation': interpret_cohens_d(effect_size),
                'significance': significance,
                'ours_is_better': bool(ours_is_better) if ours_is_better is not None else None,
                'ours_median': float(ours_median),
                'other_median': float(other_median),
            }

    return test_results


# ============================================================
# 클러스터 유사도 분석 (ARI, NMI)
# ============================================================

def calculate_clustering_similarity(
    all_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """모델 간 클러스터링 유사도 계산 (ARI, NMI)"""

    print("\n" + "="*80)
    print("클러스터링 유사도 계산 (ARI, NMI)")
    print("="*80)

    similarity_results = {
        'Louvain_vs_Ours': {'ARI': [], 'NMI': []},
        'Leiden_vs_Ours': {'ARI': [], 'NMI': []},
    }

    n_seeds = len(EXPERIMENT_CONFIG['random_seeds'])

    for i in range(n_seeds):
        louvain_labels = all_results['Louvain'][i]['cluster_labels']
        leiden_labels = all_results['Leiden'][i]['cluster_labels']
        ours_labels = all_results['Ours'][i]['cluster_labels']

        # Louvain vs Ours
        ari_louvain = adjusted_rand_score(louvain_labels, ours_labels)
        nmi_louvain = normalized_mutual_info_score(louvain_labels, ours_labels)
        similarity_results['Louvain_vs_Ours']['ARI'].append(ari_louvain)
        similarity_results['Louvain_vs_Ours']['NMI'].append(nmi_louvain)

        # Leiden vs Ours
        ari_leiden = adjusted_rand_score(leiden_labels, ours_labels)
        nmi_leiden = normalized_mutual_info_score(leiden_labels, ours_labels)
        similarity_results['Leiden_vs_Ours']['ARI'].append(ari_leiden)
        similarity_results['Leiden_vs_Ours']['NMI'].append(nmi_leiden)

    # 통계량 계산
    summary = {}
    for comparison, metrics in similarity_results.items():
        summary[comparison] = {}
        for metric_name, values in metrics.items():
            summary[comparison][metric_name] = {
                'median': float(np.median(values)),
                'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'values': [float(v) for v in values],
            }

    print(f"\nLouvain vs Ours:")
    print(f"  ARI: {summary['Louvain_vs_Ours']['ARI']['median']:.4f} ± {summary['Louvain_vs_Ours']['ARI']['iqr']:.4f}")
    print(f"  NMI: {summary['Louvain_vs_Ours']['NMI']['median']:.4f} ± {summary['Louvain_vs_Ours']['NMI']['iqr']:.4f}")

    print(f"\nLeiden vs Ours:")
    print(f"  ARI: {summary['Leiden_vs_Ours']['ARI']['median']:.4f} ± {summary['Leiden_vs_Ours']['ARI']['iqr']:.4f}")
    print(f"  NMI: {summary['Leiden_vs_Ours']['NMI']['median']:.4f} ± {summary['Leiden_vs_Ours']['NMI']['iqr']:.4f}")

    return summary


# ============================================================
# Phase 2: 질적 분석 (대표 시드 선택)
# ============================================================

def select_representative_seed(
    all_results: Dict[str, List[Dict[str, Any]]],
    all_statistics: Dict[str, Dict[str, Dict[str, float]]]
) -> int:
    """NPMI median에 가장 가까운 시드 선택"""

    print("\n" + "="*80)
    print("Phase 2: 질적 분석 - 대표 시드 선택")
    print("="*80)

    # Ours 모델의 NPMI median
    ours_median = all_statistics['Ours']['npmi']['median']
    ours_results = all_results['Ours']

    # Median에 가장 가까운 시드 찾기
    min_diff = float('inf')
    representative_seed = None
    representative_idx = None

    for idx, result in enumerate(ours_results):
        diff = abs(result['npmi'] - ours_median)
        if diff < min_diff:
            min_diff = diff
            representative_seed = result['random_seed']
            representative_idx = idx

    print(f"\n선택된 대표 시드: {representative_seed}")
    print(f"  Ours NPMI median: {ours_median:.4f}")
    print(f"  Representative NPMI: {ours_results[representative_idx]['npmi']:.4f}")
    print(f"  Difference: {min_diff:.6f}")

    return representative_idx


def extract_cluster_words(result: Dict[str, Any]) -> Dict[int, List[str]]:
    """클러스터별 단어 리스트 추출"""

    cluster_labels = result['cluster_labels']
    vocab = result['vocab']

    clusters = defaultdict(list)
    for word_idx, cluster_id in enumerate(cluster_labels):
        word = vocab[word_idx]
        clusters[int(cluster_id)].append(word)

    return dict(clusters)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard 유사도 계산"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def match_clusters_hungarian(
    model1_clusters: Dict[int, List[str]],
    model2_clusters: Dict[int, List[str]]
) -> Dict[str, Any]:
    """Hungarian Algorithm으로 최적 클러스터 매칭"""

    n1 = len(model1_clusters)
    n2 = len(model2_clusters)

    # 유사도 행렬 생성
    similarity_matrix = np.zeros((n1, n2))
    cluster1_ids = sorted(model1_clusters.keys())
    cluster2_ids = sorted(model2_clusters.keys())

    for i, c1_id in enumerate(cluster1_ids):
        for j, c2_id in enumerate(cluster2_ids):
            set1 = set(model1_clusters[c1_id])
            set2 = set(model2_clusters[c2_id])
            similarity_matrix[i, j] = jaccard_similarity(set1, set2)

    # Hungarian algorithm (최대 유사도 매칭)
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)

    # 매칭 결과
    matches = []
    for i, j in zip(row_ind, col_ind):
        c1_id = cluster1_ids[i]
        c2_id = cluster2_ids[j]
        matches.append({
            'cluster1_id': c1_id,
            'cluster2_id': c2_id,
            'jaccard': float(similarity_matrix[i, j]),
            'cluster1_size': len(model1_clusters[c1_id]),
            'cluster2_size': len(model2_clusters[c2_id]),
            'intersection_size': len(set(model1_clusters[c1_id]) & set(model2_clusters[c2_id])),
        })

    avg_jaccard = similarity_matrix[row_ind, col_ind].mean()

    return {
        'matches': matches,
        'avg_jaccard': float(avg_jaccard),
        'similarity_matrix': similarity_matrix.tolist(),
        'cluster1_ids': cluster1_ids,
        'cluster2_ids': cluster2_ids,
    }


def calculate_topk_jaccard(
    model1_clusters: Dict[int, List[str]],
    model2_clusters: Dict[int, List[str]],
    matching_result: Dict[str, Any],
    k_values: List[int]
) -> Dict[int, float]:
    """Top-K 단어 Jaccard 유사도 계산"""

    topk_results = {}

    for k in k_values:
        jaccard_values = []

        for match in matching_result['matches']:
            c1_id = match['cluster1_id']
            c2_id = match['cluster2_id']

            # Top-K 단어 추출
            topk1 = set(model1_clusters[c1_id][:k])
            topk2 = set(model2_clusters[c2_id][:k])

            jaccard = jaccard_similarity(topk1, topk2)
            jaccard_values.append(jaccard)

        topk_results[k] = float(np.mean(jaccard_values))

    return topk_results


def create_wordcloud(
    cluster_words: List[str],
    title: str,
    output_path: Path
):
    """워드 클라우드 생성"""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️  wordcloud 패키지가 없습니다.")
        return

    word_freq = {word: 1 for word in cluster_words[:EXPERIMENT_CONFIG['wordcloud_words']]}

    if not word_freq:
        return

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10,
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_qualitative_analysis(
    all_results: Dict[str, List[Dict[str, Any]]],
    representative_idx: int,
    output_dir: Path
):
    """질적 분석 수행"""

    print("\n" + "="*80)
    print("질적 분석 수행")
    print("="*80)

    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    # 대표 시드 결과 추출
    louvain_result = all_results['Louvain'][representative_idx]
    leiden_result = all_results['Leiden'][representative_idx]
    ours_result = all_results['Ours'][representative_idx]

    # 클러스터 단어 추출
    louvain_clusters = extract_cluster_words(louvain_result)
    leiden_clusters = extract_cluster_words(leiden_result)
    ours_clusters = extract_cluster_words(ours_result)

    # 1. 클러스터 단어 저장
    for model_name, clusters in [('Louvain', louvain_clusters),
                                  ('Leiden', leiden_clusters),
                                  ('Ours', ours_clusters)]:
        model_dir = qualitative_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # JSON 저장
        json_path = model_dir / "cluster_words.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clusters, f, indent=2, ensure_ascii=False)

        # CSV 요약
        summary_data = []
        for cluster_id, words in sorted(clusters.items()):
            summary_data.append({
                'cluster_id': cluster_id,
                'size': len(words),
                'top_10_words': ', '.join(words[:10]),
            })

        summary_df = pd.DataFrame(summary_data)
        csv_path = model_dir / "cluster_summary.csv"
        summary_df.to_csv(csv_path, index=False)

        print(f"\n{model_name}:")
        print(f"  Clusters: {len(clusters)}")
        print(f"  Saved: {json_path}")

        # 워드 클라우드 생성
        wordcloud_dir = model_dir / "wordclouds"
        wordcloud_dir.mkdir(parents=True, exist_ok=True)

        for cluster_id, words in clusters.items():
            wc_path = wordcloud_dir / f"cluster_{cluster_id}.png"
            create_wordcloud(words, f"{model_name} - Cluster {cluster_id}", wc_path)

    # 2. 클러스터 매칭 (Hungarian Algorithm)
    similarity_dir = output_dir / "cluster_similarity"
    similarity_dir.mkdir(parents=True, exist_ok=True)

    print("\n클러스터 매칭 분석:")

    # Louvain vs Ours
    louvain_matching = match_clusters_hungarian(louvain_clusters, ours_clusters)
    topk_louvain = calculate_topk_jaccard(
        louvain_clusters, ours_clusters, louvain_matching,
        EXPERIMENT_CONFIG['topk_jaccard_values']
    )
    louvain_matching['topk_jaccard'] = topk_louvain

    print(f"\nLouvain vs Ours:")
    print(f"  Average Jaccard: {louvain_matching['avg_jaccard']:.4f}")
    for k, jaccard in topk_louvain.items():
        print(f"  Top-{k} Jaccard: {jaccard:.4f}")

    # Leiden vs Ours
    leiden_matching = match_clusters_hungarian(leiden_clusters, ours_clusters)
    topk_leiden = calculate_topk_jaccard(
        leiden_clusters, ours_clusters, leiden_matching,
        EXPERIMENT_CONFIG['topk_jaccard_values']
    )
    leiden_matching['topk_jaccard'] = topk_leiden

    print(f"\nLeiden vs Ours:")
    print(f"  Average Jaccard: {leiden_matching['avg_jaccard']:.4f}")
    for k, jaccard in topk_leiden.items():
        print(f"  Top-{k} Jaccard: {jaccard:.4f}")

    # 저장
    matching_results = {
        'Louvain_vs_Ours': louvain_matching,
        'Leiden_vs_Ours': leiden_matching,
    }

    matching_path = similarity_dir / "cluster_matching.json"
    with open(matching_path, 'w', encoding='utf-8') as f:
        json.dump(matching_results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 클러스터 매칭 결과 저장: {matching_path}")

    return matching_results


# ============================================================
# 결과 저장
# ============================================================

def save_all_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    all_statistics: Dict[str, Dict[str, Dict[str, float]]],
    test_results: Dict[str, Any],
    similarity_results: Dict[str, Any],
    matching_results: Dict[str, Any],
    output_dir: Path
):
    """모든 결과 저장"""

    print("\n" + "="*80)
    print("결과 저장")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    statistical_dir = output_dir / "statistical"
    statistical_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raw results
    raw_path = statistical_dir / f"raw_results_{timestamp}.json"
    # cluster_labels는 numpy array이므로 변환 필요
    serializable_results = {}
    for model, results in all_results.items():
        serializable_results[model] = []
        for r in results:
            r_copy = r.copy()
            if 'cluster_labels' in r_copy:
                r_copy['cluster_labels'] = r_copy['cluster_labels'].tolist()
            if 'embeddings' in r_copy:
                del r_copy['embeddings']  # 너무 큼
            serializable_results[model].append(r_copy)

    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Raw results: {raw_path}")

    # 2. Summary statistics
    summary_data = []
    for model_name in ['Louvain', 'Leiden', 'Ours']:
        stats = all_statistics[model_name]
        row = {'Model': model_name}

        for metric in ['modularity', 'npmi', 'silhouette', 'davies_bouldin', 'calinski_harabasz']:
            if metric in stats and not np.isnan(stats[metric]['median']):
                row[f'{metric}_median'] = stats[metric]['median']
                row[f'{metric}_iqr'] = stats[metric]['iqr']
                row[f'{metric}_min'] = stats[metric]['min']
                row[f'{metric}_max'] = stats[metric]['max']
            else:
                row[f'{metric}_median'] = None
                row[f'{metric}_iqr'] = None
                row[f'{metric}_min'] = None
                row[f'{metric}_max'] = None

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = statistical_dir / f"summary_statistics_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics: {summary_path}")

    # 3. Statistical tests
    tests_path = statistical_dir / f"statistical_tests_{timestamp}.json"
    with open(tests_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistical tests: {tests_path}")

    # 4. Clustering similarity
    similarity_path = statistical_dir / f"clustering_similarity_{timestamp}.json"
    with open(similarity_path, 'w', encoding='utf-8') as f:
        json.dump(similarity_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Clustering similarity: {similarity_path}")

    # 5. Main table (논문용)
    table_path = statistical_dir / f"table_rq3_main_{timestamp}.txt"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(f"Table: RQ3 Overall Performance (Median ± IQR, n={len(EXPERIMENT_CONFIG['random_seeds'])})\n")
        f.write("="*160 + "\n\n")

        header = (f"{'Model':<12} {'NPMI ↑':>22} {'Modularity ↑':>22} "
                 f"{'ARI vs Ours':>18} {'NMI vs Ours':>18} "
                 f"{'Silhouette ↑':>22} {'DBI ↓':>22}")
        f.write(header + "\n")
        f.write("-"*160 + "\n")

        for model_name in ['Louvain', 'Leiden', 'Ours']:
            stats = all_statistics[model_name]
            row = f"{model_name:<12}"

            # NPMI
            if 'npmi' in stats:
                npmi_med = stats['npmi']['median']
                npmi_iqr = stats['npmi']['iqr']
                sig = ''
                if model_name != 'Ours' and 'metrics' in test_results:
                    if 'npmi' in test_results['metrics']:
                        sig = test_results['metrics']['npmi'][model_name]['significance']
                row += f"  {npmi_med:.4f} ± {npmi_iqr:.4f}{sig:>4}"
            else:
                row += f"  {'N/A':>22}"

            # Modularity
            if 'modularity' in stats:
                mod_med = stats['modularity']['median']
                mod_iqr = stats['modularity']['iqr']
                row += f"  {mod_med:.4f} ± {mod_iqr:.4f}"
            else:
                row += f"  {'N/A':>22}"

            # ARI, NMI
            if model_name == 'Ours':
                row += f"  {1.0:>18.4f}"
                row += f"  {1.0:>18.4f}"
            else:
                comparison_key = f"{model_name}_vs_Ours"
                ari_med = similarity_results[comparison_key]['ARI']['median']
                nmi_med = similarity_results[comparison_key]['NMI']['median']
                row += f"  {ari_med:>18.4f}"
                row += f"  {nmi_med:>18.4f}"

            # Embedding metrics (Ours만)
            for metric in ['silhouette', 'davies_bouldin']:
                if metric in stats and not np.isnan(stats[metric]['median']):
                    row += f"  {stats[metric]['median']:.4f} ± {stats[metric]['iqr']:.4f}"
                else:
                    row += f"  {'-':>22}"

            f.write(row + "\n")

        f.write("\n" + "="*160 + "\n")
        f.write(f"Bonferroni corrected α = {test_results['alpha_corrected']:.4f} ({test_results['n_comparisons']} comparisons)\n")
        f.write("Significance: *** p < α_corrected, ** p < 0.01, * p < 0.05, ns = not significant\n")
        f.write("↑ Higher is better, ↓ Lower is better\n")
        f.write("- : Not applicable (traditional methods don't produce embeddings)\n")

    print(f"✓ Main table: {table_path}")

    # 6. Cluster similarity table
    similarity_table_path = statistical_dir / f"table_cluster_similarity_{timestamp}.txt"
    with open(similarity_table_path, 'w', encoding='utf-8') as f:
        f.write(f"Table: Cluster Similarity Analysis (Median ± IQR, n={len(EXPERIMENT_CONFIG['random_seeds'])})\n")
        f.write("="*100 + "\n\n")

        header = f"{'Comparison':<20} {'ARI':>18} {'NMI':>18} {'Matched Jaccard':>18} {'Top-10 Jaccard':>18}"
        f.write(header + "\n")
        f.write("-"*100 + "\n")

        for comparison in ['Louvain_vs_Ours', 'Leiden_vs_Ours']:
            display_name = comparison.replace('_', ' ')
            ari = similarity_results[comparison]['ARI']
            nmi = similarity_results[comparison]['NMI']

            # Matched Jaccard (대표 시드)
            matched_jac = matching_results[comparison]['avg_jaccard']

            # Top-10 Jaccard (대표 시드)
            top10_jac = matching_results[comparison]['topk_jaccard'][10]

            row = (f"{display_name:<20} "
                  f"{ari['median']:8.4f}±{ari['iqr']:.4f} "
                  f"{nmi['median']:8.4f}±{nmi['iqr']:.4f} "
                  f"{matched_jac:>18.4f} "
                  f"{top10_jac:>18.4f}")

            f.write(row + "\n")

        f.write("\n" + "="*100 + "\n")
        f.write("ARI, NMI: Clustering structure similarity (higher = more similar)\n")
        f.write("Matched Jaccard: Best-matched cluster pairs (Hungarian Algorithm)\n")
        f.write("Top-10 Jaccard: Word overlap in top 10 words per cluster\n")

    print(f"✓ Cluster similarity table: {similarity_table_path}")

    # 7. Matched clusters details
    matched_table_path = statistical_dir / f"table_matched_clusters_{timestamp}.txt"
    with open(matched_table_path, 'w', encoding='utf-8') as f:
        for comparison, matching in matching_results.items():
            f.write(f"\n{comparison.replace('_', ' ')}\n")
            f.write("="*120 + "\n")

            header = (f"{'Model1 Cluster':<15} {'Model2 Cluster':<15} {'Jaccard':>10} "
                     f"{'Size1':>8} {'Size2':>8} {'Intersection':>15}")
            f.write(header + "\n")
            f.write("-"*120 + "\n")

            for match in matching['matches']:
                row = (f"{match['cluster1_id']:<15} {match['cluster2_id']:<15} "
                      f"{match['jaccard']:>10.4f} {match['cluster1_size']:>8} "
                      f"{match['cluster2_size']:>8} {match['intersection_size']:>15}")
                f.write(row + "\n")

            f.write(f"\nAverage Jaccard: {matching['avg_jaccard']:.4f}\n")
            f.write(f"\nTop-K Jaccard:\n")
            for k, jaccard in matching['topk_jaccard'].items():
                f.write(f"  Top-{k}: {jaccard:.4f}\n")
            f.write("\n")

    print(f"✓ Matched clusters table: {matched_table_path}")


# ============================================================
# 시각화
# ============================================================

def create_visualizations(
    all_statistics: Dict[str, Dict[str, Dict[str, float]]],
    similarity_results: Dict[str, Any],
    output_dir: Path
):
    """결과 시각화"""

    print("\n" + "="*80)
    print("시각화 생성")
    print("="*80)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    models = ['Louvain', 'Leiden', 'Ours']

    # 1. NPMI Boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    npmi_data = []
    for model in models:
        npmi_data.append(all_statistics[model]['npmi']['values'])

    bp = ax.boxplot(npmi_data, labels=models, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))

    ax.set_ylabel('NPMI', fontsize=14, fontweight='bold')
    ax.set_title('NPMI Distribution (n=30)', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    npmi_path = viz_dir / "npmi_boxplot.png"
    plt.savefig(npmi_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ NPMI boxplot: {npmi_path}")

    # 2. Similarity Heatmap
    fig, ax = plt.subplots(figsize=(8, 4))

    heatmap_data = []
    row_labels = []

    for comparison in ['Louvain_vs_Ours', 'Leiden_vs_Ours']:
        display_name = comparison.replace('_vs_Ours', '')
        row_labels.append(display_name)

        ari = similarity_results[comparison]['ARI']['median']
        nmi = similarity_results[comparison]['NMI']['median']

        heatmap_data.append([ari, nmi])

    heatmap_df = pd.DataFrame(heatmap_data, index=row_labels, columns=['ARI', 'NMI'])

    sns.heatmap(heatmap_df, annot=True, fmt='.4f', cmap='YlGnBu',
                cbar_kws={'label': 'Similarity Score'},
                linewidths=0.5, ax=ax, vmin=0, vmax=1)

    ax.set_title('Clustering Similarity vs Ours', fontsize=14, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.tight_layout()
    heatmap_path = viz_dir / "similarity_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Similarity heatmap: {heatmap_path}")

    # 3. Performance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # NPMI
    ax = axes[0]
    npmi_medians = [all_statistics[m]['npmi']['median'] for m in models]
    npmi_iqrs = [all_statistics[m]['npmi']['iqr'] for m in models]

    bars = ax.bar(models, npmi_medians, yerr=npmi_iqrs, capsize=5,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')

    ax.set_ylabel('NPMI (Median ± IQR)', fontsize=12, fontweight='bold')
    ax.set_title('NPMI Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Modularity
    ax = axes[1]
    mod_medians = [all_statistics[m]['modularity']['median'] for m in models]
    mod_iqrs = [all_statistics[m]['modularity']['iqr'] for m in models]

    bars = ax.bar(models, mod_medians, yerr=mod_iqrs, capsize=5,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')

    ax.set_ylabel('Modularity (Median ± IQR)', fontsize=12, fontweight='bold')
    ax.set_title('Modularity Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    comparison_path = viz_dir / "performance_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Performance comparison: {comparison_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """RQ3 통합 분석 메인 함수"""

    print("="*80)
    print("RQ3 통합 분석: 전통적 방법 vs 제안 방법")
    print("="*80)
    print(f"데이터: {EXPERIMENT_CONFIG['csv_path']}")
    print(f"문서 수: {EXPERIMENT_CONFIG['num_documents']}")
    print(f"Vocab: {EXPERIMENT_CONFIG['top_n_words']} 단어")
    print(f"반복 횟수: {len(EXPERIMENT_CONFIG['random_seeds'])}")
    print("="*80)

    # 최적 설정 로드
    optimal_config = load_optimal_config()

    # Phase 1: 정량적 비교
    all_results = run_quantitative_experiments(optimal_config)

    # 통계 분석
    print("\n" + "="*80)
    print("통계 분석")
    print("="*80)

    all_statistics = {}
    for model_name, results in all_results.items():
        all_statistics[model_name] = calculate_robust_statistics(results)
        print(f"\n{model_name}:")
        print(f"  NPMI: {all_statistics[model_name]['npmi']['median']:.4f} ± "
              f"{all_statistics[model_name]['npmi']['iqr']:.4f}")
        print(f"  Modularity: {all_statistics[model_name]['modularity']['median']:.4f} ± "
              f"{all_statistics[model_name]['modularity']['iqr']:.4f}")

    # 통계 검정
    test_results = perform_statistical_tests(all_statistics)

    # 클러스터링 유사도
    similarity_results = calculate_clustering_similarity(all_results)

    # Phase 2: 질적 분석
    output_dir = Path(EXPERIMENT_CONFIG['output_dir'])
    representative_idx = select_representative_seed(all_results, all_statistics)
    matching_results = generate_qualitative_analysis(all_results, representative_idx, output_dir)

    # 결과 저장
    save_all_results(
        all_results, all_statistics, test_results,
        similarity_results, matching_results, output_dir
    )

    # 시각화
    create_visualizations(all_statistics, similarity_results, output_dir)

    print("\n" + "="*80)
    print("✅ RQ3 통합 분석 완료!")
    print("="*80)
    print(f"결과 디렉토리: {output_dir.absolute()}")
    print(f"\n주요 출력:")
    print(f"  - statistical/: 통계 검정, 테이블")
    print(f"  - qualitative/: 클러스터 단어, 워드클라우드")
    print(f"  - cluster_similarity/: ARI, NMI, 매칭 결과")
    print(f"  - visualizations/: 박스플롯, 히트맵")


if __name__ == "__main__":
    main()
