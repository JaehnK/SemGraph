#!/usr/bin/env python3
"""
RQ3 실험: 전통적 그래프 클러스터링 vs 제안 방법 비교

연구 질문: GraphMAE2 기반 방법이 전통적 커뮤니티 탐지 알고리즘보다
           의미적으로 일관된 클러스터를 생성하는가?

비교 모델:
1. Louvain: 전통적 modularity 최적화
2. Leiden: Louvain 개선판 (더 정확한 커뮤니티 탐지)
3. Ours: GraphMAE2 + Word2Vec + DistilBERT (최적 하이퍼파라미터)

실험 설정:
- 데이터: AG News
- Vocab size: 500
- Random seeds: [42, 123, 456, 789, 101] (n=5)
- 동일한 공출현 그래프 사용 (공정한 비교)
- Edge weight threshold: 5

평가 지표:
- Modularity (구조 기반 - Louvain/Leiden에 유리)
- NPMI (의미 일관성 - 핵심 지표)
- Silhouette Score (임베딩 품질)
- Davies-Bouldin Index (임베딩 품질)
- Calinski-Harabasz Index (임베딩 품질)

통계 검정:
- Two-tailed t-test (Ours vs Leiden)
- Bonferroni correction
- Cohen's d (effect size)

예상 결과:
- Modularity: Leiden > Louvain > Ours (예상됨)
- NPMI: Ours > Leiden > Louvain (목표)
- 임베딩 품질: Ours > Leiden, Louvain (N/A for traditional)
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

    # 임베딩 (Ours 모델용)
    'embedding_method': 'attention',  # Attention fusion (RQ1 결과)

    # 최적 하이퍼파라미터 (RQ2 통계 검증 결과)
    'optimal_mask_rate': 0.3,      # ✓ 통계적으로 유의 (vs 0.9: p=0.016, d=1.92)
    'optimal_epochs': 1000,        # ✓ NPMI 최고 (0.130 vs 500:0.112), Silhouette은 동일 (p=0.954)
    'optimal_pca_dim': 256,        # ✓ 정보 손실 없음 (explained variance=1.0)

    # GraphMAE
    'encoder_type': 'gat',
    'decoder_type': 'gat',

    # 클러스터링
    'clustering_method': 'kmeans',
    'min_clusters': 5,
    'max_clusters': 20,

    # 재현성
    'random_seeds': [42, 123, 456, 789, 101, 202, 303, 404, 505, 606,
                     707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616,
                     1717, 1818, 1919, 2020, 2121, 2222, 2323, 2424, 2525, 2626],  # n=30

    # 평가
    'eval_metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

    # 전통적 방법 설정
    'louvain_resolution': 1.0,
    'leiden_resolution': 1.0,

    # 출력
    'output_dir': 'results/rq3_traditional_vs_ours',
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
    """RQ2 결과에서 통계적으로 검증된 최적 설정 로드"""

    optimal_config_path = Path("results/rq2_ablation/optimal_config.json")

    # 통계 검증 파일 경로
    stat_files = {
        'mask_rate': Path("results/rq2_ablation/mask_rate/statistical_tests_mask_rate.json"),
        'epochs': Path("results/rq2_ablation/epochs/statistical_tests_epochs.json"),
        'pca_dim': Path("results/rq2_ablation/pca_dim/statistical_tests_pca_dim.json"),
    }

    if optimal_config_path.exists():
        print(f"✓ RQ2 최적 설정 발견: {optimal_config_path}")
        with open(optimal_config_path, 'r') as f:
            optimal_config = json.load(f)

        print(f"\n최적 하이퍼파라미터:")
        print(f"  - Mask Rate: {optimal_config.get('mask_rate', 'N/A')}")
        print(f"  - Epochs: {optimal_config.get('epochs', 'N/A')}")
        print(f"  - PCA Dim: {optimal_config.get('pca_dim', 'N/A')}")

        # 통계 검증 확인
        print(f"\n통계 검증 상태:")
        all_validated = True
        for param, stat_path in stat_files.items():
            if stat_path.exists():
                print(f"  ✓ {param}: 통계 검증 완료")
                # 간단한 검증 요약 출력
                try:
                    with open(stat_path, 'r') as f:
                        stat_data = json.load(f)
                    # NPMI 기준으로 유의미한 차이 확인
                    if 'npmi' in stat_data:
                        best_param = stat_data['npmi']['best_param']
                        comparisons = stat_data['npmi']['comparisons']
                        sig_count = sum(1 for c in comparisons if c['significant'])
                        print(f"      Best: {best_param}, Significant differences: {sig_count}/{len(comparisons)}")
                except:
                    pass
            else:
                print(f"  ⚠️ {param}: 통계 검증 파일 없음!")
                all_validated = False

        if not all_validated:
            print(f"\n⚠️  일부 파라미터가 통계적으로 검증되지 않았습니다.")
            print(f"   scripts/statistical_validation.py를 실행하는 것을 권장합니다!")

        return optimal_config
    else:
        print(f"⚠️  RQ2 최적 설정 파일 없음. 기본값 사용.")
        print(f"   RQ2 실험 (experiments/rq2_ablation_study.py)을 먼저 실행하세요!")
        print(f"\n기본값 (통계 검증 결과 기반):")
        default_config = {
            'mask_rate': EXPERIMENT_CONFIG['optimal_mask_rate'],
            'epochs': EXPERIMENT_CONFIG['optimal_epochs'],
            'pca_dim': EXPERIMENT_CONFIG['optimal_pca_dim'],
        }
        print(f"  - Mask Rate: {default_config['mask_rate']}")
        print(f"  - Epochs: {default_config['epochs']}")
        print(f"  - PCA Dim: {default_config['pca_dim']}")
        return default_config


# ============================================================
# 공통 그래프 구축
# ============================================================

def build_shared_word_graph(random_seed: int) -> Tuple[WordGraph, DocumentService]:
    """모든 모델이 공유할 공출현 그래프 구축"""

    print(f"  [Seed {random_seed}] 공출현 그래프 구축 중...")

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

    print(f"    → Nodes: {word_graph.num_nodes}, Edges: {word_graph.num_edges}")

    return word_graph, doc_service


# ============================================================
# 1. Louvain 실험
# ============================================================

def run_louvain_experiment(
    word_graph: WordGraph,
    random_seed: int
) -> Dict[str, Any]:
    """Louvain 커뮤니티 탐지"""

    print(f"    [Seed {random_seed}] Louvain 실행 중...", end=' ')

    service = TraditionalGraphClusteringService(random_state=random_seed)
    cluster_labels, metrics = service.louvain_clustering(
        word_graph,
        resolution=EXPERIMENT_CONFIG['louvain_resolution']
    )

    # NPMI 계산 (MetricsService 사용)
    # Louvain은 임베딩이 없으므로 그래프 기반으로만 NPMI 계산
    metrics_service = MetricsService()

    # Dummy embedding (NPMI만 계산하기 위해)
    # 실제로는 word_graph와 cluster_labels만 사용
    npmi = metrics_service.calculate_metrics(
        embeddings=np.zeros((word_graph.num_nodes, 1)),  # dummy
        labels=cluster_labels,
        metric_names=['npmi'],
        word_graph=word_graph,
        total_docs=EXPERIMENT_CONFIG['num_documents']
    )['npmi']

    metrics['npmi'] = npmi

    result = {
        'model_name': 'Louvain',
        'random_seed': random_seed,
        'num_clusters': metrics['num_clusters'],
        'modularity': metrics['modularity'],
        'npmi': npmi,
        'metrics': metrics,
    }

    print(f"Modularity: {metrics['modularity']:.4f}, NPMI: {npmi:.4f}, Clusters: {metrics['num_clusters']}")

    return result


# ============================================================
# 2. Leiden 실험
# ============================================================

def run_leiden_experiment(
    word_graph: WordGraph,
    random_seed: int
) -> Dict[str, Any]:
    """Leiden 커뮤니티 탐지"""

    print(f"    [Seed {random_seed}] Leiden 실행 중...", end=' ')

    service = TraditionalGraphClusteringService(random_state=random_seed)
    cluster_labels, metrics = service.leiden_clustering(
        word_graph,
        resolution=EXPERIMENT_CONFIG['leiden_resolution']
    )

    # NPMI 계산
    metrics_service = MetricsService()
    npmi = metrics_service.calculate_metrics(
        embeddings=np.zeros((word_graph.num_nodes, 1)),  # dummy
        labels=cluster_labels,
        metric_names=['npmi'],
        word_graph=word_graph,
        total_docs=EXPERIMENT_CONFIG['num_documents']
    )['npmi']

    metrics['npmi'] = npmi

    result = {
        'model_name': 'Leiden',
        'random_seed': random_seed,
        'num_clusters': metrics['num_clusters'],
        'modularity': metrics['modularity'],
        'npmi': npmi,
        'metrics': metrics,
    }

    print(f"Modularity: {metrics['modularity']:.4f}, NPMI: {npmi:.4f}, Clusters: {metrics['num_clusters']}")

    return result


# ============================================================
# 3. Ours (GRACE) 실험
# ============================================================

def run_ours_experiment(
    random_seed: int,
    optimal_config: Dict[str, Any]
) -> Dict[str, Any]:
    """제안 방법 (GRACE) 실행"""

    print(f"    [Seed {random_seed}] Ours (GRACE) 실행 중...", end=' ')

    set_all_seeds(random_seed)

    # GRACE 설정
    pca_dim = optimal_config['pca_dim']
    grace_config = GRACEConfig(
        # 데이터
        csv_path=EXPERIMENT_CONFIG['csv_path'],
        num_documents=EXPERIMENT_CONFIG['num_documents'],
        text_column=EXPERIMENT_CONFIG['text_column'],

        # 그래프
        top_n_words=EXPERIMENT_CONFIG['top_n_words'],
        exclude_stopwords=EXPERIMENT_CONFIG['exclude_stopwords'],
        edge_weight_threshold=EXPERIMENT_CONFIG['edge_weight_threshold'],
        edge_top_k=-1,

        # 임베딩
        embedding_method=EXPERIMENT_CONFIG['embedding_method'],
        embed_size=pca_dim * 2,
        w2v_dim=pca_dim,
        bert_dim=pca_dim,

        # GraphMAE (최적 설정)
        graphmae_epochs=optimal_config['epochs'],
        graphmae_lr=0.001,
        mask_rate=optimal_config['mask_rate'],
        encoder_type=EXPERIMENT_CONFIG['encoder_type'],
        decoder_type=EXPERIMENT_CONFIG['decoder_type'],

        # 클러스터링
        clustering_method=EXPERIMENT_CONFIG['clustering_method'],
        num_clusters=None,
        min_clusters=EXPERIMENT_CONFIG['min_clusters'],
        max_clusters=EXPERIMENT_CONFIG['max_clusters'],

        # 평가
        eval_metrics=EXPERIMENT_CONFIG['eval_metrics'],

        # 재현성
        random_seed=random_seed,

        # 출력
        save_results=False,
        output_dir=EXPERIMENT_CONFIG['output_dir'],
        save_graph_viz=False,
        save_embeddings=False,
        verbose=EXPERIMENT_CONFIG['verbose'],
    )

    # 파이프라인 실행
    pipeline = GRACEPipeline(grace_config)
    results = pipeline.run()

    # Modularity 계산 (NetworkX 사용)
    import networkx as nx
    import community.community_louvain as community_louvain

    nx_graph = pipeline.word_graph.export_to_networkx(include_weights=True)
    partition = {i: int(pipeline.cluster_labels[i]) for i in range(len(pipeline.cluster_labels))}
    modularity = community_louvain.modularity(partition, nx_graph)

    result = {
        'model_name': 'Ours',
        'random_seed': random_seed,
        'num_clusters': results['num_clusters'],
        'modularity': modularity,
        'npmi': results['metrics'].get('npmi', 0),
        'metrics': {
            **results['metrics'],
            'modularity': modularity,
        },
    }

    print(f"Modularity: {modularity:.4f}, NPMI: {results['metrics'].get('npmi', 0):.4f}, "
          f"Clusters: {results['num_clusters']}")

    return result


# ============================================================
# 실험 실행
# ============================================================

def run_all_experiments(optimal_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """모든 모델 실험 실행"""

    all_results = {
        'Louvain': [],
        'Leiden': [],
        'Ours': [],
    }

    for seed in EXPERIMENT_CONFIG['random_seeds']:
        print(f"\n{'='*80}")
        print(f"Random Seed: {seed}")
        print(f"{'='*80}")

        # 공통 그래프 구축 (Louvain, Leiden용)
        word_graph, doc_service = build_shared_word_graph(seed)

        # 1. Louvain
        louvain_result = run_louvain_experiment(word_graph, seed)
        all_results['Louvain'].append(louvain_result)

        # 2. Leiden
        leiden_result = run_leiden_experiment(word_graph, seed)
        all_results['Leiden'].append(leiden_result)

        # 3. Ours (GRACE)
        ours_result = run_ours_experiment(seed, optimal_config)
        all_results['Ours'].append(ours_result)

    return all_results


# ============================================================
# 통계 분석
# ============================================================

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """평균, 표준편차 계산"""

    # 공통 메트릭
    common_metrics = ['modularity', 'npmi']

    # 임베딩 기반 메트릭 (Ours만 해당)
    embedding_metrics = EXPERIMENT_CONFIG['eval_metrics']

    all_metrics = common_metrics + embedding_metrics

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
        # NaN 제거 (전통적 방법에서는 임베딩 메트릭이 없음)
        valid_values = [v for v in values if not np.isnan(v)]

        if len(valid_values) > 0:
            stats[metric] = {
                'mean': np.mean(valid_values),
                'std': np.std(valid_values, ddof=1) if len(valid_values) > 1 else 0,
                'values': valid_values,
            }
        else:
            stats[metric] = {
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
    n_comparisons = n_models * n_metrics  # = 4
    alpha_corrected = alpha / n_comparisons  # Bonferroni correction

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

            # Ours가 더 나은지 판단 (메트릭 방향 고려)
            ours_mean = np.mean(ours_values)
            other_mean = np.mean(other_values)

            if metric in ['modularity', 'npmi', 'silhouette', 'calinski_harabasz']:
                # Higher is better
                ours_is_better = ours_mean > other_mean
            elif metric in ['davies_bouldin']:
                # Lower is better
                ours_is_better = ours_mean < other_mean
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
                'ours_mean': float(ours_mean),
                'other_mean': float(other_mean),
            }

    return test_results


# ============================================================
# 결과 저장
# ============================================================

def save_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    all_statistics: Dict[str, Dict[str, Dict[str, float]]],
    test_results: Dict[str, Any],
    output_dir: Path
):
    """결과 저장"""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Raw results
    raw_path = output_dir / f"raw_results_{timestamp}.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Raw results: {raw_path}")

    # 2. Summary statistics
    summary_data = []
    for model_name in ['Louvain', 'Leiden', 'Ours']:
        stats = all_statistics[model_name]
        row = {'Model': model_name}

        for metric in ['modularity', 'npmi', 'silhouette', 'davies_bouldin', 'calinski_harabasz']:
            if metric in stats and not np.isnan(stats[metric]['mean']):
                row[f'{metric}_mean'] = stats[metric]['mean']
                row[f'{metric}_std'] = stats[metric]['std']
            else:
                row[f'{metric}_mean'] = None
                row[f'{metric}_std'] = None

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary: {summary_path}")

    # 3. Statistical tests
    tests_path = output_dir / f"statistical_tests_{timestamp}.json"
    with open(tests_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistical tests: {tests_path}")

    # 4. Table 5 (논문용)
    table5_path = output_dir / f"table5_rq3_{timestamp}.txt"
    with open(table5_path, 'w', encoding='utf-8') as f:
        f.write(f"Table 5: RQ3 - Overall Performance Comparison (Mean ± Std, n={len(EXPERIMENT_CONFIG['random_seeds'])})\n")
        f.write("="*140 + "\n\n")

        header = f"{'Model':<15}  {'Modularity ↑':>18}  {'NPMI ↑':>25}  {'Silhouette ↑':>18}  {'DBI ↓':>18}  {'CHI ↑':>18}"
        f.write(header + "\n")
        f.write("-"*140 + "\n")

        for model_name in ['Louvain', 'Leiden', 'Ours']:
            stats = all_statistics[model_name]
            row = f"{model_name:<15}"

            # Modularity
            if 'modularity' in stats:
                mod_mean = stats['modularity']['mean']
                mod_std = stats['modularity']['std']
                sig = ''
                if model_name != 'Ours' and 'metrics' in test_results:
                    if 'modularity' in test_results['metrics']:
                        sig = test_results['metrics']['modularity'][model_name]['significance']
                row += f"  {mod_mean:8.3f}±{mod_std:.3f}{sig:>3}"
            else:
                row += f"  {'N/A':>18}"

            # NPMI (effect size 추가)
            if 'npmi' in stats:
                npmi_mean = stats['npmi']['mean']
                npmi_std = stats['npmi']['std']
                annotation = ''
                if model_name == 'Ours':
                    # Ours는 기준점
                    annotation = '    '
                elif 'metrics' in test_results and 'npmi' in test_results['metrics']:
                    test_info = test_results['metrics']['npmi'][model_name]
                    sig = test_info['significance']
                    d = test_info['cohens_d']
                    annotation = f"{sig:>3}d={d:+.2f}"

                row += f"  {npmi_mean:8.3f}±{npmi_std:.3f}{annotation}"
            else:
                row += f"  {'N/A':>25}"

            # Embedding metrics (Ours만)
            for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
                if metric in stats and not np.isnan(stats[metric]['mean']):
                    row += f"  {stats[metric]['mean']:8.3f}±{stats[metric]['std']:.3f}"
                else:
                    row += f"  {'-':>18}"

            f.write(row + "\n")

        f.write("\n" + "="*140 + "\n")
        f.write(f"Bonferroni corrected α = {test_results['alpha_corrected']:.4f} ({test_results['n_comparisons']} comparisons)\n")
        f.write("Significance: *** p < α_corrected, ** p < 0.01, * p < 0.05, ns = not significant\n")
        f.write("d = Cohen's d (effect size): |d| < 0.2 (negligible), < 0.5 (small), < 0.8 (medium), ≥ 0.8 (large)\n")
        f.write("↑ Higher is better, ↓ Lower is better\n")
        f.write("- : Not applicable (traditional methods don't produce embeddings)\n")

        # 통계 검정 상세 결과 추가
        f.write("\n" + "="*140 + "\n")
        f.write("Statistical Test Details (Ours vs Others)\n")
        f.write("="*140 + "\n\n")

        for metric in ['modularity', 'npmi']:
            if metric not in test_results['metrics']:
                continue

            f.write(f"\n{metric.upper()}:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Comparison':<20} {'Ours Mean':>12} {'Other Mean':>12} {'t-stat':>12} {'p-value':>12} {'d':>8} {'Better':>10}\n")
            f.write("-" * 100 + "\n")

            for other_model in ['Louvain', 'Leiden']:
                test_info = test_results['metrics'][metric][other_model]
                better = '✓ Ours' if test_info['ours_is_better'] else '✗ Other'

                f.write(
                    f"{'Ours vs ' + other_model:<20} "
                    f"{test_info['ours_mean']:12.4f} "
                    f"{test_info['other_mean']:12.4f} "
                    f"{test_info['t_statistic']:12.4f} "
                    f"{test_info['p_value']:12.4f} "
                    f"{test_info['cohens_d']:8.2f} "
                    f"{better:>10}\n"
                )

    print(f"✓ Table 5: {table5_path}")


# ============================================================
# 시각화
# ============================================================

def plot_results(
    all_statistics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path
):
    """결과 시각화"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = ['Louvain', 'Leiden', 'Ours']

    # 1. Modularity vs NPMI 비교
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Modularity
    ax = axes[0]
    mod_means = [all_statistics[m]['modularity']['mean'] for m in models]
    mod_stds = [all_statistics[m]['modularity']['std'] for m in models]

    bars = ax.bar(models, mod_means, yerr=mod_stds, capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')

    ax.set_ylabel('Modularity', fontsize=12, fontweight='bold')
    ax.set_title('Modularity Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(mod_means) * 1.2)

    # NPMI
    ax = axes[1]
    npmi_means = [all_statistics[m]['npmi']['mean'] for m in models]
    npmi_stds = [all_statistics[m]['npmi']['std'] for m in models]

    bars = ax.bar(models, npmi_means, yerr=npmi_stds, capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7, edgecolor='black')

    ax.set_ylabel('NPMI', fontsize=12, fontweight='bold')
    ax.set_title('NPMI Comparison (Semantic Coherence)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(npmi_means) * 1.2)

    plt.tight_layout()
    comparison_path = plots_dir / f"rq3_comparison_{timestamp}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot: {comparison_path}")

    # 2. Performance Heatmap (정규화)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Ours만 모든 메트릭 있음
    heatmap_data = []
    for model in models:
        row = []
        for metric in ['modularity', 'npmi', 'silhouette', 'calinski_harabasz']:
            if metric in all_statistics[model] and not np.isnan(all_statistics[model][metric]['mean']):
                row.append(all_statistics[model][metric]['mean'])
            else:
                row.append(0)  # N/A는 0으로
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=models,
        columns=['Modularity', 'NPMI', 'Silhouette', 'CHI']
    )

    # 정규화
    heatmap_normalized = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())

    sns.heatmap(heatmap_normalized, annot=heatmap_df.round(3), fmt='g',
                cmap='YlGnBu', cbar_kws={'label': 'Normalized Score'},
                linewidths=0.5, ax=ax)

    ax.set_title('RQ3: Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')

    plt.tight_layout()
    heatmap_path = plots_dir / f"rq3_heatmap_{timestamp}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap: {heatmap_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """RQ3 실험 메인 함수"""

    print("="*80)
    print("RQ3 실험: 전통적 그래프 클러스터링 vs 제안 방법")
    print("="*80)
    print(f"데이터: {EXPERIMENT_CONFIG['csv_path']}")
    print(f"문서 수: {EXPERIMENT_CONFIG['num_documents']}")
    print(f"Vocab: {EXPERIMENT_CONFIG['top_n_words']} 단어")
    print(f"Random seeds: {EXPERIMENT_CONFIG['random_seeds']}")
    print("="*80)

    # 최적 설정 로드
    optimal_config = load_optimal_config()

    # 실험 실행
    print("\n실험 시작...")
    all_results = run_all_experiments(optimal_config)

    # 통계 분석
    print("\n" + "="*80)
    print("통계 분석 중...")
    print("="*80)

    all_statistics = {}
    for model_name, results in all_results.items():
        all_statistics[model_name] = calculate_statistics(results)

    # 통계 검정
    test_results = perform_statistical_tests(all_statistics)

    # 결과 출력
    print("\n" + "="*80)
    print("결과 요약")
    print("="*80)

    for model_name in ['Louvain', 'Leiden', 'Ours']:
        stats = all_statistics[model_name]
        print(f"\n{model_name}:")
        for metric in ['modularity', 'npmi', 'silhouette']:
            if metric in stats and not np.isnan(stats[metric]['mean']):
                print(f"  {metric:20s}: {stats[metric]['mean']:.4f} ± {stats[metric]['std']:.4f}")

    # 결과 저장
    output_dir = Path(EXPERIMENT_CONFIG['output_dir'])
    save_results(all_results, all_statistics, test_results, output_dir)

    # 시각화
    print("\n시각화 생성 중...")
    plot_results(all_statistics, output_dir)

    print("\n" + "="*80)
    print("✅ RQ3 실험 완료!")
    print("="*80)
    print(f"결과 디렉토리: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
