#!/usr/bin/env python3
"""
RQ1 실험: 단일 vs 다중 임베딩 비교

연구 질문: 다중 임베딩(Word2Vec + DistilBERT)이 단일 임베딩보다
           클러스터링 성능을 향상시키는가?

비교 모델:
1. W2V-KMeans: Word2Vec만 사용 + K-means
2. BERT-KMeans: DistilBERT만 사용 + K-means
3. Concat-KMeans: Word2Vec + DistilBERT (GraphMAE 없이) + K-means
4. Ours: Word2Vec + DistilBERT + GraphMAE + K-means

실험 설정:
- 데이터: AG News
- Vocab size: 500
- Random seeds: [42, 123, 456, 789, 101] (n=5)
- PCA dim: 256 (각 임베딩)
- GraphMAE epochs: 500
- Edge weight threshold: 5

평가 지표:
- NPMI (주 지표)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

통계 검정:
- Two-tailed t-test
- Bonferroni correction
- Cohen's d (effect size)
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
from scipy import stats

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.services.GRACE import GRACEConfig, GRACEPipeline
from core.services.GRACE.ClusteringService import ClusteringService
from core.services.Metric import MetricsService


# ============================================================
# 실험 설정
# ============================================================

EXPERIMENT_CONFIG = {
    # 데이터
    'csv_path': 'data/ag_news.csv',
    'num_documents': 10000,  # 프로토콜에 따라 조정 가능
    'text_column': 'text',

    # 그래프
    'top_n_words': 500,
    'exclude_stopwords': True,
    'edge_weight_threshold': 5,

    # 임베딩 (프로토콜: PCA 256차원)
    'pca_dim': 256,  # 각 임베딩의 차원

    # GraphMAE
    'graphmae_epochs': 500,
    'mask_rate': 0.5,  # 프로토콜 기본값
    'encoder_type': 'gat',
    'decoder_type': 'gat',

    # 클러스터링
    'clustering_method': 'kmeans',
    'min_clusters': 5,
    'max_clusters': 20,

    # 재현성
    'random_seeds': [42, 123, 456, 789, 101],

    # 평가
    'eval_metrics': ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

    # 출력
    'output_dir': 'results/rq1_single_vs_multi',
    'verbose': True,
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


def create_grace_config(
    embedding_method: str,
    random_seed: int,
    use_graphmae: bool = True
) -> GRACEConfig:
    """실험용 GRACE 설정 생성"""

    if embedding_method == 'w2v':
        # Word2Vec only
        embed_size = EXPERIMENT_CONFIG['pca_dim']
        w2v_dim = EXPERIMENT_CONFIG['pca_dim']
        bert_dim = 0
    elif embedding_method == 'bert':
        # BERT only
        embed_size = EXPERIMENT_CONFIG['pca_dim']
        w2v_dim = 0
        bert_dim = EXPERIMENT_CONFIG['pca_dim']
    elif embedding_method == 'concat':
        # Word2Vec + BERT (concat)
        embed_size = EXPERIMENT_CONFIG['pca_dim'] * 2  # 512
        w2v_dim = EXPERIMENT_CONFIG['pca_dim']
        bert_dim = EXPERIMENT_CONFIG['pca_dim']
    else:
        raise ValueError(f"Unknown embedding_method: {embedding_method}")

    config = GRACEConfig(
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
        embedding_method=embedding_method,
        embed_size=embed_size,
        w2v_dim=w2v_dim,
        bert_dim=bert_dim,

        # GraphMAE
        graphmae_epochs=EXPERIMENT_CONFIG['graphmae_epochs'] if use_graphmae else 0,
        graphmae_lr=0.001,
        mask_rate=EXPERIMENT_CONFIG['mask_rate'],
        encoder_type=EXPERIMENT_CONFIG['encoder_type'],
        decoder_type=EXPERIMENT_CONFIG['decoder_type'],

        # 클러스터링
        clustering_method=EXPERIMENT_CONFIG['clustering_method'],
        num_clusters=None,  # Elbow method
        min_clusters=EXPERIMENT_CONFIG['min_clusters'],
        max_clusters=EXPERIMENT_CONFIG['max_clusters'],

        # 평가
        eval_metrics=EXPERIMENT_CONFIG['eval_metrics'],

        # 재현성
        random_seed=random_seed,

        # 출력
        save_results=False,  # 실험 스크립트에서 직접 관리
        output_dir=EXPERIMENT_CONFIG['output_dir'],
        save_graph_viz=False,
        save_embeddings=False,
        verbose=False,  # 실험 시 로그 최소화
    )

    return config


def run_single_experiment(
    model_name: str,
    embedding_method: str,
    random_seed: int,
    use_graphmae: bool = True
) -> Dict[str, Any]:
    """단일 실험 실행"""

    print(f"  [Seed {random_seed}] {model_name} 실행 중...")

    # 시드 고정
    set_all_seeds(random_seed)

    # 설정 생성
    config = create_grace_config(embedding_method, random_seed, use_graphmae)

    # 파이프라인 실행
    if use_graphmae:
        # GraphMAE 포함 (Ours 모델)
        pipeline = GRACEPipeline(config)
        results = pipeline.run()
    else:
        # GraphMAE 없이 임베딩만 사용 (Concat-KMeans)
        pipeline = GRACEPipeline(config)

        # 1-3단계만 실행 (데이터 로드 ~ 임베딩 계산)
        pipeline.load_and_preprocess()
        pipeline.build_semantic_network()
        pipeline.compute_node_features()

        # GraphMAE 없이 바로 클러스터링
        embeddings = pipeline.node_features

        # 클러스터링
        clustering_service = ClusteringService(random_state=random_seed)
        cluster_labels, best_k, _, _ = clustering_service.auto_clustering_elbow(
            embeddings,
            min_clusters=config.min_clusters,
            max_clusters=config.max_clusters,
            n_init=10
        )

        # 평가
        metrics_service = MetricsService()
        metrics = metrics_service.calculate_metrics(
            embeddings,
            cluster_labels,
            config.eval_metrics,
            word_graph=pipeline.word_graph,
            total_docs=len(pipeline.documents) if pipeline.documents else 0
        )

        results = {
            'metrics': metrics,
            'num_clusters': best_k,
        }

    # 결과 정리
    result = {
        'model_name': model_name,
        'embedding_method': embedding_method,
        'use_graphmae': use_graphmae,
        'random_seed': random_seed,
        'num_clusters': results.get('num_clusters', 0),
        'metrics': results.get('metrics', {}),
    }

    print(f"    → NPMI: {results['metrics'].get('npmi', 0):.4f}, "
          f"Silhouette: {results['metrics'].get('silhouette', 0):.4f}, "
          f"Clusters: {results.get('num_clusters', 0)}")

    return result


def run_model_experiments(
    model_name: str,
    embedding_method: str,
    use_graphmae: bool = True
) -> List[Dict[str, Any]]:
    """모델별 5회 반복 실험"""

    print(f"\n{'='*80}")
    print(f"모델: {model_name}")
    print(f"  임베딩: {embedding_method}, GraphMAE: {use_graphmae}")
    print(f"{'='*80}")

    results = []
    for seed in EXPERIMENT_CONFIG['random_seeds']:
        result = run_single_experiment(model_name, embedding_method, seed, use_graphmae)
        results.append(result)

    return results


# ============================================================
# 통계 분석
# ============================================================

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """평균, 표준편차 계산"""

    metrics_list = {metric: [] for metric in EXPERIMENT_CONFIG['eval_metrics']}

    for result in results:
        for metric in EXPERIMENT_CONFIG['eval_metrics']:
            value = result['metrics'].get(metric, np.nan)
            metrics_list[metric].append(value)

    stats_dict = {}
    for metric, values in metrics_list.items():
        stats_dict[metric] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),  # 표본 표준편차
            'values': values,
        }

    return stats_dict


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Cohen's d 효과 크기 계산"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0


def perform_statistical_tests(
    all_statistics: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Any]:
    """통계적 검정 수행 (Ours vs others)"""

    # Bonferroni correction
    n_comparisons = 3  # Ours vs (W2V, BERT, Concat)
    alpha = 0.05
    alpha_corrected = alpha / n_comparisons

    test_results = {}

    for metric in EXPERIMENT_CONFIG['eval_metrics']:
        test_results[metric] = {}

        ours_values = all_statistics['Ours'][metric]['values']

        for model_name in ['W2V-KMeans', 'BERT-KMeans', 'Concat-KMeans']:
            other_values = all_statistics[model_name][metric]['values']

            # Two-tailed t-test
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

            test_results[metric][model_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'p_value_corrected': alpha_corrected,
                'cohens_d': effect_size,
                'significance': significance,
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

    # 1. Raw results (JSON)
    raw_results_path = output_dir / f"raw_results_{timestamp}.json"
    with open(raw_results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Raw results saved: {raw_results_path}")

    # 2. Summary statistics (CSV)
    summary_data = []
    for model_name, stats in all_statistics.items():
        row = {'Model': model_name}
        for metric in EXPERIMENT_CONFIG['eval_metrics']:
            row[f'{metric}_mean'] = stats[metric]['mean']
            row[f'{metric}_std'] = stats[metric]['std']
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"summary_statistics_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary statistics saved: {summary_path}")

    # 3. Statistical tests (JSON)
    tests_path = output_dir / f"statistical_tests_{timestamp}.json"
    with open(tests_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistical tests saved: {tests_path}")

    # 4. Table 1 형식 (Markdown/LaTeX ready)
    table1_path = output_dir / f"table1_rq1_{timestamp}.txt"
    with open(table1_path, 'w', encoding='utf-8') as f:
        f.write("Table 1: RQ1 - Single vs Multi-modal Embeddings (Mean ± Std, n=5)\n")
        f.write("="*100 + "\n\n")

        # Header
        header = f"{'Model':<20}"
        for metric in EXPERIMENT_CONFIG['eval_metrics']:
            header += f"{metric.upper():^18}"
        f.write(header + "\n")
        f.write("-"*100 + "\n")

        # Rows
        for model_name in ['W2V-KMeans', 'BERT-KMeans', 'Concat-KMeans', 'Ours']:
            stats = all_statistics[model_name]
            row = f"{model_name:<20}"

            for metric in EXPERIMENT_CONFIG['eval_metrics']:
                mean = stats[metric]['mean']
                std = stats[metric]['std']

                # 유의성 표시 (Ours 제외)
                sig = ''
                if model_name != 'Ours' and test_results[metric][model_name]['significance'] != 'ns':
                    sig = test_results[metric][model_name]['significance']

                row += f"{mean:.3f}±{std:.3f}{sig:>3}   "

            f.write(row + "\n")

        f.write("\n" + "="*100 + "\n")
        f.write("Significance: *** p < 0.017 (Bonferroni corrected), ** p < 0.01, * p < 0.05\n")

    print(f"✓ Table 1 saved: {table1_path}")


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

    # Figure 2: NPMI Comparison (Bar Chart with Error Bars)
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ['W2V-KMeans', 'BERT-KMeans', 'Concat-KMeans', 'Ours']
    npmi_means = [all_statistics[m]['npmi']['mean'] for m in models]
    npmi_stds = [all_statistics[m]['npmi']['std'] for m in models]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(models, npmi_means, yerr=npmi_stds, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel('NPMI Score', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('RQ1: NPMI Comparison (Mean ± Std, n=5)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(npmi_means) * 1.2)

    plt.tight_layout()
    npmi_plot_path = plots_dir / f"fig2_npmi_comparison_{timestamp}.png"
    plt.savefig(npmi_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ NPMI plot saved: {npmi_plot_path}")

    # Performance Heatmap (모든 지표)
    fig, ax = plt.subplots(figsize=(10, 6))

    # 데이터 준비 (정규화)
    heatmap_data = []
    for model in models:
        row = []
        for metric in EXPERIMENT_CONFIG['eval_metrics']:
            row.append(all_statistics[model][metric]['mean'])
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(
        heatmap_data,
        index=models,
        columns=[m.upper() for m in EXPERIMENT_CONFIG['eval_metrics']]
    )

    # 정규화 (0-1 범위)
    heatmap_normalized = (heatmap_df - heatmap_df.min()) / (heatmap_df.max() - heatmap_df.min())

    sns.heatmap(heatmap_normalized, annot=heatmap_df.round(3), fmt='g',
                cmap='YlGnBu', cbar_kws={'label': 'Normalized Score'},
                linewidths=0.5, ax=ax)

    ax.set_title('RQ1: Performance Heatmap (All Metrics)', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xlabel('')

    plt.tight_layout()
    heatmap_path = plots_dir / f"fig_heatmap_rq1_{timestamp}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved: {heatmap_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """RQ1 실험 메인 함수"""

    print("="*80)
    print("RQ1 실험: 단일 vs 다중 임베딩 비교")
    print("="*80)
    print(f"데이터: {EXPERIMENT_CONFIG['csv_path']}")
    print(f"문서 수: {EXPERIMENT_CONFIG['num_documents']}")
    print(f"Vocab: {EXPERIMENT_CONFIG['top_n_words']} 단어")
    print(f"Random seeds: {EXPERIMENT_CONFIG['random_seeds']}")
    print(f"출력: {EXPERIMENT_CONFIG['output_dir']}")
    print("="*80)

    # 결과 저장
    all_results = {}

    # 1. W2V-KMeans
    all_results['W2V-KMeans'] = run_model_experiments(
        'W2V-KMeans', 'w2v', use_graphmae=False
    )

    # 2. BERT-KMeans
    all_results['BERT-KMeans'] = run_model_experiments(
        'BERT-KMeans', 'bert', use_graphmae=False
    )

    # 3. Concat-KMeans (GraphMAE 없이)
    all_results['Concat-KMeans'] = run_model_experiments(
        'Concat-KMeans', 'concat', use_graphmae=False
    )

    # 4. Ours (GraphMAE 포함)
    all_results['Ours'] = run_model_experiments(
        'Ours', 'concat', use_graphmae=True
    )

    # 통계 계산
    print("\n" + "="*80)
    print("통계 분석 중...")
    print("="*80)

    all_statistics = {}
    for model_name, results in all_results.items():
        all_statistics[model_name] = calculate_statistics(results)

    # 통계적 검정
    test_results = perform_statistical_tests(all_statistics)

    # 결과 출력
    print("\n" + "="*80)
    print("결과 요약")
    print("="*80)

    for model_name, stats in all_statistics.items():
        print(f"\n{model_name}:")
        for metric in EXPERIMENT_CONFIG['eval_metrics']:
            mean = stats[metric]['mean']
            std = stats[metric]['std']
            print(f"  {metric:20s}: {mean:.4f} ± {std:.4f}")

    # 결과 저장
    output_dir = Path(EXPERIMENT_CONFIG['output_dir'])
    save_results(all_results, all_statistics, test_results, output_dir)

    # 시각화
    print("\n시각화 생성 중...")
    plot_results(all_statistics, output_dir)

    print("\n" + "="*80)
    print("✅ RQ1 실험 완료!")
    print("="*80)
    print(f"결과 디렉토리: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
