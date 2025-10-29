#!/usr/bin/env python3
"""
RQ2 실험: Ablation Study - GraphMAE2 하이퍼파라미터 영향 분석

연구 질문: GraphMAE2의 하이퍼파라미터가 클러스터링 성능에 어떤 영향을 미치는가?

실험 구성:
2.1 Mask Rate 실험: [0.3, 0.5, 0.75, 0.9]
2.2 Training Epochs 실험: [250, 500, 1000, 2000]
2.3 Output Dimensionality 실험: [64, 128, 256]
    - Input 고정: Attention Fusion(256d) - RQ1과 동일
    - GraphMAE 출력: 256d
    - PCA 차원 축소: [64d, 128d, 256d (no PCA)]
    - 연구 질문: 저차원 압축이 클러스터링 성능에 미치는 영향?

실험 설정:
- 데이터: AG News
- Vocab size: 500
- Random seeds: [42, 123, 456, 789, 101] (n=5)
- 임베딩: Attention Fusion (Gated, 256d) - RQ1과 동일
- Edge weight threshold: 5

평가 지표:
- NPMI (주 지표)
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Training time
- Explained Variance (RQ2.3만)

목표:
- 각 하이퍼파라미터의 최적값 결정
- 차원 축소의 영향 분석 (RQ2.3)
- RQ3에서 사용할 최종 설정 도출
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from core.services.GRACE import GRACEConfig, GRACEPipeline
from core.services.clustering import SphericalKMeansClusteringService
from core.services.Metric import MetricsService


# ============================================================
# 실험 설정
# ============================================================

BASE_CONFIG = {
    # 데이터
    'csv_path': 'data/ag_news.csv',
    'num_documents': 10000,
    'text_column': 'text',

    # 그래프
    'top_n_words': 500,
    'exclude_stopwords': True,
    'edge_weight_threshold': 5,

    # 임베딩 (기본값) - RQ1과 동일하게 Attention Fusion
    'embedding_method': 'attention',
    'fusion_type': 'gated',
    'embed_dim': 256,  # 모든 실험에서 256d 고정 (RQ1과 동일)

    # GraphMAE (기본값)
    'graphmae_epochs': 500,  # 기본값 (2.2에서 테스트)
    'mask_rate': 0.5,  # 기본값 (2.1에서 테스트)
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
    'output_dir': 'results/rq2_ablation',
    'verbose': False,
}

# 실험 변수
ABLATION_EXPERIMENTS = {
    'mask_rate': [0.3, 0.5, 0.75, 0.9],
    'epochs': [250, 500, 1000, 2000],
    'pca_dim': [64, 128, 256],  # RQ2.3: GraphMAE 출력(256d) 후 PCA 축소 (256은 baseline)
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
    random_seed: int,
    mask_rate: float = None,
    epochs: int = None,
) -> GRACEConfig:
    """실험용 GRACE 설정 생성 (RQ1과 동일한 Attention Fusion)

    Args:
        random_seed: 랜덤 시드
        mask_rate: GraphMAE mask rate (None이면 기본값)
        epochs: GraphMAE epochs (None이면 기본값)
    """

    # 기본값 사용
    _mask_rate = mask_rate if mask_rate is not None else BASE_CONFIG['mask_rate']
    _epochs = epochs if epochs is not None else BASE_CONFIG['graphmae_epochs']

    # RQ1과 동일: Attention Fusion (256d)
    embed_size = BASE_CONFIG['embed_dim']  # 256d
    w2v_dim = BASE_CONFIG['embed_dim']     # 256d (PCA 축소)
    bert_dim = BASE_CONFIG['embed_dim']    # 256d (PCA 축소)

    config = GRACEConfig(
        # 데이터
        csv_path=BASE_CONFIG['csv_path'],
        num_documents=BASE_CONFIG['num_documents'],
        text_column=BASE_CONFIG['text_column'],

        # 그래프
        top_n_words=BASE_CONFIG['top_n_words'],
        exclude_stopwords=BASE_CONFIG['exclude_stopwords'],
        edge_weight_threshold=BASE_CONFIG['edge_weight_threshold'],
        edge_top_k=-1,

        # 임베딩 (Attention Fusion)
        embedding_method=BASE_CONFIG['embedding_method'],
        embed_size=embed_size,
        w2v_dim=w2v_dim,
        bert_dim=bert_dim,
        fusion_type=BASE_CONFIG['fusion_type'],

        # GraphMAE
        graphmae_epochs=_epochs,
        graphmae_lr=0.001,
        mask_rate=_mask_rate,
        encoder_type=BASE_CONFIG['encoder_type'],
        decoder_type=BASE_CONFIG['decoder_type'],

        # 클러스터링
        clustering_method=BASE_CONFIG['clustering_method'],
        num_clusters=None,
        min_clusters=BASE_CONFIG['min_clusters'],
        max_clusters=BASE_CONFIG['max_clusters'],

        # 평가
        eval_metrics=BASE_CONFIG['eval_metrics'],

        # 재현성
        random_seed=random_seed,

        # 출력
        save_results=False,
        output_dir=BASE_CONFIG['output_dir'],
        save_graph_viz=False,
        save_embeddings=False,
        verbose=BASE_CONFIG['verbose'],
    )

    return config


def run_single_experiment(
    config: GRACEConfig,
    experiment_name: str,
    param_value: Any,
    random_seed: int
) -> Dict[str, Any]:
    """단일 실험 실행"""

    print(f"    [Seed {random_seed}] {experiment_name}={param_value} 실행 중...", end=' ')

    # 시드 고정
    set_all_seeds(random_seed)

    # 시간 측정 시작
    start_time = time.time()

    # 파이프라인 실행
    pipeline = GRACEPipeline(config)
    results = pipeline.run()

    # 실행 시간
    elapsed_time = time.time() - start_time

    # 결과 정리
    result = {
        'experiment_name': experiment_name,
        'param_value': param_value,
        'random_seed': random_seed,
        'num_clusters': results.get('num_clusters', 0),
        'metrics': results.get('metrics', {}),
        'training_time_seconds': elapsed_time,
    }

    print(f"NPMI: {results['metrics'].get('npmi', 0):.4f}, "
          f"Time: {elapsed_time/60:.1f}min")

    return result


# ============================================================
# 2.1 Mask Rate 실험
# ============================================================

def run_mask_rate_experiments() -> List[Dict[str, Any]]:
    """Mask Rate 영향 분석"""

    print("\n" + "="*80)
    print("RQ2.1: Mask Rate 실험")
    print("="*80)
    print(f"Mask rates: {ABLATION_EXPERIMENTS['mask_rate']}")
    print(f"고정 파라미터: epochs={BASE_CONFIG['graphmae_epochs']}, "
          f"embed_dim={BASE_CONFIG['embed_dim']}d (Attention Fusion)")
    print("="*80)

    all_results = []

    for mask_rate in ABLATION_EXPERIMENTS['mask_rate']:
        print(f"\n[Mask Rate = {mask_rate}]")

        for seed in BASE_CONFIG['random_seeds']:
            config = create_grace_config(
                random_seed=seed,
                mask_rate=mask_rate,
                epochs=BASE_CONFIG['graphmae_epochs'],
            )

            result = run_single_experiment(
                config, 'mask_rate', mask_rate, seed
            )
            all_results.append(result)

    return all_results


# ============================================================
# 2.2 Training Epochs 실험
# ============================================================

def run_epochs_experiments() -> List[Dict[str, Any]]:
    """Training Epochs 영향 분석"""

    print("\n" + "="*80)
    print("RQ2.2: Training Epochs 실험")
    print("="*80)
    print(f"Epochs: {ABLATION_EXPERIMENTS['epochs']}")
    print(f"고정 파라미터: mask_rate={BASE_CONFIG['mask_rate']}, "
          f"embed_dim={BASE_CONFIG['embed_dim']}d (Attention Fusion)")
    print("="*80)

    all_results = []

    for epochs in ABLATION_EXPERIMENTS['epochs']:
        print(f"\n[Epochs = {epochs}]")

        for seed in BASE_CONFIG['random_seeds']:
            config = create_grace_config(
                random_seed=seed,
                mask_rate=BASE_CONFIG['mask_rate'],
                epochs=epochs,
            )

            result = run_single_experiment(
                config, 'epochs', epochs, seed
            )
            all_results.append(result)

    return all_results


# ============================================================
# 2.3 Embedding Dimension 실험 (GraphMAE 출력 후 PCA)
# ============================================================

def run_pca_dim_experiments() -> List[Dict[str, Any]]:
    """GraphMAE 출력 차원 축소가 클러스터링 성능에 미치는 영향 분석

    Input (고정): Attention Fusion(256d) - RQ1과 동일
    GraphMAE 출력: 256d
    PCA 축소: [64, 128, 256 (no PCA)]
    """

    print("\n" + "="*80)
    print("RQ2.3: GraphMAE 출력 차원 축소 실험")
    print("="*80)
    print(f"Input 고정: Attention Fusion(256d) - RQ1과 동일")
    print(f"GraphMAE 출력: 256d")
    print(f"PCA 축소: {ABLATION_EXPERIMENTS['pca_dim']} (256는 PCA 없음)")
    print(f"고정 파라미터: mask_rate={BASE_CONFIG['mask_rate']}, "
          f"epochs={BASE_CONFIG['graphmae_epochs']}")
    print("="*80)

    all_results = []

    for pca_dim in ABLATION_EXPERIMENTS['pca_dim']:
        print(f"\n[PCA Dim = {pca_dim}]")

        for seed in BASE_CONFIG['random_seeds']:
            print(f"    [Seed {seed}] GraphMAE(256d) → PCA({pca_dim}d) 실행 중...", end=' ')

            # 시드 고정
            set_all_seeds(seed)
            start_time = time.time()

            # 1. 파이프라인 실행 (Attention Fusion 256d - RQ1과 동일)
            config = create_grace_config(
                random_seed=seed,
                mask_rate=BASE_CONFIG['mask_rate'],
                epochs=BASE_CONFIG['graphmae_epochs'],
            )
            config.save_results = False
            config.verbose = False

            pipeline = GRACEPipeline(config)

            # GraphMAE까지만 실행
            pipeline.load_and_preprocess()
            pipeline.build_semantic_network()
            pipeline.compute_node_features()
            pipeline.train_graphmae()

            # 2. GraphMAE 출력에 PCA 적용
            from sklearn.decomposition import PCA
            from core.services.Metric import MetricsService

            embeddings = pipeline.graphmae_embeddings.cpu().numpy()

            # PCA 축소 (256이면 축소하지 않음)
            if pca_dim < embeddings.shape[1]:
                pca = PCA(n_components=pca_dim, random_state=seed)
                embeddings = pca.fit_transform(embeddings)
                explained_var = pca.explained_variance_ratio_.sum()
            else:
                explained_var = 1.0

            embeddings = torch.tensor(embeddings, dtype=torch.float32)

            # 3. 클러스터링
            from core.services.clustering import SphericalKMeansClusteringService
            clustering_service = SphericalKMeansClusteringService(random_state=seed)

            cluster_labels, best_k, _, _ = clustering_service.auto_clustering(
                embeddings,
                min_clusters=BASE_CONFIG['min_clusters'],
                max_clusters=BASE_CONFIG['max_clusters'],
                n_init=10
            )

            # 4. 평가
            metrics_service = MetricsService()
            metrics = metrics_service.calculate_metrics(
                embeddings=embeddings.numpy(),
                labels=cluster_labels,
                metric_names=BASE_CONFIG['eval_metrics'],
                word_graph=pipeline.word_graph,
                total_docs=BASE_CONFIG['num_documents']
            )

            elapsed_time = time.time() - start_time

            # 5. 결과 정리
            result = {
                'experiment_name': 'pca_dim',
                'param_value': pca_dim,
                'random_seed': seed,
                'num_clusters': best_k,
                'metrics': metrics,
                'training_time_seconds': elapsed_time,
                'explained_variance': explained_var,
            }

            print(f"NPMI: {metrics.get('npmi', 0):.4f}, "
                  f"Var: {explained_var:.3f}, "
                  f"Time: {elapsed_time/60:.1f}min")

            all_results.append(result)

    return all_results


# ============================================================
# 통계 분석
# ============================================================

def calculate_statistics_by_param(
    results: List[Dict[str, Any]],
    param_name: str
) -> Dict[Any, Dict[str, Any]]:
    """파라미터 값별 통계 계산"""

    # 파라미터 값별로 그룹화
    grouped = {}
    for result in results:
        param_value = result['param_value']
        if param_value not in grouped:
            grouped[param_value] = []
        grouped[param_value].append(result)

    # 통계 계산
    statistics = {}
    for param_value, group_results in grouped.items():
        metrics_dict = {metric: [] for metric in BASE_CONFIG['eval_metrics']}
        times = []
        explained_vars = []

        for result in group_results:
            for metric in BASE_CONFIG['eval_metrics']:
                value = result['metrics'].get(metric, np.nan)
                metrics_dict[metric].append(value)
            times.append(result['training_time_seconds'])

            # explained_variance (PCA 실험에만 있음)
            if 'explained_variance' in result:
                explained_vars.append(result['explained_variance'])

        # 평균/표준편차
        stats = {}
        for metric, values in metrics_dict.items():
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'values': values,
            }

        stats['training_time'] = {
            'mean': np.mean(times),
            'std': np.std(times, ddof=1),
            'values': times,
        }

        # explained_variance (있는 경우만)
        if explained_vars:
            stats['explained_variance'] = {
                'mean': np.mean(explained_vars),
                'std': np.std(explained_vars, ddof=1),
                'values': explained_vars,
            }

        statistics[param_value] = stats

    return statistics


def find_optimal_parameter(
    statistics: Dict[Any, Dict[str, Any]],
    metric: str = 'npmi',
    higher_is_better: bool = True
) -> Tuple[Any, float]:
    """최적 파라미터 값 찾기"""

    best_param = None
    best_value = -np.inf if higher_is_better else np.inf

    for param_value, stats in statistics.items():
        mean_value = stats[metric]['mean']

        if higher_is_better:
            if mean_value > best_value:
                best_value = mean_value
                best_param = param_value
        else:
            if mean_value < best_value:
                best_value = mean_value
                best_param = param_value

    return best_param, best_value


# ============================================================
# 결과 저장
# ============================================================

def save_ablation_results(
    experiment_name: str,
    results: List[Dict[str, Any]],
    statistics: Dict[Any, Dict[str, Any]],
    output_dir: Path
):
    """Ablation 실험 결과 저장"""

    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Raw results
    raw_path = exp_dir / f"raw_results_{timestamp}.json"
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Raw results: {raw_path}")

    # 2. Summary statistics
    summary_data = []
    for param_value, stats in sorted(statistics.items()):
        row = {experiment_name: param_value}

        for metric in BASE_CONFIG['eval_metrics']:
            row[f'{metric}_mean'] = stats[metric]['mean']
            row[f'{metric}_std'] = stats[metric]['std']

        row['time_mean'] = stats['training_time']['mean']
        row['time_std'] = stats['training_time']['std']

        # explained_variance (PCA 실험만)
        if 'explained_variance' in stats:
            row['explained_var_mean'] = stats['explained_variance']['mean']
            row['explained_var_std'] = stats['explained_variance']['std']

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = exp_dir / f"summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary: {summary_path}")

    # 3. Table 형식
    table_path = exp_dir / f"table_{experiment_name}_{timestamp}.txt"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(f"Table: Effect of {experiment_name.replace('_', ' ').title()} (Mean ± Std, n=5)\n")
        f.write("="*120 + "\n\n")

        # Header
        header = f"{experiment_name.upper():>15}"
        for metric in BASE_CONFIG['eval_metrics']:
            header += f"  {metric.upper():>18}"
        header += f"  {'TIME(min)':>18}"

        # PCA 실험이면 explained variance 추가
        has_explained_var = any('explained_variance' in stats for stats in statistics.values())
        if has_explained_var:
            header += f"  {'EXP_VAR':>18}"

        f.write(header + "\n")
        f.write("-"*120 + "\n")

        # Rows
        for param_value in sorted(statistics.keys()):
            stats = statistics[param_value]
            row = f"{str(param_value):>15}"

            for metric in BASE_CONFIG['eval_metrics']:
                mean = stats[metric]['mean']
                std = stats[metric]['std']
                row += f"  {mean:8.3f}±{std:.3f}"

            time_mean = stats['training_time']['mean'] / 60  # 분 단위
            time_std = stats['training_time']['std'] / 60
            row += f"  {time_mean:8.1f}±{time_std:.1f}"

            # explained_variance (PCA 실험만)
            if has_explained_var and 'explained_variance' in stats:
                exp_var_mean = stats['explained_variance']['mean']
                exp_var_std = stats['explained_variance']['std']
                row += f"  {exp_var_mean:8.3f}±{exp_var_std:.3f}"

            f.write(row + "\n")

    print(f"✓ Table: {table_path}")

    return exp_dir


# ============================================================
# 시각화
# ============================================================

def plot_ablation_results(
    experiment_name: str,
    statistics: Dict[Any, Dict[str, Any]],
    output_dir: Path
):
    """Ablation 결과 시각화"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = output_dir / experiment_name / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    param_values = sorted(statistics.keys())

    # 1. NPMI Line Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    npmi_means = [statistics[pv]['npmi']['mean'] for pv in param_values]
    npmi_stds = [statistics[pv]['npmi']['std'] for pv in param_values]

    ax.errorbar(param_values, npmi_means, yerr=npmi_stds,
                marker='o', markersize=8, linewidth=2, capsize=5,
                label='NPMI', color='#2ca02c')

    ax.set_xlabel(experiment_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('NPMI Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Effect of {experiment_name.replace("_", " ").title()} on NPMI',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    npmi_path = plots_dir / f"npmi_{experiment_name}_{timestamp}.png"
    plt.savefig(npmi_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ NPMI plot: {npmi_path}")

    # 2. Multi-metric Line Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics_to_plot = BASE_CONFIG['eval_metrics']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        means = [statistics[pv][metric]['mean'] for pv in param_values]
        stds = [statistics[pv][metric]['std'] for pv in param_values]

        ax.errorbar(param_values, means, yerr=stds,
                    marker='o', markersize=8, linewidth=2, capsize=5,
                    color=colors[idx])

        ax.set_xlabel(experiment_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.upper()} vs {experiment_name.replace("_", " ").title()}',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    multi_path = plots_dir / f"multi_metric_{experiment_name}_{timestamp}.png"
    plt.savefig(multi_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Multi-metric plot: {multi_path}")

    # 3. Box Plot (NPMI 분포)
    fig, ax = plt.subplots(figsize=(10, 6))

    data_for_box = []
    labels_for_box = []
    for pv in param_values:
        data_for_box.append(statistics[pv]['npmi']['values'])
        labels_for_box.append(str(pv))

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                     notch=True, showmeans=True)

    # 색상 설정
    for patch in bp['boxes']:
        patch.set_facecolor('#87CEEB')
        patch.set_alpha(0.7)

    ax.set_xlabel(experiment_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('NPMI Score', fontsize=12, fontweight='bold')
    ax.set_title(f'NPMI Distribution by {experiment_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    box_path = plots_dir / f"boxplot_{experiment_name}_{timestamp}.png"
    plt.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Box plot: {box_path}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """RQ2 실험 메인 함수"""

    print("="*80)
    print("RQ2 실험: Ablation Study - GraphMAE2 하이퍼파라미터 영향")
    print("="*80)
    print(f"데이터: {BASE_CONFIG['csv_path']}")
    print(f"문서 수: {BASE_CONFIG['num_documents']}")
    print(f"Vocab: {BASE_CONFIG['top_n_words']} 단어")
    print(f"Random seeds: {BASE_CONFIG['random_seeds']}")
    print(f"출력: {BASE_CONFIG['output_dir']}")
    print("="*80)

    output_dir = Path(BASE_CONFIG['output_dir'])
    optimal_config = {}

    # ========================================
    # 2.1 Mask Rate 실험
    # ========================================
    print("\n" + "🔬 2.1 Mask Rate 실험 시작...")
    mask_rate_results = run_mask_rate_experiments()
    mask_rate_stats = calculate_statistics_by_param(mask_rate_results, 'mask_rate')

    # 최적값 찾기
    optimal_mask_rate, optimal_npmi = find_optimal_parameter(mask_rate_stats, 'npmi')
    print(f"\n✓ 최적 Mask Rate: {optimal_mask_rate} (NPMI: {optimal_npmi:.4f})")
    optimal_config['mask_rate'] = optimal_mask_rate

    # 결과 저장 및 시각화
    save_ablation_results('mask_rate', mask_rate_results, mask_rate_stats, output_dir)
    plot_ablation_results('mask_rate', mask_rate_stats, output_dir)

    # ========================================
    # 2.2 Training Epochs 실험
    # ========================================
    print("\n" + "🔬 2.2 Training Epochs 실험 시작...")
    epochs_results = run_epochs_experiments()
    epochs_stats = calculate_statistics_by_param(epochs_results, 'epochs')

    # 최적값 찾기
    optimal_epochs, optimal_npmi = find_optimal_parameter(epochs_stats, 'npmi')
    print(f"\n✓ 최적 Epochs: {optimal_epochs} (NPMI: {optimal_npmi:.4f})")
    optimal_config['epochs'] = optimal_epochs

    # 결과 저장 및 시각화
    save_ablation_results('epochs', epochs_results, epochs_stats, output_dir)
    plot_ablation_results('epochs', epochs_stats, output_dir)

    # ========================================
    # 2.3 Embedding Dimension 실험
    # ========================================
    print("\n" + "🔬 2.3 Embedding Dimension 실험 시작...")
    pca_dim_results = run_pca_dim_experiments()
    pca_dim_stats = calculate_statistics_by_param(pca_dim_results, 'pca_dim')

    # 최적값 찾기
    optimal_pca_dim, optimal_npmi = find_optimal_parameter(pca_dim_stats, 'npmi')
    print(f"\n✓ 최적 PCA Dim: {optimal_pca_dim} (Final: {optimal_pca_dim*2}, NPMI: {optimal_npmi:.4f})")
    optimal_config['pca_dim'] = optimal_pca_dim

    # 결과 저장 및 시각화
    save_ablation_results('pca_dim', pca_dim_results, pca_dim_stats, output_dir)
    plot_ablation_results('pca_dim', pca_dim_stats, output_dir)

    # ========================================
    # 최적 설정 저장
    # ========================================
    print("\n" + "="*80)
    print("✅ RQ2 실험 완료!")
    print("="*80)
    print("\n최적 하이퍼파라미터 설정:")
    print(f"  Mask Rate: {optimal_config['mask_rate']}")
    print(f"  Epochs: {optimal_config['epochs']}")
    print(f"  PCA Dim: {optimal_config['pca_dim']} (Final: {optimal_config['pca_dim']*2})")
    print("="*80)

    # 최적 설정 저장
    optimal_config_path = output_dir / "optimal_config.json"
    with open(optimal_config_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_config, f, indent=2)
    print(f"\n✓ 최적 설정 저장: {optimal_config_path}")

    print(f"\n📁 결과 디렉토리: {output_dir.absolute()}")
    print("\n⚠️  RQ3 실험에서 위 최적 설정을 사용하세요!")


if __name__ == "__main__":
    main()
