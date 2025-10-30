#!/usr/bin/env python3
"""
Ablation Study Main Script for GRACE

체계적인 ablation study를 수행하여 각 하이퍼파라미터의 영향을 분석합니다.

Ablation Parameters:
- Embedding Input Method: w2v_only, bert_only, concat
- Input Size (concat only): 64, 128, 256
- Output Size: 64, 128, 256
- Mask Rate: 0.3, 0.5, 0.75, 0.9
- Epoch: 100, 250, 500, 1000
- Decoder Type: gcn, gat, mlp, linear
- Edge Filtering (Top-K): -1 (all), 5, 10, 20, 50

Usage:
    python ablation_main.py --all                    # 모든 ablation 실행
    python ablation_main.py --embedding              # Embedding method ablation만
    python ablation_main.py --mask-rate              # Mask rate ablation만
    python ablation_main.py --epochs                 # Epochs ablation만
    python ablation_main.py --output-size            # Output size ablation만
    python ablation_main.py --decoder                # Decoder type ablation만
    python ablation_main.py --edge-filtering         # Edge filtering ablation만
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import torch
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
from copy import deepcopy

# 재현성을 위한 멀티스레딩 제한 (import 전에 설정)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# PYTHONPATH 설정
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "core"))

from colorama import init, Fore, Style
from core.services.GRACE import GRACEPipeline, GRACEConfig

# Colorama 초기화
init(autoreset=True)


def print_banner():
    """Ablation Study 배너 출력"""
    banner = f"""
{Fore.CYAN}{'=' * 80}
{Fore.GREEN}  GRACE Ablation Study
{Fore.YELLOW}  Systematic Hyperparameter Analysis
{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}
"""
    print(banner)


def create_base_config() -> GRACEConfig:
    """기본 설정 생성 (Ablation의 베이스라인)"""

    config = GRACEConfig(
        # 데이터
        csv_path="data/reddit_mental_health_cleaned.csv",
        num_documents=10000,
        text_column='body',

        # 그래프
        top_n_words=500,
        exclude_stopwords=True,
        edge_top_k=-1,  # edge_filtering ablation에서 변경
        edge_weight_threshold=0.0,

        # 임베딩
        embedding_method='concat',  # embedding ablation에서 변경
        embed_size=128,  # output_size ablation에서 변경
        w2v_dim=64,
        bert_dim=64,

        # GraphMAE
        graphmae_epochs=100,  # epochs ablation에서 변경
        graphmae_lr=0.001,
        graphmae_device=None,
        mask_rate=0.3,  # mask_rate ablation에서 변경
        encoder_type='gcn',
        decoder_type='gcn',  # decoder ablation에서 변경

        # 클러스터링
        clustering_method='kmeans',
        num_clusters=None,
        min_clusters=3,
        max_clusters=20,

        # 평가
        eval_metrics=['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi'],

        # 출력
        save_results=True,
        output_dir="results/ablation",
        save_graph_viz=True,  # 네트워크 시각화 활성화
        save_embeddings=False,

        # 디버그
        verbose=False,  # ablation에서는 간단하게
        log_interval=50
    )

    return config


def run_single_experiment(
    config: GRACEConfig,
    experiment_name: str,
    verbose: bool = False
) -> Tuple[Dict[str, float], int]:
    """단일 실험 실행"""

    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}Running: {experiment_name}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    try:
        # 파이프라인 실행
        pipeline = GRACEPipeline(config)
        results = pipeline.run()

        # 결과 추출
        metrics = results.get('metrics', {})
        n_clusters = results.get('n_clusters', 0)

        # 간단한 결과 출력
        print(f"  ✓ Clusters: {n_clusters}")
        print(f"  ✓ Silhouette: {metrics.get('silhouette', 0.0):.4f}")
        print(f"  ✓ NPMI: {metrics.get('npmi', 0.0):.4f}")

        return metrics, n_clusters

    except Exception as e:
        print(f"{Fore.RED}  ✗ Failed: {e}{Style.RESET_ALL}")
        import traceback
        if verbose:
            traceback.print_exc()
        return {}, 0


def ablation_embedding_method(
    base_config: GRACEConfig,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Embedding Input Method Ablation"""

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.GREEN}Ablation 1: Embedding Input Method")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")

    methods = ['w2v', 'bert', 'concat']
    # concat일 때만 input_size를 변경
    input_sizes_for_concat = [64, 128, 256]

    results = {}

    for method in methods:
        if method == 'concat':
            # concat일 때는 input_size도 ablation
            for input_size in input_sizes_for_concat:
                config = deepcopy(base_config)
                config.embedding_method = method
                config.w2v_dim = input_size // 2
                config.bert_dim = input_size // 2
                config.embed_size = input_size  # w2v_dim + bert_dim
                config.output_dir = str(output_dir / f"embedding_{method}_input{input_size}")

                exp_name = f"Embedding={method}, InputSize={input_size}"
                metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

                results[f"{method}_input{input_size}"] = {
                    'method': method,
                    'input_size': input_size,
                    'n_clusters': n_clusters,
                    'metrics': metrics
                }
        else:
            # w2v, bert는 고정 크기
            config = deepcopy(base_config)
            config.embedding_method = method
            config.embed_size = 128
            config.output_dir = str(output_dir / f"embedding_{method}")

            exp_name = f"Embedding={method}"
            metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

            results[method] = {
                'method': method,
                'input_size': 128,
                'n_clusters': n_clusters,
                'metrics': metrics
            }

    return results


def ablation_output_size(
    base_config: GRACEConfig,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Output Size (embed_size) Ablation"""

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.GREEN}Ablation 2: Output Size (Embedding Dimension)")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")

    output_sizes = [64, 128, 256]
    results = {}

    for output_size in output_sizes:
        config = deepcopy(base_config)
        # concat 모드에서 출력 크기 조정
        config.w2v_dim = output_size // 2
        config.bert_dim = output_size // 2
        config.embed_size = output_size
        config.output_dir = str(output_dir / f"output_size_{output_size}")

        exp_name = f"OutputSize={output_size}"
        metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

        results[f"output_{output_size}"] = {
            'output_size': output_size,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

    return results


def ablation_mask_rate(
    base_config: GRACEConfig,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Mask Rate Ablation"""

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.GREEN}Ablation 3: Mask Rate")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")

    mask_rates = [0.3, 0.5, 0.75, 0.9]
    results = {}

    for mask_rate in mask_rates:
        config = deepcopy(base_config)
        config.mask_rate = mask_rate
        config.output_dir = str(output_dir / f"mask_rate_{mask_rate}")

        exp_name = f"MaskRate={mask_rate}"
        metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

        results[f"mask_{mask_rate}"] = {
            'mask_rate': mask_rate,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

    return results


def ablation_epochs(
    base_config: GRACEConfig,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Training Epochs Ablation"""

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.GREEN}Ablation 4: Training Epochs")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")

    epochs_list = [100, 250, 500, 1000]
    results = {}

    for epochs in epochs_list:
        config = deepcopy(base_config)
        config.graphmae_epochs = epochs
        config.output_dir = str(output_dir / f"epochs_{epochs}")

        exp_name = f"Epochs={epochs}"
        metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

        results[f"epochs_{epochs}"] = {
            'epochs': epochs,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

    return results


def ablation_decoder_type(
    base_config: GRACEConfig,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Decoder Type Ablation"""

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.GREEN}Ablation 5: Decoder Type")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")

    decoder_types = ['gcn', 'gat', 'mlp', 'linear']
    results = {}

    for decoder_type in decoder_types:
        config = deepcopy(base_config)
        config.decoder_type = decoder_type
        config.output_dir = str(output_dir / f"decoder_{decoder_type}")

        exp_name = f"DecoderType={decoder_type}"
        metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

        results[f"decoder_{decoder_type}"] = {
            'decoder_type': decoder_type,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

    return results


def ablation_edge_filtering(
    base_config: GRACEConfig,
    output_dir: Path,
    verbose: bool = False
) -> Dict[str, Any]:
    """Edge Filtering Ablation (Top-K)"""

    print(f"\n{Fore.MAGENTA}{'=' * 80}")
    print(f"{Fore.GREEN}Ablation 6: Edge Filtering (Top-K per Node)")
    print(f"{Fore.MAGENTA}{'=' * 80}{Style.RESET_ALL}\n")

    top_k_values = [-1, 5, 10, 20, 50]  # -1: no filtering
    results = {}

    for top_k in top_k_values:
        config = deepcopy(base_config)
        config.edge_top_k = top_k
        config.output_dir = str(output_dir / f"edge_top_k_{top_k}")

        exp_name = f"EdgeTopK={top_k if top_k > 0 else 'All'}"
        metrics, n_clusters = run_single_experiment(config, exp_name, verbose)

        results[f"top_k_{top_k}"] = {
            'edge_top_k': top_k,
            'n_clusters': n_clusters,
            'metrics': metrics
        }

    return results


def save_ablation_results(
    results: Dict[str, Any],
    ablation_name: str,
    output_dir: Path
):
    """Ablation 결과 저장"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"{ablation_name}_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{Fore.GREEN}✓ Results saved to: {results_file}{Style.RESET_ALL}")


def print_ablation_summary(
    results: Dict[str, Any],
    ablation_name: str,
    metric_key: str = 'silhouette'
):
    """Ablation 결과 요약 출력"""

    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"{Fore.GREEN}Summary: {ablation_name}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    # 메트릭별로 정렬
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('metrics', {}).get(metric_key, 0.0),
        reverse=True
    )

    print(f"  {'Config':<30} {'Clusters':>10} {'Silhouette':>12} {'NPMI':>10}")
    print(f"  {'-' * 70}")

    for config_name, result in sorted_results:
        n_clusters = result.get('n_clusters', 0)
        metrics = result.get('metrics', {})
        silhouette = metrics.get('silhouette', 0.0)
        npmi = metrics.get('npmi', 0.0)

        print(f"  {config_name:<30} {n_clusters:>10} {silhouette:>12.4f} {npmi:>10.4f}")

    # Best configuration
    best_config, best_result = sorted_results[0]
    print(f"\n{Fore.GREEN}Best Configuration:{Style.RESET_ALL}")
    print(f"  {best_config}")
    print(f"  {metric_key}: {best_result['metrics'].get(metric_key, 0.0):.4f}")


def run_full_ablation_study(
    base_config: GRACEConfig,
    output_dir: Path,
    selected_ablations: List[str],
    verbose: bool = False
) -> Dict[str, Any]:
    """전체 Ablation Study 실행"""

    all_results = {}

    if 'embedding' in selected_ablations or 'all' in selected_ablations:
        results_embedding = ablation_embedding_method(base_config, output_dir / "embedding", verbose)
        all_results['embedding_method'] = results_embedding
        save_ablation_results(results_embedding, "embedding_method", output_dir)
        print_ablation_summary(results_embedding, "Embedding Input Method")

    if 'output-size' in selected_ablations or 'all' in selected_ablations:
        results_output = ablation_output_size(base_config, output_dir / "output_size", verbose)
        all_results['output_size'] = results_output
        save_ablation_results(results_output, "output_size", output_dir)
        print_ablation_summary(results_output, "Output Size")

    if 'mask-rate' in selected_ablations or 'all' in selected_ablations:
        results_mask = ablation_mask_rate(base_config, output_dir / "mask_rate", verbose)
        all_results['mask_rate'] = results_mask
        save_ablation_results(results_mask, "mask_rate", output_dir)
        print_ablation_summary(results_mask, "Mask Rate")

    if 'epochs' in selected_ablations or 'all' in selected_ablations:
        results_epochs = ablation_epochs(base_config, output_dir / "epochs", verbose)
        all_results['epochs'] = results_epochs
        save_ablation_results(results_epochs, "epochs", output_dir)
        print_ablation_summary(results_epochs, "Training Epochs")

    if 'decoder' in selected_ablations or 'all' in selected_ablations:
        results_decoder = ablation_decoder_type(base_config, output_dir / "decoder", verbose)
        all_results['decoder_type'] = results_decoder
        save_ablation_results(results_decoder, "decoder_type", output_dir)
        print_ablation_summary(results_decoder, "Decoder Type")

    if 'edge-filtering' in selected_ablations or 'all' in selected_ablations:
        results_edge = ablation_edge_filtering(base_config, output_dir / "edge_filtering", verbose)
        all_results['edge_filtering'] = results_edge
        save_ablation_results(results_edge, "edge_filtering", output_dir)
        print_ablation_summary(results_edge, "Edge Filtering (Top-K)")

    # 전체 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_results_file = output_dir / f"ablation_study_full_{timestamp}.json"

    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{Fore.GREEN}{'=' * 80}")
    print(f"✓ Full ablation study completed!")
    print(f"✓ Results saved to: {final_results_file}")
    print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}\n")

    return all_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="GRACE Ablation Study - Systematic Hyperparameter Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 모든 ablation 실행
  python ablation_main.py --all

  # 특정 ablation만 실행
  python ablation_main.py --embedding --mask-rate

  # Verbose 모드
  python ablation_main.py --all --verbose

  # 작은 데이터셋으로 테스트
  python ablation_main.py --embedding --max-docs 1000
        """
    )

    # Ablation 선택
    parser.add_argument('--all', action='store_true', help='모든 ablation 실행')
    parser.add_argument('--embedding', action='store_true', help='Embedding method ablation')
    parser.add_argument('--output-size', action='store_true', help='Output size ablation')
    parser.add_argument('--mask-rate', action='store_true', help='Mask rate ablation')
    parser.add_argument('--epochs', action='store_true', help='Training epochs ablation')
    parser.add_argument('--decoder', action='store_true', help='Decoder type ablation')
    parser.add_argument('--edge-filtering', action='store_true', help='Edge filtering (top-k) ablation')

    # 공통 설정
    parser.add_argument('--data', type=str, default='data/reddit_mental_health_cleaned.csv',
                        help='데이터 파일 경로')
    parser.add_argument('--max-docs', type=int, default=10000,
                        help='최대 문서 수')
    parser.add_argument('--output', type=str, default='results/ablation',
                        help='출력 디렉토리')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cuda', 'cpu'],
                        help='디바이스')
    parser.add_argument('--verbose', action='store_true',
                        help='상세 출력')

    args = parser.parse_args()

    # 배너 출력
    print_banner()

    # 선택된 ablation 확인
    selected_ablations = []
    if args.all:
        selected_ablations = ['all']
    else:
        if args.embedding:
            selected_ablations.append('embedding')
        if args.output_size:
            selected_ablations.append('output-size')
        if args.mask_rate:
            selected_ablations.append('mask-rate')
        if args.epochs:
            selected_ablations.append('epochs')
        if args.decoder:
            selected_ablations.append('decoder')
        if args.edge_filtering:
            selected_ablations.append('edge-filtering')

    if not selected_ablations:
        print(f"{Fore.RED}Error: No ablation selected. Use --all or specific flags.{Style.RESET_ALL}")
        parser.print_help()
        sys.exit(1)

    print(f"\n{Fore.CYAN}Selected Ablations:{Style.RESET_ALL}")
    for ablation in selected_ablations:
        print(f"  ✓ {ablation}")

    # 기본 설정 생성
    base_config = create_base_config()
    base_config.csv_path = args.data
    base_config.num_documents = args.max_docs
    base_config.graphmae_device = args.device

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{Fore.CYAN}Configuration:{Style.RESET_ALL}")
    print(f"  Data: {base_config.csv_path}")
    print(f"  Max docs: {base_config.num_documents}")
    print(f"  Device: {base_config.graphmae_device if base_config.graphmae_device else 'auto'}")
    print(f"  Output: {output_dir}")

    # Ablation Study 실행
    try:
        results = run_full_ablation_study(
            base_config,
            output_dir,
            selected_ablations,
            verbose=args.verbose
        )

    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Interrupted by user.{Style.RESET_ALL}")
        sys.exit(1)

    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
