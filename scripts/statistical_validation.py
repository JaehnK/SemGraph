"""
Statistical validation for RQ2 ablation study results.
Performs t-tests with Bonferroni correction comparing best configuration vs others.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_results(json_path: str) -> List[Dict]:
    """Load raw results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_metric_by_param(results: List[Dict], param_name: str, metric_name: str) -> Dict[float, List[float]]:
    """
    Extract metric values grouped by parameter value.

    Args:
        results: List of experiment results
        param_name: Name of the parameter being varied (e.g., 'mask_rate')
        metric_name: Name of the metric (e.g., 'npmi', 'silhouette')

    Returns:
        Dictionary mapping parameter value to list of metric values
    """
    data = {}
    for result in results:
        param_value = result['param_value']
        metric_value = result['metrics'][metric_name]

        if param_value not in data:
            data[param_value] = []
        data[param_value].append(metric_value)

    return data


def find_best_config(data: Dict[float, List[float]], higher_is_better: bool = True) -> float:
    """
    Find the best configuration based on mean metric value.

    Args:
        data: Dictionary mapping parameter value to metric values
        higher_is_better: Whether higher metric values are better

    Returns:
        Parameter value with best mean performance
    """
    means = {param: np.mean(values) for param, values in data.items()}

    if higher_is_better:
        best_param = max(means, key=means.get)
    else:
        best_param = min(means, key=means.get)

    return best_param


def perform_ttest_with_bonferroni(best_values: List[float],
                                   other_data: Dict[float, List[float]],
                                   metric_name: str,
                                   higher_is_better: bool = True) -> pd.DataFrame:
    """
    Perform t-tests comparing best config vs others with Bonferroni correction.

    Args:
        best_values: Metric values for the best configuration
        other_data: Dictionary of other configurations and their metric values
        metric_name: Name of the metric being tested
        higher_is_better: Whether higher values are better

    Returns:
        DataFrame with test results
    """
    n_comparisons = len(other_data)
    bonferroni_alpha = 0.05 / n_comparisons

    results = []

    for param_value, values in other_data.items():
        # Perform independent samples t-test
        t_stat, p_value = stats.ttest_ind(best_values, values)

        # Calculate means and effect size (Cohen's d)
        mean_best = np.mean(best_values)
        mean_other = np.mean(values)

        pooled_std = np.sqrt((np.var(best_values, ddof=1) + np.var(values, ddof=1)) / 2)
        cohens_d = (mean_best - mean_other) / pooled_std if pooled_std > 0 else 0

        # Determine significance
        is_significant = p_value < bonferroni_alpha

        # Determine if best is better (considering direction)
        if higher_is_better:
            best_is_better = mean_best > mean_other
        else:
            best_is_better = mean_best < mean_other

        results.append({
            'param_value': param_value,
            'mean': mean_other,
            'std': np.std(values, ddof=1),
            't_statistic': t_stat,
            'p_value': p_value,
            'bonferroni_alpha': bonferroni_alpha,
            'significant': is_significant,
            'cohens_d': cohens_d,
            'best_is_better': best_is_better
        })

    df = pd.DataFrame(results)
    df = df.sort_values('param_value')

    return df


def analyze_parameter(json_path: str, param_name: str, output_dir: Path) -> Dict:
    """
    Analyze one parameter's ablation study with statistical tests.

    Args:
        json_path: Path to raw results JSON
        param_name: Name of parameter (e.g., 'mask_rate', 'epochs', 'pca_dim')
        output_dir: Directory to save results

    Returns:
        Dictionary with analysis results
    """
    results = load_results(json_path)

    # Metrics to analyze (metric_name, higher_is_better)
    metrics = [
        ('silhouette', True),
        ('davies_bouldin', False),
        ('calinski_harabasz', True),
        ('npmi', True)
    ]

    analysis_results = {}

    for metric_name, higher_is_better in metrics:
        print(f"\n{'='*80}")
        print(f"Analyzing {param_name.upper()} - {metric_name.upper()}")
        print(f"{'='*80}")

        # Extract data
        data = extract_metric_by_param(results, param_name, metric_name)

        # Find best configuration
        best_param = find_best_config(data, higher_is_better)
        best_values = data[best_param]

        print(f"\nBest {param_name}: {best_param}")
        print(f"Best mean {metric_name}: {np.mean(best_values):.4f} ± {np.std(best_values, ddof=1):.4f}")

        # Prepare other configs for comparison
        other_data = {k: v for k, v in data.items() if k != best_param}

        if len(other_data) == 0:
            print("No other configurations to compare.")
            continue

        # Perform t-tests
        test_results = perform_ttest_with_bonferroni(
            best_values, other_data, metric_name, higher_is_better
        )

        print(f"\nComparisons (n={len(best_values)} per group, Bonferroni α={test_results['bonferroni_alpha'].iloc[0]:.4f}):")
        print(test_results.to_string(index=False))

        # Save results
        analysis_results[metric_name] = {
            'best_param': best_param,
            'best_mean': float(np.mean(best_values)),
            'best_std': float(np.std(best_values, ddof=1)),
            'comparisons': test_results.to_dict('records')
        }

    # Save to JSON
    output_file = output_dir / f'statistical_tests_{param_name}.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✓ Saved statistical tests to {output_file}")

    return analysis_results


def create_summary_table(param_name: str, analysis_results: Dict) -> str:
    """Create a formatted summary table for the parameter."""

    lines = []
    lines.append(f"\n{'='*100}")
    lines.append(f"Statistical Validation Summary: {param_name.upper()}")
    lines.append(f"{'='*100}\n")

    for metric_name, results in analysis_results.items():
        best_param = results['best_param']
        best_mean = results['best_mean']
        best_std = results['best_std']

        lines.append(f"\n{metric_name.upper()} (Best: {param_name}={best_param}, {best_mean:.4f}±{best_std:.4f})")
        lines.append("-" * 100)
        cohens_d_header = "Cohen's d"
        lines.append(f"{'Param':<10} {'Mean±Std':<20} {'t-stat':<12} {'p-value':<12} {'Significant':<12} {cohens_d_header:<12}")
        lines.append("-" * 100)

        for comp in results['comparisons']:
            sig_marker = '***' if comp['significant'] else 'ns'
            lines.append(
                f"{comp['param_value']:<10} "
                f"{comp['mean']:.4f}±{comp['std']:.4f}   "
                f"{comp['t_statistic']:>10.4f}  "
                f"{comp['p_value']:>10.4f}  "
                f"{sig_marker:<12} "
                f"{comp['cohens_d']:>10.4f}"
            )

    return '\n'.join(lines)


def main():
    """Main analysis function."""
    candidates = [
        ROOT / "artifacts" / "runs" / "results" / "rq2_ablation",
        ROOT / "results" / "rq2_ablation",
    ]
    results_dir = next((p for p in candidates if p.exists()), candidates[0])

    # Analyze each parameter
    parameters = [
        ('mask_rate', results_dir / 'mask_rate' / 'raw_results_20251021_202751.json'),
        ('epochs', results_dir / 'epochs' / 'raw_results_20251021_221124.json'),
        ('pca_dim', results_dir / 'pca_dim' / 'raw_results_20251021_232532.json')
    ]

    all_summaries = []

    for param_name, json_path in parameters:
        print(f"\n\n{'#'*100}")
        print(f"# PARAMETER: {param_name.upper()}")
        print(f"{'#'*100}")

        output_dir = json_path.parent
        analysis_results = analyze_parameter(str(json_path), param_name, output_dir)

        # Create and save summary table
        summary = create_summary_table(param_name, analysis_results)
        all_summaries.append(summary)

        summary_file = output_dir / f'statistical_summary_{param_name}.txt'
        with open(summary_file, 'w') as f:
            f.write(summary)

        print(f"✓ Saved summary to {summary_file}")

    # Save combined summary
    combined_file = results_dir / 'statistical_validation_summary.txt'
    with open(combined_file, 'w') as f:
        f.write('\n\n'.join(all_summaries))

    print(f"\n\n{'='*100}")
    print(f"✓ All statistical validation completed!")
    print(f"✓ Combined summary saved to {combined_file}")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
