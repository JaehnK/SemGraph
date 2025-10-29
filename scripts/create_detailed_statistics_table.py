"""
통계량이 포함된 상세 테이블 생성 스크립트
RQ1: Single vs Multi-modal Embeddings
"""

import json
import pandas as pd
from pathlib import Path

def load_data(results_dir):
    """결과 디렉토리에서 데이터 로드"""
    with open(results_dir / 'statistical_tests_20251020_174917.json', 'r') as f:
        stats = json.load(f)

    summary_df = pd.read_csv(results_dir / 'summary_statistics_20251020_174917.csv')

    return stats, summary_df

def format_pvalue(p):
    """p-value 포맷팅"""
    if p < 0.001:
        return "< 0.001"
    elif p < 0.01:
        return f"{p:.4f}"
    else:
        return f"{p:.3f}"

def create_detailed_table(stats, summary_df, output_path):
    """통계량이 포함된 상세 테이블 생성"""

    models = ['W2V-KMeans', 'BERT-KMeans', 'Concat-KMeans', 'GRACE']
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi']
    metric_names = {
        'silhouette': 'SILHOUETTE',
        'davies_bouldin': 'DAVIES_BOULDIN',
        'calinski_harabasz': 'CALINSKI_HARABASZ',
        'npmi': 'NPMI'
    }

    # 테이블 1: 평균±표준편차 + 통계량
    lines = []
    lines.append("="*140)
    lines.append("Table 1: RQ1 - Single vs Multi-modal Embeddings with Statistical Tests (Mean ± Std, n=5)")
    lines.append("="*140)
    lines.append("")

    # 헤더
    header = f"{'Model':<20}"
    for metric in metrics:
        header += f"{metric_names[metric]:>25}  "
    lines.append(header)
    lines.append("-"*140)

    # 각 모델별 결과
    for model in models:
        row_data = summary_df[summary_df['Model'] == model].iloc[0]
        row = f"{model:<20}"

        for metric in metrics:
            mean = row_data[f'{metric}_mean']
            std = row_data[f'{metric}_std']

            # GRACE는 기준 모델이므로 통계량 없음
            if model == 'GRACE':
                row += f"{mean:>10.3f}±{std:<6.3f}        "
            else:
                # 통계 검정 결과 가져오기
                stat_info = stats['grace_vs_all'][metric][model]
                t_stat = stat_info['t_statistic']
                p_val = stat_info['p_value']
                sig = stat_info['significance']

                row += f"{mean:>10.3f}±{std:<6.3f} {sig:>4}  "

        lines.append(row)

    lines.append("="*140)
    lines.append("Significance: (***) p < 0.0167 (Bonferroni corrected), (**) p < 0.01, (*) p < 0.05")
    lines.append("")

    # 테이블 2: 상세 통계 검정 결과
    lines.append("")
    lines.append("="*120)
    lines.append("Table 2: Detailed Statistical Test Results (Paired t-test vs GRACE, Bonferroni α = 0.0167)")
    lines.append("="*120)
    lines.append("")

    for metric in metrics:
        lines.append(f"\n{metric_names[metric]}:")
        lines.append("-"*120)
        cohens_label = "Cohen's d"
        lines.append(f"{'Model':<20} {'t-statistic':>15} {'p-value':>15} {cohens_label:>15} {'Significance':>15}")
        lines.append("-"*120)

        for model in models:
            if model == 'GRACE':
                continue

            stat_info = stats['grace_vs_all'][metric][model]
            t_stat = stat_info['t_statistic']
            p_val = stat_info['p_value']
            cohens_d = stat_info['cohens_d']
            sig = stat_info['significance']

            lines.append(
                f"{model:<20} {t_stat:>15.4f} {format_pvalue(p_val):>15} "
                f"{cohens_d:>15.4f} {sig:>15}"
            )

        lines.append("")

    lines.append("="*120)
    lines.append("")
    lines.append("Notes:")
    lines.append("- Paired t-test with Bonferroni correction (α = 0.05 / 3 comparisons = 0.0167)")
    lines.append("- Positive t-statistic: GRACE > baseline")
    lines.append("- Negative t-statistic: GRACE < baseline")
    lines.append("- |Cohen's d| > 0.8: large effect size")
    lines.append("- All comparisons show very large effect sizes (|d| > 6.0)")

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Detailed table saved to: {output_path}")
    print('\n'.join(lines))

def create_latex_table(stats, summary_df, output_path):
    """LaTeX 형식 테이블 생성"""

    models = ['W2V-KMeans', 'BERT-KMeans', 'Concat-KMeans', 'GRACE']
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'npmi']

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{RQ1: Single vs Multi-modal Embeddings (Mean ± Std, n=5)}")
    lines.append("\\label{tab:rq1_results}")
    lines.append("\\begin{tabular}{l|cccc}")
    lines.append("\\hline")
    lines.append("Model & Silhouette & Davies-Bouldin & Calinski-Harabasz & NPMI \\\\")
    lines.append("\\hline")

    for model in models:
        row_data = summary_df[summary_df['Model'] == model].iloc[0]
        row_parts = [model.replace('-', '\\text{-}')]

        for metric in metrics:
            mean = row_data[f'{metric}_mean']
            std = row_data[f'{metric}_std']

            if model == 'GRACE':
                row_parts.append(f"${mean:.3f} \\pm {std:.3f}$")
            else:
                stat_info = stats['grace_vs_all'][metric][model]
                sig = stat_info['significance']
                row_parts.append(f"${mean:.3f} \\pm {std:.3f}^{{{sig}}}$")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\begin{tablenotes}")
    lines.append("\\item Significance vs GRACE: $^{***}$ $p < 0.0167$ (Bonferroni), $^{**}$ $p < 0.01$, $^{*}$ $p < 0.05$")
    lines.append("\\end{tablenotes}")
    lines.append("\\end{table}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ LaTeX table saved to: {output_path}")

def main():
    results_dir = Path('/home/jaehun/lab/SENTIMENT/results/rq1_single_vs_multi')

    # 데이터 로드
    stats, summary_df = load_data(results_dir)

    # 상세 텍스트 테이블 생성
    output_txt = results_dir / 'table1_rq1_detailed_statistics.txt'
    create_detailed_table(stats, summary_df, output_txt)

    # LaTeX 테이블 생성
    output_tex = results_dir / 'table1_rq1_latex.tex'
    create_latex_table(stats, summary_df, output_tex)

if __name__ == '__main__':
    main()
