import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


class PlotGenerator:
    """
    개별 플롯 생성을 담당하는 클래스.
    논문 수준의 고품질 시각화 제공.
    """

    # 논문용 컬러 팔레트 (colorblind-friendly)
    COLORS = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'sequential': 'viridis',
        'diverging': 'RdYlBu_r'
    }

    # 논문용 스타일 설정
    FIGURE_SIZE = (8, 6)
    DPI = 300
    FONT_SIZE = 12
    FONT_FAMILY = 'DejaVu Sans'  # Times New Roman 대안

    @staticmethod
    def _setup_style():
        """논문용 matplotlib 스타일 설정"""
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.size': PlotGenerator.FONT_SIZE,
            'font.family': PlotGenerator.FONT_FAMILY,
            'axes.labelsize': PlotGenerator.FONT_SIZE,
            'axes.titlesize': PlotGenerator.FONT_SIZE + 2,
            'xtick.labelsize': PlotGenerator.FONT_SIZE - 1,
            'ytick.labelsize': PlotGenerator.FONT_SIZE - 1,
            'legend.fontsize': PlotGenerator.FONT_SIZE - 1,
            'figure.dpi': PlotGenerator.DPI,
            'savefig.dpi': PlotGenerator.DPI,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    @staticmethod
    def plot_scatter(
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
        save_path: Union[str, Path],
        title: str = "Embedding Visualization",
        method_name: str = "t-SNE",
        legend_title: str = "Cluster",
        figsize: tuple = FIGURE_SIZE
    ):
        """
        2D 임베딩을 scatter plot으로 시각화.

        Args:
            embeddings_2d: 2D 축소된 임베딩 (n_samples, 2)
            labels: 클러스터 레이블 (n_samples,)
            save_path: 저장 경로
            title: 플롯 제목
            method_name: 차원 축소 방법 이름 (축 레이블용)
            legend_title: 범례 제목
            figsize: 그림 크기
        """
        PlotGenerator._setup_style()

        fig, ax = plt.subplots(figsize=figsize)

        # 클러스터별로 scatter
        unique_labels = np.unique(labels)
        colors = PlotGenerator.COLORS['primary']

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=colors[i % len(colors)],
                label=f'{legend_title} {label}',
                alpha=0.7,
                s=30,
                edgecolors='none'
            )

        ax.set_xlabel(f'{method_name} Dimension 1')
        ax.set_ylabel(f'{method_name} Dimension 2')
        ax.set_title(title)
        ax.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_comparison_bar(
        results_dict: Dict[str, Dict[str, float]],
        save_path: Union[str, Path],
        metrics: Optional[List[str]] = None,
        figsize: tuple = (12, 8)
    ):
        """
        여러 방법의 메트릭을 막대그래프로 비교.

        Args:
            results_dict: {method_name: {metric: value}} 형식
                예: {'GRACE': {'silhouette': 0.45, 'davies_bouldin': 1.2}}
            save_path: 저장 경로
            metrics: 표시할 메트릭 리스트 (None이면 모든 메트릭)
            figsize: 그림 크기
        """
        PlotGenerator._setup_style()

        # 메트릭 추출
        all_metrics = set()
        for method_results in results_dict.values():
            all_metrics.update(method_results.keys())

        if metrics is None:
            metrics = sorted(list(all_metrics))

        # subplot 개수 계산 (2열 그리드)
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        methods = list(results_dict.keys())
        colors = PlotGenerator.COLORS['primary']

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # 각 메서드의 해당 메트릭 값 추출
            values = [results_dict[m].get(metric, 0) for m in methods]

            # 막대 그래프
            bars = ax.bar(
                range(len(methods)),
                values,
                color=colors[:len(methods)],
                alpha=0.7
            )

            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=PlotGenerator.FONT_SIZE - 2
                )

            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

        # 빈 subplot 제거
        for idx in range(n_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Performance Comparison Across Methods', fontsize=PlotGenerator.FONT_SIZE + 4, y=1.00)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_ablation_heatmap(
        ablation_results: Dict[str, float],
        save_path: Union[str, Path],
        param_names: List[str],
        title: str = "Ablation Study Heatmap",
        figsize: tuple = (10, 8)
    ):
        """
        하이퍼파라미터 조합별 성능을 히트맵으로 시각화.

        Args:
            ablation_results: {config_str: metric_value} 형식
                예: {'lr=0.001_dim=128': 0.45}
            save_path: 저장 경로
            param_names: 파라미터 이름 리스트 (순서대로)
            title: 플롯 제목
            figsize: 그림 크기
        """
        PlotGenerator._setup_style()

        # config_str 파싱하여 데이터 구조화
        data = defaultdict(dict)

        for config_str, value in ablation_results.items():
            # 'lr=0.001_dim=128' -> {'lr': '0.001', 'dim': '128'}
            params = {}
            for part in config_str.split('_'):
                if '=' in part:
                    key, val = part.split('=')
                    params[key] = val

            # 2개 파라미터 가정 (확장 가능)
            if len(param_names) >= 2:
                row_key = params.get(param_names[0], 'unknown')
                col_key = params.get(param_names[1], 'unknown')
                data[row_key][col_key] = value

        # DataFrame 형태로 변환
        rows = sorted(data.keys())
        cols = sorted(set(k for v in data.values() for k in v.keys()))

        heatmap_data = []
        for row in rows:
            heatmap_data.append([data[row].get(col, np.nan) for col in cols])

        heatmap_array = np.array(heatmap_data)

        # 히트맵 그리기
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            heatmap_array,
            annot=True,
            fmt='.3f',
            cmap=PlotGenerator.COLORS['sequential'],
            xticklabels=cols,
            yticklabels=rows,
            cbar_kws={'label': 'Performance Score'},
            ax=ax
        )

        ax.set_xlabel(param_names[1] if len(param_names) > 1 else 'Parameter 2')
        ax.set_ylabel(param_names[0])
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_cluster_wordcloud(
        cluster_words: Dict[int, List[str]],
        save_path: Union[str, Path],
        max_words: int = 50,
        figsize: tuple = (16, 12)
    ):
        """
        각 클러스터의 대표 단어를 워드클라우드로 시각화.

        Args:
            cluster_words: {cluster_id: [word1, word2, ...]} 형식
            save_path: 저장 경로
            max_words: 워드클라우드 최대 단어 수
            figsize: 그림 크기
        """
        if not WORDCLOUD_AVAILABLE:
            raise ImportError(
                "wordcloud가 설치되지 않았습니다. "
                "'pip install wordcloud'로 설치하세요."
            )

        PlotGenerator._setup_style()

        n_clusters = len(cluster_words)
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (cluster_id, words) in enumerate(sorted(cluster_words.items())):
            ax = axes[idx]

            # 단어 빈도 계산 (리스트에서 중복 단어가 빈도)
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            # 워드클라우드 생성
            wc = WordCloud(
                width=800,
                height=600,
                background_color='white',
                max_words=max_words,
                colormap=PlotGenerator.COLORS['sequential'],
                relative_scaling=0.5
            ).generate_from_frequencies(word_freq)

            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(f'Cluster {cluster_id}', fontsize=PlotGenerator.FONT_SIZE + 2)
            ax.axis('off')

        # 빈 subplot 제거
        for idx in range(n_clusters, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle('Cluster Word Clouds', fontsize=PlotGenerator.FONT_SIZE + 4, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_elbow_curve(
        k_range: List[int],
        inertias: List[float],
        save_path: Union[str, Path],
        title: str = "Elbow Method for Optimal K",
        figsize: tuple = FIGURE_SIZE
    ):
        """
        K-means Elbow curve 시각화.

        Args:
            k_range: K 값 리스트
            inertias: 각 K의 inertia 값
            save_path: 저장 경로
            title: 플롯 제목
            figsize: 그림 크기
        """
        PlotGenerator._setup_style()

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # 데이터 포인트에 값 표시
        for k, inertia in zip(k_range, inertias):
            ax.annotate(
                f'{inertia:.1f}',
                (k, inertia),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=PlotGenerator.FONT_SIZE - 2
            )

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_metric_evolution(
        metric_history: Dict[str, List[float]],
        save_path: Union[str, Path],
        title: str = "Metric Evolution During Training",
        figsize: tuple = (12, 6)
    ):
        """
        학습 중 메트릭 변화를 선 그래프로 시각화.

        Args:
            metric_history: {metric_name: [values]} 형식
            save_path: 저장 경로
            title: 플롯 제목
            figsize: 그림 크기
        """
        PlotGenerator._setup_style()

        fig, ax = plt.subplots(figsize=figsize)

        colors = PlotGenerator.COLORS['primary']

        for idx, (metric_name, values) in enumerate(metric_history.items()):
            epochs = range(1, len(values) + 1)
            ax.plot(
                epochs,
                values,
                label=metric_name,
                color=colors[idx % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        labels: List[str],
        save_path: Union[str, Path],
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8)
    ):
        """
        Confusion matrix를 히트맵으로 시각화.

        Args:
            cm: Confusion matrix (n_classes, n_classes)
            labels: 클래스 레이블 리스트
            save_path: 저장 경로
            title: 플롯 제목
            figsize: 그림 크기
        """
        PlotGenerator._setup_style()

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'},
            ax=ax
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_network_graph(
        word_graph,
        cluster_labels: np.ndarray,
        save_path: Union[str, Path],
        title: str = "Semantic Network with Clusters",
        figsize: tuple = (16, 12),
        node_size_scale: float = 300,
        edge_width_scale: float = 0.5,
        k: float = 2.0,
        max_edges: Optional[int] = None,
        random_seed: int = 42
    ):
        """
        의미연결망을 클러스터별 색상으로 시각화.

        Args:
            word_graph: WordGraph 객체
            cluster_labels: 클러스터 레이블 (n_nodes,)
            save_path: 저장 경로
            title: 플롯 제목
            figsize: 그림 크기
            node_size_scale: 노드 크기 스케일 (빈도에 따라)
            edge_width_scale: 엣지 두께 스케일 (가중치에 따라)
            k: 노드 간 거리 (spring layout)
            max_edges: 표시할 최대 엣지 수 (None이면 모두 표시)
            random_seed: 재현성을 위한 랜덤 시드
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx가 설치되지 않았습니다. "
                "'pip install networkx'로 설치하세요."
            )

        PlotGenerator._setup_style()

        # NetworkX 그래프 생성
        G = nx.Graph()

        # 노드 추가 (단어 및 빈도)
        for idx, word in enumerate(word_graph.words):
            G.add_node(idx, label=word.content, freq=word.freq, cluster=int(cluster_labels[idx]))

        # 엣지 추가 (co-occurrence weight) - 최적화: sparse matrix 활용
        edges_with_weights = []
        edge_matrix = word_graph.edge_matrix

        # sparse matrix인 경우 nonzero() 사용
        if hasattr(edge_matrix, 'nonzero'):
            rows, cols = edge_matrix.nonzero()
            for idx in range(len(rows)):
                i, j = rows[idx], cols[idx]
                if i < j:  # 상삼각만
                    weight = edge_matrix[i, j]
                    edges_with_weights.append((i, j, weight))
        else:
            # dense matrix인 경우 기존 방식
            for i in range(word_graph.num_nodes):
                for j in range(i + 1, word_graph.num_nodes):
                    weight = edge_matrix[i, j]
                    if weight > 0:
                        edges_with_weights.append((i, j, weight))

        # 엣지 가중치로 정렬하여 상위 N개만 표시 (선택사항)
        if max_edges is not None and len(edges_with_weights) > max_edges:
            edges_with_weights.sort(key=lambda x: x[2], reverse=True)
            edges_with_weights = edges_with_weights[:max_edges]

        for i, j, weight in edges_with_weights:
            G.add_edge(i, j, weight=weight)

        # 레이아웃 계산 (Spectral layout - 빠르고 커뮤니티 구조 강조)
        pos = nx.spectral_layout(G)

        # 클러스터별 색상
        unique_clusters = np.unique(cluster_labels)
        colors = PlotGenerator.COLORS['primary']
        cluster_colors = {int(c): colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

        # 노드 색상 및 크기
        node_colors = [cluster_colors[G.nodes[node]['cluster']] for node in G.nodes()]
        # 노드 크기를 줄여서 겹침 방지 (빈도에 로그 스케일 적용)
        node_sizes = [np.log1p(G.nodes[node]['freq']) * node_size_scale * 0.5 for node in G.nodes()]

        # 엣지 가중치 정규화 (더 얇게)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        if edge_weights:
            max_weight = max(edge_weights)
            # 엣지 두께를 더 얇게 조정
            edge_widths = [w / max_weight * edge_width_scale * 3 for w in edge_weights]
        else:
            edge_widths = [0.5]

        # 플롯 그리기
        fig, ax = plt.subplots(figsize=figsize)

        # 엣지 그리기 (더 투명하게)
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.1,  # 훨씬 더 투명하게
            edge_color='gray',
            ax=ax
        )

        # 노드 그리기 (동심원 형태)
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            node_shape='o',  # 원형
            alpha=0.9,  # 노드는 더 불투명하게
            linewidths=2,  # 테두리 두께 증가
            edgecolors='black',  # 검정 테두리로 동심원 효과
            ax=ax
        )

        # 레이블 그리기 (상위 빈도 노드만, 더 적게)
        top_n_labels = min(20, len(G.nodes()) // 10)  # 노드 수의 10% 또는 최대 20개
        node_freqs = [(node, G.nodes[node]['freq']) for node in G.nodes()]
        node_freqs.sort(key=lambda x: x[1], reverse=True)
        top_nodes = {node for node, _ in node_freqs[:top_n_labels]}

        labels = {node: G.nodes[node]['label'] for node in G.nodes() if node in top_nodes}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=9,
            font_weight='bold',
            font_color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
            ax=ax
        )

        # 범례 추가 (클러스터)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cluster_colors[c], label=f'Cluster {c}', alpha=0.8)
            for c in sorted(cluster_colors.keys())
        ]
        ax.legend(
            handles=legend_elements,
            title='Clusters',
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=PlotGenerator.FONT_SIZE - 2
        )

        ax.set_title(title, fontsize=PlotGenerator.FONT_SIZE + 4)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
