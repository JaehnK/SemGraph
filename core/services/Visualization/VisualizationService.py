import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from .DimensionReducer import DimensionReducer
from .PlotGenerator import PlotGenerator


class VisualizationService:
    """
    실험 결과 시각화를 위한 통합 서비스.

    주요 기능:
    - 임베딩 차원 축소 및 시각화 (t-SNE, UMAP)
    - 메트릭 비교 그래프
    - Ablation study 히트맵
    - 클러스터 워드클라우드
    - Elbow curve
    """

    def __init__(self, output_dir: Union[str, Path] = "outputs/figures"):
        """
        Args:
            output_dir: 시각화 파일 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"VisualizationService initialized. Output: {self.output_dir}")

    def visualize_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        method: str = "tsne",
        filename: str = "embeddings_visualization.png",
        title: Optional[str] = None,
        **kwargs
    ) -> Path:
        """
        임베딩을 2D로 축소하여 scatter plot 생성.

        Args:
            embeddings: 고차원 임베딩 (n_samples, n_features)
            labels: 클러스터 레이블 (n_samples,)
            method: 차원 축소 방법 ('tsne', 'umap', 'pca', 'auto')
            filename: 저장 파일명
            title: 플롯 제목 (None이면 자동 생성)
            **kwargs: 차원 축소 및 플롯 파라미터

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Visualizing embeddings with {method}. "
            f"Shape: {embeddings.shape}, Labels: {len(np.unique(labels))} clusters"
        )

        # 차원 축소
        if method == "tsne":
            embeddings_2d = DimensionReducer.reduce_tsne(embeddings, **kwargs)
            method_name = "t-SNE"
        elif method == "umap":
            embeddings_2d = DimensionReducer.reduce_umap(embeddings, **kwargs)
            method_name = "UMAP"
        elif method == "pca":
            embeddings_2d = DimensionReducer.reduce_pca(embeddings, **kwargs)
            method_name = "PCA"
        elif method == "auto":
            embeddings_2d, method_name = DimensionReducer.auto_select_method(
                embeddings, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Choose from ['tsne', 'umap', 'pca', 'auto']"
            )

        # 제목 생성
        if title is None:
            title = f"{method_name} Visualization of Embeddings"

        # 플롯 생성
        save_path = self.output_dir / filename
        PlotGenerator.plot_scatter(
            embeddings_2d=embeddings_2d,
            labels=labels,
            save_path=save_path,
            title=title,
            method_name=method_name
        )

        self.logger.info(f"Saved visualization to {save_path}")
        return save_path

    def compare_methods(
        self,
        results_dict: Dict[str, Dict[str, float]],
        filename: str = "method_comparison.png",
        metrics: Optional[List[str]] = None
    ) -> Path:
        """
        여러 방법의 메트릭을 막대그래프로 비교.

        Args:
            results_dict: {method_name: {metric: value}} 형식
                예: {
                    'SemGraph': {'silhouette': 0.45, 'davies_bouldin': 1.2},
                    'TF-IDF': {'silhouette': 0.30, 'davies_bouldin': 1.8}
                }
            filename: 저장 파일명
            metrics: 표시할 메트릭 (None이면 모두)

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Comparing {len(results_dict)} methods "
            f"across {len(metrics) if metrics else 'all'} metrics"
        )

        save_path = self.output_dir / filename
        PlotGenerator.plot_comparison_bar(
            results_dict=results_dict,
            save_path=save_path,
            metrics=metrics
        )

        self.logger.info(f"Saved comparison to {save_path}")
        return save_path

    def visualize_ablation(
        self,
        ablation_results: Dict[str, float],
        param_names: List[str],
        filename: str = "ablation_heatmap.png",
        title: str = "Ablation Study Results"
    ) -> Path:
        """
        Ablation study 결과를 히트맵으로 시각화.

        Args:
            ablation_results: {config_str: metric_value} 형식
                예: {'lr=0.001_dim=128': 0.45}
            param_names: 파라미터 이름 리스트 (x, y축용)
            filename: 저장 파일명
            title: 플롯 제목

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Visualizing ablation study with {len(ablation_results)} configurations"
        )

        save_path = self.output_dir / filename
        PlotGenerator.plot_ablation_heatmap(
            ablation_results=ablation_results,
            save_path=save_path,
            param_names=param_names,
            title=title
        )

        self.logger.info(f"Saved ablation heatmap to {save_path}")
        return save_path

    def visualize_cluster_words(
        self,
        cluster_words: Dict[int, List[str]],
        filename: str = "cluster_wordclouds.png",
        max_words: int = 50
    ) -> Path:
        """
        각 클러스터의 대표 단어를 워드클라우드로 시각화.

        Args:
            cluster_words: {cluster_id: [word1, word2, ...]} 형식
            filename: 저장 파일명
            max_words: 워드클라우드 최대 단어 수

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Generating word clouds for {len(cluster_words)} clusters"
        )

        save_path = self.output_dir / filename
        PlotGenerator.plot_cluster_wordcloud(
            cluster_words=cluster_words,
            save_path=save_path,
            max_words=max_words
        )

        self.logger.info(f"Saved word clouds to {save_path}")
        return save_path

    def plot_elbow(
        self,
        k_range: List[int],
        inertias: List[float],
        filename: str = "elbow_curve.png",
        title: str = "Elbow Method for Optimal K"
    ) -> Path:
        """
        K-means Elbow curve 시각화.

        Args:
            k_range: K 값 리스트
            inertias: 각 K의 inertia 값
            filename: 저장 파일명
            title: 플롯 제목

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Plotting elbow curve for K={k_range[0]} to {k_range[-1]}"
        )

        save_path = self.output_dir / filename
        PlotGenerator.plot_elbow_curve(
            k_range=k_range,
            inertias=inertias,
            save_path=save_path,
            title=title
        )

        self.logger.info(f"Saved elbow curve to {save_path}")
        return save_path

    def plot_training_metrics(
        self,
        metric_history: Dict[str, List[float]],
        filename: str = "training_metrics.png",
        title: str = "Training Metrics Evolution"
    ) -> Path:
        """
        학습 중 메트릭 변화를 시각화.

        Args:
            metric_history: {metric_name: [epoch_values]} 형식
            filename: 저장 파일명
            title: 플롯 제목

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Plotting training metrics for {len(metric_history)} metrics"
        )

        save_path = self.output_dir / filename
        PlotGenerator.plot_metric_evolution(
            metric_history=metric_history,
            save_path=save_path,
            title=title
        )

        self.logger.info(f"Saved training metrics to {save_path}")
        return save_path

    def generate_full_report(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        results_dict: Dict[str, Dict[str, float]],
        ablation_results: Optional[Dict[str, float]] = None,
        param_names: Optional[List[str]] = None,
        cluster_words: Optional[Dict[int, List[str]]] = None,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """
        모든 시각화를 한 번에 생성 (full report).

        Args:
            embeddings: 임베딩 벡터
            labels: 클러스터 레이블
            results_dict: 메서드별 메트릭 결과
            ablation_results: Ablation study 결과 (optional)
            param_names: Ablation 파라미터 이름 (optional)
            cluster_words: 클러스터별 단어 (optional)
            prefix: 파일명 prefix

        Returns:
            {visualization_type: file_path} 딕셔너리
        """
        self.logger.info("Generating full visualization report...")

        report_paths = {}

        # 1. t-SNE 시각화
        try:
            report_paths['tsne'] = self.visualize_embeddings(
                embeddings=embeddings,
                labels=labels,
                method='tsne',
                filename=f"{prefix}tsne_visualization.png" if prefix else "tsne_visualization.png"
            )
        except Exception as e:
            self.logger.warning(f"t-SNE visualization failed: {e}")

        # 2. UMAP 시각화 (optional)
        try:
            report_paths['umap'] = self.visualize_embeddings(
                embeddings=embeddings,
                labels=labels,
                method='umap',
                filename=f"{prefix}umap_visualization.png" if prefix else "umap_visualization.png"
            )
        except Exception as e:
            self.logger.warning(f"UMAP visualization failed (may need umap-learn): {e}")

        # 3. 메서드 비교
        try:
            report_paths['comparison'] = self.compare_methods(
                results_dict=results_dict,
                filename=f"{prefix}method_comparison.png" if prefix else "method_comparison.png"
            )
        except Exception as e:
            self.logger.warning(f"Method comparison failed: {e}")

        # 4. Ablation heatmap (if provided)
        if ablation_results and param_names:
            try:
                report_paths['ablation'] = self.visualize_ablation(
                    ablation_results=ablation_results,
                    param_names=param_names,
                    filename=f"{prefix}ablation_heatmap.png" if prefix else "ablation_heatmap.png"
                )
            except Exception as e:
                self.logger.warning(f"Ablation heatmap failed: {e}")

        # 5. Cluster word clouds (if provided)
        if cluster_words:
            try:
                report_paths['wordclouds'] = self.visualize_cluster_words(
                    cluster_words=cluster_words,
                    filename=f"{prefix}cluster_wordclouds.png" if prefix else "cluster_wordclouds.png"
                )
            except Exception as e:
                self.logger.warning(f"Word clouds failed (may need wordcloud package): {e}")

        self.logger.info(f"Full report generated with {len(report_paths)} visualizations")
        return report_paths

    def visualize_network(
        self,
        word_graph,
        cluster_labels: np.ndarray,
        filename: str = "network_visualization.png",
        title: str = "Semantic Network with Clusters",
        node_size_scale: float = 300,
        edge_width_scale: float = 0.5,
        k: float = 2.0,
        max_edges: Optional[int] = None,
        random_seed: int = 42
    ) -> Path:
        """
        의미연결망을 클러스터별 색상으로 시각화.

        Args:
            word_graph: WordGraph 객체
            cluster_labels: 클러스터 레이블 (n_nodes,)
            filename: 저장 파일명
            title: 플롯 제목
            node_size_scale: 노드 크기 스케일 (빈도에 따라)
            edge_width_scale: 엣지 두께 스케일 (가중치에 따라)
            k: 노드 간 거리 (spring layout)
            max_edges: 표시할 최대 엣지 수 (None이면 모두 표시)
            random_seed: 재현성을 위한 랜덤 시드

        Returns:
            저장된 파일 경로
        """
        self.logger.info(
            f"Visualizing network with {word_graph.num_nodes} nodes "
            f"and {word_graph.num_edges} edges"
        )

        save_path = self.output_dir / filename
        PlotGenerator.plot_network_graph(
            word_graph=word_graph,
            cluster_labels=cluster_labels,
            save_path=save_path,
            title=title,
            node_size_scale=node_size_scale,
            edge_width_scale=edge_width_scale,
            k=k,
            max_edges=max_edges,
            random_seed=random_seed
        )

        self.logger.info(f"Saved network visualization to {save_path}")
        return save_path
