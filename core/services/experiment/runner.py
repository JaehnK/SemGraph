"""Execution runner for SemGraph experiment specs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .datasets import DatasetMaterializer, PreparedDataset
from .specs import ExperimentRunSpec
from ..Metric import MetricsService
from ..clustering import SphericalKMeansClusteringService
from ..semgraph import SemGraphConfig, SemGraphPipeline, TraditionalGraphClusteringService


class ExperimentRunner:
    """Run SemGraph preliminary, main, and ablation experiment specs."""

    def __init__(self, output_dir: str = "artifacts/experiments"):
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "datasets"
        self.results_dir = self.output_dir / "results"
        self.dataset_materializer = DatasetMaterializer(self.dataset_dir)
        self.metrics_service = MetricsService()

    def run_many(
        self,
        specs: Iterable[ExperimentRunSpec],
        max_runs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        specs_list = list(specs)
        if max_runs is not None:
            specs_list = specs_list[:max_runs]

        summaries = [self.run(spec) for spec in specs_list]
        self._write_flat_summary(summaries)
        return summaries

    def run(self, spec: ExperimentRunSpec) -> Dict[str, Any]:
        run_start = time.perf_counter()
        run_dir = self.results_dir / spec.preset / spec.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        prepared = self.dataset_materializer.prepare(spec.dataset, spec.seed)
        config = self._build_semgraph_config(spec, prepared, run_dir / "semgraph")

        pipeline = SemGraphPipeline(config)
        semgraph_result = pipeline.run()

        model_records: List[Dict[str, Any]] = []
        labels_by_model: Dict[str, np.ndarray] = {}

        if "semgraph" in spec.models:
            labels = np.asarray(pipeline.cluster_labels)
            labels_by_model["semgraph"] = labels
            model_records.append(
                self._build_model_record(
                    model_name="semgraph",
                    labels=labels,
                    metrics=semgraph_result.get("metrics", {}),
                    modularity=self._compute_modularity(pipeline.word_graph, labels),
                    selection=self._selection_payload(pipeline.clustering_service, spec),
                    skipped=False,
                )
            )

        if "node_embedding_kmeans" in spec.models:
            record = self._run_node_embedding_kmeans(pipeline, spec)
            model_records.append(record)
            if "labels" in record:
                labels_by_model["node_embedding_kmeans"] = record.pop("labels")

        if "louvain" in spec.models:
            record = self._run_traditional(pipeline, spec, method="louvain")
            model_records.append(record)
            if "labels" in record:
                labels_by_model["louvain"] = record.pop("labels")

        if "leiden" in spec.models:
            record = self._run_traditional(pipeline, spec, method="leiden")
            model_records.append(record)
            if "labels" in record:
                labels_by_model["leiden"] = record.pop("labels")

        summary = {
            "run": spec.to_dict(),
            "dataset": prepared.to_dict(),
            "duration_seconds": time.perf_counter() - run_start,
            "models": model_records,
            "comparisons": self._pairwise_comparisons(labels_by_model),
        }
        self._write_json(run_dir / "experiment_summary.json", summary)
        return summary

    def _run_node_embedding_kmeans(
        self,
        pipeline: SemGraphPipeline,
        spec: ExperimentRunSpec,
    ) -> Dict[str, Any]:
        if pipeline.node_features is None:
            return self._skipped_record("node_embedding_kmeans", "node features are missing")

        clustering = SphericalKMeansClusteringService(random_state=spec.seed)
        labels, best_k, inertias, silhouette_scores = clustering.auto_clustering(
            pipeline.node_features,
            min_clusters=spec.dataset.min_clusters,
            max_clusters=spec.dataset.max_clusters,
        )
        metrics = self.metrics_service.calculate_metrics(
            pipeline.node_features,
            labels,
            pipeline.config.eval_metrics,
            word_graph=pipeline.word_graph,
            total_docs=len(pipeline.documents) if pipeline.documents else 0,
        )
        record = self._build_model_record(
            model_name="node_embedding_kmeans",
            labels=labels,
            metrics=metrics,
            modularity=self._compute_modularity(pipeline.word_graph, labels),
            selection={
                "selected_k": int(best_k),
                "k_range": [spec.dataset.min_clusters, spec.dataset.max_clusters],
                "inertias": _float_list(inertias),
                "silhouette_scores": _float_list(silhouette_scores),
                "selection_method": clustering.k_selection_method,
                "fallback_used": clustering.k_selection_fallback_used,
            },
            skipped=False,
        )
        record["labels"] = np.asarray(labels)
        return record

    def _run_traditional(
        self,
        pipeline: SemGraphPipeline,
        spec: ExperimentRunSpec,
        method: str,
    ) -> Dict[str, Any]:
        if pipeline.word_graph is None:
            return self._skipped_record(method, "word graph is missing")

        service = TraditionalGraphClusteringService(random_state=spec.seed)
        try:
            if method == "louvain":
                labels, structural_metrics = service.louvain_clustering(pipeline.word_graph)
            elif method == "leiden":
                labels, structural_metrics = service.leiden_clustering(pipeline.word_graph)
            else:
                raise ValueError(f"Unsupported traditional method: {method}")
        except ImportError as exc:
            return self._skipped_record(method, str(exc))

        metrics = self.metrics_service.calculate_metrics(
            pipeline.node_features,
            labels,
            pipeline.config.eval_metrics,
            word_graph=pipeline.word_graph,
            total_docs=len(pipeline.documents) if pipeline.documents else 0,
        )
        record = self._build_model_record(
            model_name=method,
            labels=labels,
            metrics=metrics,
            modularity=float(structural_metrics.get("modularity", math.nan)),
            selection={
                "selected_k": int(len(np.unique(labels))),
                "k_range": None,
                "inertias": [],
                "silhouette_scores": [],
                "selection_method": "community_detection",
                "fallback_used": False,
            },
            skipped=False,
        )
        record["labels"] = np.asarray(labels)
        return record

    def _build_semgraph_config(
        self,
        spec: ExperimentRunSpec,
        prepared: PreparedDataset,
        output_dir: Path,
    ) -> SemGraphConfig:
        params = spec.params
        return SemGraphConfig(
            csv_path=prepared.csv_path,
            num_documents=prepared.num_documents,
            text_column=prepared.text_column,
            top_n_words=params.top_n_words,
            exclude_stopwords=True,
            edge_top_k=-1,
            edge_weight_threshold=params.edge_weight_threshold,
            embed_size=params.embed_size,
            graphmae_epochs=params.graphmae_epochs,
            graphmae_lr=params.graphmae_lr,
            graphmae_weight_decay=params.graphmae_weight_decay,
            graphmae_device=None,
            mask_rate=params.mask_rate,
            encoder_type=params.encoder_type,
            decoder_type=params.decoder_type,
            clustering_method="kmeans",
            num_clusters=None,
            min_clusters=spec.dataset.min_clusters,
            max_clusters=spec.dataset.max_clusters,
            eval_metrics=["silhouette", "davies_bouldin", "calinski_harabasz", "npmi"],
            save_results=True,
            output_dir=str(output_dir),
            save_graph_viz=params.save_graph_viz,
            save_network_viz=False,
            save_embeddings=params.save_embeddings,
            verbose=params.verbose,
            log_interval=params.log_interval,
            random_seed=spec.seed,
        )

    def _build_model_record(
        self,
        model_name: str,
        labels: np.ndarray,
        metrics: Dict[str, Any],
        modularity: float,
        selection: Dict[str, Any],
        skipped: bool,
    ) -> Dict[str, Any]:
        metrics = {key: _json_float(value) for key, value in metrics.items()}
        metrics["modularity"] = _json_float(modularity)
        return {
            "model": model_name,
            "skipped": skipped,
            "num_clusters": int(len(np.unique(labels))),
            "cluster_distribution": _cluster_distribution(labels),
            "metrics": metrics,
            "k_selection": selection,
        }

    def _selection_payload(
        self,
        clustering: SphericalKMeansClusteringService,
        spec: ExperimentRunSpec,
    ) -> Dict[str, Any]:
        return {
            "selected_k": int(clustering.best_k) if clustering.best_k is not None else None,
            "k_range": [spec.dataset.min_clusters, spec.dataset.max_clusters],
            "inertias": _float_list(clustering.inertias or []),
            "silhouette_scores": _float_list(clustering.silhouette_scores or []),
            "selection_method": clustering.k_selection_method,
            "fallback_used": clustering.k_selection_fallback_used,
        }

    def _compute_modularity(self, word_graph: Any, labels: np.ndarray) -> float:
        if word_graph is None:
            return math.nan
        try:
            nx_graph = word_graph.export_to_networkx(include_weights=True)
            communities = [
                set(np.where(labels == cluster_id)[0])
                for cluster_id in np.unique(labels)
            ]
            return float(nx.algorithms.community.modularity(nx_graph, communities, weight="weight"))
        except Exception:
            return math.nan

    def _pairwise_comparisons(
        self,
        labels_by_model: Dict[str, np.ndarray],
    ) -> List[Dict[str, Any]]:
        if "semgraph" not in labels_by_model:
            return []

        semgraph_labels = labels_by_model["semgraph"]
        comparisons = []
        for model_name, labels in labels_by_model.items():
            if model_name == "semgraph":
                continue
            comparisons.append(
                {
                    "pair": f"semgraph-vs-{model_name}",
                    "ari": _json_float(adjusted_rand_score(semgraph_labels, labels)),
                    "nmi": _json_float(normalized_mutual_info_score(semgraph_labels, labels)),
                    "jaccard": _json_float(_mean_best_jaccard(semgraph_labels, labels)),
                }
            )
        return comparisons

    def _skipped_record(self, model_name: str, reason: str) -> Dict[str, Any]:
        return {
            "model": model_name,
            "skipped": True,
            "reason": reason,
            "num_clusters": None,
            "cluster_distribution": {},
            "metrics": {},
            "k_selection": {},
        }

    def _write_flat_summary(self, summaries: List[Dict[str, Any]]) -> None:
        rows = []
        for summary in summaries:
            run = summary["run"]
            dataset = summary["dataset"]
            for model in summary["models"]:
                row = {
                    "run_id": run["run_id"],
                    "preset": run["preset"],
                    "dataset": dataset["name"],
                    "seed": run["seed"],
                    "ablation_axis": run.get("ablation_axis"),
                    "ablation_value": run.get("ablation_value"),
                    "model": model["model"],
                    "skipped": model["skipped"],
                    "num_clusters": model["num_clusters"],
                    "selected_k": model.get("k_selection", {}).get("selected_k"),
                    "selection_method": model.get("k_selection", {}).get("selection_method"),
                    "fallback_used": model.get("k_selection", {}).get("fallback_used"),
                    "duration_seconds": summary["duration_seconds"],
                }
                row.update(model.get("metrics", {}))
                rows.append(row)

        if not rows:
            return
        self.results_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.results_dir / "experiment_summary.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _cluster_distribution(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def _mean_best_jaccard(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    communities_a = _communities(labels_a)
    communities_b = _communities(labels_b)
    scores = []
    for community_a in communities_a:
        best = 0.0
        for community_b in communities_b:
            union = community_a | community_b
            if union:
                best = max(best, len(community_a & community_b) / len(union))
        scores.append(best)
    return float(np.mean(scores)) if scores else math.nan


def _communities(labels: np.ndarray) -> List[set]:
    return [set(np.where(labels == label)[0]) for label in np.unique(labels)]


def _float_list(values: Iterable[Any]) -> List[Optional[float]]:
    return [_json_float(value) for value in values]


def _json_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _json_float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
