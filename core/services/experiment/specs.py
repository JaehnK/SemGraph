"""Experiment presets for preliminary, main, and ablation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATASET_NAMES = ("ag_news", "20_newsgroups", "arxiv")
MODEL_NAMES = ("semgraph", "node_embedding_kmeans", "louvain", "leiden")
ABLATION_AXES = (
    "mask_rate",
    "graphmae_epochs",
    "embed_size",
    "top_n_words",
    "edge_weight_threshold",
    "auto_k_range",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset sampling and auto-k bounds."""

    name: str
    num_documents: int
    min_clusters: int
    max_clusters: int
    balanced: bool = True


@dataclass(frozen=True)
class SemGraphRunParams:
    """SemGraph parameters that are varied by the experiment protocol."""

    top_n_words: int
    edge_weight_threshold: float
    embed_size: int
    graphmae_epochs: int
    mask_rate: float
    graphmae_lr: float = 0.001
    graphmae_weight_decay: float = 0.0
    encoder_type: str = "gat"
    decoder_type: str = "gat"
    save_graph_viz: bool = False
    save_embeddings: bool = False
    verbose: bool = True
    log_interval: int = 50


@dataclass(frozen=True)
class ExperimentRunSpec:
    """A single dataset/seed/configuration execution unit."""

    preset: str
    dataset: DatasetSpec
    seed: int
    params: SemGraphRunParams
    models: Tuple[str, ...]
    ablation_axis: Optional[str] = None
    ablation_value: Optional[str] = None

    @property
    def run_id(self) -> str:
        suffix = ""
        if self.ablation_axis:
            suffix = f"__{self.ablation_axis}-{self.ablation_value}"
        return f"{self.preset}__{self.dataset.name}__seed-{self.seed}{suffix}"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["run_id"] = self.run_id
        return payload


PRELIMINARY_DATASETS = {
    "ag_news": DatasetSpec("ag_news", 1_000, 3, 10),
    "20_newsgroups": DatasetSpec("20_newsgroups", 1_000, 3, 10),
    "arxiv": DatasetSpec("arxiv", 1_000, 3, 10),
}

MAIN_DATASETS = {
    "ag_news": DatasetSpec("ag_news", 50_000, 3, 15),
    "20_newsgroups": DatasetSpec("20_newsgroups", 20_000, 5, 30),
    "arxiv": DatasetSpec("arxiv", 100_000, 5, 50),
}

PRELIMINARY_PARAMS = SemGraphRunParams(
    top_n_words=150,
    edge_weight_threshold=5.0,
    embed_size=64,
    graphmae_epochs=50,
    mask_rate=0.3,
    log_interval=10,
)

MAIN_PARAMS = SemGraphRunParams(
    top_n_words=1_000,
    edge_weight_threshold=5.0,
    embed_size=256,
    graphmae_epochs=1_000,
    mask_rate=0.3,
    log_interval=100,
)

PRELIMINARY_SEEDS = (42, 43, 44)
MAIN_SEEDS = (13, 29, 42, 73, 101, 137, 193, 257, 389, 521)

ABLATION_VALUES: Dict[str, Tuple[Any, ...]] = {
    "mask_rate": (0.1, 0.3, 0.5, 0.75),
    "graphmae_epochs": (250, 500, 1_000, 2_000),
    "embed_size": (64, 128, 256),
    "top_n_words": (500, 1_000, 2_000),
    "edge_weight_threshold": (3.0, 5.0, 10.0),
    "auto_k_range": ((3, 15), (5, 30), (5, 50)),
}


def build_run_specs(
    preset: str,
    dataset_names: Optional[Sequence[str]] = None,
    axes: Optional[Sequence[str]] = None,
    seed_limit: Optional[int] = None,
    include_baselines: bool = True,
) -> List[ExperimentRunSpec]:
    """Build run specs for a named experiment preset.

    ``smoke`` is accepted as an alias for ``preliminary`` because the project
    uses the term casually while the protocol treats it as a preliminary run.
    """

    normalized = "preliminary" if preset == "smoke" else preset
    if normalized not in {"preliminary", "main", "ablation"}:
        raise ValueError(f"Unsupported preset: {preset}")

    if normalized == "preliminary":
        return _build_standard_specs(
            preset="preliminary",
            dataset_specs=_select_datasets(PRELIMINARY_DATASETS, dataset_names),
            params=PRELIMINARY_PARAMS,
            seeds=_limit_seeds(PRELIMINARY_SEEDS, seed_limit),
            include_baselines=include_baselines,
        )

    if normalized == "main":
        return _build_standard_specs(
            preset="main",
            dataset_specs=_select_datasets(MAIN_DATASETS, dataset_names),
            params=MAIN_PARAMS,
            seeds=_limit_seeds(MAIN_SEEDS, seed_limit),
            include_baselines=include_baselines,
        )

    return _build_ablation_specs(
        dataset_specs=_select_datasets(MAIN_DATASETS, dataset_names),
        axes=axes or ABLATION_AXES,
        seeds=_limit_seeds(MAIN_SEEDS, seed_limit),
        include_baselines=include_baselines,
    )


def _build_standard_specs(
    preset: str,
    dataset_specs: Iterable[DatasetSpec],
    params: SemGraphRunParams,
    seeds: Sequence[int],
    include_baselines: bool,
) -> List[ExperimentRunSpec]:
    models = MODEL_NAMES if include_baselines else ("semgraph",)
    return [
        ExperimentRunSpec(
            preset=preset,
            dataset=dataset,
            seed=seed,
            params=params,
            models=models,
        )
        for dataset in dataset_specs
        for seed in seeds
    ]


def _build_ablation_specs(
    dataset_specs: Iterable[DatasetSpec],
    axes: Sequence[str],
    seeds: Sequence[int],
    include_baselines: bool,
) -> List[ExperimentRunSpec]:
    models = MODEL_NAMES if include_baselines else ("semgraph",)
    specs: List[ExperimentRunSpec] = []
    for axis in axes:
        if axis not in ABLATION_VALUES:
            raise ValueError(f"Unsupported ablation axis: {axis}")
        for dataset in dataset_specs:
            for seed in seeds:
                for value in ABLATION_VALUES[axis]:
                    params = _params_for_ablation(axis, value)
                    adjusted_dataset = _dataset_for_ablation(dataset, axis, value)
                    specs.append(
                        ExperimentRunSpec(
                            preset="ablation",
                            dataset=adjusted_dataset,
                            seed=seed,
                            params=params,
                            models=models,
                            ablation_axis=axis,
                            ablation_value=_format_ablation_value(value),
                        )
                    )
    return specs


def _params_for_ablation(axis: str, value: Any) -> SemGraphRunParams:
    if axis == "auto_k_range":
        return MAIN_PARAMS
    return replace(MAIN_PARAMS, **{axis: value})


def _dataset_for_ablation(
    dataset: DatasetSpec,
    axis: str,
    value: Any,
) -> DatasetSpec:
    if axis != "auto_k_range":
        return dataset
    min_clusters, max_clusters = value
    return replace(dataset, min_clusters=min_clusters, max_clusters=max_clusters)


def _format_ablation_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "-".join(str(item) for item in value)
    return str(value).replace(".", "p")


def _select_datasets(
    dataset_specs: Dict[str, DatasetSpec],
    dataset_names: Optional[Sequence[str]],
) -> List[DatasetSpec]:
    names = tuple(dataset_names or DATASET_NAMES)
    unknown = sorted(set(names) - set(dataset_specs))
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {', '.join(unknown)}")
    return [dataset_specs[name] for name in names]


def _limit_seeds(seeds: Sequence[int], seed_limit: Optional[int]) -> Tuple[int, ...]:
    if seed_limit is None:
        return tuple(seeds)
    if seed_limit <= 0:
        raise ValueError("seed_limit must be positive")
    return tuple(seeds[:seed_limit])
