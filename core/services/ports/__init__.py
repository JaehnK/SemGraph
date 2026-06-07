"""
Application-facing service ports for SemGraph.

These protocols intentionally avoid importing concrete runtime dependencies
such as spaCy, transformers, DGL, GraphMAE2, or matplotlib.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Mapping, Optional, Protocol, Sequence, Union


PathValue = Union[str, PathLike]


class CorpusRepository(Protocol):
    """Loads raw text documents for a pipeline run."""

    def load_documents(
        self,
        path: PathValue,
        text_column: str,
        limit: Optional[int] = None,
    ) -> Sequence[str]:
        ...


class TextPreprocessor(Protocol):
    """Transforms raw documents into the corpus state used by graph building."""

    def process_documents(self, documents: Sequence[str]) -> Any:
        ...


class EmbeddingProvider(Protocol):
    """Creates fixed-width term embeddings for graph nodes."""

    def embed_terms(self, terms: Sequence[Any], target_dim: int) -> Any:
        ...


class RepresentationLearner(Protocol):
    """Learns graph-aware node representations from graph state and features."""

    def fit_transform(self, graph: Any, node_features: Any) -> Any:
        ...


class Clusterer(Protocol):
    """Assigns representation vectors to clusters."""

    def fit_predict(self, embeddings: Any, num_clusters: Optional[int] = None) -> Any:
        ...


class ArtifactWriter(Protocol):
    """Persists pipeline artifacts without coupling the pipeline to a format."""

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> str:
        ...

