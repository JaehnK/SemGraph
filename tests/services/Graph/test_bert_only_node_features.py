from pathlib import Path

import pytest

from core.services.semgraph.SemGraphConfig import SemGraphConfig


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_node_feature_handler_has_no_legacy_embedding_paths():
    source = (PROJECT_ROOT / "core/services/Graph/NodeFeatureHandler.py").read_text(encoding="utf-8")

    forbidden_tokens = [
        "Word2Vec",
        "_get_w2v_embeddings",
        "_get_concat_embeddings",
        "_get_attention_embeddings",
        "AttentionFusion",
        "self.w2v",
    ]

    assert [token for token in forbidden_tokens if token in source] == []


def test_semgraph_config_rejects_legacy_embedding_methods():
    assert SemGraphConfig(csv_path="data/test.csv").embedding_method == "bert"

    with pytest.raises(ValueError, match="BERT-only"):
        SemGraphConfig(csv_path="data/test.csv", embedding_method="concat")
