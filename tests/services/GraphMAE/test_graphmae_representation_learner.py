import subprocess
import sys
from types import SimpleNamespace

import torch

from core.services.semgraph.SemGraphConfig import SemGraphConfig
from core.services.semgraph.SemGraphPipeline import SemGraphPipeline


def test_graphmae_adapter_import_does_not_load_dgl():
    code = """
import sys
from core.services.adapters.graphmae_representation_learner import GraphMAERepresentationLearner

loaded = sorted(
    name for name in sys.modules
    if name == "dgl" or name.startswith("dgl.")
)

assert GraphMAERepresentationLearner.__name__ == "GraphMAERepresentationLearner"
assert loaded == [], loaded
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


class RecordingLearner:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def fit_transform(self, graph, node_features):
        self.calls.append((graph, node_features))
        return self.output


def test_pipeline_trains_through_injected_representation_learner():
    node_features = torch.ones(3, 4)
    output = torch.zeros(3, 8)
    learner = RecordingLearner(output)
    config = SemGraphConfig(csv_path="unused.csv", graphmae_device="cpu", verbose=False)
    pipeline = SemGraphPipeline(config, representation_learner=learner)
    pipeline.word_graph = SimpleNamespace(node_features=node_features)
    pipeline.graph_service = object()

    pipeline.train_graphmae()

    assert learner.calls == [(pipeline.word_graph, node_features)]
    assert pipeline.graphmae_embeddings is output
