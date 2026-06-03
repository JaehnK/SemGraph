import json
from pathlib import Path

from core.services.adapters import JsonArtifactWriter


def test_json_artifact_writer_creates_output_dir_and_preserves_payload(tmp_path):
    output_dir = tmp_path / "semgraph_output"
    payload = {
        "num_clusters": 2,
        "metrics": {
            "silhouette": 0.42,
        },
    }

    writer = JsonArtifactWriter(str(output_dir))
    path = writer.write_json("semgraph_results_test.json", payload)

    assert Path(path).parent == output_dir
    assert json.loads(Path(path).read_text(encoding="utf-8")) == payload
