import json
import math

from core.services.experiment.aggregate import (
    aggregate_over_seeds,
    flatten_summaries,
    write_summary_tables,
)


def _summary(seed, silhouette, preset="preliminary", axis=None, value=None, skipped_leiden=False):
    run_id = f"{preset}__ag_news__seed-{seed}"
    leiden = {
        "model": "leiden",
        "skipped": skipped_leiden,
        "num_clusters": None if skipped_leiden else 4,
        "metrics": {} if skipped_leiden else {"silhouette": 0.1, "modularity": 0.4},
        "k_selection": {},
    }
    return {
        "run": {
            "run_id": run_id,
            "preset": preset,
            "seed": seed,
            "ablation_axis": axis,
            "ablation_value": value,
        },
        "dataset": {"name": "ag_news"},
        "duration_seconds": 10.0,
        "models": [
            {
                "model": "semgraph",
                "skipped": False,
                "num_clusters": 5,
                "metrics": {"silhouette": silhouette, "npmi": 0.05, "modularity": 0.3},
                "k_selection": {"selected_k": 5, "selection_method": "kneedle"},
            },
            leiden,
        ],
    }


def test_aggregate_computes_mean_and_sample_std_over_seeds():
    flat = flatten_summaries([_summary(42, 0.2), _summary(43, 0.4)])

    aggregate = aggregate_over_seeds(flat)
    semgraph = aggregate[aggregate["model"] == "semgraph"].iloc[0]

    assert semgraph["n_seeds"] == 2
    assert math.isclose(semgraph["silhouette_mean"], 0.3)
    assert math.isclose(semgraph["silhouette_std"], math.sqrt(0.02))
    assert semgraph["selected_k_mean"] == 5


def test_aggregate_excludes_skipped_models_and_keeps_ablation_groups():
    flat = flatten_summaries(
        [
            _summary(42, 0.2, preset="ablation", axis="mask_rate", value=0.1, skipped_leiden=True),
            _summary(42, 0.5, preset="ablation", axis="mask_rate", value=0.5),
        ]
    )

    aggregate = aggregate_over_seeds(flat)
    leiden = aggregate[aggregate["model"] == "leiden"]
    semgraph = aggregate[aggregate["model"] == "semgraph"]

    assert list(leiden["ablation_value"]) == [0.5]
    assert sorted(semgraph["ablation_value"]) == [0.1, 0.5]
    assert (semgraph["n_seeds"] == 1).all()


def test_write_summary_tables_reads_all_runs_on_disk(tmp_path):
    for seed, silhouette in ((42, 0.2), (43, 0.4), (44, 0.6)):
        run_dir = tmp_path / "preliminary" / f"preliminary__ag_news__seed-{seed}"
        run_dir.mkdir(parents=True)
        (run_dir / "experiment_summary.json").write_text(json.dumps(_summary(seed, silhouette)))

    paths = write_summary_tables(tmp_path)

    assert paths["flat"].exists()
    assert paths["aggregate"].exists()
    assert paths["aggregate"].read_text().count("\n") == 3  # header + semgraph + leiden


def test_write_summary_tables_without_runs_writes_nothing(tmp_path):
    assert write_summary_tables(tmp_path) == {}
    assert not (tmp_path / "experiment_summary.csv").exists()
