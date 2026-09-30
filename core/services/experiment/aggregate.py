"""Flatten run summaries and aggregate them across seeds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


GROUP_COLUMNS = ["preset", "dataset", "ablation_axis", "ablation_value", "model"]
VALUE_COLUMNS = [
    "num_clusters",
    "selected_k",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "npmi",
    "modularity",
    "duration_seconds",
]


def flatten_summaries(summaries: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Turn run summaries into one row per run and model."""

    rows = []
    for summary in summaries:
        run = summary["run"]
        dataset = summary["dataset"]
        for model in summary["models"]:
            k_selection = model.get("k_selection") or {}
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
                "selected_k": k_selection.get("selected_k"),
                "selection_method": k_selection.get("selection_method"),
                "fallback_used": k_selection.get("fallback_used"),
                "duration_seconds": summary["duration_seconds"],
            }
            row.update(model.get("metrics", {}))
            rows.append(row)
    return pd.DataFrame(rows)


def load_run_summaries(results_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``<preset>/<run_id>/experiment_summary.json`` under results_dir."""

    paths = sorted(Path(results_dir).glob("*/*/experiment_summary.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def aggregate_over_seeds(flat: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, sample std, and seed count per preset/dataset/config/model.

    Skipped model records are excluded. ``duration_seconds`` is the whole run's
    wall time, so it is shared by every model in a run.
    """

    columns = GROUP_COLUMNS + ["n_seeds"] + [
        f"{name}_{stat}" for name in VALUE_COLUMNS for stat in ("mean", "std")
    ]
    if flat.empty:
        return pd.DataFrame(columns=columns)

    frame = flat[~flat["skipped"].astype(bool)].copy()
    for name in VALUE_COLUMNS:
        if name not in frame.columns:
            frame[name] = float("nan")
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    # groupby drops NaN keys, and standard runs have no ablation axis.
    for name in ("ablation_axis", "ablation_value"):
        column = frame[name].astype(object)
        frame[name] = column.where(column.notna(), "")

    grouped = frame.groupby(GROUP_COLUMNS, sort=True)
    stats = grouped[VALUE_COLUMNS].agg(["mean", "std"])
    stats.columns = [f"{name}_{stat}" for name, stat in stats.columns]
    stats.insert(0, "n_seeds", grouped["seed"].nunique())
    return stats.reset_index()[columns]


def write_summary_tables(results_dir: Path) -> Dict[str, Path]:
    """Rebuild the flat and seed-aggregated CSVs from all run summaries on disk."""

    results_dir = Path(results_dir)
    flat = flatten_summaries(load_run_summaries(results_dir))
    if flat.empty:
        return {}

    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "flat": results_dir / "experiment_summary.csv",
        "aggregate": results_dir / "experiment_aggregate.csv",
    }
    flat.to_csv(paths["flat"], index=False)
    aggregate_over_seeds(flat).to_csv(paths["aggregate"], index=False)
    return paths
