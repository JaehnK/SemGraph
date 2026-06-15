#!/usr/bin/env python3
"""SemGraph experiment runner.

Use ``plan`` before ``run`` to inspect the generated preliminary, main, or
ablation protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "core"))

from core.services.experiment import ABLATION_AXES, DATASET_NAMES, build_run_specs
from core.services.experiment.runner import ExperimentRunner


def main() -> int:
    args = _parse_args()
    dataset_names = _normalise_multi(args.dataset, DATASET_NAMES)
    axes = _normalise_multi(args.axis, ABLATION_AXES)

    specs = build_run_specs(
        preset=args.preset,
        dataset_names=dataset_names,
        axes=axes,
        seed_limit=args.seed_limit,
        include_baselines=not args.no_baselines,
    )

    if args.max_runs is not None:
        specs = specs[:args.max_runs]

    if args.command == "plan" or args.dry_run:
        _print_plan(specs, as_json=args.json)
        return 0

    preset = "preliminary" if args.preset == "smoke" else args.preset
    if preset in {"main", "ablation"} and args.max_runs is None and not args.allow_large:
        raise SystemExit(
            "main/ablation runs are large. Pass --allow-large or use --max-runs."
        )

    runner = ExperimentRunner(output_dir=args.output_dir)
    summaries = runner.run_many(specs)
    print(f"Completed {len(summaries)} run(s).")
    print(f"Results: {Path(args.output_dir) / 'results'}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SemGraph experiment protocols.")
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument(
        "--preset",
        choices=("smoke", "preliminary", "main", "ablation"),
        default="preliminary",
        help="Experiment preset. smoke is an alias for preliminary.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=("all", *DATASET_NAMES),
        help="Dataset to include. Repeat for multiple datasets. Defaults to all.",
    )
    parser.add_argument(
        "--axis",
        action="append",
        choices=("all", *ABLATION_AXES),
        help="Ablation axis to include. Defaults to all for ablation.",
    )
    parser.add_argument("--seed-limit", type=int, help="Use only the first N preset seeds.")
    parser.add_argument("--max-runs", type=int, help="Cap generated runs for debugging.")
    parser.add_argument("--output-dir", default="artifacts/experiments")
    parser.add_argument("--no-baselines", action="store_true", help="Run SemGraph only.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running.")
    parser.add_argument("--json", action="store_true", help="Print plan as JSON.")
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Allow uncapped main or ablation execution.",
    )
    return parser.parse_args()


def _normalise_multi(
    values: Optional[List[str]],
    choices: tuple[str, ...],
) -> Optional[List[str]]:
    if not values or "all" in values:
        return None
    return values


def _print_plan(specs: List[Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps([spec.to_dict() for spec in specs], indent=2, ensure_ascii=False))
        return

    by_preset = Counter(spec.preset for spec in specs)
    by_dataset = Counter(spec.dataset.name for spec in specs)
    by_axis = Counter(spec.ablation_axis or "none" for spec in specs)

    print(f"Total runs: {len(specs)}")
    print(f"Presets: {_format_counter(by_preset)}")
    print(f"Datasets: {_format_counter(by_dataset)}")
    print(f"Ablation axes: {_format_counter(by_axis)}")
    if specs:
        print("\nFirst run:")
        print(json.dumps(specs[0].to_dict(), indent=2, ensure_ascii=False))


def _format_counter(counter: Counter) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counter.items()))


if __name__ == "__main__":
    raise SystemExit(main())
