"""Dataset materialization for SemGraph experiment runs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .specs import DatasetSpec


TEXT_COLUMN = "body"
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class PreparedDataset:
    """CSV dataset prepared for the existing SemGraph pipeline."""

    name: str
    csv_path: str
    text_column: str
    num_documents: int
    source: str
    label_column: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetMaterializer:
    """Load public corpora and write the text column expected by SemGraph."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, spec: DatasetSpec, seed: int) -> PreparedDataset:
        csv_path = self.output_dir / f"{spec.name}_{spec.num_documents}_seed{seed}.csv"
        metadata_path = csv_path.with_suffix(".metadata.json")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return PreparedDataset(
                name=spec.name,
                csv_path=str(csv_path),
                text_column=TEXT_COLUMN,
                num_documents=len(df),
                source=_source_name(spec.name),
                label_column=LABEL_COLUMN if LABEL_COLUMN in df.columns else None,
                metadata={"reused": True, "metadata_path": str(metadata_path)},
            )

        if spec.name == "ag_news":
            df = self._load_ag_news(spec, seed)
        elif spec.name == "20_newsgroups":
            df = self._load_20_newsgroups(spec, seed)
        elif spec.name == "arxiv":
            df = self._load_arxiv(spec, seed)
        else:
            raise ValueError(f"Unsupported dataset: {spec.name}")

        df = self._sample(df, spec.num_documents, seed, spec.balanced)
        if df.empty:
            raise RuntimeError(f"No documents loaded for dataset: {spec.name}")

        df.to_csv(csv_path, index=False)
        metadata = {
            "name": spec.name,
            "source": _source_name(spec.name),
            "requested_documents": spec.num_documents,
            "actual_documents": len(df),
            "balanced": spec.balanced,
            "seed": seed,
            "label_counts": (
                df[LABEL_COLUMN].value_counts().sort_index().to_dict()
                if LABEL_COLUMN in df.columns
                else {}
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return PreparedDataset(
            name=spec.name,
            csv_path=str(csv_path),
            text_column=TEXT_COLUMN,
            num_documents=len(df),
            source=_source_name(spec.name),
            label_column=LABEL_COLUMN if LABEL_COLUMN in df.columns else None,
            metadata=metadata,
        )

    def _load_ag_news(self, spec: DatasetSpec, seed: int) -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install the research dependencies to load AG News.") from exc

        dataset = load_dataset("ag_news", split="train")
        df = dataset.to_pandas()[["text", "label"]]
        df = df.rename(columns={"text": TEXT_COLUMN, "label": LABEL_COLUMN})
        return _clean_frame(df)

    def _load_20_newsgroups(self, spec: DatasetSpec, seed: int) -> pd.DataFrame:
        from sklearn.datasets import fetch_20newsgroups

        dataset = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=seed,
        )
        df = pd.DataFrame(
            {
                TEXT_COLUMN: dataset.data,
                LABEL_COLUMN: [dataset.target_names[i] for i in dataset.target],
            }
        )
        return _clean_frame(df)

    def _load_arxiv(self, spec: DatasetSpec, seed: int) -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install the research dependencies to load arXiv.") from exc

        stream = load_dataset(
            "gfissore/arxiv-abstracts-2021",
            split="train",
            streaming=True,
        )
        stream = stream.shuffle(seed=seed, buffer_size=10_000)

        records: List[Dict[str, str]] = []
        per_label = max(1, math.ceil(spec.num_documents / len(_ARXIV_TOP_LEVELS)))
        label_counts = {label: 0 for label in _ARXIV_TOP_LEVELS}
        max_scan = max(10_000, spec.num_documents * 200)

        for index, row in enumerate(stream):
            if index >= max_scan or len(records) >= spec.num_documents:
                break

            label = _primary_arxiv_category(row.get("categories"))
            if spec.balanced and label not in label_counts:
                continue
            if spec.balanced and label_counts[label] >= per_label:
                continue

            text = _normalise_text(f"{row.get('title', '')}. {row.get('abstract', '')}")
            if not text:
                continue

            records.append({TEXT_COLUMN: text, LABEL_COLUMN: label})
            if label in label_counts:
                label_counts[label] += 1

        return _clean_frame(pd.DataFrame(records))

    def _sample(
        self,
        df: pd.DataFrame,
        num_documents: int,
        seed: int,
        balanced: bool,
    ) -> pd.DataFrame:
        if len(df) <= num_documents:
            return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        if not balanced or LABEL_COLUMN not in df.columns:
            return df.sample(n=num_documents, random_state=seed).reset_index(drop=True)

        labels = sorted(df[LABEL_COLUMN].dropna().unique())
        per_label = max(1, num_documents // len(labels))
        sampled = []
        sampled_indices: List[int] = []
        for label in labels:
            group = df[df[LABEL_COLUMN] == label]
            sample = group.sample(n=min(per_label, len(group)), random_state=seed)
            sampled.append(sample)
            sampled_indices.extend(sample.index.tolist())

        result = pd.concat(sampled, ignore_index=True)
        if len(result) < num_documents:
            remaining = df.drop(sampled_indices, errors="ignore")
            fill_count = min(num_documents - len(result), len(remaining))
            if fill_count > 0:
                result = pd.concat(
                    [result, remaining.sample(n=fill_count, random_state=seed)],
                    ignore_index=True,
                )
        return result.sample(frac=1.0, random_state=seed).head(num_documents).reset_index(drop=True)


_ARXIV_TOP_LEVELS = ("cs", "math", "physics", "stat", "q-bio", "q-fin", "econ", "eess")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Expected text column '{TEXT_COLUMN}'")
    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].map(_normalise_text)
    df = df[df[TEXT_COLUMN].str.len() > 0]
    return df.reset_index(drop=True)


def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _primary_arxiv_category(value: Any) -> str:
    if isinstance(value, Iterable) and not isinstance(value, str):
        value = next(iter(value), "")
    token = str(value or "").split()[0]
    if token.startswith("physics."):
        return "physics"
    return token.split(".")[0] if token else "unknown"


def _source_name(dataset_name: str) -> str:
    return {
        "ag_news": "Hugging Face ag_news",
        "20_newsgroups": "scikit-learn 20 Newsgroups",
        "arxiv": "Hugging Face gfissore/arxiv-abstracts-2021",
    }[dataset_name]
