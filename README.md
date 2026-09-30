# SemGraph

SemGraph clusters the vocabulary of a text corpus into topics. It builds a word
co-occurrence graph, gives each word a DistilBERT node feature, learns node
representations with a GraphMAE2 masked autoencoder, and clusters those
representations with spherical k-means. The number of clusters is chosen
automatically.

```
corpus (CSV)
  → preprocessing (spaCy)
  → word co-occurrence graph (top-N words, weighted edges)
  → DistilBERT node features, projected to embed_size
  → GraphMAE2 self-supervised training (edge-weight aware)
  → spherical k-means with automatic k (Kneedle)
  → metrics + artifacts (JSON, CSV, figures)
```

## Requirements

- Linux x86_64, Python 3.9–3.11
- [uv](https://docs.astral.sh/uv/)
- A C/C++ compiler for the Cython extensions
- An NVIDIA GPU with a driver that supports CUDA 12.1. The lock file installs
  the CUDA 12.1 builds of torch and DGL. Without a usable GPU, GraphMAE
  training falls back to the CPU.

The dependency versions are pinned in `pyproject.toml` and `uv.lock`. A few of
them are pinned because the others require it:

| Package | Pin | Reason |
|---|---|---|
| `torch` | `2.2.1` | DGL 2.1 GraphBolt wheels exist only for torch 2.0.0–2.2.1 |
| `dgl` | `2.1.0+cu121` | Required by GraphMAE2. The CUDA build is only on `data.dgl.ai`, which `pyproject.toml` sets as the index for `dgl` |
| `spacy` / `en_core_web_sm` | `3.7.x` / `3.7.1` | Python 3.9 wheels and thinc ABI compatibility |
| `numpy` | `<2` | Required by the spaCy/thinc versions above |

## Setup

```bash
git clone https://github.com/JaehnK/SemGraph.git
cd SemGraph

# 1. GraphMAE2 (see the note below)
git clone https://github.com/THUDM/GraphMAE2.git core/GraphMAE2
git -C core/GraphMAE2 checkout bc473e8

# 2. Python environment
uv sync --group research   # runtime + dev + experiment deps (datasets, Louvain/Leiden)

# 3. Cython extensions (trie, co-occurrence counting)
cd core/entities && uv run python _cython_setup.py build_ext --inplace && cd ../..
```

> **GraphMAE2 note.** `core/GraphMAE2` is a gitlink to upstream THUDM/GraphMAE2
> at commit `bc473e8`. There is no `.gitmodules` entry, so `git clone` leaves
> the directory empty and you must clone it by hand as shown above. SemGraph
> also makes edge-weight changes to `models/edcoder.py` and `models/gcn.py` in
> that checkout. These changes are not published yet, so the upstream commit
> alone does not reproduce edge-weighted training.

DistilBERT (`distilbert-base-uncased`) is downloaded from the Hugging Face Hub
the first time it is used and is cached under `DistillBERT/model/`.

The code imports modules both as `core.services...` and as `services...`, so
set `PYTHONPATH=core` for every command:

```bash
export PYTHONPATH=core
```

## Running a single pipeline

The input is a CSV file with one document per row. The text column defaults to
`body`.

```bash
# Small CPU run
uv run python pipelines/main.py --data data/ag_news.csv --max-docs 1000 --epochs 100 --device cpu

# Full options
uv run python pipelines/main.py --help
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--embed-size` | Size of the BERT node features after projection |
| `--epochs` | Number of GraphMAE training epochs |
| `--mask-rate` | GraphMAE node mask rate |
| `--encoder-type`, `--decoder-type` | `gat`, `tsgat`, `gcn`, `mlp`, `linear` |
| `--edge-weight-threshold` | Minimum co-occurrence count for an edge to be kept |
| `--edge-top-k` | Maximum number of edges kept per node (`-1` keeps all) |
| `--random-seed` | Seed for Python, NumPy, torch and DGL |
| `--mode compare-traditional` | Also run the Louvain/Leiden baselines |

The results are written to `--output` (default `./semgraph_output`).

## Running experiments

`pipelines/experiment.py` runs the experiment protocol. A preset expands to one
run per dataset, seed and configuration. The datasets (AG News,
20 Newsgroups, arXiv abstracts) are downloaded and sampled with a balanced
split per label.

| Preset | Documents per dataset | Seeds | Purpose |
|---|---|---|---|
| `preliminary` (alias `smoke`) | 1,000 | 3 | Checking the setup |
| `main` | 20k–100k | 10 | Main results |
| `ablation` | same as `main` | 10 | Varies one axis at a time: `mask_rate`, `graphmae_epochs`, `embed_size`, `top_n_words`, `edge_weight_threshold`, `auto_k_range` |

By default every run also evaluates these baselines on the same graph:
k-means on the raw node features, Louvain and Leiden.

```bash
# Show the runs a preset expands to
uv run python pipelines/experiment.py plan --preset preliminary

# Run the preliminary preset on one dataset and one seed
uv run python pipelines/experiment.py run --preset preliminary --dataset ag_news --seed-limit 1

# main and ablation are large and are capped unless --allow-large is given
uv run python pipelines/experiment.py run --preset ablation --axis mask_rate --max-runs 4

# Rebuild the summary tables from every run on disk
uv run python pipelines/experiment.py aggregate
```

The output is written under `artifacts/experiments/`:

```
artifacts/experiments/
├── datasets/                       # sampled CSVs + metadata per dataset/seed
└── results/
    ├── experiment_summary.csv      # one row per run and model
    ├── experiment_aggregate.csv    # mean/std over seeds per preset, dataset, config and model
    └── <preset>/<run_id>/
        ├── experiment_summary.json
        └── semgraph/               # full pipeline artifacts
```

GPU runs with the same seed give the same cluster assignments, so NPMI and
modularity match exactly. Metrics computed from the embedding values
(Silhouette, Davies–Bouldin, Calinski–Harabasz) can differ around the 7th
significant digit, because DGL's CUDA kernels do not add numbers in a fixed
order. CPU and GPU runs of the same seed give different results. Do not mix
them in one table.

Each run reports Silhouette, Davies–Bouldin, Calinski–Harabasz, NPMI and graph
modularity for every model. It also compares the SemGraph word clusters with
each baseline's clusters using ARI, NMI and mean best-match Jaccard.

## Project layout

```
core/
├── entities/            # domain model: Corpus/Document, Sentence, Word, WordGraph (+ Cython)
├── services/
│   ├── semgraph/        # SemGraphConfig, SemGraphPipeline (6-stage orchestration)
│   ├── ports/           # Protocol interfaces (TextPreprocessor, EmbeddingProvider, …)
│   ├── adapters/        # BERT embedding provider, artifact writer
│   ├── Document/ Sentence/ Word/   # preprocessing services (spaCy)
│   ├── Graph/           # graph construction, node features, DGL/PyG conversion
│   ├── DBert/           # DistilBERT service
│   ├── GraphMAE/        # GraphMAE2 wrapper
│   ├── clustering/      # spherical k-means, auto-k, Louvain/Leiden baselines
│   ├── Metric/          # clustering and topic-coherence metrics
│   ├── Visualization/   # t-SNE, word clouds
│   └── experiment/      # presets, dataset loading, experiment runner
└── GraphMAE2/           # upstream GraphMAE2 checkout (see Setup)
pipelines/
├── main.py              # single-pipeline CLI
└── experiment.py        # experiment protocol CLI
tests/
```

## Tests

```bash
PYTHONPATH=core uv run python -m pytest -q tests
```

The tests in the repository are smoke tests. They check imports, the ports,
the BERT-only node feature path, the artifact writer and how presets expand
into runs. They do not need a GPU.
