# Multimodal RAPTOR Evaluation

Evaluation harness for the paper **"Multimodal RAPTOR: Context-Aware Hierarchical Retrieval for Technical Document QA"**. Benchmarks a multimodal RAPTOR retrieval system across four evaluation settings:

- **QASPER** — academic paper QA (1,005 questions, 393 papers)
- **SSTQA** — slide-structure table QA with table-heavy documents
- **MP-DocVQA** — multi-page document visual QA (5,187 questions, 927 documents)
- **Domain Custom Dataset** — proprietary aircraft maintenance manuals with ablation study (300 queries, 2 documents)

---

## Repository Structure

```
RAPTOR-Evaluation/
├── run_eval.py                        # CLI: academic benchmarks (QASPER, SSTQA, MP-DocVQA)
├── run_custom_eval_pipeline.py        # CLI: domain custom dataset evaluation
├── metrics.py                         # ANLS / Token F1 / ROUGE-L / Exact Match scoring
├── rebuild_missing_trees.py           # Utility: rebuild failed ablation trees
├── requirements.txt                   # Python dependencies
├── benchmarks/
│   ├── base_benchmark.py              # Abstract base: load / run / evaluate / checkpoint
│   ├── qasper_benchmark.py            # QASPER benchmark implementation
│   ├── sstqa_benchmark.py             # SSTQA benchmark implementation
│   ├── mpdocvqa_benchmark.py          # MP-DocVQA benchmark implementation
│   └── raptor_tree.py                 # In-memory RAPTOR tree builder
├── configs/                           # YAML configs for each benchmark variant
│   ├── collapsed_tree.yaml            # QASPER Row B: collapsed RAPTOR
│   ├── flat_retrieval.yaml            # QASPER Row A: flat, no tree
│   ├── collapsed_no_multimodal.yaml   # QASPER ablation: text-only embeddings
│   ├── collapsed_no_table_pc.yaml     # QASPER ablation: no table parent-child
│   ├── collapsed_no_expansion.yaml    # QASPER ablation: no parent-swap expansion
│   ├── qasper_collapsed_multimodal.yaml # QASPER Row C: full system
│   ├── sstqa_flat.yaml                 # SSTQA Row A: flat retrieval
│   ├── sstqa_collapsed.yaml            # SSTQA Row B: collapsed, no table P-C
│   ├── sstqa_collapsed_table_pc.yaml   # SSTQA Row C: collapsed + table P-C
│   ├── mpdocvqa_flat_textonly.yaml     # MP-DocVQA Row A: OCR text, flat retrieval
│   └── mpdocvqa_full_system.yaml      # MP-DocVQA Row C: multimodal + RAPTOR
├── evaluation/                        # Custom dataset evaluation engine
│   ├── eval_runner.py                 # Seq 1/2/3 runner with dataset normalisation
│   ├── retrieval_evaluator.py         # Embed query -> cosine search -> metrics
│   ├── metrics.py                     # Recall@K, MRR, NDCG@K
│   ├── reranker.py                    # LLM-based reranker (optional)
│   └── statistical_tests.py          # Bootstrap significance tests
├── ingestion/                         # Document ingestion pipeline
│   ├── orchestrator.py                # Docling parse -> tree build -> GCS upload
│   ├── ablation_configs.py            # Ablation configuration registry
│   ├── tree_builder.py                # RAPTOR tree construction per ablation
│   └── tree_resolver.py              # Load trees from GCS with local cache
├── data/                              # Intermediate files for image query evaluation
├── datasets/                          # Dataset cache (auto-downloaded at runtime)
├── results/                           # Committed evaluation results
├── scripts/                           # Analysis scripts
│   └── qasper_table_density_analysis.py
├── backend/evaluation/                # Image query evaluation scripts
├── workers/                           # Cloud Run Job worker
└── tests/                             # Integration tests
```

---

## Results

### QASPER (Zero-Shot, 1,005 questions)

| Configuration | F1 | EM |
|---|---|---|
| A: Flat retrieval (level-0 only) | 0.4489 | 0.2249 |
| B: Vanilla RAPTOR (collapsed) | 0.4435 | 0.2209 |
| C: Full system (table P-C + parent-swap) | 0.4491 | 0.2289 |

The QASPER null result is explained by table sparsity: the dataset is 95.4% table-free (see `results/qasper_table_density_analysis.tex`), confirming that the system's gains activate on table-dense documents.

### SSTQA (Zero-Shot)

| Configuration | Acc (%) | R-L (%) | ANLS (%) | F1 (%) |
|---|---|---|---|---|
| A: Flat leaf-only | 47.56 | 57.66 | 56.90 | 58.49 |
| B: Collapsed (no table P-C) | 47.43 | 57.19 | 56.57 | 57.98 |
| C: Collapsed + table P-C | 52.44 | 62.88 | 62.20 | 63.45 |

Table parent-child chunking activates on table-heavy documents: +5.01 pp Accuracy over flat retrieval.

### MP-DocVQA (Zero-Shot, 5,187 questions)

| Configuration | ANLS (%) | Page Acc (%) |
|---|---|---|
| A: Flat text, no tree | 61.49 | 70.85 |
| B: Leaf-only multimodal | 61.34 | 75.92 |
| C: Full system (collapsed) | 57.70 | 63.72 |

The zero-shot flat configuration (61.49% ANLS) exceeds the fine-tuned RAG-VT5 baseline (58.23% ANLS).

### Domain Custom Dataset (Retrieval-Only)

Ablation study on two aircraft maintenance manuals (AS-AMM-01-000 and SC10000AMM) with retrieval metrics (Recall@K, MRR, NDCG@10) and bootstrap significance tests. See `evaluation/results/` for full results.

Key findings:
- Full system vs. text-only RAPTOR: +0.195 NDCG@10 on SC10000AMM (p < .001)
- Context-aware vs. semantic chunking: +0.549 NDCG@10 on SC10000AMM (p < .001)
- All three sub-innovations significant on SC10000AMM (p < .001)

---

## Setup

```bash
git clone https://github.com/fyp-group18/RAPTOR-Evaluation.git
cd RAPTOR-Evaluation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The RAPTOR backend must be on PYTHONPATH
export PYTHONPATH=/path/to/raptor/backend:$PYTHONPATH
```

### Environment Variables

| Variable | Required by | Description |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | all pipelines | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | all pipelines | Path to service-account JSON |
| `DATABASE_URL` | setup scripts, tests | PostgreSQL connection string |

---

## Running Academic Benchmarks

### QASPER

```bash
# Row A: flat retrieval baseline
python run_eval.py \
    --benchmark qasper \
    --config configs/flat_retrieval.yaml \
    --output results/qasper_dev_flat.csv

# Row B: vanilla RAPTOR (collapsed)
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --output results/qasper_dev_collapsed.csv

# Row C: full system (table P-C + expansion)
python run_eval.py \
    --benchmark qasper \
    --config configs/qasper_collapsed_multimodal.yaml \
    --output results/qasper_dev_collapsed_multimodal.csv

# Quick smoke test (3 papers)
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --sample-size 3
```

### SSTQA

```bash
# Row A: flat leaf-only
python run_eval.py \
    --benchmark sstqa \
    --config configs/sstqa_flat.yaml \
    --output results/sstqa_flat.csv

# Row B: collapsed RAPTOR (no table P-C)
python run_eval.py \
    --benchmark sstqa \
    --config configs/sstqa_collapsed.yaml \
    --output results/sstqa_collapsed.csv

# Row C: collapsed + table parent-child
python run_eval.py \
    --benchmark sstqa \
    --config configs/sstqa_collapsed_table_pc.yaml \
    --output results/sstqa_collapsed_table_pc.csv
```

SSTQA tables and questions are auto-downloaded from the [ST-Raptor GitHub repository](https://github.com/OpenDataBox/ST-Raptor).

### MP-DocVQA

```bash
# Row A: flat text-only baseline
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_flat_textonly.csv

# Row C: full system
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_full_system.yaml \
    --output results/mpdocvqa_full_system.csv
```

Parallel sharding for large runs:

```bash
# Terminal 1
python run_eval.py --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_flat.csv --shard 0/2

# Terminal 2
python run_eval.py --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_flat.csv --shard 1/2
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--benchmark` | required | `qasper`, `sstqa`, or `mpdocvqa` |
| `--config` | required | Path to YAML config |
| `--output` | auto | Output CSV path |
| `--data-dir` | `datasets/` | Dataset cache directory |
| `--sample-size` | all | Limit to N items |
| `--top-k` | 5 | Chunks per question |
| `--shard` | off | Parallel split: `N/M` |
| `--rebuild-trees` | off | Force-rebuild RAPTOR trees |

Datasets are auto-downloaded on first run: QASPER and MP-DocVQA from HuggingFace, SSTQA from GitHub. All benchmarks support checkpoint-resume via `.checkpoint.json` files.

---

## Running Custom Dataset Evaluation

Evaluates retrieval quality on proprietary aircraft maintenance manuals using pre-built ablation trees. Source PDFs are not included in this repository.

```bash
python run_custom_eval_pipeline.py
```

Runs three evaluation sequences against two documents:

| Sequence | Comparison |
|---|---|
| Seq 1 | `full_context_aware` vs `original_raptor_text_only` |
| Seq 2 | `full_context_aware` vs `semantic_chunking_baseline` |
| Seq 3 | `full_context_aware` vs each sub-innovation ablation |

### Ablation Configurations

| Name | Description |
|---|---|
| `full_context_aware` | Full system: all features enabled |
| `original_raptor_text_only` | RAPTOR tree, text-only embeddings |
| `semantic_chunking_baseline` | Semantic chunking, full features |
| `baseline_naive_chunking` | Naive fixed-size chunking |
| `no_table_parent_child` | Disable table parent-child links |
| `no_header_propagation` | Disable header propagation |
| `no_caption_folding` | Disable figure-caption folding |
| `flat_no_raptor` | Flat retrieval, no tree |

### Retrieval Metrics

| Metric | Description |
|---|---|
| Recall@K | Fraction of queries with relevant chunk in top-K |
| MRR | Mean Reciprocal Rank |
| NDCG@K | Normalised Discounted Cumulative Gain |

Results include bootstrap p-values for pairwise comparisons.

---

## Table Density Analysis

The QASPER null result is explained by a per-paper table density analysis:

```bash
python scripts/qasper_table_density_analysis.py
```

Produces `results/qasper_per_paper_table_density.csv` and `results/qasper_table_density_analysis.tex`.

---

## Gitignored Files

The following are not committed and must be provided or generated locally:

- `credentials.json` — GCP service-account key
- `datasets/*.json` — evaluation datasets (anchored and un-anchored)
- `.tree-cache/` — local mirror of GCS tree artifacts
- `evaluation/results/` — per-run evaluation outputs
- `*.pkl` — tree pickle files
- `*.checkpoint.json` — benchmark resume state
- `datasets/qasper_*/`, `datasets/sstqa_*/`, `datasets/mpdocvqa_cache/` — dataset and embedding caches
