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
├── run_custom_eval_pipeline.py        # CLI: domain custom dataset (Seq 1-4)
├── run_cr_eval.py                     # CLI: Contextual Retrieval baseline evaluation
├── run_cr_significance.py             # CLI: CR significance tests with pinned tree run_ids
├── run_cr_significance_from_saved.py  # CLI: CR significance from saved per-query scores
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
│   ├── eval_runner.py                 # Seq 1-4 runner with dataset normalisation
│   ├── retrieval_evaluator.py         # Embed query -> cosine search -> metrics
│   ├── metrics.py                     # Recall@K, MRR, NDCG@K
│   ├── reranker.py                    # LLM-based reranker (optional)
│   └── statistical_tests.py          # Bootstrap significance tests
├── ingestion/                         # Document ingestion pipeline
│   ├── orchestrator.py                # Docling parse -> tree build -> GCS upload
│   ├── ablation_configs.py            # Ablation configuration registry (10 configs)
│   ├── tree_builder.py                # RAPTOR tree construction per ablation
│   ├── tree_resolver.py              # Load trees from GCS with local cache
│   ├── contextual_retrieval.py       # CR context generation (Gemini Flash)
│   ├── reanchor_per_ablation.py      # Re-anchor ground truth per ablation tree
│   └── gcs_cache.py                  # GCS upload/download with local mirror
├── data/                              # Intermediate files for image query evaluation
├── datasets/                          # Dataset cache (auto-downloaded at runtime)
├── results/                           # Committed benchmark results (academic)
├── scripts/                           # Analysis scripts
│   └── qasper_table_density_analysis.py
├── backend/evaluation/                # Image query evaluation scripts
├── workers/                           # Cloud Run Job worker
│   └── eval_worker.py                # Entrypoint: rebuild_tree / reanchor / evaluate / cr_eval
├── deploy_cloudrun.sh                 # Build + push Docker image + create Cloud Run jobs
├── trigger_pipeline.sh                # Trigger full evaluation pipeline on Cloud Run
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

Ablation study on two aircraft maintenance manuals (AS-AMM-01-000: 154 queries, SC10000AMM: 146 queries) with retrieval metrics (Recall@K, MRR, NDCG@10) and paired bootstrap significance tests (10K iterations, seed 42). All rows use per-ablation anchored ground-truth chunk IDs for internally consistent comparisons.

**Proposed System vs. Text-Only RAPTOR (Seq 1):**

| Document | Config | R@1 | R@5 | R@10 | MRR | NDCG@10 | p |
|---|---|---|---|---|---|---|---|
| AS-AMM | Text-only RAPTOR | .214 | .435 | .500 | .315 | .355 | — |
| AS-AMM | Proposed system | .286 | .545 | .604 | .400 | .445 | .005 |
| SC10000 | Text-only RAPTOR | .253 | .493 | .548 | .361 | .405 | — |
| SC10000 | Proposed system | .534 | .637 | .658 | .585 | .600 | .000 |

**Sub-Innovation Ablation — SC10000AMM (Seq 3):**

| Config | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| Proposed (full) | .534 | .637 | .658 | .585 | .600 |
| −Table P-C (2a) | .274 | .541 | .610 | .398 | .446 |
| −Headers (2b) | .260 | .507 | .582 | .364 | .414 |
| −Captions (2c) | .274 | .486 | .562 | .370 | .413 |
| −All (naive) | .226 | .425 | .452 | .313 | .347 |

**Contextual Retrieval Baseline:**

Replaces deterministic section-header propagation with LLM-generated chunk context (Gemini 2.5 Flash, Anthropic CR method). Uses naive chunking + CR context + multimodal embeddings + RAPTOR tree.

| Document | Config | R@1 | R@5 | R@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| AS-AMM | Naive baseline | .166 | .335 | .387 | .244 | .276 |
| AS-AMM | **CR baseline** | **.154** | **.363** | **.425** | **.247** | **.286** |
| AS-AMM | Proposed system | .243 | .470 | .519 | .343 | .382 |
| SC10000 | Naive baseline | .201 | .378 | .402 | .279 | .309 |
| SC10000 | **CR baseline** | **.140** | **.348** | **.384** | **.226** | **.263** |
| SC10000 | Proposed system | .476 | .567 | .585 | .521 | .534 |

| Comparison | AS-AMM Δ NDCG | p | SC10000 Δ NDCG | p |
|---|---|---|---|---|
| CR vs naive | +.010 | .580 | −.046 | .003 |
| CR vs proposed | +.096 | .001 | +.272 | .000 |
| CR vs −headers | −.085 | .006 | −.106 | .000 |

CR provides no significant improvement over naive chunking (AS-AMM p=.580) and significantly hurts on SC10000AMM (p=.003). The deterministic structural approach outperforms LLM-generated context.

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

Runs four evaluation sequences plus a CR baseline against two documents:

| Sequence | Comparison |
|---|---|
| Seq 1 | `full_context_aware` vs `original_raptor_text_only` |
| Seq 2 | `full_context_aware` vs `semantic_chunking_baseline` |
| Seq 3 | `full_context_aware` vs each sub-innovation ablation |
| Seq 4 | 2x2 factorial: embedding type x chunking strategy |
| CR | `contextual_retrieval` vs naive / proposed / no_header_prop |

### Ablation Configurations

| Name | Description |
|---|---|
| `full_context_aware` | Full system: all features enabled |
| `original_raptor_text_only` | RAPTOR tree, text-only embeddings |
| `semantic_chunking_baseline` | Semantic chunking, no context features |
| `baseline_naive_chunking` | Naive fixed-size chunking |
| `no_table_parent_child` | Disable table parent-child links |
| `no_header_propagation` | Disable header propagation |
| `no_caption_folding` | Disable figure-caption folding |
| `flat_no_raptor` | Flat retrieval, no tree |
| `text_context_aware` | Context-aware chunking, text-only embeddings |
| `contextual_retrieval` | LLM-generated CR context + naive chunking |

### Retrieval Metrics

| Metric | Description |
|---|---|
| Recall@K | Fraction of queries with relevant chunk in top-K |
| MRR | Mean Reciprocal Rank |
| NDCG@K | Normalised Discounted Cumulative Gain |

Results include bootstrap p-values for pairwise comparisons.

### Contextual Retrieval Baseline

The CR baseline implements Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) method: for each naive chunk, Gemini 2.5 Flash generates a short context snippet situating the chunk within the full document. The context is prepended to chunk text before embedding.

**Step 1: Build CR trees** (requires Gemini Flash API access, ~1-2 hours per document):

```bash
# Build CR tree for each document
python -m ingestion.orchestrator \
    --document "datasets/AS AMM 01 000 I1 R1 Feb 2 2018_compressed.pdf" \
    --gcs-bucket raptor-assets \
    --ablations contextual_retrieval

python -m ingestion.orchestrator \
    --document "datasets/SC10000AMM Rev J.pdf" \
    --gcs-bucket raptor-assets \
    --ablations contextual_retrieval
```

CR context generation saves incremental checkpoints to GCS every 25 chunks. If interrupted, re-running the same command resumes from the last checkpoint.

**Step 2: Re-anchor ground truth for CR:**

```bash
python -m ingestion.reanchor_per_ablation \
    --input datasets/combined_eval_dataset.json \
    --output-dir datasets/ \
    --cache-keys "SC10000AMM:<sc_cache_key>,AS-AMM-01-000:<as_cache_key>" \
    --ablations contextual_retrieval
```

**Step 3: Evaluate CR baseline:**

```bash
python run_cr_eval.py
```

Evaluates CR against naive baseline, proposed system, and no_header_prop with paired bootstrap significance tests. Outputs paper-ready formatted results and LaTeX table rows.

**Cloud Run deployment** (for running on server):

```bash
# Deploy worker image
GCP_PROJECT_ID=<project> RAPTOR_BACKEND_SRC=/path/to/backend bash deploy_cloudrun.sh

# Trigger CR tree build (both documents in sequence)
gcloud run jobs execute raptor-tree-rebuild --region <region> --tasks 1 \
    --update-env-vars "TASK_TYPE=rebuild_tree,TASK_CONFIGS=<cache_key>:contextual_retrieval:<filename>"

# Trigger CR evaluation (reanchor + evaluate + upload results to GCS)
gcloud run jobs execute raptor-evaluation --region <region> --tasks 1 \
    --update-env-vars "TASK_TYPE=cr_eval,CACHE_KEYS=AS-AMM-01-000:<as_key>|SC10000AMM:<sc_key>"
```

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
- `.cr-checkpoints/` — incremental CR context generation checkpoints
- `evaluation/results/` — per-run evaluation outputs
- `*.pkl` — tree pickle files
- `*.checkpoint.json` — benchmark resume state
- `datasets/qasper_*/`, `datasets/sstqa_*/`, `datasets/mpdocvqa_cache/` — dataset and embedding caches
