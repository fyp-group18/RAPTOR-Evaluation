# RAPTOR Evaluation Harness

Evaluation harness for benchmarking a multimodal RAPTOR retrieval system. Two separate pipelines are provided:

- **Academic benchmarks** (`run_eval.py`) — QASPER, MP-DocVQA, DocVQA against published baselines
- **Custom dataset evaluation** (`run_custom_eval_pipeline.py` / `python -m evaluation.eval_runner`) — ablation study on proprietary aircraft maintenance manuals with retrieval-only metrics

---

## Repository Map

| Path | Role |
|------|------|
| `run_eval.py` | CLI entry point — academic benchmarks |
| `run_custom_eval_pipeline.py` | CLI entry point — custom dataset evaluation (both documents) |
| `rebuild_missing_trees.py` | Utility: rebuild failed trees from cached Docling output |
| `metrics.py` | ANLS / Token F1 / Exact Match used by academic benchmarks |
| `requirements.txt` | Python dependencies |
| `requirements-cloudrun.txt` | Cloud Run subset (no heavy ML libs) |
| `Dockerfile` | Container image for Cloud Run deployment |
| `deploy_cloudrun.sh` | Builds and deploys the Cloud Run Job |
| `trigger_pipeline.sh` | Submits a Cloud Run Job execution |
| `credentials.json` | GCP service-account key — **gitignored, never committed** |
| **`benchmarks/`** | |
| `benchmarks/base_benchmark.py` | Abstract base class: load / run / evaluate |
| `benchmarks/qasper_benchmark.py` | QASPER benchmark |
| `benchmarks/mpdocvqa_benchmark.py` | MP-DocVQA benchmark |
| `benchmarks/docvqa_benchmark.py` | DocVQA benchmark |
| `benchmarks/raptor_tree.py` | In-memory RAPTOR tree builder (QASPER, no DB/GCS) |
| **`configs/`** | |
| `configs/collapsed_tree.yaml` | QASPER full system — collapsed + RAPTOR |
| `configs/flat_retrieval.yaml` | QASPER baseline — flat, no tree |
| `configs/collapsed_no_multimodal.yaml` | QASPER ablation — text-only embeddings |
| `configs/collapsed_no_table_pc.yaml` | QASPER ablation — no table parent-child |
| `configs/collapsed_no_expansion.yaml` | QASPER ablation — no parent-swap expansion |
| `configs/qasper_collapsed_multimodal.yaml` | QASPER multimodal embeddings variant |
| `configs/mpdocvqa_flat_textonly.yaml` | MP-DocVQA Row A — OCR text, flat retrieval |
| `configs/mpdocvqa_full_system.yaml` | MP-DocVQA Row C — multimodal + RAPTOR + table P-C |
| **`evaluation/`** | |
| `evaluation/eval_runner.py` | Sequences 1/2/3 runner; dataset normalisation; result serialiser |
| `evaluation/retrieval_evaluator.py` | Core: embed query → cosine search tree → compute metrics |
| `evaluation/metrics.py` | Retrieval metrics: Recall@K, MRR, NDCG |
| `evaluation/reranker.py` | LLM-based reranker (optional `embedding_reranked` mode) |
| `evaluation/statistical_tests.py` | Bootstrap significance tests |
| `evaluation/__main__.py` | `python -m evaluation.eval_runner` entry point |
| `evaluation/results/` | Output directory — **gitignored** |
| **`ingestion/`** | |
| `ingestion/orchestrator.py` | CLI: Docling parse → ablation tree build → GCS upload |
| `ingestion/ablation_configs.py` | Ablation configuration registry |
| `ingestion/docling_runner.py` | Wraps Docling document parser |
| `ingestion/gcs_cache.py` | GCS read/write for Docling output and tree artifacts |
| `ingestion/tree_builder.py` | Builds a RAPTOR tree for one ablation config |
| `ingestion/tree_resolver.py` | Loads tree artifacts from GCS with local `.tree-cache/` mirror |
| `ingestion/reanchor_chunk_ids.py` | **Setup script** — re-anchors chunk IDs from production DB |
| `ingestion/reanchor_per_ablation.py` | **Setup script** — re-anchors chunk IDs per ablation tree |
| `ingestion/export_production_tree.py` | **Setup script** — exports production DB tree to a pickle |
| **`datasets/`** | |
| `datasets/AS AMM 01 000 I1 R1 Feb 2 2018_compressed.pdf` | Source PDF — document AS-AMM-01-000 |
| `datasets/SC10000AMM Rev J.pdf` | Source PDF — document SC10000AMM |
| `datasets/combined_eval_dataset.json` | Base QA dataset (un-anchored) — **gitignored** |
| `datasets/combined_eval_dataset_anchored.json` | Primary input for `run_custom_eval_pipeline.py` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_full.json` | Per-ablation: `full_context_aware` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_flat_retrieval.json` | Per-ablation: `flat_retrieval` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_text_only_raptor.json` | Per-ablation: `original_raptor_text_only` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_semantic_chunking.json` | Per-ablation: `semantic_chunking_baseline` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_no_table_pc.json` | Per-ablation: `no_table_parent_child` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_no_header_prop.json` | Per-ablation: `no_header_propagation` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_no_caption_fold.json` | Per-ablation: `no_caption_folding` — **gitignored** |
| `datasets/combined_eval_dataset_anchored_no_context_aware.json` | Per-ablation: `baseline_naive_chunking` — **gitignored** |
| `datasets/image_eval_dataset.json` | Image/figure query dataset — **gitignored** |
| **`data/`** | Intermediate files for backend image evaluation |
| `data/page_mapping_AS-AMM-01-000.json` | Chapter-page → PDF-page mapping for doc 1 |
| `data/page_mapping_SC10000AMM.json` | Chapter-page → PDF-page mapping for doc 2 |
| `data/image_eval_dataset_fixed.json` | Image dataset with corrected chunk annotations |
| `data/image_eval_gt_per_ablation.json` | Per-ablation ground truth for image queries |
| `data/image_queries_redesigned.json` | Redesigned content-based image queries |
| `data/image_query_node_content.json` | Intermediate: node content for image query matching |
| `data/anchoring_report.json` | Report from the chunk-ID anchoring process |
| **`backend/evaluation/`** | |
| `backend/evaluation/fix_image_query_annotations.py` | **Setup script** — anchors image query chunk IDs to trees |
| `backend/evaluation/redesign_image_queries.py` | **Setup script** — creates content-based image queries |
| `backend/evaluation/run_image_eval.py` | Runs image-query evaluation (Seq 1 & Seq 3) |
| `backend/evaluation/run_image_eval_redesigned.py` | Runs evaluation on redesigned image queries |
| **`workers/`** | |
| `workers/eval_worker.py` | Cloud Run Job worker — dispatches by `CLOUD_RUN_TASK_INDEX` |
| **`tests/`** | |
| `tests/test_collapsed_retrieval.py` | Live-DB test: verifies collapsed tree retrieval works |

---

## Prerequisites

- Python 3.12+
- Access to the [raptor](https://github.com/fyp-group18/raptor) backend (`core.*` and `modules.*` must be importable)
- GCP service account with read access to the `raptor-assets` GCS bucket
- `GOOGLE_APPLICATION_CREDENTIALS` pointing to `credentials.json`

---

## Setup

```bash
git clone https://github.com/fyp-group18/RAPTOR-Evaluation.git
cd RAPTOR-Evaluation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put the backend on PYTHONPATH
export PYTHONPATH=/path/to/raptor/backend:$PYTHONPATH
```

Required environment variables:

| Variable | Required by | Description |
|----------|-------------|-------------|
| `GOOGLE_CLOUD_PROJECT` | all pipelines | GCP project ID |
| `GOOGLE_APPLICATION_CREDENTIALS` | all pipelines | Path to service-account JSON |
| `DATABASE_URL` | `tests/test_collapsed_retrieval.py`, setup scripts that read the DB | PostgreSQL connection string |

> **Note:** `datasets/*.json` files are gitignored. To reproduce the custom-dataset evaluation, either obtain the pre-built anchored datasets out-of-band, or regenerate them with the setup scripts described below.

---

## Pipeline 1 — Ingestion (Build Ablation Trees)

Parse a PDF with Docling, build RAPTOR trees under every ablation configuration, and upload artifacts to GCS.

```bash
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
GOOGLE_CLOUD_PROJECT=raptor-496700 \
PYTHONPATH="/path/to/raptor/backend:." \
python -m ingestion.orchestrator \
    --document "datasets/SC10000AMM Rev J.pdf" \
    --gcs-bucket raptor-assets

# Build specific ablations only
python -m ingestion.orchestrator \
    --document "datasets/SC10000AMM Rev J.pdf" \
    --gcs-bucket raptor-assets \
    --ablations full_context_aware,no_table_parent_child

# Parse only (skip tree building)
python -m ingestion.orchestrator \
    --document "datasets/SC10000AMM Rev J.pdf" \
    --gcs-bucket raptor-assets \
    --parse-only

# Register a pre-built tree pickle (skip rebuilding)
python -m ingestion.orchestrator \
    --document "datasets/SC10000AMM Rev J.pdf" \
    --gcs-bucket raptor-assets \
    --register-existing-tree production_tree.pkl \
    --ablation-label full_context_aware

# Dry run: see what would happen without executing
python -m ingestion.orchestrator \
    --document "datasets/SC10000AMM Rev J.pdf" \
    --gcs-bucket raptor-assets \
    --dry-run
```

Available ablation names (passed as comma-separated list to `--ablations`):

| Name | Description |
|------|-------------|
| `full_context_aware` | Full system: all features enabled |
| `original_raptor_text_only` | RAPTOR tree, text-only embeddings |
| `semantic_chunking_baseline` | Semantic chunking, full features |
| `baseline_naive_chunking` | Naive fixed-size chunking |
| `no_table_parent_child` | No table parent-child links |
| `no_header_propagation` | No header propagation |
| `no_caption_folding` | No figure-caption folding |
| `flat_no_raptor` | Flat retrieval, no tree |

Trees are stored in GCS at `gs://raptor-assets/trees/{cache_key}/{run_id}/{label}/tree.pkl` and mirrored locally to `.tree-cache/`.

---

## Pipeline 2 — Custom Dataset Evaluation

Evaluates retrieval quality on proprietary aircraft maintenance manuals using pre-built ablation trees. Requires trees to be present in `.tree-cache/` (built by Pipeline 1 or downloaded from GCS automatically).

### Quick start (both documents, all sequences)

```bash
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
GOOGLE_CLOUD_PROJECT=raptor-496700 \
PYTHONPATH="/path/to/raptor/backend:." \
python run_custom_eval_pipeline.py
```

This runs three evaluation sequences against two documents (AS-AMM-01-000 and SC10000AMM):

| Sequence | Comparison |
|----------|-----------|
| Seq 1 | `full_context_aware` vs `original_raptor_text_only` (multimodal RAPTOR contribution) |
| Seq 2 | `full_context_aware` vs `semantic_chunking_baseline` (context-aware chunking contribution) |
| Seq 3 | `full_context_aware` vs each ablation (per sub-innovation contribution) |

Results are written to `evaluation/results/{document}/`.

### Fine-grained control (`eval_runner` CLI)

```bash
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
GOOGLE_CLOUD_PROJECT=raptor-496700 \
PYTHONPATH="/path/to/raptor/backend:." \
python -m evaluation.eval_runner \
    --combined-dataset datasets/combined_eval_dataset_anchored.json \
    --image-dataset datasets/image_eval_dataset.json \
    --gcs-bucket raptor-assets \
    --cache-keys "SC10000AMM:c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0,AS-AMM-01-000:112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448" \
    --sequences 1,2,3 \
    --top-k 1,3,5,10 \
    --dataset-dir datasets/ \
    --retrieval-mode embedding_only
```

`eval_runner` CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--combined-dataset` | — | Path to `combined_eval_dataset_anchored.json` |
| `--image-dataset` | — | Path to `image_eval_dataset.json` |
| `--gcs-bucket` | required | GCS bucket holding tree artifacts |
| `--cache-key` | — | Single document cache key (legacy) |
| `--cache-keys` | — | `source:key,source:key` pairs for multi-doc |
| `--run-id` | latest | Pin to a specific ingestion run_id |
| `--sequences` | `1,2,3` | Which sequences to run |
| `--top-k` | `1,3,5,10` | K values for Recall@K / NDCG metrics |
| `--dataset-dir` | auto-detected | Directory for per-ablation anchored datasets |
| `--retrieval-mode` | `embedding_only` | `embedding_only`, `embedding_reranked`, or `both` |
| `--k-prime` | `30` | Cosine candidate pool size before reranking |
| `--results-dir` | `evaluation/results/` | Output directory |

### Per-ablation dataset anchoring (setup, one-time)

The `datasets/combined_eval_dataset_anchored*.json` files must have `ground_truth_chunk_id` values that match the actual chunk IDs in each ablation tree. Generate them with:

```bash
# Step 1: Re-anchor from GCS trees (preferred)
python -m ingestion.reanchor_per_ablation \
    --input datasets/combined_eval_dataset.json \
    --output-dir datasets/ \
    --cache-keys "SC10000AMM:c313...,AS-AMM-01-000:112e..." \
    --gcs-bucket raptor-assets

# Step 2 (alternative): Re-anchor from production DB
python -m ingestion.reanchor_chunk_ids \
    --input datasets/combined_eval_dataset.json \
    --output datasets/combined_eval_dataset_anchored.json \
    --doc-map "AS-AMM-01-000:87,SC10000AMM:88"
```

### Metrics

| Metric | Description |
|--------|-------------|
| `Recall@K` | Fraction of queries where the relevant chunk appears in top-K results |
| `MRR` | Mean Reciprocal Rank |
| `NDCG@K` | Normalised Discounted Cumulative Gain |

Results include bootstrap p-values for each pairwise comparison.

---

## Pipeline 3 — Academic Benchmarks

Validates RAPTOR retrieval against QASPER, MP-DocVQA, and DocVQA. Datasets are auto-downloaded from HuggingFace on first run.

### Dependency

Requires the [raptor](https://github.com/fyp-group18/raptor) backend on `PYTHONPATH` (`core.*` and `modules.*`).

### Running benchmarks

```bash
# QASPER: quick smoke test (3 papers)
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --sample-size 3

# QASPER: full validation set
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --output results/qasper_collapsed.csv

# QASPER: flat retrieval baseline
python run_eval.py \
    --benchmark qasper \
    --config configs/flat_retrieval.yaml \
    --output results/qasper_flat.csv

# MP-DocVQA Row A: flat text-only baseline
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_val_flat_textonly.csv

# MP-DocVQA Row C: full system
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_full_system.yaml \
    --output results/mpdocvqa_val_full_system.csv
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark` | required | `qasper`, `mpdocvqa`, or `docvqa` |
| `--config` | required | Path to YAML config file |
| `--output` | auto-generated | Output CSV path |
| `--data-dir` | `datasets/` | Dataset cache directory |
| `--sample-size` | all | Limit to N items |
| `--top-k` | 5 | Chunks retrieved per question |
| `--split` | varies | Dataset split (`validation`, `test`, `train`) |
| `--rebuild-trees` | off | Force-rebuild RAPTOR trees |
| `--shard` | off | Split across terminals: `N/M` (e.g. `0/2` and `1/2`) |

### Parallel sharding (MP-DocVQA)

```bash
# Terminal 1 — first half
python run_eval.py --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_flat.csv --shard 0/2

# Terminal 2 — second half
python run_eval.py --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_flat.csv --shard 1/2
```

Each shard writes to its own file (`*_shard0of2.csv`) with its own checkpoint.

### How QASPER works

1. Download papers from HuggingFace; chunk with `RecursiveCharacterTextSplitter(1600/200)`
2. Embed chunks with `gemini-embedding-2-preview` (3072-D); cached to `datasets/qasper_embeddings/`
3. Per question: embed query → cosine search → top-K chunks
4. Generate answer with Gemini Flash
5. Score: Token F1 + Exact Match (max over annotator answers)

### How MP-DocVQA works

Per-document retrieval over up to 20 pages per document (~927 documents).

| Row | Config | Features |
|-----|--------|----------|
| A | `mpdocvqa_flat_textonly.yaml` | OCR text → flat chunks → text-only embed |
| C | `mpdocvqa_full_system.yaml` | Multimodal page leaves + table P-C + RAPTOR tree |

Caching: `datasets/mpdocvqa_cache/` stores OCR, captions, table detection, embeddings, and trees. Cached Gemini calls are skipped on re-runs.

**External comparison (López et al., 2025):**

| System | ANLS |
|--------|------|
| RAG-VT5 base (text RAG, no reranker) | 58.23% |
| RAG-VT5 + reranker | 61.06% |
| RAG-Pix2Struct (visual RAG) | 54.10% |

### Datasets auto-downloaded from HuggingFace

| Benchmark | HuggingFace ID | Split |
|-----------|----------------|-------|
| QASPER | `allenai/qasper` | validation |
| MP-DocVQA | `lmms-lab/MP-DocVQA` | val |
| DocVQA | `lmms-lab/DocVQA` | test |

### Resume support

All benchmarks write a `.checkpoint.json` next to the output CSV. Re-running the same command resumes from the last completed question.

### Output format

```csv
index,prediction,ground_truths,f1,exact_match
0,"predicted answer","gt1|gt2",...
AVERAGE,,,0.7234,0.5100
```

A `.summary.txt` with config info and aggregate scores is written alongside each CSV.

---

## Backend Image Query Evaluation

Evaluates retrieval on figure/image queries separately from text queries.

### Setup scripts (one-time)

These scripts are run once to produce the data files under `data/`:

```bash
# Fix image query chunk annotations
PYTHONPATH="/path/to/raptor/backend:." \
python -m backend.evaluation.fix_image_query_annotations

# Redesign queries to be content-based rather than symptom-based
PYTHONPATH="/path/to/raptor/backend:." \
python -m backend.evaluation.redesign_image_queries
```

### Evaluation runs

```bash
# Original image queries (Seq 1 & Seq 3)
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
GOOGLE_CLOUD_PROJECT=raptor-496700 \
PYTHONPATH="/path/to/raptor/backend:." \
python -m backend.evaluation.run_image_eval

# Redesigned content-based image queries with old/new comparison
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
GOOGLE_CLOUD_PROJECT=raptor-496700 \
PYTHONPATH="/path/to/raptor/backend:." \
python -m backend.evaluation.run_image_eval_redesigned
```

Output goes to `results/image_queries/` and `results/image_queries_redesigned/`.

---

## Cloud Run Deployment

```bash
# Build and deploy the Cloud Run Job
./deploy_cloudrun.sh

# Trigger a pipeline run
./trigger_pipeline.sh
```

The `workers/eval_worker.py` reads `CLOUD_RUN_TASK_INDEX` and dispatches either a tree rebuild or an evaluation task based on `TASK_TYPE`, `TASK_CONFIGS`, and `EVAL_TASKS` environment variables.

---

## Tests

```bash
# Requires DATABASE_URL to be set
python tests/test_collapsed_retrieval.py
```

Verifies:
1. Chunk level distribution and embedding coverage
2. Table parent nodes have NULL embeddings
3. Flat vs collapsed retrieval comparison for a test query

---

## Gitignore notes

The following are **not committed** and must be provided or generated locally:

- `credentials.json` — GCP service-account key
- `datasets/*.json` — all evaluation datasets (anchored and un-anchored)
- `.tree-cache/` — local mirror of GCS tree artifacts (auto-populated at runtime)
- `evaluation/results/` — run outputs
- `*.pkl` — tree pickle files
- `*.checkpoint.json` — benchmark resume state
- `datasets/qasper_*/`, `datasets/mpdocvqa_cache/` — HuggingFace and Gemini caches
