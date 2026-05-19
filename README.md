# RAPTOR Evaluation Harness

Evaluation harness for benchmarking the multimodal RAPTOR retrieval system against published academic baselines.

This harness validates RAPTOR's hierarchical retrieval against three benchmarks: QASPER, MP-DocVQA, and DocVQA, with full ablation configuration support.

## Dependency

This harness imports from the [raptor](https://github.com/fyp-group18/raptor) backend (`core.*` and `modules.*` packages). The main backend must be importable at runtime.

## Setup

```bash
# Clone this repo alongside (or inside) the main backend
git clone https://github.com/fyp-group18/RAPTOR-Evaluation.git

# Install eval-specific dependencies
pip install -r requirements.txt

# Ensure the raptor backend is on PYTHONPATH
export PYTHONPATH=/path/to/raptor/backend:$PYTHONPATH
```

Required environment variables (from the main backend):
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `DATABASE_URL` (for `test_collapsed_retrieval.py` only)

## Datasets

Datasets are auto-downloaded from HuggingFace and cached in `datasets/` (gitignored).

| Benchmark | HuggingFace ID | Split | Notes |
|-----------|----------------|-------|-------|
| QASPER | `allenai/qasper` | validation | NLP papers with QA; embeddings cached per-paper |
| MP-DocVQA | `lmms-lab/MP-DocVQA` | val | Multi-page document VQA; ~927 docs, up to 20 pages each |
| DocVQA | `lmms-lab/DocVQA` | test | Document visual QA |

## Running Benchmarks

```bash
# Quick smoke test: 3 papers from QASPER
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --sample-size 3

# Full QASPER benchmark (validation split, all papers)
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --output results/qasper_collapsed.csv

# QASPER with flat retrieval baseline
python run_eval.py \
    --benchmark qasper \
    --config configs/flat_retrieval.yaml \
    --output results/qasper_flat.csv

# Custom retrieval depth
python run_eval.py \
    --benchmark qasper \
    --config configs/collapsed_tree.yaml \
    --top-k 10

# MP-DocVQA smoke test (5 documents, both configs)
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --sample-size 5

python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_full_system.yaml \
    --sample-size 5

# MP-DocVQA Row A: full text-only baseline (external comparison: RAG-VT5 58.23% ANLS)
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

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--benchmark` | (required) | `qasper`, `mpdocvqa`, or `docvqa` |
| `--config` | (required) | Path to YAML config file |
| `--output` | auto-generated | Path for output CSV |
| `--data-dir` | `datasets/` | Dataset cache directory |
| `--sample-size` | all | Limit to N papers (for quick testing) |
| `--top-k` | 5 | Chunks to retrieve per question |
| `--split` | varies | Dataset split (`validation`, `test`, `train`) |
| `--shard` | off | Split dataset across terminals: `N/M` (0-indexed). E.g. `0/2` and `1/2` |

### Parallel Sharding (MP-DocVQA)

MP-DocVQA loads all 927 documents before answering questions. With `--shard` you can split this work across two terminals to roughly halve wall-clock time:

```bash
# Terminal 1 — docs 0-463
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_val_flat_textonly.csv \
    --shard 0/2

# Terminal 2 — docs 464-926
python run_eval.py \
    --benchmark mpdocvqa \
    --config configs/mpdocvqa_flat_textonly.yaml \
    --output results/mpdocvqa_val_flat_textonly.csv \
    --shard 1/2
```

Each shard writes to its own file (`*_shard0of2.csv`, `*_shard1of2.csv`) with its own checkpoint — stopping and restarting a shard resumes exactly where it left off. Both shards share the same `mpdocvqa_cache/` directory safely since they process disjoint document sets.

## How QASPER Works

1. **Load**: Download papers from HuggingFace, extract full text, chunk with `RecursiveCharacterTextSplitter(1600/200)`
2. **Embed**: Each paper's chunks are embedded with `gemini-embedding-2-preview` (3072-D). Cached to `datasets/qasper_embeddings/` as `.npz` files for re-runs
3. **Retrieve**: For each question, embed the query and cosine-search against the paper's chunk embeddings (top-K)
4. **Generate**: Feed retrieved chunks + question to Gemini Flash with a concise QA prompt
5. **Evaluate**: Token F1 + Exact Match against ground truth (max over multiple annotator answers)

### Resume Support

Benchmarks save a checkpoint after each prediction (`.checkpoint.json` next to the output CSV). If the run crashes, re-running the same command resumes from the last completed question.

## How MP-DocVQA Works

MP-DocVQA tests multi-page document understanding (up to 20 pages/document, ~927 documents).
Per-document retrieval is used — only chunks from the query document are searched — isolating
comprehension quality from corpus-level retrieval.

Two rows are evaluated:

| Row | Config | Features |
|-----|--------|----------|
| A | `mpdocvqa_flat_textonly.yaml` | OCR text -> flat chunks -> text-only embed -> flat retrieval |
| C | `mpdocvqa_full_system.yaml` | Multimodal page leaves + table P-C + RAPTOR tree -> collapsed retrieval |

Delta C-A = total contribution of multimodal + RAPTOR + table-aware on real visually rich documents.

**External comparison** (Lopez et al., 2025):
| System | ANLS |
|--------|------|
| RAG-VT5 base (text RAG, no reranker) | 58.23% |
| RAG-VT5 + reranker | 61.06% |
| RAG-Pix2Struct (visual RAG) | 54.10% |

### Caching

All intermediate results are cached in `datasets/mpdocvqa_cache/`:
- `ocr/` — Gemini Flash OCR text per page
- `captions/` — Gemini Flash page captions (Mode C)
- `table_detection/` — YES/NO table detection per page (Mode C)
- `tables/` — Extracted table markdown (Mode C)
- `page_images/` — Saved PNG images for multimodal embedding
- `embeddings/` — Leaf chunk embeddings per document per mode
- `trees/` — RAPTOR trees per document (Mode C)

On re-runs, cached Gemini calls are skipped entirely. Embeddings and trees are also cached.

## Configuration

Each YAML config controls which features are active during evaluation:

**QASPER configs:**

| Config File | Retrieval Mode | Description |
|-------------|---------------|-------------|
| `collapsed_tree.yaml` | collapsed | Full system (default) |
| `flat_retrieval.yaml` | flat | Leaf-only (pre-fix baseline) |
| `collapsed_no_multimodal.yaml` | collapsed | Text-only embeddings |
| `collapsed_no_table_pc.yaml` | collapsed | No table parent-child |
| `collapsed_no_expansion.yaml` | collapsed | No parent swap |

**MP-DocVQA configs:**

| Config File | Retrieval Mode | Description |
|-------------|---------------|-------------|
| `mpdocvqa_flat_textonly.yaml` | flat | Row A: OCR text, no tree |
| `mpdocvqa_full_system.yaml` | collapsed | Row C: multimodal + RAPTOR + table P-C |

Ablation toggles (`use_multimodal_embed`, `use_table_parent_child`, `use_retrieval_expansion`) define intent but are not yet implemented in the main codebase. The benchmark runner logs a warning when these are set to non-default values.

## Metrics

| Metric | Function | Used By |
|--------|----------|---------|
| ANLS | `anls_score()` | MP-DocVQA, DocVQA |
| Page Accuracy | per-question hit rate | MP-DocVQA (secondary) |
| Token F1 | `f1_token_score()` | QASPER |
| Exact Match | `exact_match()` | QASPER |

All metrics normalize text (lowercase, strip, remove articles/punctuation) before comparison. When multiple ground truths exist, the max score is taken.

## Output Format

Results are saved as CSV:

```csv
index,prediction,ground_truths,f1,exact_match
0,"predicted answer","gt1|gt2"
1,...
AVERAGE,,,0.7234,0.5100
```

A `.summary.txt` file is also created alongside each CSV with config info and aggregate scores.

## Testing Collapsed Retrieval

Run the diagnostic test script to verify collapsed tree retrieval works against a live database:

```bash
python tests/test_collapsed_retrieval.py
```

This will:
1. Query the database for chunk level distribution and embedding coverage
2. Verify table parent nodes have NULL embeddings
3. Compare flat vs collapsed retrieval for a test query
4. Print level distribution of results in each mode
