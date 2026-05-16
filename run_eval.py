"""
CLI entry point for running RAPTOR evaluation benchmarks.

Usage:
    cd backend

    # Build RAPTOR trees + run collapsed retrieval (first run: ~44min tree build)
    PYTHONPATH=. uv run python RAPTOR-evaluation/run_eval.py \\
        --benchmark qasper \\
        --config RAPTOR-evaluation/configs/collapsed_tree.yaml \\
        --output RAPTOR-evaluation/results/qasper_dev_collapsed.csv

    # Trees cached — only retrieval + generation (fast re-run)
    PYTHONPATH=. uv run python RAPTOR-evaluation/run_eval.py \\
        --benchmark qasper \\
        --config RAPTOR-evaluation/configs/collapsed_tree.yaml \\
        --output RAPTOR-evaluation/results/qasper_dev_collapsed.csv

    # Force rebuild all RAPTOR trees even if cached
    PYTHONPATH=. uv run python RAPTOR-evaluation/run_eval.py \\
        --benchmark qasper \\
        --config RAPTOR-evaluation/configs/collapsed_tree.yaml \\
        --rebuild-trees

    # Quick smoke test with 5 papers
    PYTHONPATH=. uv run python RAPTOR-evaluation/run_eval.py \\
        --benchmark qasper \\
        --config RAPTOR-evaluation/configs/collapsed_tree.yaml \\
        --sample-size 5
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# Ensure backend is on sys.path for core.* imports
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Ensure RAPTOR-evaluation is on sys.path for benchmarks.* imports
eval_dir = str(Path(__file__).resolve().parent)
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "qasper": "benchmarks.qasper_benchmark.QasperBenchmark",
    "docvqa": "benchmarks.docvqa_benchmark.DocvqaBenchmark",
    "mpdocvqa": "benchmarks.mpdocvqa_benchmark.MpDocVqaBenchmark",
}


def _import_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_scores_from_csv(csv_path: str) -> dict[str, float] | None:
    """Extract average metric scores from a benchmark output CSV.

    The CSV written by BaseBenchmark._save_results has the format:
        index,prediction,ground_truths,<metric1>,<metric2>,...
        0,...
        AVERAGE,,,<val1>,<val2>,...

    Returns dict of metric→score, or None if file not found/unreadable.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers is None:
                return None
            metric_cols = headers[3:]  # first 3: index, prediction, ground_truths
            for row in reader:
                if row and row[0] == "AVERAGE":
                    scores = {}
                    for col, val in zip(metric_cols, row[3:]):
                        try:
                            scores[col] = float(val)
                        except (ValueError, IndexError):
                            pass
                    return scores if scores else None
        return None
    except (FileNotFoundError, OSError):
        return None


def _find_flat_baseline(output_path: str, config: dict) -> str | None:
    """Locate the flat baseline CSV for comparison, in priority order:
    1. flat_baseline_csv key in config
    2. <results_dir>/qasper_dev_flat.csv
    3. <results_dir>/qasper_flat_retrieval.csv
    4. Any *flat*.csv in the results dir
    """
    # Explicit config key
    if "flat_baseline_csv" in config:
        return config["flat_baseline_csv"]

    results_dir = Path(output_path).parent

    # Named candidates
    for candidate in (
        results_dir / "qasper_dev_flat.csv",
        results_dir / "qasper_flat_retrieval.csv",
    ):
        if candidate.exists():
            return str(candidate)

    # Glob fallback
    matches = sorted(results_dir.glob("*flat*.csv"))
    if matches:
        return str(matches[0])

    return None


def _find_collapsed_baseline(output_path: str, config: dict) -> str | None:
    """Locate the vanilla-collapsed (Row B) CSV for 3-row comparison."""
    if "collapsed_baseline_csv" in config:
        return config["collapsed_baseline_csv"]

    results_dir = Path(output_path).parent
    for candidate in (
        results_dir / "qasper_dev_collapsed.csv",
        results_dir / "qasper_collapsed.csv",
    ):
        if candidate.exists():
            return str(candidate)

    matches = sorted(results_dir.glob("*collapsed*.csv"))
    # Exclude the current output file itself
    matches = [str(p) for p in matches if str(p) != output_path]
    return matches[0] if matches else None


def _print_comparison_table(
    collapsed_scores: dict[str, float],
    flat_scores: dict[str, float] | None,
    flat_label: str = "Flat (no tree)",
    collapsed_label: str = "Collapsed (RAPTOR)",
) -> None:
    """Print a comparison table between flat and collapsed retrieval results."""
    metrics = list(collapsed_scores.keys())

    col_w = 22
    val_w = 8

    header_cells = ["Configuration".ljust(col_w)] + [
        m.upper().center(val_w) for m in metrics
    ]
    divider = "─" * (col_w + 2 + (val_w + 3) * len(metrics))

    print()
    print("QASPER Dev Set Results:")
    print("┌" + divider + "┐")
    print("│ " + " │ ".join(header_cells) + " │")
    print("├" + divider + "┤")

    if flat_scores:
        flat_vals = [
            f"{flat_scores.get(m, float('nan')):.4f}".center(val_w) for m in metrics
        ]
        print("│ " + flat_label.ljust(col_w) + " │ " + " │ ".join(flat_vals) + " │")

    collapsed_vals = [
        f"{collapsed_scores.get(m, float('nan')):.4f}".center(val_w) for m in metrics
    ]
    print(
        "│ " + collapsed_label.ljust(col_w) + " │ " + " │ ".join(collapsed_vals) + " │"
    )

    if flat_scores:
        print("├" + divider + "┤")
        delta_vals = []
        for m in metrics:
            diff = collapsed_scores.get(m, 0.0) - flat_scores.get(m, 0.0)
            sign = "+" if diff >= 0 else ""
            delta_vals.append(f"{sign}{diff * 100:.2f}%".center(val_w))
        print("│ " + "Delta".ljust(col_w) + " │ " + " │ ".join(delta_vals) + " │")

    print("└" + divider + "┘")
    print()


def _print_three_row_comparison(
    row_c_scores: dict[str, float],
    row_b_scores: dict[str, float] | None,
    row_a_scores: dict[str, float] | None,
) -> None:
    """Print the QASPER ablation table for all three rows (A, B, C) with deltas."""
    metrics = list(row_c_scores.keys())

    col_w = 47
    val_w = 8
    header_cells = ["Configuration".ljust(col_w)] + [
        m.upper().center(val_w) for m in metrics
    ]
    divider = "─" * (col_w + 2 + (val_w + 3) * len(metrics))

    def _fmt(scores: dict | None, m: str) -> str:
        if scores is None:
            return "  N/A  ".center(val_w)
        return f"{scores.get(m, float('nan')):.4f}".center(val_w)

    def _delta(a: dict | None, b: dict | None, m: str) -> str:
        if a is None or b is None:
            return "  N/A  ".center(val_w)
        diff = b.get(m, 0.0) - a.get(m, 0.0)
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff * 100:.2f}%".center(val_w)

    print()
    print("QASPER Dev Set — All Rows:")
    print("┌" + divider + "┐")
    print("│ " + " │ ".join(header_cells) + " │")
    print("├" + divider + "┤")

    rows = [
        ("Row A: Flat retrieval (no tree)", row_a_scores),
        ("Row B: Vanilla RAPTOR (collapsed)", row_b_scores),
        ("Row C: Full system (table P-C + expansion)", row_c_scores),
    ]
    for label, sc in rows:
        vals = [_fmt(sc, m) for m in metrics]
        print("│ " + label.ljust(col_w) + " │ " + " │ ".join(vals) + " │")

    print("├" + divider + "┤")
    delta_rows = [
        ("Delta (B-A): RAPTOR tree contribution", row_a_scores, row_b_scores),
        ("Delta (C-B): Table-aware contribution", row_b_scores, row_c_scores),
        ("Delta (C-A): Total improvement", row_a_scores, row_c_scores),
    ]
    for label, src, dst in delta_rows:
        vals = [_delta(src, dst, m) for m in metrics]
        print("│ " + label.ljust(col_w) + " │ " + " │ ".join(vals) + " │")

    print("└" + divider + "┘")
    print()


def _find_mpdocvqa_flat_baseline(output_path: str, config: dict) -> str | None:
    """Locate the MP-DocVQA flat baseline CSV for comparison."""
    if "flat_baseline_csv" in config:
        return config["flat_baseline_csv"]

    results_dir = Path(output_path).parent
    for candidate in (
        results_dir / "mpdocvqa_mpdocvqa_flat_textonly.csv",
        results_dir / "mpdocvqa_flat_textonly.csv",
        results_dir / "mpdocvqa_flat.csv",
    ):
        if candidate.exists():
            return str(candidate)

    matches = sorted(results_dir.glob("mpdocvqa*flat*.csv"))
    if matches:
        return str(matches[0])

    return None


def _print_mpdocvqa_comparison(
    current_scores: dict[str, float],
    flat_scores: dict[str, float] | None,
    current_config: dict,
) -> None:
    """Print a 2-row comparison table for MP-DocVQA (Row A vs Row C) with external baselines."""
    is_mode_c = current_config.get("doc_processing") == "multimodal_page"

    row_a_scores = flat_scores if is_mode_c else current_scores
    row_c_scores = current_scores if is_mode_c else None

    col_w = 46
    anls_w = 8
    page_w = 12
    divider = "─" * (col_w + 2 + anls_w + 3 + page_w + 2)

    def _fmt_anls(scores: dict | None) -> str:
        if scores is None:
            return "  N/A  ".center(anls_w)
        return f"{scores.get('anls', float('nan')):.4f}".center(anls_w)

    def _fmt_page(scores: dict | None) -> str:
        if scores is None:
            return "     N/A     ".center(page_w)
        v = scores.get("page_accuracy", float("nan"))
        return f"{v * 100:.1f}%".center(page_w)

    def _delta_anls(a: dict | None, c: dict | None) -> str:
        if a is None or c is None:
            return "  N/A  ".center(anls_w)
        diff = c.get("anls", 0.0) - a.get("anls", 0.0)
        s = "+" if diff >= 0 else ""
        return f"{s}{diff * 100:.2f}%".center(anls_w)

    def _delta_page(a: dict | None, c: dict | None) -> str:
        if a is None or c is None:
            return "     N/A     ".center(page_w)
        diff = c.get("page_accuracy", 0.0) - a.get("page_accuracy", 0.0)
        s = "+" if diff >= 0 else ""
        return f"{s}{diff * 100:.1f}%".center(page_w)

    print()
    print("MP-DocVQA Validation Results:")
    print("┌" + divider + "┐")
    header = "│ " + "Configuration".ljust(col_w) + " │ " + "ANLS".center(anls_w) + " │ " + "Page Acc.".center(page_w) + " │"
    print(header)
    print("├" + divider + "┤")

    row_a_label = "Row A: Flat, text-only OCR"
    row_c_label = "Row C: Full system (multimodal+RAPTOR+table)"

    print("│ " + row_a_label.ljust(col_w) + " │ " + _fmt_anls(row_a_scores) + " │ " + _fmt_page(row_a_scores) + " │")
    print("│ " + row_c_label.ljust(col_w) + " │ " + _fmt_anls(row_c_scores) + " │ " + _fmt_page(row_c_scores) + " │")

    print("├" + divider + "┤")
    print("│ " + "Delta (C-A): Total improvement".ljust(col_w) + " │ " + _delta_anls(row_a_scores, row_c_scores) + " │ " + _delta_page(row_a_scores, row_c_scores) + " │")
    print("└" + divider + "┘")

    print()
    print("External Comparison (López et al., 2025):")
    print("  RAG-VT5 base (text RAG, no reranker):  58.23% ANLS")
    print("  RAG-VT5 + reranker:                    61.06% ANLS")
    print("  RAG-Pix2Struct (visual RAG):            54.10% ANLS")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run RAPTOR evaluation benchmarks against published datasets."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=list(BENCHMARKS.keys()),
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., configs/collapsed_tree.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for output CSV (default: results/<benchmark>_<config>.csv)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for dataset cache (default: RAPTOR-evaluation/datasets/)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit to N papers/items (for quick testing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to use (default: validation for QASPER)",
    )
    parser.add_argument(
        "--rebuild-trees",
        action="store_true",
        default=False,
        help="Force rebuild RAPTOR trees even if disk cache exists",
    )
    parser.add_argument(
        "--shard",
        default=None,
        metavar="N/M",
        help="Process shard N of M (0-indexed). E.g. --shard 0/2 and --shard 1/2 split across two terminals.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    shard_id: int | None = None
    num_shards: int | None = None
    if args.shard:
        try:
            parts = args.shard.split("/")
            shard_id = int(parts[0])
            num_shards = int(parts[1])
            if not (0 <= shard_id < num_shards):
                raise ValueError(f"shard_id {shard_id} out of range for num_shards {num_shards}")
        except (ValueError, IndexError) as exc:
            logger.error(f"Invalid --shard value '{args.shard}'. Use format N/M (0-indexed), e.g. 0/2. ({exc})")
            sys.exit(1)

    if args.output is None:
        config_stem = Path(args.config).stem
        results_dir = Path(eval_dir) / "results"
        results_dir.mkdir(exist_ok=True)
        args.output = str(results_dir / f"{args.benchmark}_{config_stem}.csv")

    # Auto-suffix output path per shard so each shard writes its own file + checkpoint
    if shard_id is not None and num_shards is not None:
        p = Path(args.output)
        args.output = str(p.with_stem(f"{p.stem}_shard{shard_id}of{num_shards}"))

    if args.data_dir is None:
        args.data_dir = str(Path(eval_dir) / "datasets")

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Merge CLI overrides into config
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size
    if args.top_k is not None:
        config["top_k"] = args.top_k
    if args.split is not None:
        config["split"] = args.split
    if args.rebuild_trees:
        config["rebuild_trees"] = True
    if shard_id is not None:
        config["shard_id"] = shard_id
        config["num_shards"] = num_shards

    retrieval_mode = config.get("retrieval_mode", "collapsed")
    os.environ["RETRIEVAL_MODE"] = retrieval_mode
    logger.info(f"Set RETRIEVAL_MODE={retrieval_mode}")

    build_raptor = config.get("build_raptor_tree", False)
    if build_raptor:
        logger.info("RAPTOR tree construction: ENABLED")
    if args.rebuild_trees:
        logger.info("--rebuild-trees: will force-rebuild all tree caches")
    if shard_id is not None:
        logger.info(f"Shard: {shard_id + 1}/{num_shards} → output: {args.output}")

    # Write merged config to a temp file so the benchmark sees CLI overrides
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="raptor_eval_"
    ) as tmp:
        yaml.dump(config, tmp)
        merged_config_path = tmp.name

    try:
        benchmark_class = _import_class(BENCHMARKS[args.benchmark])
        benchmark = benchmark_class(merged_config_path)

        logger.info(f"Benchmark: {args.benchmark}")
        logger.info(f"Config: {benchmark.description}")
        logger.info(f"Retrieval mode: {benchmark.retrieval_mode}")
        if config.get("sample_size"):
            logger.info(f"Sample size: {config['sample_size']}")
        if config.get("top_k"):
            logger.info(f"Top-K: {config['top_k']}")
        logger.info(f"Output: {args.output}")

        benchmark.load_dataset(data_dir=args.data_dir)
        scores = benchmark.run_all(output_path=args.output)

        print("\n" + "=" * 50)
        print(f"Results for {args.benchmark} ({benchmark.retrieval_mode}):")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
        print("=" * 50)

        # Comparison tables
        if args.benchmark == "mpdocvqa":
            # MP-DocVQA 2-row comparison (Row A vs Row C) with external baselines
            flat_csv = _find_mpdocvqa_flat_baseline(args.output, config)
            flat_scores = _load_scores_from_csv(flat_csv) if flat_csv else None
            if flat_csv and flat_scores:
                logger.info(f"MP-DocVQA Row A baseline: {flat_csv}")
            elif config.get("doc_processing") == "multimodal_page":
                logger.info(
                    "No flat baseline CSV found. Run mpdocvqa_flat_textonly config first "
                    "or set flat_baseline_csv in the config to enable comparison."
                )
            _print_mpdocvqa_comparison(scores, flat_scores, config)

        elif build_raptor and retrieval_mode == "collapsed":
            is_row_c = config.get("use_table_parent_child") or config.get(
                "use_retrieval_expansion"
            )

            flat_csv = _find_flat_baseline(args.output, config)
            flat_scores = _load_scores_from_csv(flat_csv) if flat_csv else None

            if is_row_c:
                # 3-row comparison: Row A (flat), Row B (vanilla collapsed), Row C (current)
                collapsed_csv = _find_collapsed_baseline(args.output, config)
                collapsed_scores_b = (
                    _load_scores_from_csv(collapsed_csv) if collapsed_csv else None
                )
                if collapsed_csv and collapsed_scores_b:
                    logger.info(f"Row B baseline: {collapsed_csv}")
                else:
                    logger.info(
                        "No vanilla-collapsed CSV found for Row B. Save Row B results as "
                        "'qasper_dev_collapsed.csv' or set 'collapsed_baseline_csv' in config."
                    )
                if flat_csv and flat_scores:
                    logger.info(f"Row A baseline: {flat_csv}")

                _print_three_row_comparison(scores, collapsed_scores_b, flat_scores)

                # Table detection impact summary
                if hasattr(benchmark, "print_table_detection_impact"):
                    benchmark.print_table_detection_impact()
            else:
                # Original 2-row comparison (Row B vs Row A)
                if flat_scores:
                    logger.info(f"Comparing against flat baseline: {flat_csv}")
                else:
                    logger.info(
                        "No flat baseline CSV found. To enable comparison, save the flat "
                        "results as 'qasper_dev_flat.csv' in the same results directory, "
                        "or set flat_baseline_csv in the config."
                    )
                _print_comparison_table(scores, flat_scores)

    finally:
        os.unlink(merged_config_path)


if __name__ == "__main__":
    main()
