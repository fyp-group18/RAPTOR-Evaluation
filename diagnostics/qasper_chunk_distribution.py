"""
Diagnostic: QASPER chunk count distribution.

Counts chunks per paper (using the same splitter as QasperBenchmark) to determine
whether RAPTOR tree construction is viable for the QASPER dev split.

Run from backend/:
    PYTHONPATH=. uv run python RAPTOR-evaluation/diagnostics/qasper_chunk_distribution.py
"""

import statistics
import sys
from pathlib import Path

# Mirror the sys.path bootstrap from run_eval.py so benchmarks.* imports resolve.
_eval_dir = str(Path(__file__).resolve().parent.parent)  # backend/RAPTOR-evaluation/
_backend_dir = str(Path(__file__).resolve().parent.parent.parent)  # backend/
for _d in (_backend_dir, _eval_dir):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from benchmarks.qasper_benchmark import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _extract_paper_text,
    _load_qasper_json,
)

# Points to backend/RAPTOR-evaluation/datasets/ where qasper_raw/ already lives.
_DATA_DIR = str(Path(__file__).resolve().parent.parent / "datasets")
_SPLIT = "dev"


def main() -> None:
    print(f"Loading QASPER ({_SPLIT} split) from {_DATA_DIR} ...")
    papers_dict = _load_qasper_json(_DATA_DIR, _SPLIT)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunk_counts: list[int] = []
    skipped_no_text = 0

    for paper in papers_dict.values():
        full_text = _extract_paper_text(paper)
        if not full_text.strip():
            skipped_no_text += 1
            continue
        chunks = splitter.split_text(full_text)
        chunk_counts.append(len(chunks))

    if not chunk_counts:
        print("No papers with text found — cannot compute distribution.")
        return

    total_papers = len(chunk_counts)
    total_chunks = sum(chunk_counts)
    mean_val = statistics.mean(chunk_counts)
    median_val = statistics.median(chunk_counts)
    min_val = min(chunk_counts)
    max_val = max(chunk_counts)
    stdev_val = statistics.stdev(chunk_counts) if total_papers > 1 else 0.0

    one = sum(1 for c in chunk_counts if c == 1)
    two_four = sum(1 for c in chunk_counts if 2 <= c <= 4)
    five_nine = sum(1 for c in chunk_counts if 5 <= c <= 9)
    ten_nineteen = sum(1 for c in chunk_counts if 10 <= c <= 19)
    twenty_plus = sum(1 for c in chunk_counts if c >= 20)

    def pct(n: int) -> str:
        return f"{n / total_papers * 100:.1f}%"

    # Format median without trailing .0 for whole numbers.
    median_str = str(int(median_val)) if median_val == int(median_val) else f"{median_val:.1f}"

    print(f"\nTotal papers: {total_papers}")
    if skipped_no_text:
        print(f"Skipped (no text): {skipped_no_text}")
    print(f"Total chunks: {total_chunks}")
    print(f"\nChunks per paper:")
    print(f"  Mean:   {mean_val:.1f}")
    print(f"  Median: {median_str}")
    print(f"  Min:    {min_val}")
    print(f"  Max:    {max_val}")
    print(f"  Std:    {stdev_val:.1f}")
    print(f"\nDistribution:")
    print(f"  1 chunk:      {one} papers ({pct(one)})")
    print(f"  2-4 chunks:   {two_four} papers ({pct(two_four)})")
    print(f"  5-9 chunks:   {five_nine} papers ({pct(five_nine)})")
    print(f"  10-19 chunks: {ten_nineteen} papers ({pct(ten_nineteen)})")
    print(f"  20+ chunks:   {twenty_plus} papers ({pct(twenty_plus)})")

    print(f"\nDecision: ", end="")
    if median_val >= 10:
        print("PROCEED with RAPTOR tree build — papers are long enough for meaningful hierarchies")
    elif median_val >= 5:
        print("BORDERLINE — RAPTOR may produce shallow trees (1 level). Proceed with caution.")
    else:
        print("SKIP — pivot to DocVQA. Papers too short for RAPTOR clustering to produce meaningful hierarchies.")

    eligible_chunks = sum(c for c in chunk_counts if c >= 5)
    estimated_calls = eligible_chunks // 6
    estimated_minutes = estimated_calls / (0.4 * 60)
    print(f"\nEstimated Gemini summarization calls for tree build: ~{estimated_calls}")
    print(f"Estimated time at 0.4 q/s: ~{estimated_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
