#!/usr/bin/env python3
"""QASPER table-density ΔF1 analysis.

Bins 281 QASPER dev papers by table count, recomputes per-question F1 from
stored prediction CSVs (Row A = flat, Row C = full system), and reports
per-bin mean F1 and ΔF1 with statistical significance.

Outputs:
  - Console summary table
  - results/qasper_table_density_analysis.tex  (booktabs LaTeX)
  - results/qasper_per_paper_table_density.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup — import from existing codebase
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.qasper_benchmark import (  # noqa: E402
    _extract_ground_truths,
    _extract_paper_text,
    _load_qasper_json,
    detect_text_tables,
)
from metrics import f1_token_score, score_with_multiple_gts  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = str(_PROJECT_ROOT / "datasets")
ROW_A_CSV = _PROJECT_ROOT / "results" / "qasper_dev_flat.csv"
ROW_C_CSV = _PROJECT_ROOT / "results" / "qasper_dev_collapsed_multimodal.csv"
OUTPUT_TEX = _PROJECT_ROOT / "results" / "qasper_table_density_analysis.tex"
OUTPUT_CSV = _PROJECT_ROOT / "results" / "qasper_per_paper_table_density.csv"

EXPECTED_F1 = {"row_a": 0.4489, "row_c": 0.4491}
TOLERANCE = 0.001

BIN_EDGES = [(0, 0, "0"), (1, 2, "1--2"), (3, float("inf"), "3+")]


def load_csv_predictions(csv_path: Path) -> list[tuple[str, list[str]]]:
    """Parse a results CSV, returning (prediction, ground_truths) per question."""
    rows: list[tuple[str, list[str]]] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # noqa: F841
        for row in reader:
            idx = row[0]
            if idx == "AVERAGE":
                break
            prediction = row[1]
            gts = row[2].split("|") if row[2] else []
            rows.append((prediction, gts))
    return rows


def compute_per_question_f1(
    predictions: list[tuple[str, list[str]]],
) -> list[float]:
    """Recompute F1 for each (prediction, ground_truths) pair."""
    return [
        score_with_multiple_gts(pred, gts, f1_token_score)
        for pred, gts in predictions
    ]


def verify_aggregate(scores: list[float], expected: float, label: str) -> None:
    """Abort if recomputed mean F1 diverges from stored aggregate."""
    computed = np.mean(scores)
    if abs(computed - expected) > TOLERANCE:
        print(
            f"VERIFICATION FAILED for {label}: "
            f"computed={computed:.6f}, expected={expected:.4f}, "
            f"diff={abs(computed - expected):.6f} > tolerance={TOLERANCE}"
        )
        sys.exit(1)
    print(f"  {label}: computed={computed:.6f}, expected={expected:.4f} ✓")


def assign_bin(table_count: int) -> str:
    """Map a table count to its bin label."""
    for lo, hi, label in BIN_EDGES:
        if lo <= table_count <= hi:
            return label
    return BIN_EDGES[-1][2]


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1-2: Load QASPER dev JSON
    # ------------------------------------------------------------------
    print("Loading QASPER dev split...")
    papers_dict = _load_qasper_json(DATA_DIR, "dev")
    print(f"  {len(papers_dict)} papers in JSON")

    # ------------------------------------------------------------------
    # Step 3-4: Count tables per paper + build question-index → paper_id
    # ------------------------------------------------------------------
    print("Counting tables and building question index...")
    paper_table_counts: dict[str, int] = {}
    question_to_paper: dict[int, str] = {}
    q_idx = 0

    for paper_id in papers_dict:
        paper = papers_dict[paper_id]
        text = _extract_paper_text(paper)

        if not text.strip():
            continue

        tables = detect_text_tables(text)
        paper_table_counts[paper_id] = len(tables)

        for qa in paper.get("qas", []):
            question = qa.get("question", "")
            if not question:
                continue
            gts = _extract_ground_truths(qa)
            if not gts:
                continue
            question_to_paper[q_idx] = paper_id
            q_idx += 1

    n_papers = len(paper_table_counts)
    n_questions = len(question_to_paper)
    print(f"  {n_papers} papers with text, {n_questions} questions mapped")

    # ------------------------------------------------------------------
    # Step 5: Parse CSVs
    # ------------------------------------------------------------------
    print("Parsing prediction CSVs...")
    row_a_preds = load_csv_predictions(ROW_A_CSV)
    row_c_preds = load_csv_predictions(ROW_C_CSV)
    print(f"  Row A: {len(row_a_preds)} questions")
    print(f"  Row C: {len(row_c_preds)} questions")

    if len(row_a_preds) != n_questions or len(row_c_preds) != n_questions:
        print(
            f"ERROR: question count mismatch — "
            f"mapping={n_questions}, Row A CSV={len(row_a_preds)}, "
            f"Row C CSV={len(row_c_preds)}"
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Recompute per-question F1 and verify
    # ------------------------------------------------------------------
    print("Recomputing per-question F1...")
    f1_a = compute_per_question_f1(row_a_preds)
    f1_c = compute_per_question_f1(row_c_preds)

    print("Verifying aggregates:")
    verify_aggregate(f1_a, EXPECTED_F1["row_a"], "Row A")
    verify_aggregate(f1_c, EXPECTED_F1["row_c"], "Row C")

    # ------------------------------------------------------------------
    # Step 7: Aggregate per paper
    # ------------------------------------------------------------------
    paper_f1_a: dict[str, list[float]] = defaultdict(list)
    paper_f1_c: dict[str, list[float]] = defaultdict(list)

    for qi in range(n_questions):
        pid = question_to_paper[qi]
        paper_f1_a[pid].append(f1_a[qi])
        paper_f1_c[pid].append(f1_c[qi])

    # Per-paper means
    per_paper_data: list[dict] = []
    for pid in paper_table_counts:
        if pid not in paper_f1_a:
            # Paper had text but all questions were skipped (no valid GT)
            continue
        mean_a = np.mean(paper_f1_a[pid])
        mean_c = np.mean(paper_f1_c[pid])
        per_paper_data.append({
            "paper_id": pid,
            "table_count": paper_table_counts[pid],
            "n_questions": len(paper_f1_a[pid]),
            "f1_row_a": mean_a,
            "f1_row_c": mean_c,
            "delta_f1": mean_c - mean_a,
        })

    # ------------------------------------------------------------------
    # Step 8: Bin by table count
    # ------------------------------------------------------------------
    bins: dict[str, list[dict]] = defaultdict(list)
    for ppd in per_paper_data:
        bl = assign_bin(ppd["table_count"])
        bins[bl].append(ppd)

    bin_order = [label for _, _, label in BIN_EDGES]
    bin_results: list[dict] = []

    for label in bin_order:
        papers_in_bin = bins.get(label, [])
        n = len(papers_in_bin)
        if n == 0:
            bin_results.append({
                "bin": label,
                "n_papers": 0,
                "n_questions": 0,
                "mean_f1_a": float("nan"),
                "mean_f1_c": float("nan"),
                "mean_delta": float("nan"),
                "std_delta": float("nan"),
                "p_value": float("nan"),
                "note": "",
            })
            continue

        nq = sum(p["n_questions"] for p in papers_in_bin)
        f1_as = np.array([p["f1_row_a"] for p in papers_in_bin])
        f1_cs = np.array([p["f1_row_c"] for p in papers_in_bin])
        deltas = f1_cs - f1_as

        # Statistical test: Wilcoxon signed-rank (paired, non-parametric)
        if n >= 10:
            # Filter out zero differences for Wilcoxon
            nonzero_diffs = deltas[deltas != 0]
            if len(nonzero_diffs) >= 5:
                _, p_val = stats.wilcoxon(f1_as, f1_cs)
            else:
                _, p_val = stats.ttest_rel(f1_as, f1_cs)
        else:
            p_val = float("nan")

        note = ""
        if n < 10:
            note = "n<10"

        bin_results.append({
            "bin": label,
            "n_papers": n,
            "n_questions": nq,
            "mean_f1_a": float(np.mean(f1_as)),
            "mean_f1_c": float(np.mean(f1_cs)),
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas, ddof=1)) if n > 1 else 0.0,
            "p_value": float(p_val),
            "note": note,
        })

    # ------------------------------------------------------------------
    # Step 9: Table count distribution
    # ------------------------------------------------------------------
    tc_hist: dict[int, int] = defaultdict(int)
    for ppd in per_paper_data:
        tc = ppd["table_count"]
        tc_hist[min(tc, 5)] += 1  # bucket 5+

    # ------------------------------------------------------------------
    # Output: Console table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("QASPER Table-Density Analysis — Per-Bin Results")
    print("=" * 90)
    header = (
        f"{'Bin':>8}  {'Papers':>6}  {'Questions':>9}  "
        f"{'F1(A)':>7}  {'F1(C)':>7}  {'ΔF1':>7}  "
        f"{'σ(ΔF1)':>7}  {'p-val':>8}  {'Note':>6}"
    )
    print(header)
    print("-" * 90)
    for br in bin_results:
        p_str = (
            f"{br['p_value']:.4f}" if not np.isnan(br["p_value"]) else "   n/a"
        )
        print(
            f"{br['bin']:>8}  {br['n_papers']:>6}  {br['n_questions']:>9}  "
            f"{br['mean_f1_a']:>7.4f}  {br['mean_f1_c']:>7.4f}  "
            f"{br['mean_delta']:>+7.4f}  {br['std_delta']:>7.4f}  "
            f"{p_str:>8}  {br['note']:>6}"
        )

    # Totals
    total_papers = sum(br["n_papers"] for br in bin_results)
    total_q = sum(br["n_questions"] for br in bin_results)
    print("-" * 90)
    print(f"{'Total':>8}  {total_papers:>6}  {total_q:>9}")

    # Distribution
    print(f"\nTable Count Distribution (n={total_papers} papers):")
    for tc in sorted(tc_hist):
        label = f"{tc}" if tc < 5 else "5+"
        bar = "█" * tc_hist[tc]
        print(f"  {label:>3} tables: {tc_hist[tc]:>4} papers  {bar}")

    # ------------------------------------------------------------------
    # Output: Per-paper CSV
    # ------------------------------------------------------------------
    per_paper_data.sort(key=lambda x: x["paper_id"])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "table_count", "n_questions",
            "f1_row_a", "f1_row_c", "delta_f1",
        ])
        for ppd in per_paper_data:
            writer.writerow([
                ppd["paper_id"],
                ppd["table_count"],
                ppd["n_questions"],
                f"{ppd['f1_row_a']:.6f}",
                f"{ppd['f1_row_c']:.6f}",
                f"{ppd['delta_f1']:.6f}",
            ])
    print(f"\nPer-paper CSV saved to: {OUTPUT_CSV}")

    # ------------------------------------------------------------------
    # Output: LaTeX table (booktabs)
    # ------------------------------------------------------------------
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{QASPER performance by table density. "
        r"$\Delta$F1 = Row~C $-$ Row~A (full system vs.\ flat retrieval).}",
        r"  \label{tab:table-density}",
        r"  \begin{tabular}{l r r c c c c}",
        r"    \toprule",
        r"    Tables & Papers & Questions & F1 (A) & F1 (C) "
        r"& $\Delta$F1 & $p$ \\",
        r"    \midrule",
    ]
    for br in bin_results:
        p_str = (
            f"{br['p_value']:.3f}" if not np.isnan(br["p_value"]) else "---"
        )
        # Add dagger for insufficient n
        note = r"$^\dagger$" if br["note"] == "n<10" else ""
        lines.append(
            f"    {br['bin']}{note} & {br['n_papers']} & {br['n_questions']} "
            f"& {br['mean_f1_a']:.3f} & {br['mean_f1_c']:.3f} "
            f"& {br['mean_delta']:+.3f} & {p_str} \\\\"
        )
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
    ])
    # Add footnote if any bin has n<10
    if any(br["note"] == "n<10" for br in bin_results):
        lines.append(
            r"  \vspace{2pt}",
        )
        lines.append(
            r"  {\footnotesize $^\dagger$Fewer than 10 papers; "
            r"insufficient for reliable inference.}",
        )
    lines.extend([
        r"\end{table}",
    ])

    with open(OUTPUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to:   {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
