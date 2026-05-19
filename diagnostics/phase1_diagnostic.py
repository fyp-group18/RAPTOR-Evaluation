"""Phase 1: Diagnose tree inventory, page metadata, and anchoring status.

Run:
    GCS_BUCKET=raptor-assets python diagnostics/phase1_diagnostic.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.tree_resolver import resolve_tree
from ingestion.gcs_cache import list_run_ids_for_ablation

BUCKET = os.environ.get("GCS_BUCKET", "raptor-assets")

CACHE_KEYS = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

ABLATION_LABELS = [
    "full_context_aware",
    "no_table_parent_child",
    "no_header_propagation",
    "no_caption_folding",
    "baseline_naive_chunking",
    "semantic_chunking_baseline",
    "flat_no_raptor",
    "original_raptor_text_only",
]

ABLATION_KEY_TO_LABEL = {
    "full": "full_context_aware",
    "no_table_pc": "no_table_parent_child",
    "no_header_prop": "no_header_propagation",
    "no_caption_fold": "no_caption_folding",
    "no_context_aware": "baseline_naive_chunking",
    "semantic_chunking": "semantic_chunking_baseline",
    "flat_retrieval": "flat_no_raptor",
    "text_only_raptor": "original_raptor_text_only",
}

LABEL_TO_KEY = {v: k for k, v in ABLATION_KEY_TO_LABEL.items()}


# ───────────────────────────────────────────────────────────────────
# Phase 1.1: Inventory
# ───────────────────────────────────────────────────────────────────

def phase_1_1():
    print("\n" + "=" * 90)
    print("PHASE 1.1: GCS Tree Inventory")
    print("=" * 90)

    rows = []
    missing = []
    resolved_trees: dict[tuple[str, str], tuple[bytes, dict]] = {}

    for doc, cache_key in CACHE_KEYS.items():
        for label in ABLATION_LABELS:
            try:
                tree_bytes, manifest = resolve_tree(
                    cache_key=cache_key,
                    ablation_label=label,
                    bucket=BUCKET,
                )
                run_id = manifest.get("run_id", "?")
                num_leaf = manifest.get("num_leaf_nodes", "?")
                num_summary = manifest.get("num_summary_nodes", "?")
                created = manifest.get("created_at", "?")
                if isinstance(created, str) and len(created) > 19:
                    created = created[:19]

                total_nodes = "?"
                try:
                    nodes = pickle.loads(tree_bytes)
                    total_nodes = len(nodes)
                except Exception:
                    pass

                rows.append((doc, label, run_id, total_nodes, num_leaf, num_summary, created))
                resolved_trees[(doc, label)] = (tree_bytes, manifest)
            except FileNotFoundError:
                missing.append((doc, label))
                rows.append((doc, label, "MISSING", "-", "-", "-", "-"))

    # Print table
    print(f"\n{'Document':<16s} {'Ablation':<30s} {'Run ID':<28s} {'Nodes':>6s} {'Leaf':>6s} {'Summ':>6s} {'Created':<20s}")
    print("─" * 120)
    for doc, label, run_id, total, leaf, summ, created in rows:
        print(f"{doc:<16s} {label:<30s} {str(run_id):<28s} {str(total):>6s} {str(leaf):>6s} {str(summ):>6s} {str(created):<20s}")

    if missing:
        print(f"\n*** STOP: {len(missing)} trees MISSING from GCS ***")
        for doc, label in missing:
            print(f"  - {doc} / {label}")
        return None

    print(f"\nAll {len(rows)} trees found on GCS.")
    return resolved_trees


# ───────────────────────────────────────────────────────────────────
# Phase 1.2: Page metadata check
# ───────────────────────────────────────────────────────────────────

def phase_1_2(resolved_trees):
    print("\n" + "=" * 90)
    print("PHASE 1.2: Page Metadata in Every Tree")
    print("=" * 90)

    broken = []
    print(f"\n{'Document':<16s} {'Ablation':<26s} {'Total':>6s} {'Pages':>7s} {'PgStrt':>7s} {'Embed':>7s} {'ParID':>7s} {'IsPar':>6s}")
    print("─" * 90)

    for doc, cache_key in CACHE_KEYS.items():
        for label in ABLATION_LABELS:
            tree_bytes, manifest = resolved_trees[(doc, label)]
            nodes = pickle.loads(tree_bytes)

            total = len(nodes)
            has_pages = sum(1 for n in nodes if n.get("pages") and n["pages"] != [0] and n["pages"] != [])
            has_page_start = sum(1 for n in nodes if n.get("page_start") and n["page_start"] > 0)
            has_embedding = sum(1 for n in nodes if n.get("embedding") is not None)
            has_parent_id = sum(1 for n in nodes if n.get("parent_id"))
            is_parent_count = sum(1 for n in nodes if n.get("is_parent"))

            print(f"{doc:<16s} {label:<26s} {total:>6d} {has_pages:>7d} {has_page_start:>7d} {has_embedding:>7d} {has_parent_id:>7d} {is_parent_count:>6d}")

            if has_pages == 0 or has_embedding == 0:
                broken.append((doc, label, has_pages, has_embedding))

    if broken:
        print(f"\n*** STOP: {len(broken)} trees have 0 pages or 0 embeddings ***")
        for doc, label, pg, emb in broken:
            print(f"  - {doc}/{label}: pages={pg}, embeddings={emb}")
        return False

    print("\nAll trees have non-zero pages and embeddings.")
    return True


# ───────────────────────────────────────────────────────────────────
# Phase 1.3: Anchoring diagnostic
# ───────────────────────────────────────────────────────────────────

def phase_1_3(resolved_trees):
    print("\n" + "=" * 90)
    print("PHASE 1.3: Anchoring Diagnostic (before re-anchoring)")
    print("=" * 90)

    datasets_dir = ROOT / "datasets"

    print(f"\n{'Document':<16s} {'Ablation':<26s} {'Total':>6s} {'Exact':>7s} {'PgFall':>7s} {'Unmatched':>10s}")
    print("─" * 82)

    for doc, cache_key in CACHE_KEYS.items():
        for label in ABLATION_LABELS:
            ablation_key = LABEL_TO_KEY.get(label)
            if not ablation_key:
                continue

            # Load per-ablation anchored dataset if it exists
            anchored_file = datasets_dir / f"combined_eval_dataset_anchored_{ablation_key}.json"
            if not anchored_file.exists():
                print(f"{doc:<16s} {label:<26s} {'NO FILE':>6s}")
                continue

            queries = json.loads(anchored_file.read_text())
            # Filter to this document
            doc_queries = [q for q in queries if q.get("manual_source") == doc]
            if not doc_queries:
                print(f"{doc:<16s} {label:<26s} {'0':>6s} {'0':>7s} {'0':>7s} {'0':>10s}")
                continue

            # Load tree nodes
            tree_bytes, _ = resolved_trees[(doc, label)]
            nodes = pickle.loads(tree_bytes)
            node_ids = {str(n.get("id", "")) for n in nodes}

            # Build page lookup
            page_to_nodes = defaultdict(set)
            for n in nodes:
                pages = n.get("pages") or []
                if isinstance(pages, list):
                    for p in pages:
                        if isinstance(p, int) and p > 0:
                            page_to_nodes[p].add(str(n.get("id", "")))
                ps = n.get("page_start")
                pe = n.get("page_end")
                if ps and pe and ps > 0:
                    for p in range(ps, pe + 1):
                        page_to_nodes[p].add(str(n.get("id", "")))

            exact = 0
            page_fallback = 0
            unmatched = 0

            for q in doc_queries:
                gt_id = q.get("ground_truth_chunk_id")
                if gt_id and str(gt_id) in node_ids:
                    exact += 1
                    continue

                # Try page overlap
                page_ref = q.get("page_reference", "")
                target_pages = []
                if page_ref and page_ref.strip().isdigit():
                    target_pages = [int(page_ref.strip())]

                if target_pages:
                    found = False
                    for p in target_pages:
                        if page_to_nodes.get(p):
                            page_fallback += 1
                            found = True
                            break
                    if found:
                        continue

                unmatched += 1

            total = len(doc_queries)
            print(f"{doc:<16s} {label:<26s} {total:>6d} {exact:>7d} {page_fallback:>7d} {unmatched:>10d}")

    print()


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    resolved = phase_1_1()
    if resolved is None:
        print("Aborting: missing trees on GCS.")
        sys.exit(1)

    ok = phase_1_2(resolved)
    if not ok:
        print("Aborting: broken trees detected.")
        sys.exit(1)

    phase_1_3(resolved)
    print("Phase 1 diagnostic complete.")
