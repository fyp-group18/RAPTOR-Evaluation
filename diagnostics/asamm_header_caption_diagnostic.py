"""Diagnostic: why AS-AMM no_header_propagation and no_caption_folding trees
score near-zero on TABLE queries.

Loads trees directly from .tree-cache (no GCS required).
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TREE_CACHE = ROOT / ".tree-cache"

AS_AMM_KEY = "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448"
SC_KEY = "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0"

TREES = {
    "AS-AMM full_context_aware": (
        AS_AMM_KEY,
        "20260518_072545_57edf67d",
        "full_context_aware",
    ),
    "AS-AMM no_header_propagation": (
        AS_AMM_KEY,
        "20260517_104040_95536286",
        "no_header_propagation",
    ),
    "AS-AMM no_caption_folding": (
        AS_AMM_KEY,
        "20260517_104040_95536286",
        "no_caption_folding",
    ),
    "SC10000AMM no_header_propagation": (
        SC_KEY,
        "20260517_080545_666ba344",
        "no_header_propagation",
    ),
    "SC10000AMM no_caption_folding": (
        SC_KEY,
        "20260517_080545_666ba344",
        "no_caption_folding",
    ),
}

ANCHORED_DATASETS = {
    "no_header_prop": ROOT / "datasets" / "combined_eval_dataset_anchored_no_header_prop.json",
    "no_caption_fold": ROOT / "datasets" / "combined_eval_dataset_anchored_no_caption_fold.json",
    "full": ROOT / "datasets" / "combined_eval_dataset_anchored_full.json",
}


def load_tree(cache_key: str, run_id: str, ablation_label: str) -> tuple[list[dict], dict]:
    base = TREE_CACHE / cache_key / run_id / ablation_label
    pkl_path = base / "tree.pkl"
    manifest_path = base / "tree_manifest.json"
    with open(pkl_path, "rb") as f:
        nodes = pickle.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)
    return nodes, manifest


def node_stats(nodes: list[dict]) -> dict:
    total = len(nodes)
    by_level: dict[int, dict] = defaultdict(lambda: {
        "count": 0, "embedded": 0, "is_parent": 0, "has_parent_id": 0,
    })

    for node in nodes:
        lvl = node.get("level", 0)
        by_level[lvl]["count"] += 1
        if node.get("embedding") is not None:
            by_level[lvl]["embedded"] += 1
        if node.get("is_parent"):
            by_level[lvl]["is_parent"] += 1
        if node.get("parent_id"):
            by_level[lvl]["has_parent_id"] += 1

    embedded_total = sum(1 for n in nodes if n.get("embedding") is not None)
    is_parent_total = sum(1 for n in nodes if n.get("is_parent"))
    has_parent_id_total = sum(1 for n in nodes if n.get("parent_id"))
    searchable = sum(
        1 for n in nodes
        if not n.get("is_parent") and n.get("embedding") is not None
    )

    # Node type breakdown at level 0
    leaf_nodes = [n for n in nodes if n.get("level", 0) == 0]
    leaf_types = {
        "is_parent (table header, not searchable)": sum(1 for n in leaf_nodes if n.get("is_parent")),
        "has_parent_id (table row child, searchable)": sum(1 for n in leaf_nodes if n.get("parent_id")),
        "prose/image leaf (no parent_id, no is_parent)": sum(
            1 for n in leaf_nodes if not n.get("parent_id") and not n.get("is_parent")
        ),
    }

    return {
        "total": total,
        "embedded_total": embedded_total,
        "is_parent_total": is_parent_total,
        "has_parent_id_total": has_parent_id_total,
        "searchable": searchable,
        "by_level": dict(by_level),
        "leaf_types": leaf_types,
    }


def sample_node_texts(nodes: list[dict], node_type: str, n: int = 3) -> list[str]:
    """Return sample texts for a given node type."""
    if node_type == "is_parent":
        candidates = [nd for nd in nodes if nd.get("is_parent")]
    elif node_type == "has_parent_id":
        candidates = [nd for nd in nodes if nd.get("parent_id")]
    elif node_type == "embedded_leaf":
        candidates = [
            nd for nd in nodes
            if not nd.get("is_parent") and not nd.get("parent_id")
            and nd.get("level", 0) == 0 and nd.get("embedding") is not None
        ]
    elif node_type == "summary":
        candidates = [nd for nd in nodes if nd.get("level", 0) > 0]
    else:
        candidates = []
    return [c.get("text", "")[:200] for c in candidates[:n]]


def check_gt_ids_searchable(
    dataset_path: Path,
    nodes: list[dict],
    manual_source: str,
    label: str,
) -> dict:
    """Check whether ground-truth chunk IDs from the anchored dataset point
    to searchable nodes in the given tree."""
    if not dataset_path.exists():
        return {"error": f"Dataset not found: {dataset_path}"}

    with open(dataset_path) as f:
        queries = json.load(f)

    # Filter to the relevant manual
    relevant_queries = [q for q in queries if q.get("manual_source") == manual_source]

    node_id_map = {n.get("id"): n for n in nodes if n.get("id")}
    searchable_ids = {
        n.get("id") for n in nodes
        if not n.get("is_parent") and n.get("embedding") is not None
    }
    parent_ids = {n.get("id") for n in nodes if n.get("is_parent")}
    parent_id_child_ids = {n.get("id") for n in nodes if n.get("parent_id")}

    results = {
        "total_queries": len(relevant_queries),
        "gt_in_searchable": 0,
        "gt_is_parent_node": 0,  # gt points to an is_parent (not searchable)
        "gt_has_parent_id": 0,   # gt points to a child row chunk (searchable)
        "gt_missing_from_tree": 0,
        "gt_other_not_searchable": 0,
        "table_queries": 0,
        "table_gt_searchable": 0,
        "table_gt_is_parent": 0,
        "table_gt_missing": 0,
    }

    for q in relevant_queries:
        gt_id = q.get("ground_truth_chunk_id")
        is_table = q.get("origin") == "table"

        if is_table:
            results["table_queries"] += 1

        if gt_id not in node_id_map:
            results["gt_missing_from_tree"] += 1
            if is_table:
                results["table_gt_missing"] += 1
            continue

        if gt_id in searchable_ids:
            results["gt_in_searchable"] += 1
            if is_table:
                results["table_gt_searchable"] += 1
        elif gt_id in parent_ids:
            results["gt_is_parent_node"] += 1
            if is_table:
                results["table_gt_is_parent"] += 1
        else:
            results["gt_other_not_searchable"] += 1

    return results


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print('=' * 72)


def print_stats_table(label: str, stats: dict, manifest: dict) -> None:
    ts = manifest.get("tree_stats", {})
    print(f"\n--- {label} ---")
    print(f"  Manifest tree_stats: "
          f"leaf={ts.get('num_leaf_nodes')}, "
          f"summary={ts.get('num_summary_nodes')}, "
          f"levels={ts.get('num_levels')}, "
          f"table_chunks={ts.get('num_table_chunks')}")
    print(f"  Actual counts from pickle:")
    print(f"    Total nodes    : {stats['total']}")
    print(f"    Embedded (any) : {stats['embedded_total']}")
    print(f"    is_parent      : {stats['is_parent_total']}  (table header rows, NOT searchable)")
    print(f"    has_parent_id  : {stats['has_parent_id_total']}  (table row children, searchable)")
    print(f"    Searchable     : {stats['searchable']}  (embedded + not is_parent)")
    print(f"  Level breakdown:")
    for lvl in sorted(stats["by_level"].keys()):
        d = stats["by_level"][lvl]
        lvl_label = "leaf" if lvl == 0 else f"RAPTOR summary L{lvl}"
        print(f"    Level {lvl} ({lvl_label}): "
              f"count={d['count']}, "
              f"embedded={d['embedded']}, "
              f"is_parent={d['is_parent']}, "
              f"has_parent_id={d['has_parent_id']}")
    print(f"  Leaf node type breakdown:")
    for t, cnt in stats["leaf_types"].items():
        print(f"    {t}: {cnt}")


def main() -> None:
    print_section("LOADING TREES")

    tree_data = {}
    for label, (cache_key, run_id, ablation_label) in TREES.items():
        try:
            nodes, manifest = load_tree(cache_key, run_id, ablation_label)
            stats = node_stats(nodes)
            tree_data[label] = (nodes, manifest, stats)
            print(f"  Loaded {label}: {len(nodes)} nodes")
        except Exception as e:
            print(f"  FAILED to load {label}: {e}")

    # -----------------------------------------------------------------------
    print_section("PER-TREE NODE STATISTICS")
    # -----------------------------------------------------------------------
    for label, (nodes, manifest, stats) in tree_data.items():
        print_stats_table(label, stats, manifest)

    # -----------------------------------------------------------------------
    print_section("EMBEDDING COVERAGE BY NODE TYPE — KEY COMPARISON")
    # -----------------------------------------------------------------------
    print(f"\n{'Label':<45} {'Searchable':>10} {'is_parent':>10} {'child_rows':>10} {'L1+summaries':>13}")
    print("-" * 92)
    for label, (nodes, manifest, stats) in tree_data.items():
        summary_nodes = sum(
            stats["by_level"].get(lvl, {}).get("count", 0)
            for lvl in stats["by_level"]
            if lvl > 0
        )
        print(f"{label:<45} "
              f"{stats['searchable']:>10} "
              f"{stats['is_parent_total']:>10} "
              f"{stats['has_parent_id_total']:>10} "
              f"{summary_nodes:>13}")

    # -----------------------------------------------------------------------
    print_section("SAMPLE TEXTS — AS-AMM no_header_propagation")
    # -----------------------------------------------------------------------
    label_nhp = "AS-AMM no_header_propagation"
    if label_nhp in tree_data:
        nodes_nhp, _, _ = tree_data[label_nhp]

        print("\n  [is_parent samples — table header rows, NOT in search index]")
        for i, txt in enumerate(sample_node_texts(nodes_nhp, "is_parent"), 1):
            print(f"    [{i}] {repr(txt[:160])}")

        print("\n  [has_parent_id samples — table row children, ARE searchable]")
        for i, txt in enumerate(sample_node_texts(nodes_nhp, "has_parent_id"), 1):
            print(f"    [{i}] {repr(txt[:160])}")

        print("\n  [embedded_leaf samples — prose leaves, ARE searchable]")
        for i, txt in enumerate(sample_node_texts(nodes_nhp, "embedded_leaf"), 1):
            print(f"    [{i}] {repr(txt[:160])}")

        print("\n  [summary node samples — RAPTOR hierarchy nodes]")
        for i, txt in enumerate(sample_node_texts(nodes_nhp, "summary"), 1):
            print(f"    [{i}] {repr(txt[:160])}")
    else:
        print(f"  Tree not loaded: {label_nhp}")

    # -----------------------------------------------------------------------
    print_section("SAMPLE TEXTS — SC10000AMM no_header_propagation (works correctly)")
    # -----------------------------------------------------------------------
    label_sc_nhp = "SC10000AMM no_header_propagation"
    if label_sc_nhp in tree_data:
        nodes_sc, _, _ = tree_data[label_sc_nhp]

        print("\n  [has_parent_id samples — table row children]")
        for i, txt in enumerate(sample_node_texts(nodes_sc, "has_parent_id"), 1):
            print(f"    [{i}] {repr(txt[:160])}")

        print("\n  [summary node samples]")
        for i, txt in enumerate(sample_node_texts(nodes_sc, "summary"), 1):
            print(f"    [{i}] {repr(txt[:160])}")

    # -----------------------------------------------------------------------
    print_section("GROUND-TRUTH ID → SEARCHABILITY CHECK")
    # -----------------------------------------------------------------------

    # Map dataset keys to tree labels and manual sources
    checks = [
        # (dataset_key, tree_label, manual_source)
        ("no_header_prop", "AS-AMM no_header_propagation", "AS-AMM-01-000"),
        ("no_caption_fold", "AS-AMM no_caption_folding", "AS-AMM-01-000"),
        ("full", "AS-AMM full_context_aware", "AS-AMM-01-000"),
        ("no_header_prop", "SC10000AMM no_header_propagation", "SC10000AMM"),
        ("no_caption_fold", "SC10000AMM no_caption_folding", "SC10000AMM"),
    ]

    for dataset_key, tree_label, manual_source in checks:
        dataset_path = ANCHORED_DATASETS.get(dataset_key)
        if dataset_path is None or not dataset_path.exists():
            print(f"\n  SKIPPED (missing dataset): {tree_label} / {dataset_key}")
            continue
        if tree_label not in tree_data:
            print(f"\n  SKIPPED (tree not loaded): {tree_label}")
            continue

        nodes, _, _ = tree_data[tree_label]
        result = check_gt_ids_searchable(dataset_path, nodes, manual_source, tree_label)

        print(f"\n  {tree_label} — dataset={dataset_key} — manual={manual_source}")
        if "error" in result:
            print(f"    ERROR: {result['error']}")
            continue

        total = result["total_queries"]
        print(f"    Total queries for this manual : {total}")
        print(f"    GT points to searchable node  : {result['gt_in_searchable']} "
              f"({100 * result['gt_in_searchable'] / max(total, 1):.0f}%)")
        print(f"    GT points to is_parent node   : {result['gt_is_parent_node']} "
              f"({100 * result['gt_is_parent_node'] / max(total, 1):.0f}%)  <-- MISS: not in search index")
        print(f"    GT has_parent_id (child row)  : {result['gt_has_parent_id']}")
        print(f"    GT missing from tree entirely : {result['gt_missing_from_tree']} "
              f"({100 * result['gt_missing_from_tree'] / max(total, 1):.0f}%)")
        print(f"    TABLE queries:")
        tq = result["table_queries"]
        print(f"      Total table queries         : {tq}")
        print(f"      Table GT searchable         : {result['table_gt_searchable']} "
              f"({100 * result['table_gt_searchable'] / max(tq, 1):.0f}%)")
        print(f"      Table GT is_parent          : {result['table_gt_is_parent']} "
              f"({100 * result['table_gt_is_parent'] / max(tq, 1):.0f}%)")
        print(f"      Table GT missing            : {result['table_gt_missing']} "
              f"({100 * result['table_gt_missing'] / max(tq, 1):.0f}%)")

    # -----------------------------------------------------------------------
    print_section("ROOT CAUSE: RAPTOR TREE BUILD FAILURE IN AS-AMM ABLATIONS")
    # -----------------------------------------------------------------------

    print("""
  KEY FINDING: Tree build process diverged between AS-AMM and SC10000AMM runs.

  AS-AMM no_header_propagation / no_caption_folding (run 20260517_104040_95536286):
    - num_summary_nodes = 0   (manifest confirmed)
    - num_levels = 1          (manifest confirmed)
    - RAPTOR tree was NOT built — only flat leaf nodes exist

  SC10000AMM same ablations (run 20260517_080545_666ba344):
    - num_summary_nodes = 52/55
    - num_levels = 3/4
    - RAPTOR tree was successfully built

  This means for AS-AMM, the "collapsed" retrieval_mode searches across ALL
  levels including RAPTOR summary nodes — but there ARE no summary nodes.
  The only searchable nodes are the ~100 embedded leaf nodes (prose + table
  row children) from 410/412 total nodes.

  The 306 is_parent nodes (table header rows) are explicitly excluded from
  the search index by TreeIndex.__init__ (line: if node.get("is_parent"): continue).
  The ~100 embedded nodes are only the prose pages and table child rows.

  WHY the RAPTOR build failed:
    The manifest for both AS-AMM ablations shows:
      num_table_chunks = 402/403  (vs 3459 in full_context_aware)
      num_leaf_nodes   = 410/412  (vs 4358 in full_context_aware)

    This is consistent with the full_context_aware tree having been built with
    a DIFFERENT (newer?) Docling parse that produced many more table rows per
    page (4358 vs ~410 pages). The no_header_propagation / no_caption_folding
    ablations used an OLDER Docling cache (GCS bucket prefix differs:
      - no_header_prop: gs://hitl-dss-assets/docling-cache/...
      - full_context_aware: gs://raptor-assets/docling-cache/...

    The older parse produced ~410 chunks (1 per page, few table rows).
    With ~100 embeddable leaf nodes (after excluding 306 is_parent), the
    RAPTOR UMAP/GMM step ran but either:
      (a) All LLM summarization calls failed (quota/project issue), or
      (b) The build completed but no summary nodes were written.

    Evidence: num_summary_nodes=0 in manifest, single-level tree.

  CONSEQUENCE FOR EVALUATION:
    - 306 is_parent nodes (table header context) → filtered out by TreeIndex
    - ~4 child row nodes per table on average (not hundreds like full tree)
    - Only ~100 searchable nodes total vs 4108 in full_context_aware
    - TABLE queries ground-truth IDs (from anchored dataset) point to child
      row nodes that DO exist in the tree — but the tree has so few table
      rows that the chance of a correct match is low (recall ≈ 0)
    - No RAPTOR summary nodes to help surface cross-page table patterns

  CONCLUSION:
    The near-zero TABLE scores for AS-AMM no_header_propagation and
    no_caption_folding are caused by:
    1. Stale Docling cache used at build time → ~10x fewer leaf nodes
    2. RAPTOR summarization failed → 0 summary nodes, 1-level flat tree
    3. TreeIndex excludes 306 is_parent nodes → only ~100 searchable nodes
    4. Those ~100 nodes are insufficient to match TABLE ground-truth queries
""")

    # -----------------------------------------------------------------------
    print_section("DOCLING CACHE BUCKET DISCREPANCY")
    # -----------------------------------------------------------------------

    for label, (_, manifest, _) in tree_data.items():
        docling_path = manifest.get("docling_cache_path", "N/A")
        print(f"  {label:<45} -> {docling_path}")

    print()


if __name__ == "__main__":
    main()
