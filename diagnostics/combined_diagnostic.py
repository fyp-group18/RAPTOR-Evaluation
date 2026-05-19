"""Combined diagnostic for Fixes 1-4.

Since the embedding API is down (GCP project suspended), we use the GT chunk's
own embedding as a proxy query to test retrieval mechanics.

Checks:
  Fix 1: Self-retrieval test (does GT chunk rank #1 when we use its own embedding?)
  Fix 2: ID overlap between anchored dataset and each ablation tree
  Fix 3: Page metadata and embedding coverage for zero-scoring ablations
  Fix 4: Medium-confidence reanchoring spot-check
"""

import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "datasets" / "combined_eval_dataset_anchored.json"

# Both cache keys
CACHE_KEYS = {
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
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


def find_tree(cache_key: str, ablation_label: str) -> Path | None:
    tree_dir = ROOT / ".tree-cache" / cache_key
    candidates = sorted(tree_dir.glob(f"*/{ablation_label}/tree.pkl"))
    return candidates[-1] if candidates else None


def load_nodes(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def cosine_sim_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-10:
        return np.zeros(matrix.shape[0])
    q_hat = query / q_norm
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms = np.maximum(m_norms, 1e-10)
    m_hat = matrix / m_norms
    return m_hat @ q_hat


def main():
    # Load dataset
    with open(DATASET_PATH) as f:
        all_queries = json.load(f)
    print(f"Total queries: {len(all_queries)}")

    queries_by_manual = defaultdict(list)
    for q in all_queries:
        queries_by_manual[q["manual_source"]].append(q)
    for m, qs in queries_by_manual.items():
        print(f"  {m}: {len(qs)} queries")

    # =========================================================================
    # FIX 1: Self-retrieval test on full_context_aware
    # =========================================================================
    print("\n" + "=" * 80)
    print("FIX 1: Self-retrieval test (using GT chunk embedding as query)")
    print("=" * 80)

    for manual, cache_key in CACHE_KEYS.items():
        tree_path = find_tree(cache_key, "full_context_aware")
        if not tree_path:
            print(f"\n  {manual}: No full_context_aware tree found")
            continue

        nodes = load_nodes(tree_path)
        node_by_id = {n["id"]: n for n in nodes if "id" in n}

        # Build searchable index
        searchable = [
            n for n in nodes
            if not n.get("is_parent") and n.get("embedding") is not None
        ]
        if not searchable:
            print(f"\n  {manual}: No searchable nodes!")
            continue

        emb_matrix = np.array([n["embedding"] for n in searchable], dtype=np.float32)
        searchable_ids = [n["id"] for n in searchable]
        searchable_id_set = set(searchable_ids)

        manual_queries = queries_by_manual.get(manual, [])
        high_conf = [
            q for q in manual_queries
            if q.get("_reanchor_metadata", {}).get("confidence") == "high"
        ]

        print(f"\n  {manual}: {len(nodes)} nodes, {len(searchable)} searchable, "
              f"{len(high_conf)} high-confidence queries")

        # Check ID overlap for this manual's queries
        all_gt_ids = {q["ground_truth_chunk_id"] for q in manual_queries}
        gt_in_tree = all_gt_ids & set(node_by_id.keys())
        gt_in_searchable = all_gt_ids & searchable_id_set
        print(f"    GT IDs in tree: {len(gt_in_tree)}/{len(all_gt_ids)}")
        print(f"    GT IDs in searchable: {len(gt_in_searchable)}/{len(all_gt_ids)}")

        # Check: for high-conf queries whose GT chunk is in the searchable set,
        # use the GT chunk's embedding as query. Does it rank #1?
        random.seed(42)
        testable = [
            q for q in high_conf
            if q["ground_truth_chunk_id"] in searchable_id_set
        ]
        print(f"    Testable (GT in searchable): {len(testable)}")
        sample = random.sample(testable, min(5, len(testable)))

        ranks = []
        for q in sample:
            gt_id = q["ground_truth_chunk_id"]
            gt_node = node_by_id[gt_id]
            gt_emb = np.array(gt_node["embedding"], dtype=np.float32)

            scores = cosine_sim_batch(gt_emb, emb_matrix)
            sorted_indices = np.argsort(scores)[::-1]
            retrieved_ids = [searchable[idx]["id"] for idx in sorted_indices]

            if gt_id in retrieved_ids:
                rank = retrieved_ids.index(gt_id) + 1
            else:
                rank = -1
            ranks.append(rank)

            gt_score = float(scores[sorted_indices[rank - 1]]) if rank > 0 else 0.0
            print(f"    {q['id']}: GT chunk rank={rank}, "
                  f"top score={scores[sorted_indices[0]]:.4f}, "
                  f"GT score={gt_score:.4f}")
            if rank > 1:
                # Show what outranked it
                for r in range(min(3, rank - 1)):
                    better_node = searchable[sorted_indices[r]]
                    print(f"      Rank {r+1}: level={better_node.get('level',0)} "
                          f"text={better_node['text'][:80].replace(chr(10), ' ')}")

        in_top1 = sum(1 for r in ranks if r == 1)
        in_top5 = sum(1 for r in ranks if 1 <= r <= 5)
        print(f"    Self-retrieval: {in_top1}/{len(ranks)} rank=1, {in_top5}/{len(ranks)} in top-5")

    # =========================================================================
    # FIX 2: ID overlap check for ALL ablation trees
    # =========================================================================
    print("\n" + "=" * 80)
    print("FIX 2: Chunk ID overlap between anchored dataset and each ablation tree")
    print("=" * 80)

    for manual, cache_key in CACHE_KEYS.items():
        manual_queries = queries_by_manual.get(manual, [])
        anchored_ids = {q["ground_truth_chunk_id"] for q in manual_queries}
        print(f"\n  {manual} ({len(anchored_ids)} anchored chunk IDs):")

        for ablation_label in ABLATION_LABELS:
            tree_path = find_tree(cache_key, ablation_label)
            if not tree_path:
                print(f"    {ablation_label:<30s}: NO TREE FOUND")
                continue

            nodes = load_nodes(tree_path)
            tree_ids = {n.get("id") for n in nodes if n.get("id")}
            overlap = anchored_ids & tree_ids
            searchable = [
                n for n in nodes
                if not n.get("is_parent") and n.get("embedding") is not None
            ]
            searchable_ids = {n.get("id") for n in searchable if n.get("id")}
            searchable_overlap = anchored_ids & searchable_ids

            status = "OK" if len(overlap) == len(anchored_ids) else "MISMATCH"
            print(f"    {ablation_label:<30s}: {len(overlap):>3}/{len(anchored_ids)} IDs match "
                  f"({len(searchable_overlap)} in searchable) [{status}] "
                  f"({len(nodes)} total, {len(searchable)} searchable)")

    # =========================================================================
    # FIX 3: Page metadata and embeddings for zero-scoring ablations
    # =========================================================================
    print("\n" + "=" * 80)
    print("FIX 3: Node diagnostics for potentially zero-scoring ablations")
    print("=" * 80)

    problem_ablations = [
        "original_raptor_text_only",
        "semantic_chunking_baseline",
        "baseline_naive_chunking",
    ]

    for manual, cache_key in CACHE_KEYS.items():
        for ablation_label in problem_ablations:
            tree_path = find_tree(cache_key, ablation_label)
            if not tree_path:
                print(f"\n  {manual} / {ablation_label}: NO TREE")
                continue

            nodes = load_nodes(tree_path)
            print(f"\n  {manual} / {ablation_label}:")
            print(f"    Total nodes: {len(nodes)}")

            # Count by level
            level_counts = Counter(n.get("level", 0) for n in nodes)
            print(f"    By level: {dict(sorted(level_counts.items()))}")

            # Page metadata
            has_pages = sum(1 for n in nodes if n.get("pages"))
            has_page_start = sum(1 for n in nodes if n.get("page_start"))
            has_page_end = sum(1 for n in nodes if n.get("page_end"))
            has_embedding = sum(1 for n in nodes if n.get("embedding") is not None)
            is_parent = sum(1 for n in nodes if n.get("is_parent"))

            print(f"    pages populated: {has_pages}/{len(nodes)}")
            print(f"    page_start populated: {has_page_start}/{len(nodes)}")
            print(f"    page_end populated: {has_page_end}/{len(nodes)}")
            print(f"    has embedding: {has_embedding}/{len(nodes)}")
            print(f"    is_parent: {is_parent}/{len(nodes)}")

            # Check for zero-valued page fields
            zero_page_start = sum(1 for n in nodes if n.get("page_start") == 0)
            empty_pages = sum(1 for n in nodes if n.get("pages") == [])
            print(f"    page_start=0: {zero_page_start}/{len(nodes)}")
            print(f"    pages=[]: {empty_pages}/{len(nodes)}")

            # Print 3 sample nodes
            print(f"    Sample nodes:")
            for n in nodes[:3]:
                print(f"      id={n.get('id', '?')[:12]}... "
                      f"level={n.get('level', '?')} "
                      f"pages={n.get('pages')} "
                      f"page_start={n.get('page_start')} "
                      f"page_end={n.get('page_end')} "
                      f"has_emb={'yes' if n.get('embedding') is not None else 'NO'} "
                      f"is_parent={n.get('is_parent', False)} "
                      f"section={n.get('section_header', '')[:40] if n.get('section_header') else 'None'}")
                print(f"        text={n.get('text', '')[:120].replace(chr(10), ' ')}")

    # =========================================================================
    # FIX 4: Medium-confidence reanchoring spot-check
    # =========================================================================
    print("\n" + "=" * 80)
    print("FIX 4: Medium-confidence reanchoring spot-check (10 samples)")
    print("=" * 80)

    medium_conf = [
        q for q in all_queries
        if q.get("_reanchor_metadata", {}).get("confidence") == "medium"
    ]
    print(f"Total medium-confidence queries: {len(medium_conf)}")

    # Check how many have fault_code == "NONE" or "None"
    none_fault = [
        q for q in medium_conf
        if (q.get("ground_truth_fault_code") or "").strip().upper() == "NONE"
    ]
    print(f"  With fault_code=NONE: {len(none_fault)}")

    random.seed(123)
    sample = random.sample(medium_conf, min(10, len(medium_conf)))

    # Load trees for chunk text lookup
    tree_cache = {}
    for manual, cache_key in CACHE_KEYS.items():
        tree_path = find_tree(cache_key, "full_context_aware")
        if tree_path:
            nodes = load_nodes(tree_path)
            for n in nodes:
                if n.get("id"):
                    tree_cache[n["id"]] = n

    for i, q in enumerate(sample):
        meta = q.get("_reanchor_metadata", {})
        gt_id = q["ground_truth_chunk_id"]
        matched_node = tree_cache.get(gt_id)

        print(f"\n  [{i+1}] {q['id']} ({q['manual_source']})")
        print(f"      Origin: {q['origin']}")
        print(f"      Fault code: \"{q['ground_truth_fault_code']}\"")
        print(f"      Page ref: {q['page_reference']}")
        print(f"      Match method: {meta.get('match_method')}, score: {meta.get('match_score')}")
        print(f"      Observation: {q['observation'][:200]}")
        if matched_node:
            print(f"      Matched chunk text: {matched_node['text'][:200].replace(chr(10), ' ')}")
            print(f"      Chunk pages: {matched_node.get('pages')} "
                  f"(start={matched_node.get('page_start')}, end={matched_node.get('page_end')})")
        else:
            print(f"      Matched chunk: NOT IN TREE (id={gt_id[:20]}...)")
        print(f"      >>> PLAUSIBLE? (needs human review)")


if __name__ == "__main__":
    main()
