"""Fix 1 diagnostic: Check if retrieval actually works on the full_context_aware tree.

Picks 5 high-confidence queries, embeds them, retrieves top-10 from the tree,
and checks whether the ground truth chunk (from anchored dataset) appears.

IMPORTANT HYPOTHESIS: The anchored dataset has chunk IDs from the production DB,
but the tree-cache pickles have independently generated UUIDs. If this is the case,
exact chunk ID matching will ALWAYS fail, and the retrieval evaluator silently
falls back to page-level matching (or scores zero when no pages are available).

This script checks both:
  (a) Whether the retrieval function itself returns sensible results
  (b) Whether chunk ID matching between dataset and tree is broken
"""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root and backend to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BACKEND = Path(os.environ.get(
    "RAPTOR_BACKEND",
    str(ROOT.parent / "hitl-dss-react" / "backend"),
))
if BACKEND.exists() and str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# SC10000AMM cache key (most queries target this manual)
CACHE_KEY_SC = "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0"
TREE_DIR_SC = ROOT / ".tree-cache" / CACHE_KEY_SC

DATASET_PATH = ROOT / "datasets" / "combined_eval_dataset_anchored.json"


def find_latest_tree(tree_dir: Path, ablation_label: str) -> Path | None:
    """Find the most recent tree.pkl for a given ablation label."""
    candidates = sorted(tree_dir.glob(f"*/{ ablation_label}/tree.pkl"))
    return candidates[-1] if candidates else None


def load_tree_nodes(pkl_path: Path) -> list[dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def embed_query(text: str) -> np.ndarray | None:
    """Embed a single query using the same function the evaluator uses."""
    from modules.embeddings import embed
    result = embed(text=text)
    if result is not None:
        return np.array(result, dtype=np.float32)
    return None


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
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

    # Filter high-confidence queries for SC10000AMM
    high_conf = [
        q for q in all_queries
        if q.get("_reanchor_metadata", {}).get("confidence") == "high"
        and q.get("manual_source") == "SC10000AMM"
    ]
    print(f"High-confidence SC10000AMM queries: {len(high_conf)}")

    # Pick 5 representative queries (spread across origins)
    import random
    random.seed(42)
    sample = random.sample(high_conf, min(5, len(high_conf)))

    # Load full_context_aware tree
    tree_path = find_latest_tree(TREE_DIR_SC, "full_context_aware")
    if not tree_path:
        print("ERROR: No full_context_aware tree found for SC10000AMM")
        return
    print(f"\nTree path: {tree_path}")

    nodes = load_tree_nodes(tree_path)
    print(f"Total nodes in tree: {len(nodes)}")

    # Build searchable index (same logic as TreeIndex)
    searchable = []
    for node in nodes:
        if node.get("is_parent"):
            continue
        if node.get("embedding") is None:
            continue
        searchable.append(node)

    print(f"Searchable nodes: {len(searchable)}")

    # Build node lookup by ID
    node_by_id = {n.get("id", f"node_{i}"): n for i, n in enumerate(nodes)}
    tree_ids = set(node_by_id.keys())

    # Build embedding matrix
    emb_matrix = np.array([n["embedding"] for n in searchable], dtype=np.float32)

    # Check: are ANY anchored chunk IDs present in this tree?
    anchored_ids = {q["ground_truth_chunk_id"] for q in all_queries if q.get("manual_source") == "SC10000AMM"}
    overlap = anchored_ids & tree_ids
    print(f"\n=== CRITICAL CHECK ===")
    print(f"Anchored dataset chunk IDs (SC10000AMM): {len(anchored_ids)}")
    print(f"Tree node IDs: {len(tree_ids)}")
    print(f"OVERLAP (IDs present in both): {len(overlap)}")
    if len(overlap) == 0:
        print(">>> CONFIRMED: Zero overlap. Anchored IDs are from production DB, tree has independent UUIDs.")
        print(">>> This means exact chunk ID matching ALWAYS fails. All scoring is broken.")
    print()

    # Now run retrieval for each sample query
    in_top10 = 0
    in_top50 = 0
    in_top100 = 0

    for i, query in enumerate(sample):
        print(f"{'='*80}")
        print(f"Query {i+1}: {query['id']}")
        print(f"  Text: {query['observation'][:120]}...")
        print(f"  Ground truth chunk_id: {query['ground_truth_chunk_id']}")
        print(f"  Fault code: {query['ground_truth_fault_code']}")
        print(f"  Page reference: {query['page_reference']}")

        # Check if GT chunk exists in tree
        gt_id = query["ground_truth_chunk_id"]
        gt_in_tree = gt_id in node_by_id
        print(f"  GT chunk in tree? {gt_in_tree}")

        if gt_in_tree:
            gt_node = node_by_id[gt_id]
            print(f"  GT chunk text (first 200): {gt_node['text'][:200]}")
        else:
            # Find the chunk by text content instead — look for fault code in tree
            fault_lower = query["ground_truth_fault_code"].lower().strip()
            text_matches = [
                n for n in searchable
                if fault_lower in n.get("text", "").lower()
            ]
            print(f"  Text-match search (fault code in tree nodes): {len(text_matches)} nodes contain the fault code")
            if text_matches:
                best = text_matches[0]
                print(f"    First match ID: {best['id']}")
                print(f"    First match text (first 200): {best['text'][:200]}")
                print(f"    First match pages: {best.get('pages', [])} page_start={best.get('page_start')}")

        # Embed query
        print(f"\n  Embedding query...")
        q_emb = embed_query(query["observation"])
        if q_emb is None:
            print("  FAILED to embed query")
            continue

        # Retrieve top-10
        scores = cosine_similarity_batch(q_emb, emb_matrix)
        top_indices = np.argsort(scores)[::-1]

        print(f"\n  Top-10 retrieved:")
        for rank, idx in enumerate(top_indices[:10]):
            node = searchable[idx]
            pages = node.get("pages", [])
            print(f"    #{rank+1}: score={scores[idx]:.4f} | id={node['id'][:12]}... | "
                  f"level={node.get('level',0)} | pages={pages} | "
                  f"text={node['text'][:100].replace(chr(10), ' ')}")

        # Check: is GT chunk in results? (it won't be if IDs don't match)
        retrieved_ids = [searchable[idx]["id"] for idx in top_indices]
        if gt_id in retrieved_ids[:10]:
            in_top10 += 1
            print(f"  >>> GT chunk found at rank {retrieved_ids.index(gt_id) + 1}")
        elif gt_id in retrieved_ids[:50]:
            in_top50 += 1
            print(f"  >>> GT chunk found at rank {retrieved_ids.index(gt_id) + 1}")
        elif gt_id in retrieved_ids[:100]:
            in_top100 += 1
            print(f"  >>> GT chunk found at rank {retrieved_ids.index(gt_id) + 1}")
        elif gt_id in retrieved_ids:
            rank = retrieved_ids.index(gt_id) + 1
            print(f"  >>> GT chunk found at rank {rank} (outside top-100)")
        else:
            print(f"  >>> GT chunk NOT FOUND in tree at all (ID mismatch)")

        # Instead, check if any retrieved node contains the fault code text
        fault_lower = query["ground_truth_fault_code"].lower().strip()
        if fault_lower and fault_lower != "none":
            text_match_ranks = []
            for rank, idx in enumerate(top_indices[:100]):
                if fault_lower in searchable[idx]["text"].lower():
                    text_match_ranks.append(rank + 1)
            if text_match_ranks:
                print(f"  >>> TEXT MATCH: Fault code found in retrieved nodes at ranks: {text_match_ranks[:5]}")
            else:
                print(f"  >>> TEXT MATCH: Fault code NOT found in top-100 retrieved nodes")

        # Check page overlap
        try:
            gt_page = int(query["page_reference"])
        except (ValueError, TypeError):
            gt_page = None
        if gt_page:
            page_match_ranks = []
            for rank, idx in enumerate(top_indices[:100]):
                node_pages = searchable[idx].get("pages", [])
                ps = searchable[idx].get("page_start", 0)
                pe = searchable[idx].get("page_end", 0)
                all_pages = set(node_pages) | set(range(ps, pe + 1)) if ps else set(node_pages)
                if gt_page in all_pages:
                    page_match_ranks.append(rank + 1)
            if page_match_ranks:
                print(f"  >>> PAGE MATCH: Page {gt_page} found in top-100 at ranks: {page_match_ranks[:5]}")

    n = len(sample)
    print(f"\n{'='*80}")
    print(f"SUMMARY ({n} queries):")
    print(f"  GT chunk in top-10:  {in_top10}/{n}")
    print(f"  GT chunk in top-50:  {in_top10 + in_top50}/{n}")
    print(f"  GT chunk in top-100: {in_top10 + in_top50 + in_top100}/{n}")
    print()
    print("NOTE: If overlap is 0, these counts reflect the ID mismatch bug,")
    print("not retrieval quality. The TEXT MATCH and PAGE MATCH lines above")
    print("show whether retrieval is actually finding relevant content.")


if __name__ == "__main__":
    main()
