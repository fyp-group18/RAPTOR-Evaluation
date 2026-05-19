"""Diagnose embedding space mismatch between full_context_aware and ablation trees.

Checks:
  1-3. Embedding dimensions, norms, sample values across trees
  4-5. Query embedding vs ground-truth chunk cosine similarity
  6.   Model strings used by tree_builder, retrieval_evaluator, and production
  7.   Multimodal vs text-only embedding similarity comparison
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TREE_CACHE = PROJECT_ROOT / ".tree-cache"
DATASETS = PROJECT_ROOT / "datasets"

# Cache keys
CACHE_KEYS = {
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
}

ABLATION_LABEL = {
    "full": "full_context_aware",
    "no_table_pc": "no_table_parent_child",
    "text_only_raptor": "original_raptor_text_only",
}


def find_latest_tree(cache_key: str, ablation_label: str) -> Path | None:
    """Find the most recent tree.pkl for a given cache_key + ablation."""
    base = TREE_CACHE / cache_key
    if not base.exists():
        return None
    candidates = []
    for run_dir in sorted(base.iterdir(), reverse=True):
        pkl = run_dir / ablation_label / "tree.pkl"
        if pkl.exists():
            candidates.append(pkl)
    return candidates[0] if candidates else None


def load_tree(pkl_path: Path) -> list[dict]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_leaf_nodes_with_embeddings(nodes: list[dict], n: int = 3) -> list[dict]:
    """Return first n leaf nodes (level=0) with non-None embeddings."""
    leaves = []
    for node in nodes:
        if node.get("level", 0) == 0 and node.get("embedding") is not None:
            leaves.append(node)
            if len(leaves) >= n:
                break
    return leaves


def print_embedding_info(label: str, nodes: list[dict]):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, node in enumerate(nodes):
        emb = node["embedding"]
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        elif not isinstance(emb, np.ndarray):
            print(f"  Node {i}: embedding type={type(emb).__name__} — unexpected!")
            continue
        norm = np.linalg.norm(emb)
        print(f"  Node {i}: dim={emb.shape[0]}, L2 norm={norm:.6f}, "
              f"first 5={emb[:5].tolist()}")
        # Also check dtype
        print(f"           dtype={emb.dtype}, storage type={type(node['embedding']).__name__}")


def cosine_sim(a, b):
    a = np.array(a, dtype=np.float32) if not isinstance(a, np.ndarray) else a
    b = np.array(b, dtype=np.float32) if not isinstance(b, np.ndarray) else b
    dot = np.dot(a, b)
    return dot / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def main():
    # Use SC10000AMM (more queries target it)
    cache_key = CACHE_KEYS["SC10000AMM"]
    source = "SC10000AMM"

    # ─── Steps 1-3: Load trees, inspect embeddings ───
    trees = {}
    for ablation_key, ablation_label in ABLATION_LABEL.items():
        pkl = find_latest_tree(cache_key, ablation_label)
        if pkl is None:
            print(f"WARNING: No tree found for {source}/{ablation_label}")
            continue
        nodes = load_tree(pkl)
        trees[ablation_key] = nodes
        # Check manifest for provenance
        manifest_path = pkl.parent / "tree_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            registered = manifest.get("registered_from")
            run_id = manifest.get("run_id", "?")
            emb_model = manifest.get("embedding_model", "?")
            print(f"\n[{ablation_label}] run_id={run_id}, "
                  f"embedding_model={emb_model}, "
                  f"registered_from={'YES' if registered else 'no (fresh build)'}, "
                  f"total nodes={len(nodes)}")
            if registered:
                print(f"  Registered from: {registered}")

        sample = get_leaf_nodes_with_embeddings(nodes, n=3)
        if not sample:
            print(f"  WARNING: No leaf nodes with embeddings found!")
            # Check what we have
            with_emb = [n for n in nodes if n.get("embedding") is not None]
            print(f"  Nodes with embeddings: {len(with_emb)}/{len(nodes)}")
            if with_emb:
                sample = with_emb[:3]
        print_embedding_info(f"{ablation_label} ({source})", sample)

    # ─── Step 4: Embed a query ───
    print(f"\n\n{'='*60}")
    print(f"  STEP 4-5: Query embedding + cosine similarity")
    print(f"{'='*60}")

    # Load anchored datasets
    anchored = {}
    for ablation_key in ABLATION_LABEL:
        path = DATASETS / f"combined_eval_dataset_anchored_{ablation_key}.json"
        if path.exists():
            anchored[ablation_key] = json.loads(path.read_text())

    # Pick a query that targets SC10000AMM
    test_query = None
    for q in anchored.get("full", []):
        if q.get("manual_source") == source:
            test_query = q
            break

    if test_query is None:
        print("ERROR: No query found for SC10000AMM")
        sys.exit(1)

    qid_key = "id" if "id" in test_query else "query_id"
    text_key = "observation" if "observation" in test_query else "query_text"

    print(f"\nTest query: {test_query[qid_key]}")
    print(f"  Text: {test_query[text_key][:100]}...")
    print(f"  GT chunk (full): {test_query.get('ground_truth_chunk_id', '?')}")

    # Embed query using the eval harness's function
    sys.path.insert(0, str(PROJECT_ROOT.parent))
    sys.path.insert(0, str(PROJECT_ROOT))
    from modules.embeddings import embed

    query_emb_raw = embed(text=test_query[text_key])
    if query_emb_raw is None:
        print("ERROR: Failed to embed query")
        sys.exit(1)
    query_emb = np.array(query_emb_raw, dtype=np.float32)
    print(f"\n  Query embedding: dim={query_emb.shape[0]}, L2 norm={np.linalg.norm(query_emb):.6f}")
    print(f"  First 5 values: {query_emb[:5].tolist()}")

    # ─── Step 5: Cosine similarity against GT chunks ───
    print(f"\n  Cosine similarity: query vs ground-truth chunk")
    for ablation_key, ablation_label in ABLATION_LABEL.items():
        if ablation_key not in trees or ablation_key not in anchored:
            continue
        # Find the GT chunk ID for this ablation
        gt_chunk_id = None
        for q in anchored[ablation_key]:
            if q[qid_key] == test_query[qid_key]:
                gt_chunk_id = q.get("ground_truth_chunk_id")
                break
        if gt_chunk_id is None:
            print(f"  {ablation_label}: GT chunk ID not found")
            continue

        # Find the chunk in the tree
        gt_node = None
        for n in trees[ablation_key]:
            if str(n.get("id")) == str(gt_chunk_id):
                gt_node = n
                break
        if gt_node is None:
            print(f"  {ablation_label}: GT chunk {gt_chunk_id} not found in tree")
            continue
        if gt_node.get("embedding") is None:
            print(f"  {ablation_label}: GT chunk {gt_chunk_id} has NO embedding")
            continue

        gt_emb = gt_node["embedding"]
        if isinstance(gt_emb, list):
            gt_emb = np.array(gt_emb, dtype=np.float32)
        sim = cosine_sim(query_emb, gt_emb)
        print(f"  {ablation_label:<30}: cosine_sim={sim:.6f} "
              f"(chunk dim={gt_emb.shape[0]}, norm={np.linalg.norm(gt_emb):.6f})")

    # ─── Step 6: Model strings ───
    print(f"\n\n{'='*60}")
    print(f"  STEP 6: Embedding model identification")
    print(f"{'='*60}")

    from core.config import MODEL_EMBEDDING
    print(f"  core.config.MODEL_EMBEDDING = {MODEL_EMBEDDING!r}")
    print(f"  (Used by both tree_builder.py and retrieval_evaluator.py via modules.embeddings)")

    # Check production tree manifests
    for ablation_key, ablation_label in ABLATION_LABEL.items():
        pkl = find_latest_tree(cache_key, ablation_label)
        if pkl:
            manifest_path = pkl.parent / "tree_manifest.json"
            if manifest_path.exists():
                m = json.loads(manifest_path.read_text())
                print(f"  {ablation_label} manifest.embedding_model = {m.get('embedding_model', 'NOT SET')!r}")

    # ─── Step 7: Multimodal vs text-only similarity comparison ───
    print(f"\n\n{'='*60}")
    print(f"  STEP 7: Multimodal vs text-only embedding similarity (20 queries)")
    print(f"{'='*60}")

    # Collect queries for SC10000AMM
    full_queries = [q for q in anchored.get("full", []) if q.get("manual_source") == source]
    text_only_queries = [q for q in anchored.get("text_only_raptor", []) if q.get("manual_source") == source]

    # Detect key names from data
    _qid = "id" if full_queries and "id" in full_queries[0] else "query_id"
    _qtxt = "observation" if full_queries and "observation" in full_queries[0] else "query_text"

    # Build lookup by query id
    text_only_by_id = {q[_qid]: q for q in text_only_queries}

    full_tree = trees.get("full", [])
    text_only_tree = trees.get("text_only_raptor", [])

    # Build node lookups
    full_node_map = {str(n.get("id")): n for n in full_tree}
    text_only_node_map = {str(n.get("id")): n for n in text_only_tree}

    sims_full = []
    sims_text_only = []
    sample_count = 0

    for q in full_queries:
        if sample_count >= 20:
            break

        qid = q[_qid]
        gt_full_id = q.get("ground_truth_chunk_id")
        text_only_q = text_only_by_id.get(qid)
        if text_only_q is None:
            continue
        gt_text_only_id = text_only_q.get("ground_truth_chunk_id")

        # Get nodes
        full_node = full_node_map.get(str(gt_full_id))
        text_only_node = text_only_node_map.get(str(gt_text_only_id))

        if full_node is None or text_only_node is None:
            continue
        if full_node.get("embedding") is None or text_only_node.get("embedding") is None:
            continue

        # Embed query
        q_emb = embed(text=q[_qtxt])
        if q_emb is None:
            continue
        q_emb = np.array(q_emb, dtype=np.float32)

        full_emb = full_node["embedding"]
        if isinstance(full_emb, list):
            full_emb = np.array(full_emb, dtype=np.float32)

        text_only_emb = text_only_node["embedding"]
        if isinstance(text_only_emb, list):
            text_only_emb = np.array(text_only_emb, dtype=np.float32)

        sim_f = cosine_sim(q_emb, full_emb)
        sim_t = cosine_sim(q_emb, text_only_emb)
        sims_full.append(sim_f)
        sims_text_only.append(sim_t)

        if sample_count < 3:
            print(f"  Query {qid}: full={sim_f:.4f}, text_only={sim_t:.4f}, Δ={sim_f - sim_t:+.4f}")
        sample_count += 1

    if sims_full:
        print(f"\n  {'Metric':<35} {'full_context_aware':>20} {'text_only_raptor':>20}")
        print(f"  {'-'*35} {'-'*20} {'-'*20}")
        print(f"  {'Avg cosine sim (query vs GT chunk)':<35} {np.mean(sims_full):>20.6f} {np.mean(sims_text_only):>20.6f}")
        print(f"  {'Std dev':<35} {np.std(sims_full):>20.6f} {np.std(sims_text_only):>20.6f}")
        print(f"  {'Min':<35} {np.min(sims_full):>20.6f} {np.min(sims_text_only):>20.6f}")
        print(f"  {'Max':<35} {np.max(sims_full):>20.6f} {np.max(sims_text_only):>20.6f}")
        print(f"  {'n':<35} {len(sims_full):>20} {len(sims_text_only):>20}")
    else:
        print("  No valid query-chunk pairs found for comparison")

    # ─── Bonus: check if full_context_aware embeddings look like a different model ───
    print(f"\n\n{'='*60}")
    print(f"  BONUS: Cross-tree embedding comparison")
    print(f"{'='*60}")

    # Pick a leaf from full_context_aware and text_only_raptor that cover similar text
    # and compute cosine similarity between the NODE embeddings
    if "full" in trees and "text_only_raptor" in trees:
        full_leaves = [n for n in trees["full"]
                       if n.get("level", 0) == 0 and n.get("embedding") is not None
                       and not n.get("is_parent")]
        text_leaves = [n for n in trees["text_only_raptor"]
                       if n.get("level", 0) == 0 and n.get("embedding") is not None]

        print(f"  full_context_aware: {len(full_leaves)} embeddable leaves")
        print(f"  text_only_raptor:  {len(text_leaves)} embeddable leaves")

        # Compute average norm for each tree
        full_norms = [np.linalg.norm(np.array(n["embedding"], dtype=np.float32) if isinstance(n["embedding"], list) else n["embedding"])
                      for n in full_leaves[:100]]
        text_norms = [np.linalg.norm(np.array(n["embedding"], dtype=np.float32) if isinstance(n["embedding"], list) else n["embedding"])
                      for n in text_leaves[:100]]

        print(f"\n  Average L2 norm (first 100 leaves):")
        print(f"    full_context_aware:  {np.mean(full_norms):.6f} (std={np.std(full_norms):.6f})")
        print(f"    text_only_raptor:    {np.mean(text_norms):.6f} (std={np.std(text_norms):.6f})")

        # Cross-tree similarity: embed the TEXT of a full_context_aware node freshly
        # and compare to its stored embedding
        print(f"\n  Fresh re-embedding test (5 full_context_aware nodes):")
        print(f"  Embed node text fresh → compare to stored embedding")
        for node in full_leaves[:5]:
            fresh = embed(text=node["text"])
            if fresh is None:
                continue
            fresh_emb = np.array(fresh, dtype=np.float32)
            stored_emb = np.array(node["embedding"], dtype=np.float32) if isinstance(node["embedding"], list) else node["embedding"]
            sim = cosine_sim(fresh_emb, stored_emb)
            print(f"    Node {node['id'][:12]}...: stored_norm={np.linalg.norm(stored_emb):.4f}, "
                  f"fresh_norm={np.linalg.norm(fresh_emb):.4f}, cosine_sim={sim:.6f}")


if __name__ == "__main__":
    main()
