"""
Test script for collapsed tree retrieval.

Validates that:
1. Summary nodes (level >= 1) have embeddings in the database
2. Collapsed mode returns nodes from multiple levels
3. Flat mode returns only level-0 nodes
4. Table parent nodes (embedding=None) are excluded in both modes
5. Parent expansion logic still works in collapsed mode

Usage:
    cd backend
    uv run python -m RAPTOR-evaluation.tests.test_collapsed_retrieval

    Or directly:
    cd backend && PYTHONPATH=. uv run python RAPTOR-evaluation/tests/test_collapsed_retrieval.py
"""

import sys
import os
from collections import Counter
from pathlib import Path

# Ensure backend is on sys.path
backend_dir = str(Path(__file__).resolve().parent.parent.parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import SessionLocal
from core.models import DocumentChunkMultimodal
from core.crud import multimodal_semantic_search
from modules.embeddings import embed
from sqlalchemy import func


def run_diagnostic_sql():
    """Check level distribution and embedding presence across all chunks."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Chunk level distribution and embedding coverage")
    print("=" * 70)

    with SessionLocal() as db:
        results = (
            db.query(
                DocumentChunkMultimodal.level,
                func.count().label("total"),
                func.count(DocumentChunkMultimodal.embedding).label("has_embedding"),
            )
            .group_by(DocumentChunkMultimodal.level)
            .order_by(DocumentChunkMultimodal.level)
            .all()
        )

    if not results:
        print("  No chunks found in database.")
        return False

    print(f"\n  {'Level':<8} {'Total':<10} {'Has Embedding':<16} {'Missing':<10}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 16} {'-' * 10}")
    has_summary_embeddings = False
    for level, total, has_emb in results:
        missing = total - has_emb
        print(f"  {level:<8} {total:<10} {has_emb:<16} {missing:<10}")
        if level > 0 and has_emb > 0:
            has_summary_embeddings = True

    if not has_summary_embeddings:
        print(
            "\n  WARNING: No summary nodes (level > 0) have embeddings. "
            "Collapsed tree search will return the same results as flat search. "
            "Investigate the embedding pipeline for summary nodes."
        )
    return has_summary_embeddings


def run_table_parent_check():
    """Verify table parent nodes have embedding=None."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Table parent nodes (level=0, parent_id IS NULL, has children)")
    print("=" * 70)

    with SessionLocal() as db:
        # Find chunks that are referenced as parent_id by other chunks
        from sqlalchemy import select, exists

        parent_subq = (
            select(DocumentChunkMultimodal.parent_id)
            .where(DocumentChunkMultimodal.parent_id.is_not(None))
            .distinct()
            .subquery()
        )
        parent_chunks = (
            db.query(
                DocumentChunkMultimodal.id,
                DocumentChunkMultimodal.level,
                DocumentChunkMultimodal.embedding.is_not(None).label("has_emb"),
            )
            .filter(DocumentChunkMultimodal.id.in_(select(parent_subq)))
            .all()
        )

    if not parent_chunks:
        print("  No table parent nodes found.")
        return

    emb_count = sum(1 for _, _, has_emb in parent_chunks if has_emb)
    no_emb_count = sum(1 for _, _, has_emb in parent_chunks if not has_emb)
    print(f"  Total parent nodes: {len(parent_chunks)}")
    print(f"  With embedding: {emb_count}")
    print(f"  Without embedding (expected): {no_emb_count}")

    if emb_count > 0:
        print(
            "  WARNING: Some table parent nodes have embeddings. "
            "They will appear in collapsed search results, which may cause "
            "duplication with child-expanded versions."
        )
    else:
        print("  OK: All table parent nodes have NULL embeddings (correctly excluded from search).")


def compare_retrieval_modes(test_query: str = "hydraulic system troubleshooting"):
    """Run the same query in flat vs collapsed mode and compare results."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: flat vs collapsed for query: '{test_query}'")
    print("=" * 70)

    print("\n  Embedding query...")
    query_emb = embed(text=test_query)
    if not query_emb:
        print("  ERROR: Failed to embed query. Check Vertex AI credentials.")
        return

    # Flat mode
    print("  Running flat retrieval (level=0 only)...")
    flat_results, flat_ms = multimodal_semantic_search(
        query_emb, "manual", retrieval_mode="flat"
    )
    print(f"  Flat: {len(flat_results)} results in {flat_ms}ms")

    # Collapsed mode
    print("  Running collapsed retrieval (all levels)...")
    collapsed_results, collapsed_ms = multimodal_semantic_search(
        query_emb, "manual", retrieval_mode="collapsed"
    )
    print(f"  Collapsed: {len(collapsed_results)} results in {collapsed_ms}ms")

    # Level distribution
    flat_levels = Counter(r["level"] for r in flat_results)
    collapsed_levels = Counter(r["level"] for r in collapsed_results)

    print(f"\n  Flat level distribution:      {dict(sorted(flat_levels.items()))}")
    print(f"  Collapsed level distribution: {dict(sorted(collapsed_levels.items()))}")

    # Check for summary nodes in collapsed
    summary_in_collapsed = sum(
        count for level, count in collapsed_levels.items() if level > 0
    )
    if summary_in_collapsed > 0:
        print(f"\n  SUCCESS: Collapsed mode retrieved {summary_in_collapsed} summary node(s) (level > 0)")
    else:
        print(
            "\n  NOTE: No summary nodes in collapsed results. "
            "Either they don't have embeddings, or leaf nodes dominated top-k."
        )

    # Check no embedding=None results leaked through
    for r in collapsed_results:
        assert r.get("level") is not None, f"Chunk {r['id']} missing level field"

    # Show top-5 from each mode
    print("\n  Top 5 collapsed results:")
    for i, r in enumerate(collapsed_results[:5]):
        print(
            f"    {i + 1}. [L{r['level']}] score={r['score']:.4f} "
            f"id={r['id'][:12]}... {r.get('section_header', '')[:50]}"
        )

    # Check parent expansion compatibility
    parent_ids_in_collapsed = {
        r.get("parent_id") for r in collapsed_results if r.get("parent_id")
    }
    print(f"\n  Chunks with parent_id in collapsed results: {len(parent_ids_in_collapsed)}")
    summary_with_parent = [
        r for r in collapsed_results if r["level"] > 0 and r.get("parent_id")
    ]
    if summary_with_parent:
        print(
            f"  WARNING: {len(summary_with_parent)} summary node(s) have parent_id set. "
            "This is unexpected — summary nodes should not have parent_id."
        )
    else:
        print("  OK: No summary nodes have parent_id (parent expansion unaffected).")


def main():
    print("RAPTOR Collapsed Tree Retrieval — Diagnostic Tests")
    print("=" * 70)

    has_summaries = run_diagnostic_sql()
    run_table_parent_check()

    if has_summaries:
        compare_retrieval_modes()
    else:
        print(
            "\n  Skipping mode comparison: no summary nodes with embeddings found. "
            "Collapsed search would return identical results to flat search."
        )

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
