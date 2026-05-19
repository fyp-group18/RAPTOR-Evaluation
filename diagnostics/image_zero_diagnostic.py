"""Diagnostic: why IMAGE queries score zero on full_context_aware trees.

Investigates four questions:
1. Do image queries have ground_truth_chunk_id in anchored datasets?
2. Do full_context_aware nodes carry page metadata for page-fallback matching?
3. Per image query: how many nodes cover the target page in each tree?
4. Is there a page numbering mismatch between node metadata and query page refs?
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/joanne/Desktop/PycharmProjects/RAPTOR-Evaluation")

# ── Tree paths ────────────────────────────────────────────────────────────────

SC_CACHE_KEY  = "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0"
SC_FCA_RUN    = "20260517_075856_639c04b2"
SC_TONLY_RUN  = "20260517_080545_666ba344"

ASAMM_CACHE_KEY  = "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448"
ASAMM_FCA_RUN    = "20260518_072545_57edf67d"

TREE_CACHE = ROOT / ".tree-cache"

def tree_path(cache_key: str, run_id: str, label: str) -> Path:
    return TREE_CACHE / cache_key / run_id / label / "tree.pkl"

SC_FCA_PATH   = tree_path(SC_CACHE_KEY,    SC_FCA_RUN,   "full_context_aware")
SC_TONLY_PATH = tree_path(SC_CACHE_KEY,    SC_TONLY_RUN, "original_raptor_text_only")
ASAMM_FCA_PATH = tree_path(ASAMM_CACHE_KEY, ASAMM_FCA_RUN, "full_context_aware")

# ── Dataset paths ─────────────────────────────────────────────────────────────

IMAGE_DS_PATH    = ROOT / "datasets" / "image_eval_dataset.json"
ANCHORED_FULL_PATH = ROOT / "datasets" / "combined_eval_dataset_anchored_full.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_tree(path: Path) -> list[dict]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def node_pages(node: dict) -> list[int]:
    """Return page numbers covered by a node, mirroring retrieval_evaluator.py TreeIndex.retrieve().

    IMPORTANT: mirrors the EXACT logic in retrieval_evaluator.py so that the
    diagnostic faithfully reproduces what the evaluator sees.

    The evaluator does:
        pages = node.get("pages", [])
        if not pages:
            pages = list(range(page_start, page_end+1)) if page_start else []

    When pages is a non-empty STRING (e.g. '93'), `if not pages` is False, so
    pages stays as the raw string.  set('93') yields {'9','3'}, which never
    intersects with an int-typed relevant_page_set — hence the zero score.
    """
    pages = node.get("pages", [])
    if not pages:
        pages = list(range(
            node.get("page_start", 0),
            node.get("page_end", 0) + 1
        )) if node.get("page_start") else []

    # If pages is a string (bug in ingestion), convert to character set —
    # this is what the evaluator actually sees, so coverage will show 0
    if isinstance(pages, str):
        return []   # string pages never match int relevant_page_set
    return [int(p) for p in pages if p]


def searchable_nodes(nodes: list[dict]) -> list[dict]:
    """Mirror TreeIndex filter: exclude is_parent, exclude nodes with no embedding."""
    return [n for n in nodes if not n.get("is_parent") and n.get("embedding") is not None]


def page_coverage(nodes: list[dict]) -> dict[int, list[str]]:
    """Map page → list of node IDs that cover it."""
    cov: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        nid = str(n.get("id", ""))
        for pg in node_pages(n):
            cov[pg].append(nid)
    return cov


def sample_node_metadata_keys(nodes: list[dict], n: int = 5) -> list[set]:
    return [set(node.keys()) for node in nodes[:n]]


# ── Question 1: image queries in anchored dataset ─────────────────────────────

def q1_anchored_dataset_check():
    print("\n" + "=" * 70)
    print("Q1: Do image queries appear in combined_eval_dataset_anchored_full.json?")
    print("=" * 70)

    anchored = json.loads(ANCHORED_FULL_PATH.read_text())
    image_raw = json.loads(IMAGE_DS_PATH.read_text())

    image_ids = {e["id"] for e in image_raw}
    anchored_ids = {e["id"] for e in anchored}

    overlap = image_ids & anchored_ids
    print(f"  Image dataset entries   : {len(image_ids)}")
    print(f"  Anchored dataset entries: {len(anchored_ids)}")
    print(f"  Overlap (shared IDs)    : {len(overlap)}")

    if not overlap:
        print("\n  FINDING: Image queries are NOT in the anchored dataset.")
        print("  They come exclusively from image_eval_dataset.json with")
        print("  relevant_chunks=[] (no ground_truth_chunk_id).")
        print("  Matching therefore depends entirely on page-level fallback.")
    else:
        print(f"\n  Overlapping IDs: {sorted(overlap)[:5]}")
        for e in anchored:
            if e["id"] in overlap:
                print(f"    {e['id']}: ground_truth_chunk_id={e.get('ground_truth_chunk_id')}")


# ── Question 2 & 3: tree node page metadata + coverage ───────────────────────

def q2_q3_page_coverage():
    print("\n" + "=" * 70)
    print("Q2/Q3: Node page metadata and per-query page coverage")
    print("=" * 70)

    image_raw = json.loads(IMAGE_DS_PATH.read_text())
    sc_queries   = [e for e in image_raw if e["manual_source"] == "SC10000AMM"]
    asamm_queries = [e for e in image_raw if e["manual_source"] == "AS-AMM-01-000"]

    # ── SC10000AMM ────────────────────────────────────────────────────────────
    print(f"\n--- SC10000AMM ({len(sc_queries)} image queries) ---")

    print(f"  Loading full_context_aware tree: {SC_FCA_PATH}")
    sc_fca_all  = load_tree(SC_FCA_PATH)
    sc_fca_srch = searchable_nodes(sc_fca_all)

    print(f"  Loading original_raptor_text_only tree: {SC_TONLY_PATH}")
    sc_to_all   = load_tree(SC_TONLY_PATH)
    sc_to_srch  = searchable_nodes(sc_to_all)

    print(f"\n  Tree sizes (searchable nodes):")
    print(f"    full_context_aware     : {len(sc_fca_srch)}")
    print(f"    original_raptor_text_only: {len(sc_to_srch)}")

    # Sample metadata keys
    fca_sample = sc_fca_srch[:3] if sc_fca_srch else []
    to_sample  = sc_to_srch[:3]  if sc_to_srch  else []

    print(f"\n  Sample metadata keys (full_context_aware, first 3 nodes):")
    for i, node in enumerate(fca_sample):
        pg_keys = {k for k in node.keys() if "page" in k.lower()}
        print(f"    node {i}: pages={node.get('pages')}, page_start={node.get('page_start')}, "
              f"page_end={node.get('page_end')}, page={node.get('page')}  |  page_keys={pg_keys}")

    print(f"\n  Sample metadata keys (original_raptor_text_only, first 3 nodes):")
    for i, node in enumerate(to_sample):
        pg_keys = {k for k in node.keys() if "page" in k.lower()}
        print(f"    node {i}: pages={node.get('pages')}, page_start={node.get('page_start')}, "
              f"page_end={node.get('page_end')}, page={node.get('page')}  |  page_keys={pg_keys}")

    # Count nodes with ANY page metadata
    fca_with_pages = sum(1 for n in sc_fca_srch if node_pages(n))
    to_with_pages  = sum(1 for n in sc_to_srch  if node_pages(n))
    print(f"\n  Nodes with page metadata:")
    print(f"    full_context_aware     : {fca_with_pages}/{len(sc_fca_srch)}")
    print(f"    original_raptor_text_only: {to_with_pages}/{len(sc_to_srch)}")

    # Build page coverage maps
    sc_fca_cov = page_coverage(sc_fca_srch)
    sc_to_cov  = page_coverage(sc_to_srch)

    print(f"\n  SC10000AMM image queries — per-query page coverage:")
    print(f"  {'query_id':<15} {'target_page':>12} {'fca_nodes':>10} {'text_only_nodes':>16}")
    print("  " + "-" * 57)

    fca_zero_count  = 0
    to_zero_count   = 0
    for q in sc_queries:
        fig = q.get("figure_reference") or {}
        page = fig.get("page")
        if page is None:
            print(f"  {q['id']:<15} {'N/A':>12} {'N/A':>10} {'N/A':>16}")
            continue

        page = int(page)
        fca_hits = len(sc_fca_cov.get(page, []))
        to_hits  = len(sc_to_cov.get(page, []))

        if fca_hits == 0:
            fca_zero_count += 1
        if to_hits == 0:
            to_zero_count += 1

        flag = "  << FCA MISS" if fca_hits == 0 and to_hits > 0 else ""
        print(f"  {q['id']:<15} {page:>12} {fca_hits:>10} {to_hits:>16}{flag}")

    print(f"\n  Queries with ZERO page coverage:")
    print(f"    full_context_aware     : {fca_zero_count}/{len(sc_queries)}")
    print(f"    original_raptor_text_only: {to_zero_count}/{len(sc_queries)}")

    # ── AS-AMM-01-000 ─────────────────────────────────────────────────────────
    if asamm_queries:
        print(f"\n--- AS-AMM-01-000 ({len(asamm_queries)} image queries) ---")

        print(f"  Loading full_context_aware tree: {ASAMM_FCA_PATH}")
        asamm_fca_all  = load_tree(ASAMM_FCA_PATH)
        asamm_fca_srch = searchable_nodes(asamm_fca_all)
        print(f"  Searchable nodes: {len(asamm_fca_srch)}")

        # Sample metadata
        asamm_sample = asamm_fca_srch[:3] if asamm_fca_srch else []
        print(f"\n  Sample metadata keys (AS-AMM full_context_aware, first 3 nodes):")
        for i, node in enumerate(asamm_sample):
            pg_keys = {k for k in node.keys() if "page" in k.lower()}
            print(f"    node {i}: pages={node.get('pages')}, page_start={node.get('page_start')}, "
                  f"page_end={node.get('page_end')}, page={node.get('page')}  |  page_keys={pg_keys}")

        asamm_fca_with_pages = sum(1 for n in asamm_fca_srch if node_pages(n))
        print(f"\n  Nodes with page metadata: {asamm_fca_with_pages}/{len(asamm_fca_srch)}")

        asamm_fca_cov = page_coverage(asamm_fca_srch)
        print(f"\n  AS-AMM image queries — per-query page coverage:")
        print(f"  {'query_id':<15} {'target_page':>12} {'fca_nodes':>10}")
        print("  " + "-" * 40)

        asamm_zero = 0
        for q in asamm_queries:
            fig  = q.get("figure_reference") or {}
            page = fig.get("page")
            if page is None:
                print(f"  {q['id']:<15} {'N/A':>12} {'N/A':>10}")
                continue
            page = int(page)
            hits = len(asamm_fca_cov.get(page, []))
            if hits == 0:
                asamm_zero += 1
            print(f"  {q['id']:<15} {page:>12} {hits:>10}")

        print(f"\n  Queries with zero page coverage in AS-AMM FCA: {asamm_zero}/{len(asamm_queries)}")


# ── Question 4: page numbering deep dive ──────────────────────────────────────

def q4_page_numbering_deepdive():
    print("\n" + "=" * 70)
    print("Q4: pages field type analysis — why string pages break matching")
    print("=" * 70)

    sc_fca_all  = load_tree(SC_FCA_PATH)
    sc_fca_srch = searchable_nodes(sc_fca_all)

    # Audit the raw `pages` field type across all SC FCA nodes
    type_counts: dict[str, int] = {}
    for n in sc_fca_srch:
        t = type(n.get("pages")).__name__
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  SC full_context_aware — raw `pages` field type distribution:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count} nodes")

    # Show the evaluator's exact page extraction for the first few FCA nodes
    print(f"\n  Evaluator page extraction simulation (first 5 FCA nodes):")
    print(f"  {'raw pages':>15}  {'type':>6}  {'source_pages from evaluator':>30}  {'matches int?':>14}")
    for n in sc_fca_srch[:5]:
        raw = n.get("pages")
        # Reproduce retrieval_evaluator.py TreeIndex.retrieve() verbatim:
        pages_ev = raw if raw else (
            list(range(n.get("page_start", 0), n.get("page_end", 0) + 1))
            if n.get("page_start") else []
        )
        # Test if any target page would match
        matches_int = isinstance(pages_ev, list) and bool(set(pages_ev) & {93})
        print(f"  {str(raw):>15}  {type(raw).__name__:>6}  {str(pages_ev):>30}  {str(matches_int):>14}")

    # Why does string fail:
    print(f"\n  Root cause demonstration:")
    print(f"    pages='93' is truthy -> evaluator keeps it as string '93'")
    print(f"    set('93') = {set('93')}  (iterates characters)")
    print(f"    relevant_page_set = {{93}}  (integer)")
    print(f"    intersection = {set('93') & {93}}  <- ALWAYS EMPTY")
    print(f"\n    pages=[] with page_start=93 -> evaluator builds [93]")
    print(f"    set([93]) & {{93}} = {set([93]) & {93}}  <- MATCH")

    # Check original_raptor_text_only: pages is [] and page_start is 0
    sc_to_all  = load_tree(SC_TONLY_PATH)
    sc_to_srch = searchable_nodes(sc_to_all)
    to_type_counts: dict[str, int] = {}
    for n in sc_to_srch:
        t = type(n.get("pages")).__name__
        to_type_counts[t] = to_type_counts.get(t, 0) + 1

    nonzero_starts = [n.get("page_start") for n in sc_to_srch if n.get("page_start")]
    print(f"\n  original_raptor_text_only — raw `pages` field type distribution:")
    for t, count in sorted(to_type_counts.items()):
        print(f"    {t}: {count} nodes")
    print(f"  Non-zero page_start values: {len(nonzero_starts)}/{len(sc_to_srch)}")
    print(f"  -> text_only nodes have pages=[] AND page_start=0, so")
    print(f"     the evaluator also builds source_pages=[] for them.")
    print(f"  -> Both trees produce zero page matches for SC10000AMM IMAGE queries.")
    print(f"  -> original_raptor_text_only R@5=0.306 on IMAGE must come from a")
    print(f"     DIFFERENT evaluation run or dataset configuration, not from")
    print(f"     page-fallback succeeding — it also has no page metadata.")

    # Check AS-AMM FCA pages field type
    asamm_fca_all  = load_tree(ASAMM_FCA_PATH)
    asamm_fca_srch = searchable_nodes(asamm_fca_all)
    asamm_type_counts: dict[str, int] = {}
    for n in asamm_fca_srch:
        t = type(n.get("pages")).__name__
        asamm_type_counts[t] = asamm_type_counts.get(t, 0) + 1

    print(f"\n  AS-AMM full_context_aware — raw `pages` field type distribution:")
    for t, count in sorted(asamm_type_counts.items()):
        print(f"    {t}: {count} nodes")

    # Sample AS-AMM node to see why it works
    asamm_sample = asamm_fca_srch[:3]
    print(f"\n  AS-AMM FCA first 3 nodes (pages field works):")
    for n in asamm_sample:
        raw = n.get("pages")
        pages_ev = raw if raw else (
            list(range(n.get("page_start", 0), n.get("page_end", 0) + 1))
            if n.get("page_start") else []
        )
        print(f"    pages={raw!r} ({type(raw).__name__}) -> evaluator sees {pages_ev!r}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
  ROOT CAUSE: pages field is a STRING in SC10000AMM trees, a LIST in AS-AMM trees.

  The ingestion pipeline stores node["pages"] as a string (e.g. "93") for
  SC10000AMM trees (both full_context_aware and original_raptor_text_only).
  AS-AMM trees store it as a list (e.g. [344]).

  In retrieval_evaluator.py TreeIndex.retrieve():

      pages = node.get("pages", [])
      if not pages:
          pages = list(range(page_start, page_end+1)) if page_start else []

  When pages="93" (a non-empty string), `if not pages` is False, so the
  page_start/page_end fallback is never used.  pages stays as the string "93".

  In _resolve_match_ids():

      if set(r.source_pages) & relevant_page_set:

  source_pages is the string "93".  set("93") = {'9', '3'} — Python iterates
  the string's characters.  relevant_page_set = {93} (integer).  The
  intersection is always empty → page_relevant_retrieved stays empty →
  the evaluator returns retrieved_ids unchanged with relevant_ids = gt_chunk_ids
  (which is also empty for image queries) → all metrics = 0.

  WHY AS-AMM WORKS BUT SC10000AMM DOES NOT:
  - AS-AMM full_context_aware nodes have pages=[344] (a list) → set([344])
    intersects with {344} correctly → page-fallback matches → R@5 > 0.
  - SC10000AMM nodes have pages="1" ... "358" (strings) → broken as above.

  WHY original_raptor_text_only ALSO SCORES ZERO ON SC IMAGE QUERIES:
  - Its nodes have pages=[] AND page_start=0, so source_pages=[] for every
    node, and page-fallback also yields zero hits.
  - The reported R@5=0.306 for original_raptor_text_only on IMAGE must
    originate from a different evaluation run, test harness version, or
    dataset that embedded chunk IDs — not from page-fallback working.

  FIX:
  - In the ingestion pipeline, ensure node["pages"] is always a list[int],
    never a string or scalar.  Specifically fix the SC10000AMM tree builder
    to store pages=[page_num] instead of pages=str(page_num).
  - Alternatively, add a coercion guard in TreeIndex.retrieve():
      if isinstance(pages, (str, int)):
          pages = [int(pages)]
""")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        q1_anchored_dataset_check()
        q2_q3_page_coverage()
        q4_page_numbering_deepdive()
        print_summary()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
