"""Fix image query ground truth by locating figures in tree nodes and
re-anchoring image queries to real tree chunk IDs.

Approach:
  1. For each image query, search tree nodes for the figure label text
     (e.g., "Figure 6.3.15.1") to find the correct Docling page
  2. Verify/supplement with PyMuPDF text extraction from source PDFs
  3. Re-anchor per ablation tree (chunk IDs differ across ablations)

Reads from:
  - .tree-cache/  (tree pickles — local only, no GCS)
  - datasets/image_eval_dataset.json
  - datasets/*.pdf (source PDFs for PyMuPDF verification)

Writes to (all new files — nothing existing is touched):
  - data/page_mapping_{manual_name}.json
  - data/image_eval_dataset_fixed.json
  - data/image_eval_gt_per_ablation.json
  - data/anchoring_report.json

Usage:
    python -m backend.evaluation.fix_image_query_annotations
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import fitz  # PyMuPDF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TREE_CACHE_DIR = PROJECT_ROOT / ".tree-cache"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "data"

DOC_MAP = {
    "AS-AMM-01-000": {
        "cache_key": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
        "pdf_filename": "AS AMM 01 000 I1 R1 Feb 2 2018_compressed.pdf",
    },
    "SC10000AMM": {
        "cache_key": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
        "pdf_filename": "SC10000AMM Rev J.pdf",
    },
}

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


# ---------------------------------------------------------------------------
# Tree loading (local only — no GCS)
# ---------------------------------------------------------------------------


def load_tree_local(cache_key: str, ablation_label: str) -> tuple[list[dict], dict]:
    """Load tree pickle from local .tree-cache/ without resolve_tree/GCS."""
    cache_dir = TREE_CACHE_DIR / cache_key
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    candidates: list[tuple[str, Path, dict]] = []
    for run_dir in cache_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "docling-cache":
            continue
        manifest_path = run_dir / ablation_label / "tree_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        created_at = manifest.get("created_at", "")
        candidates.append((created_at, run_dir, manifest))

    if not candidates:
        raise FileNotFoundError(
            f"No tree found for cache_key={cache_key[:12]}..., "
            f"ablation={ablation_label}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_run_dir, manifest = candidates[0]

    tree_path = best_run_dir / ablation_label / "tree.pkl"
    if not tree_path.exists():
        raise FileNotFoundError(f"tree.pkl not found at {tree_path}")

    nodes = pickle.loads(tree_path.read_bytes())
    logger.info(
        f"  Loaded {ablation_label} from {best_run_dir.name}: "
        f"{len(nodes)} nodes"
    )
    return nodes, manifest


# ---------------------------------------------------------------------------
# Node utility functions
# ---------------------------------------------------------------------------


def _extract_page_from_node(node: dict) -> int | None:
    """Extract Docling page number from a leaf node."""
    pages = node.get("pages")
    if isinstance(pages, list) and pages:
        return int(pages[0])
    if isinstance(pages, (int, float)) and pages > 0:
        return int(pages)
    if isinstance(pages, str) and pages.strip().isdigit():
        return int(pages.strip())

    ps = node.get("page_start")
    if ps is not None and ps > 0:
        return int(ps)

    text = node.get("text", "")
    m = re.match(r"^\[.*?\|\s*Page\s+(\d+)\]", text)
    if m:
        return int(m.group(1))
    return None


def _strip_header_prefix(text: str) -> str:
    """Strip the [Section | Page N] prefix from node text."""
    m = re.match(r"^\[.*?\]\s*\n*", text)
    return text[m.end():] if m else text


def _get_leaf_nodes(nodes: list[dict]) -> list[dict]:
    """Filter to level-0 non-parent nodes."""
    return [
        n for n in nodes
        if n.get("level", 0) == 0 and not n.get("is_parent")
    ]


def _get_nodes_on_page(nodes: list[dict], docling_page: int) -> list[dict]:
    """Find level-0 non-parent nodes on a specific Docling page."""
    return [
        n for n in _get_leaf_nodes(nodes)
        if _extract_page_from_node(n) == docling_page
    ]


def _node_has_images(node: dict) -> bool:
    """Check if a node has image content (from caption folding)."""
    images = node.get("images")
    if isinstance(images, list) and len(images) > 0:
        return True
    return "Diagrams on this page:" in node.get("text", "")


def _extract_figure_description(node: dict) -> str:
    """Extract descriptive text snippet from the node."""
    text = _strip_header_prefix(node.get("text", ""))
    marker_idx = text.find("Diagrams on this page:")
    desc = text[:marker_idx].strip() if marker_idx > 0 else text.strip()
    return desc[:500] + "..." if len(desc) > 500 else desc


def _extract_section_number(section_system: str) -> str | None:
    m = re.search(r"\((Ch\s*)?(\d[\d.]*)\)", section_system)
    if m:
        return m.group(2)
    m = re.search(r"\d[\d.]*", section_system)
    return m.group() if m else None


def _chunk_matches_section(chunk: dict, section_prefix: str) -> bool:
    chunk_header = chunk.get("section_header") or ""
    if section_prefix in chunk_header:
        return True
    text = chunk.get("text", "")
    m = re.match(r"^\[([^|]+)", text)
    if m and section_prefix in m.group(1).strip():
        return True
    return False


def _fault_code_overlap_score(fault_code: str, node_text: str) -> float:
    fault_lower = fault_code.lower().strip()
    text_lower = node_text.lower()
    if not fault_lower or fault_lower == "none":
        return 0.0
    if fault_lower in text_lower:
        return 100.0
    fault_stripped = re.sub(r"\s*\([^)]*\)\s*", " ", fault_lower).strip()
    if fault_stripped != fault_lower and fault_stripped in text_lower:
        return 90.0
    fault_words = set(re.findall(r"\w+", fault_lower))
    text_words = set(re.findall(r"\w+", text_lower))
    if fault_words:
        return len(fault_words & text_words) / len(fault_words) * 50.0
    return 0.0


# ---------------------------------------------------------------------------
# Step 1: Figure-label-first page discovery
# ---------------------------------------------------------------------------


def _normalize_figure_label(label: str) -> str:
    """Normalize figure label for flexible matching.

    "Fig.  28-1" → "fig 28-1"
    "Figure 6.3.15.1" → "figure 6.3.15.1"
    """
    return re.sub(r"\s+", " ", label.strip().lower())


def _find_figure_in_nodes(
    figure_label: str,
    nodes: list[dict],
) -> list[dict]:
    """Search all nodes for ones containing the figure label text."""
    normalized = _normalize_figure_label(figure_label)
    matches = []
    for node in _get_leaf_nodes(nodes):
        node_text_normalized = re.sub(r"\s+", " ", node.get("text", "").lower())
        if normalized in node_text_normalized:
            matches.append(node)
    return matches


def _find_figure_in_pdf(
    figure_label: str,
    pdf_path: Path,
) -> list[int]:
    """Search PDF pages for the figure label text. Returns matching page numbers (1-indexed)."""
    normalized = _normalize_figure_label(figure_label)
    doc = fitz.open(str(pdf_path))
    matching_pages = []
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text("text")
        text_normalized = re.sub(r"\s+", " ", text.lower())
        if normalized in text_normalized:
            matching_pages.append(page_idx + 1)  # 1-indexed
    doc.close()
    return matching_pages


def build_page_mapping_from_queries(
    image_queries: list[dict],
    trees: dict[str, list[dict]],
) -> tuple[dict[str, dict[int, int]], dict[str, dict]]:
    """Build PDF→Docling page mapping by locating figure labels in both
    tree nodes and PDFs.

    Returns:
        page_mappings: {manual_name: {pdf_page: docling_page}}
        stats: {manual_name: mapping statistics}
    """
    page_mappings: dict[str, dict[int, int]] = defaultdict(dict)
    match_details: dict[str, list[dict]] = defaultdict(list)

    for query in image_queries:
        manual = query["manual_source"]
        fig_ref = query.get("figure_reference", {})
        figure_label = query.get("ground_truth_figure", "")
        pdf_page = fig_ref.get("page")
        nodes = trees.get(manual, [])

        if not figure_label or pdf_page is None:
            continue

        # Find figure in tree nodes → get Docling page
        node_matches = _find_figure_in_nodes(figure_label, nodes)

        if node_matches:
            # Get the Docling page from the matched node
            # Prefer nodes with images
            image_matches = [n for n in node_matches if _node_has_images(n)]
            best_match = image_matches[0] if image_matches else node_matches[0]
            docling_page = _extract_page_from_node(best_match)

            if docling_page is not None:
                page_mappings[manual][pdf_page] = docling_page
                match_details[manual].append({
                    "query_id": query["id"],
                    "figure_label": figure_label,
                    "pdf_page": pdf_page,
                    "docling_page": docling_page,
                    "offset": pdf_page - docling_page,
                    "method": "figure_label_in_tree",
                    "has_images": bool(image_matches),
                })
                continue

        # Fallback: try identity mapping (offset 0) — check if PDF page
        # exists in tree and has relevant content
        identity_nodes = _get_nodes_on_page(nodes, pdf_page)
        if identity_nodes:
            page_mappings[manual][pdf_page] = pdf_page
            match_details[manual].append({
                "query_id": query["id"],
                "figure_label": figure_label,
                "pdf_page": pdf_page,
                "docling_page": pdf_page,
                "offset": 0,
                "method": "identity_fallback",
                "has_images": any(_node_has_images(n) for n in identity_nodes),
            })
            continue

        # Secondary fallback: search PDF for figure label to find the true
        # PDF page, then try matching with tree nodes on nearby pages
        pdf_path = DATASETS_DIR / DOC_MAP[manual]["pdf_filename"]
        if pdf_path.exists():
            pdf_pages_found = _find_figure_in_pdf(figure_label, pdf_path)
            if pdf_pages_found:
                for found_pdf_page in pdf_pages_found:
                    # The annotated pdf_page and found_pdf_page should match
                    # Try found_pdf_page as Docling page too (identity)
                    found_nodes = _get_nodes_on_page(nodes, found_pdf_page)
                    if found_nodes:
                        page_mappings[manual][pdf_page] = found_pdf_page
                        match_details[manual].append({
                            "query_id": query["id"],
                            "figure_label": figure_label,
                            "pdf_page": pdf_page,
                            "docling_page": found_pdf_page,
                            "offset": pdf_page - found_pdf_page,
                            "method": "pdf_search_identity",
                            "has_images": any(_node_has_images(n) for n in found_nodes),
                        })
                        break

        # If still not mapped, note as unmapped
        if pdf_page not in page_mappings.get(manual, {}):
            match_details[manual].append({
                "query_id": query["id"],
                "figure_label": figure_label,
                "pdf_page": pdf_page,
                "docling_page": None,
                "offset": None,
                "method": "unmatched",
                "has_images": False,
            })

    # Compute statistics
    all_stats: dict[str, dict] = {}
    for manual in DOC_MAP:
        details = match_details.get(manual, [])
        offsets = [d["offset"] for d in details if d["offset"] is not None]
        offset_counter = Counter(offsets)
        most_common = offset_counter.most_common(1)[0] if offsets else (0, 0)
        n_matched = sum(1 for d in details if d["docling_page"] is not None)
        n_unmatched = sum(1 for d in details if d["docling_page"] is None)
        method_counts = Counter(d["method"] for d in details)

        all_stats[manual] = {
            "total_queries": len(details),
            "matched": n_matched,
            "unmatched": n_unmatched,
            "most_common_offset": most_common[0],
            "offset_is_constant": len(offset_counter) <= 1 and n_matched > 0,
            "offset_distribution": dict(offset_counter),
            "method_distribution": dict(method_counts),
            "details": details,
        }

    return dict(page_mappings), all_stats


# ---------------------------------------------------------------------------
# Step 2: Enrich image annotations
# ---------------------------------------------------------------------------


def enrich_image_annotations(
    image_queries: list[dict],
    page_mappings: dict[str, dict[int, int]],
    trees: dict[str, list[dict]],
) -> list[dict]:
    """Enrich image queries with corrected page numbers and chunk IDs."""
    enriched = []

    for query in image_queries:
        manual = query["manual_source"]
        fig_ref = query.get("figure_reference", {})
        pdf_page = fig_ref.get("page")
        figure_label = query.get("ground_truth_figure", "")
        fault_code = query.get("ground_truth_fault_code", "")
        nodes = trees.get(manual, [])
        mapping = page_mappings.get(manual, {})

        # Resolve Docling page
        docling_page = mapping.get(pdf_page)

        # Primary: find node via figure label (most reliable)
        best_node = None
        match_reason = ""

        if figure_label and nodes:
            fig_matches = _find_figure_in_nodes(figure_label, nodes)
            if fig_matches:
                # Prefer matches with images
                img_matches = [n for n in fig_matches if _node_has_images(n)]
                if img_matches:
                    best_node = img_matches[0]
                    match_reason = "figure_label+image"
                else:
                    best_node = fig_matches[0]
                    match_reason = "figure_label"

                # Update docling_page from the matched node
                node_page = _extract_page_from_node(best_node)
                if node_page is not None:
                    docling_page = node_page

        # Secondary: page-based search with image preference
        if best_node is None and docling_page is not None:
            page_nodes = _get_nodes_on_page(nodes, docling_page)
            if page_nodes:
                img_page_nodes = [n for n in page_nodes if _node_has_images(n)]
                if img_page_nodes:
                    best_node = img_page_nodes[0]
                    match_reason = "page+image"
                else:
                    # Score by fault code overlap
                    scored = [
                        (n, _fault_code_overlap_score(fault_code, n.get("text", "")))
                        for n in page_nodes
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    best_node = scored[0][0]
                    match_reason = "page_only"

        # Tertiary: fault code text match across all nodes
        if best_node is None and fault_code and nodes:
            fault_lower = fault_code.lower().strip()
            if fault_lower and fault_lower != "none":
                text_hits = [
                    n for n in _get_leaf_nodes(nodes)
                    if fault_lower in (n.get("text") or "").lower()
                ]
                if text_hits:
                    best_node = text_hits[0]
                    match_reason = "fault_code_global"
                    if docling_page is None:
                        docling_page = _extract_page_from_node(best_node)

        # Determine confidence
        if best_node is None:
            confidence = "unmatched"
            chunk_id = None
            fig_desc = ""
        elif "figure_label" in match_reason:
            confidence = "high"
            chunk_id = best_node.get("id")
            fig_desc = _extract_figure_description(best_node)
        elif "image" in match_reason:
            confidence = "high"
            chunk_id = best_node.get("id")
            fig_desc = _extract_figure_description(best_node)
        else:
            confidence = "medium"
            chunk_id = best_node.get("id")
            fig_desc = _extract_figure_description(best_node)

        enriched.append({
            "id": query["id"],
            "observation": query["observation"],
            "query_type": "image",
            "ground_truth_chunk_id": chunk_id,
            "page_reference": docling_page,
            "page_reference_original": pdf_page,
            "figure_description": fig_desc,
            "confidence": confidence,
            "manual_source": manual,
            "ground_truth_fault_code": fault_code,
            "ground_truth_figure": query.get("ground_truth_figure", ""),
            "section_system": query.get("section_system", ""),
            "difficulty_tier": query.get("difficulty_tier", ""),
            "figure_reference": fig_ref,
            "_match_reason": match_reason,
        })

    return enriched


# ---------------------------------------------------------------------------
# Step 3: Per-ablation re-anchoring
# ---------------------------------------------------------------------------


def anchor_image_query_to_ablation(
    query: dict,
    nodes: list[dict],
    has_caption_folding: bool,
) -> dict:
    """Anchor one image query to the best matching node in an ablation tree."""
    result = {
        "matched_chunk_id": None,
        "match_score": 0.0,
        "confidence": "none",
        "match_method": None,
        "candidates_count": 0,
    }

    figure_label = query.get("ground_truth_figure", "")
    fault_code = query.get("ground_truth_fault_code", "")
    docling_page = query.get("page_reference")
    section_system = query.get("section_system", "")
    sec_code = _extract_section_number(section_system)
    leaf_nodes = _get_leaf_nodes(nodes)

    # Strategy 1: Figure label in node text (strongest signal)
    if figure_label:
        fig_matches = _find_figure_in_nodes(figure_label, nodes)
        if fig_matches:
            img_matches = [n for n in fig_matches if _node_has_images(n)]
            best = img_matches[0] if (has_caption_folding and img_matches) else fig_matches[0]
            result.update({
                "matched_chunk_id": best.get("id"),
                "match_score": 200.0,
                "confidence": "high",
                "match_method": "figure_label" + ("+image" if img_matches else ""),
                "candidates_count": len(fig_matches),
            })
            return result

    # Strategy 2: Page match + scoring cascade
    if docling_page is not None:
        page_candidates = _get_nodes_on_page(nodes, docling_page)
        if page_candidates:
            scored = []
            for node in page_candidates:
                score = 10.0  # Base score for page match
                parts = ["page"]

                if has_caption_folding and _node_has_images(node):
                    score += 50.0
                    parts.append("image")

                fc_score = _fault_code_overlap_score(fault_code, node.get("text", ""))
                score += fc_score
                if fc_score >= 90:
                    parts.append("fault")

                if sec_code and _chunk_matches_section(node, sec_code):
                    score += 30.0
                    parts.append("section")

                scored.append((node, score, "+".join(parts)))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_node, best_score, method = scored[0]

            confidence = "high" if best_score >= 60 else "medium"
            result.update({
                "matched_chunk_id": best_node.get("id"),
                "match_score": round(best_score, 1),
                "confidence": confidence,
                "match_method": method,
                "candidates_count": len(page_candidates),
            })
            return result

    # Strategy 3: Fault code text match (no page)
    fault_lower = fault_code.lower().strip()
    if fault_lower and fault_lower != "none":
        text_matches = [
            n for n in leaf_nodes
            if fault_lower in (n.get("text") or "").lower()
        ]
        if text_matches:
            best = text_matches[0]
            if sec_code:
                sec_filtered = [n for n in text_matches if _chunk_matches_section(n, sec_code)]
                if sec_filtered:
                    best = sec_filtered[0]

            result.update({
                "matched_chunk_id": best.get("id"),
                "match_score": 100.0,
                "confidence": "medium",
                "match_method": "fault_code_no_page",
                "candidates_count": len(text_matches),
            })
            return result

    # Strategy 4: Section-only
    if sec_code:
        sec_matches = [n for n in leaf_nodes if _chunk_matches_section(n, sec_code)]
        if sec_matches:
            result.update({
                "matched_chunk_id": sec_matches[0].get("id"),
                "match_score": 30.0,
                "confidence": "low",
                "match_method": "section_only",
                "candidates_count": len(sec_matches),
            })
            return result

    result["match_method"] = "unmatched"
    return result


def reanchor_per_ablation(
    fixed_queries: list[dict],
) -> dict[str, dict[str, str | None]]:
    """Re-anchor all image queries across all ablation trees."""
    gt_per_ablation: dict[str, dict[str, str | None]] = {}

    caption_folding_labels = {
        "full_context_aware", "no_table_parent_child",
        "no_header_propagation", "flat_no_raptor",
    }

    for ablation_key, ablation_label in ABLATION_KEY_TO_LABEL.items():
        has_cf = ablation_label in caption_folding_labels
        confidence_counts = Counter()

        logger.info(f"\nAnchoring to {ablation_label} (caption_folding={has_cf})")

        trees_by_manual: dict[str, list[dict]] = {}
        for manual, doc_info in DOC_MAP.items():
            try:
                nodes, _ = load_tree_local(doc_info["cache_key"], ablation_label)
                trees_by_manual[manual] = nodes
            except FileNotFoundError:
                logger.warning(f"  No tree for {manual}/{ablation_label}")
                trees_by_manual[manual] = []

        for query in fixed_queries:
            qid = query["id"]
            manual = query["manual_source"]
            nodes = trees_by_manual.get(manual, [])

            if qid not in gt_per_ablation:
                gt_per_ablation[qid] = {}

            if not nodes:
                gt_per_ablation[qid][ablation_label] = None
                confidence_counts["none"] += 1
                continue

            result = anchor_image_query_to_ablation(query, nodes, has_cf)
            gt_per_ablation[qid][ablation_label] = result["matched_chunk_id"]
            confidence_counts[result["confidence"]] += 1

        logger.info(
            f"  Confidence: high={confidence_counts.get('high', 0)}, "
            f"medium={confidence_counts.get('medium', 0)}, "
            f"low={confidence_counts.get('low', 0)}, "
            f"none={confidence_counts.get('none', 0)}"
        )

    return gt_per_ablation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 72)
    logger.info("Fix Image Query Annotations")
    logger.info("=" * 72)

    # Load image dataset
    image_dataset_path = DATASETS_DIR / "image_eval_dataset.json"
    image_queries = json.loads(image_dataset_path.read_text())
    logger.info(f"Loaded {len(image_queries)} image queries")

    by_source = Counter(q["manual_source"] for q in image_queries)
    for source, count in by_source.items():
        logger.info(f"  {source}: {count} queries")

    # -----------------------------------------------------------------------
    # Step 1: Build page mappings via figure label search
    # -----------------------------------------------------------------------
    logger.info("\n[Step 1] Building page mappings via figure label search...")

    trees: dict[str, list[dict]] = {}
    for manual_name, doc_info in DOC_MAP.items():
        nodes, _ = load_tree_local(doc_info["cache_key"], "full_context_aware")
        trees[manual_name] = nodes

    page_mappings, mapping_stats = build_page_mapping_from_queries(
        image_queries, trees,
    )

    # Save page mappings
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for manual_name in DOC_MAP:
        mapping = page_mappings.get(manual_name, {})
        stats = mapping_stats.get(manual_name, {})
        out_path = DATA_DIR / f"page_mapping_{manual_name}.json"
        out_data = {
            "manual_name": manual_name,
            "mapping": {str(k): v for k, v in mapping.items()},
            "stats": {k: v for k, v in stats.items() if k != "details"},
            "details": stats.get("details", []),
        }
        out_path.write_text(json.dumps(out_data, indent=2))
        logger.info(f"  Saved {out_path}")

    # Print mapping summary
    print("\n" + "=" * 72)
    print("PAGE MAPPING STATISTICS")
    print("=" * 72)
    for manual_name, stats in mapping_stats.items():
        print(f"\n  {manual_name}:")
        print(f"    Queries: {stats['total_queries']}")
        print(f"    Matched: {stats['matched']}")
        print(f"    Unmatched: {stats['unmatched']}")
        if stats["offset_is_constant"]:
            print(f"    Offset: constant = {stats['most_common_offset']}")
            print(f"    Formula: Docling page = PDF page - {stats['most_common_offset']}")
        else:
            print(f"    Offset distribution: {stats['offset_distribution']}")
        print(f"    Methods: {stats['method_distribution']}")

    # -----------------------------------------------------------------------
    # Step 2: Enrich annotations
    # -----------------------------------------------------------------------
    logger.info("\n[Step 2] Enriching image annotations...")

    fixed_queries = enrich_image_annotations(image_queries, page_mappings, trees)

    fixed_path = DATA_DIR / "image_eval_dataset_fixed.json"
    fixed_path.write_text(json.dumps(fixed_queries, indent=2))
    logger.info(f"Saved fixed dataset to {fixed_path}")

    # Anchoring summary
    conf_counts = Counter(q["confidence"] for q in fixed_queries)
    conf_by_source: dict[str, Counter] = defaultdict(Counter)
    reason_counts = Counter(q.get("_match_reason", "") for q in fixed_queries)
    for q in fixed_queries:
        conf_by_source[q["manual_source"]][q["confidence"]] += 1

    print("\n" + "=" * 72)
    print("ANCHORING STATISTICS (full_context_aware)")
    print("=" * 72)
    print(f"  Total: {len(fixed_queries)}")
    for level in ["high", "medium", "unmatched"]:
        print(f"  {level}: {conf_counts.get(level, 0)}")
    print(f"\n  Match reasons: {dict(reason_counts)}")
    for source, counts in conf_by_source.items():
        print(f"\n  {source}:")
        for level in ["high", "medium", "unmatched"]:
            print(f"    {level}: {counts.get(level, 0)}")

    unmatched_count = conf_counts.get("unmatched", 0)
    if unmatched_count > 5:
        unmatched_queries = [q for q in fixed_queries if q["confidence"] == "unmatched"]
        print(f"\nWARNING: {unmatched_count} unmatched queries (>5):")
        for q in unmatched_queries:
            print(f"  {q['id']}: figure={q['ground_truth_figure']}, "
                  f"pdf_page={q['page_reference_original']}, "
                  f"manual={q['manual_source']}")
        print("Proceeding despite unmatched queries.")

    # -----------------------------------------------------------------------
    # Step 3: Per-ablation re-anchoring
    # -----------------------------------------------------------------------
    logger.info("\n[Step 3] Per-ablation re-anchoring...")
    gt_per_ablation = reanchor_per_ablation(fixed_queries)

    gt_path = DATA_DIR / "image_eval_gt_per_ablation.json"
    gt_path.write_text(json.dumps(gt_per_ablation, indent=2))
    logger.info(f"Saved per-ablation ground truth to {gt_path}")

    # Save anchoring report
    report = {
        "page_mapping_stats": {
            k: {kk: vv for kk, vv in v.items() if kk != "details"}
            for k, v in mapping_stats.items()
        },
        "anchoring_stats": {
            "total_queries": len(fixed_queries),
            "overall": dict(conf_counts),
            "by_document": {
                src: dict(counts) for src, counts in conf_by_source.items()
            },
            "match_reasons": dict(reason_counts),
        },
        "unmatched_queries": [
            {"id": q["id"], "manual": q["manual_source"],
             "figure": q.get("ground_truth_figure", ""),
             "original_page": q["page_reference_original"]}
            for q in fixed_queries if q["confidence"] == "unmatched"
        ],
    }
    report_path = DATA_DIR / "anchoring_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved anchoring report to {report_path}")

    logger.info("\nDone. Outputs in data/")


if __name__ == "__main__":
    main()
