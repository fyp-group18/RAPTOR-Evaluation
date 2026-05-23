"""Redesign image-bearing node queries from symptom-based to content-based.

Steps:
  1. Extract caption content from ground truth nodes in the full_context_aware tree
  2. Call Gemini Flash to rewrite queries based on image/caption content
  3. Quality-check the results and output a comparison table

Reads from:
  - image_node_eval/data/image_node_eval_dataset_fixed.json
  - image_node_eval/data/image_node_eval_gt_per_ablation.json
  - .tree-cache/ (full_context_aware tree pickles)

Writes to (all NEW files):
  - image_node_eval/data/image_node_query_content.json  (Step 1 output)
  - image_node_eval/data/image_node_queries_redesigned.json  (Step 2 output)
  - results/image_node_queries_redesigned/query_comparison_table.txt  (Step 3 output)

Usage:
    GOOGLE_APPLICATION_CREDENTIALS=./credentials.json \
    GOOGLE_CLOUD_PROJECT=<your-gcp-project> \
    PYTHONPATH="/path/to/raptor/backend:." \
    python -m image_node_eval.redesign_image_node_queries
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TREE_CACHE_DIR = PROJECT_ROOT / ".tree-cache"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "image_node_queries_redesigned"

DOC_MAP = {
    "AS-AMM-01-000": "112e157071358d5cec81351ddf80cc19bd7d9fa07b365048854e13d380806448",
    "SC10000AMM": "c313e3155133a7293e44a4a5166ce889a1bd33f5d28fecad5cab62813f5a40e0",
}

MODEL_FLASH = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tree loading (local only — same pattern as fix_image_node_query_annotations.py)
# ---------------------------------------------------------------------------


def load_tree_local(cache_key: str, ablation_label: str) -> list[dict]:
    """Load tree pickle from local .tree-cache/ without GCS."""
    cache_dir = TREE_CACHE_DIR / cache_key
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    candidates: list[tuple[str, Path]] = []
    for run_dir in cache_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "docling-cache":
            continue
        manifest_path = run_dir / ablation_label / "tree_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        created_at = manifest.get("created_at", "")
        candidates.append((created_at, run_dir))

    if not candidates:
        raise FileNotFoundError(
            f"No tree found for cache_key={cache_key[:12]}..., "
            f"ablation={ablation_label}"
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, best_run_dir = candidates[0]

    tree_path = best_run_dir / ablation_label / "tree.pkl"
    if not tree_path.exists():
        raise FileNotFoundError(f"tree.pkl not found at {tree_path}")

    nodes = pickle.loads(tree_path.read_bytes())
    logger.info(f"  Loaded {ablation_label} from {best_run_dir.name}: {len(nodes)} nodes")
    return nodes


# ---------------------------------------------------------------------------
# Step 1: Extract caption content from ground truth nodes
# ---------------------------------------------------------------------------


def _extract_section_header(text: str) -> str:
    """Extract the [Section Heading | Page N] prefix from node text."""
    m = re.match(r"^\[([^\]]+)\]", text)
    return m.group(1).strip() if m else ""


def _extract_caption_text(node: dict) -> str:
    """Extract caption text from a node.

    Checks two sources:
      1. The `images` field (list of dicts with 'caption' keys)
      2. Text after "Diagrams on this page:" marker in node text
    """
    # Source 1: structured images field
    images = node.get("images", [])
    if isinstance(images, list) and images:
        captions = [img.get("caption", "") for img in images if img.get("caption")]
        if captions:
            return " | ".join(captions)

    # Source 2: text marker
    text = node.get("text", "")
    marker = "Diagrams on this page:"
    idx = text.find(marker)
    if idx >= 0:
        caption_portion = text[idx + len(marker):].strip()
        # Clean up: remove trailing whitespace, limit length
        lines = [line.strip() for line in caption_portion.split("\n") if line.strip()]
        return " | ".join(lines)

    return ""


def _extract_figure_label(text: str) -> str:
    """Extract figure label like 'Figure 5.4.14.1' or 'Fig. 28-1' from text."""
    m = re.search(r"(Fig(?:ure)?\.?\s*\d[\d.\-]*)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_node_content(
    fixed_queries: list[dict],
    trees_by_manual: dict[str, list[dict]],
) -> dict[str, dict]:
    """Step 1: For each image-bearing node query, find the GT node and extract content."""
    node_content: dict[str, dict] = {}

    for query in fixed_queries:
        qid = query["id"]
        if query.get("confidence") == "unmatched":
            logger.info(f"  Skipping unmatched query {qid}")
            continue

        manual = query["manual_source"]
        chunk_id = query["ground_truth_chunk_id"]
        nodes = trees_by_manual.get(manual, [])

        # Find the GT node by ID
        gt_node = None
        for n in nodes:
            if n.get("id") == chunk_id:
                gt_node = n
                break

        if gt_node is None:
            logger.warning(f"  {qid}: GT node {chunk_id} not found in tree")
            continue

        node_text = gt_node.get("text", "")
        caption_text = _extract_caption_text(gt_node)
        section_header = _extract_section_header(node_text)
        figure_label = query.get("ground_truth_figure", "") or _extract_figure_label(node_text)

        # Use figure_reference caption as supplement if no caption from node
        if not caption_text:
            fig_ref = query.get("figure_reference", {})
            caption_text = fig_ref.get("caption", "")

        node_content[qid] = {
            "query_id": qid,
            "original_query": query["observation"],
            "ground_truth_chunk_id": chunk_id,
            "node_text": node_text[:2000],  # cap for readability
            "caption_text": caption_text,
            "section_header": section_header,
            "figure_label": figure_label,
            "manual_source": manual,
            "figure_description": query.get("figure_description", ""),
            "image_type": query.get("figure_reference", {}).get("image_type", ""),
        }

    return node_content


# ---------------------------------------------------------------------------
# Step 2: Generate content-based queries via Gemini Flash
# ---------------------------------------------------------------------------


def _build_rewrite_prompt(entry: dict) -> str:
    """Build the LLM prompt for rewriting one query."""
    return (
        "You are rewriting evaluation queries for a document retrieval system.\n\n"
        f'Original query: "{entry["original_query"]}"\n'
        f'The correct answer is a document chunk containing this image caption: "{entry["caption_text"]}"\n'
        f'It is located in section: "{entry["section_header"]}"\n'
        f'Figure label: "{entry["figure_label"]}"\n'
        f'Image type: {entry["image_type"]}\n'
        f'Figure description: "{entry["figure_description"]}"\n\n'
        "Rewrite the query so it asks about what the image/diagram SHOWS or DEPICTS, "
        "not about fault symptoms or troubleshooting procedures. The rewritten query "
        "should be answerable by finding the image node based on its visual content "
        "and caption.\n\n"
        "Examples:\n"
        '- BAD: "What diagram helps troubleshoot a no-display fault?" (symptom-based)\n'
        '- GOOD: "Show me the wiring diagram for the display module connections" (content-based)\n'
        '- BAD: "What figure relates to hydraulic pump failure?" (symptom-based)\n'
        '- GOOD: "Where is the hydraulic pump assembly diagram showing pipe routing?" (content-based)\n\n'
        "Output ONLY the rewritten query, nothing else."
    )


def _fallback_template_query(entry: dict) -> str:
    """Template-based fallback when Gemini Flash is unavailable."""
    caption = entry.get("caption_text", "")
    figure_label = entry.get("figure_label", "")
    image_type = entry.get("image_type", "diagram")

    if caption and figure_label:
        return f"Show me {figure_label}, the {image_type} described as: {caption}"
    elif caption:
        return f"Show me the {image_type} described as: {caption}"
    elif figure_label:
        fig_desc = entry.get("figure_description", "")
        if fig_desc:
            return f"Where is {figure_label}: {fig_desc}?"
        return f"Where is {figure_label}?"
    else:
        return f"Show me the {image_type} in section: {entry.get('section_header', 'unknown')}"


def _generate_with_retry(
    client,
    model: str,
    contents: list,
    config,
    max_retries: int = 5,
    base_delay: float = 2.0,
):
    """Retry wrapper with exponential backoff on 429/5xx."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e)
            retryable = any(code in err_str for code in ("429", "500", "503"))
            if retryable and attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {err_str[:100]}"
                )
                time.sleep(delay)
            else:
                raise


def rewrite_queries_with_llm(
    node_content: dict[str, dict],
) -> dict[str, dict]:
    """Step 2: Call Gemini Flash to rewrite each query."""
    from google import genai
    from google.genai import types

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

    redesigned: dict[str, dict] = {}
    use_llm = True

    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )
        # Quick health check
        logger.info("Gemini Flash client initialized, testing connection...")
    except Exception as e:
        logger.warning(f"Cannot initialize Gemini client: {e}")
        logger.warning("Falling back to template-based rewriting")
        use_llm = False

    total = len(node_content)
    for i, (qid, entry) in enumerate(sorted(node_content.items()), 1):
        if use_llm:
            try:
                prompt = _build_rewrite_prompt(entry)
                res = _generate_with_retry(
                    client=client,
                    model=MODEL_FLASH,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                    ),
                )
                rewritten = res.text.strip().strip('"').strip("'")
                method = "gemini_flash"
                logger.info(f"  [{i}/{total}] {qid}: {rewritten[:80]}...")
            except Exception as e:
                logger.warning(f"  [{i}/{total}] {qid}: LLM failed ({e}), using template")
                rewritten = _fallback_template_query(entry)
                method = "template_fallback"
        else:
            rewritten = _fallback_template_query(entry)
            method = "template_fallback"

        redesigned[qid] = {
            "query_id": qid,
            "original_query": entry["original_query"],
            "redesigned_query": rewritten,
            "ground_truth_chunk_id": entry["ground_truth_chunk_id"],
            "caption_text": entry["caption_text"],
            "section_header": entry["section_header"],
            "figure_label": entry["figure_label"],
            "manual_source": entry["manual_source"],
            "rewrite_method": method,
        }

        # Rate limiting: ~2 QPS for Flash
        if use_llm and i < total:
            time.sleep(0.5)

    return redesigned


# ---------------------------------------------------------------------------
# Step 3: Quality check
# ---------------------------------------------------------------------------

# Symptom/fault keywords that should NOT appear in redesigned queries
_SYMPTOM_WORDS = {
    "fault", "failure", "error", "malfunction", "troubleshoot", "troubleshooting",
    "defect", "symptom", "fail", "fails", "failing", "issue", "problem",
    "inoperative", "inop", "intermittent", "abnormal", "leak", "leaking",
    "crack", "cracked", "broken", "stuck", "jammed", "binding", "vibration",
    "noise", "excessive", "low pressure", "high pressure", "overheat",
    "does not", "doesn't", "won't", "unable",
}


def quality_check(
    redesigned: dict[str, dict],
    node_content: dict[str, dict],
) -> tuple[list[str], str]:
    """Step 3: Check redesigned queries for quality issues.

    Returns:
        flagged_ids: list of query IDs that were flagged
        table_text: formatted comparison table
    """
    flagged: list[str] = []
    rows: list[str] = []

    header = (
        f"{'ID':<14s} | {'Original Query':<60s} | {'Redesigned Query':<60s} | "
        f"{'Caption (ground truth)':<50s} | {'OK?':>4s}"
    )
    separator = "-" * len(header)
    rows.append(separator)
    rows.append(header)
    rows.append(separator)

    for qid in sorted(redesigned.keys()):
        entry = redesigned[qid]
        content = node_content.get(qid, {})

        original = entry["original_query"][:57] + "..." if len(entry["original_query"]) > 60 else entry["original_query"]
        rewritten = entry["redesigned_query"][:57] + "..." if len(entry["redesigned_query"]) > 60 else entry["redesigned_query"]
        caption = (content.get("caption_text", "") or "")[:47] + "..." if len(content.get("caption_text", "") or "") > 50 else (content.get("caption_text", "") or "")

        # Flag checks
        flags: list[str] = []
        query_lower = entry["redesigned_query"].lower()

        # Check for remaining symptom language
        for word in _SYMPTOM_WORDS:
            if word in query_lower:
                flags.append(f"symptom:{word}")
                break

        # Check if too generic (< 5 words)
        word_count = len(entry["redesigned_query"].split())
        if word_count < 5:
            flags.append("too_short")

        # Check if caption content has no overlap with redesigned query
        caption_words = set(re.findall(r"\w+", (content.get("caption_text", "") or "").lower()))
        query_words = set(re.findall(r"\w+", query_lower))
        if caption_words and len(caption_words & query_words) == 0:
            flags.append("no_caption_overlap")

        ok = "FAIL" if flags else "OK"
        if flags:
            flagged.append(qid)

        rows.append(
            f"{qid:<14s} | {original:<60s} | {rewritten:<60s} | "
            f"{caption:<50s} | {ok:>4s}"
        )
        if flags:
            rows.append(f"{'':>14s}   ^ FLAGS: {', '.join(flags)}")

    rows.append(separator)
    rows.append(f"\nTotal: {len(redesigned)} image-bearing node queries, {len(flagged)} flagged")

    table_text = "\n".join(rows)
    return flagged, table_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logger.info("=" * 72)
    logger.info("Redesign Image-Bearing Node Queries: Symptom-based → Content-based")
    logger.info("=" * 72)

    # Load fixed dataset
    fixed_path = DATA_DIR / "image_node_eval_dataset_fixed.json"
    if not fixed_path.exists():
        logger.error(f"Fixed dataset not found at {fixed_path}")
        sys.exit(1)

    fixed_queries = json.loads(fixed_path.read_text())
    logger.info(f"Loaded {len(fixed_queries)} image-bearing node queries")

    # Load per-ablation GT (we only need it for passing through, not for Step 1)
    gt_path = DATA_DIR / "image_node_eval_gt_per_ablation.json"
    if not gt_path.exists():
        logger.error(f"Per-ablation GT not found at {gt_path}")
        sys.exit(1)

    gt_per_ablation = json.loads(gt_path.read_text())

    # -----------------------------------------------------------------------
    # Step 1: Extract caption content
    # -----------------------------------------------------------------------
    logger.info("\n[Step 1] Extracting caption content from ground truth nodes...")

    trees_by_manual: dict[str, list[dict]] = {}
    for manual, cache_key in DOC_MAP.items():
        logger.info(f"  Loading {manual} full_context_aware tree...")
        trees_by_manual[manual] = load_tree_local(cache_key, "full_context_aware")

    node_content = extract_node_content(fixed_queries, trees_by_manual)
    logger.info(f"  Extracted content for {len(node_content)} queries")

    # Save Step 1 output
    node_content_path = DATA_DIR / "image_node_query_content.json"
    node_content_path.write_text(json.dumps(node_content, indent=2))
    logger.info(f"  Saved to {node_content_path}")

    # -----------------------------------------------------------------------
    # Step 2: Rewrite queries via Gemini Flash
    # -----------------------------------------------------------------------
    logger.info("\n[Step 2] Rewriting queries via Gemini Flash...")

    redesigned = rewrite_queries_with_llm(node_content)
    logger.info(f"  Rewrote {len(redesigned)} queries")

    # Save Step 2 output
    redesigned_path = DATA_DIR / "image_node_queries_redesigned.json"
    redesigned_path.write_text(json.dumps(redesigned, indent=2))
    logger.info(f"  Saved to {redesigned_path}")

    # -----------------------------------------------------------------------
    # Step 3: Quality check
    # -----------------------------------------------------------------------
    logger.info("\n[Step 3] Quality checking redesigned image-bearing node queries...")

    flagged, table_text = quality_check(redesigned, node_content)

    # Print table
    print("\n" + table_text)

    # Save table
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_path = RESULTS_DIR / "query_comparison_table.txt"
    table_path.write_text(table_text)
    logger.info(f"\n  Saved comparison table to {table_path}")

    if len(flagged) > 5:
        logger.warning(
            f"\n  {len(flagged)} queries flagged (>5 threshold). "
            f"Review query_comparison_table.txt before running evaluation."
        )
        logger.warning(f"  Flagged IDs: {', '.join(flagged)}")
        print(f"\nSTOP: {len(flagged)} queries flagged — review before proceeding.")
    else:
        logger.info(f"\n  {len(flagged)} queries flagged (within threshold)")

    # Report rewrite methods
    methods = {}
    for entry in redesigned.values():
        m = entry.get("rewrite_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
    logger.info(f"  Rewrite methods: {methods}")

    logger.info("\nDone. Run run_image_node_eval_redesigned.py next to evaluate.")


if __name__ == "__main__":
    main()
