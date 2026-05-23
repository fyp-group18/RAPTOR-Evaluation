"""Tree builder: chunking + embedding + RAPTOR, parameterized by ablation config.

Takes a serialized DoclingDocument dict and an ablation config, produces a
pickled tree (list of node dicts with embeddings) and stats.
"""

from __future__ import annotations

import io
import logging
import pickle
import re
import sys
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Ensure backend is importable
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_CHUNK_SIZE = 1600
_CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# Docling document iteration → pages_dict
# ---------------------------------------------------------------------------


def _extract_pages_from_docling(docling_dict: dict) -> tuple[dict, dict[int, list[bytes]]]:
    """Reconstruct page structure from a serialized DoclingDocument.

    Returns:
        pages_dict: {page_no: {"text_blocks": [...], "tables": [...], "images": [...], "section_header": ...}}
        page_images: {page_no: [png_bytes, ...]} for multimodal embedding
    """
    from docling_core.types.doc import DoclingDocument as DoclingDocumentModel

    doc = DoclingDocumentModel.model_validate(docling_dict)

    from docling_core.types.doc import (
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
    )
    from collections import Counter

    # Detect procedure heading level (same heuristic as backend)
    heading_level_counts: Counter[int] = Counter()
    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) and item.text.strip():
            heading_level_counts[item.level] += 1

    max_section_level = 2
    for lvl in sorted(heading_level_counts):
        if heading_level_counts[lvl] > 1:
            max_section_level = lvl
            break

    pages_dict: dict[int, dict] = {}
    page_images: dict[int, list[bytes]] = {}
    current_section: str | None = None

    for item, level in doc.iterate_items():
        page_no = 1
        if hasattr(item, "prov") and item.prov:
            page_no = item.prov[0].page_no
        if page_no not in pages_dict:
            pages_dict[page_no] = {"text_blocks": [], "images": [], "tables": []}

        if isinstance(item, SectionHeaderItem) and item.text.strip():
            if item.level <= max_section_level:
                current_section = item.text.strip()

        pages_dict[page_no]["section_header"] = current_section

        if isinstance(item, TextItem) and item.text.strip():
            pages_dict[page_no]["text_blocks"].append(item.text.strip())
        elif isinstance(item, TableItem):
            try:
                table_md = item.export_to_markdown()
                if table_md and table_md.strip():
                    pages_dict[page_no]["tables"].append(
                        {"markdown": table_md, "section_header": current_section}
                    )
            except Exception as e:
                logger.warning(f"Table export failed on page {page_no}: {e}")
        elif isinstance(item, PictureItem):
            try:
                img = item.get_image(doc)
                if img and img.size[0] > 150 and img.size[1] > 150:
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format="PNG")
                    img_bytes = img_byte_arr.getvalue()

                    page_images.setdefault(page_no, []).append(img_bytes)
                    # Store a placeholder caption (will be generated if needed)
                    pages_dict[page_no]["images"].append(
                        {"path": f"page_{page_no}_img_{len(page_images[page_no])}", "caption": ""}
                    )
            except Exception as e:
                logger.warning(f"Image extraction failed on page {page_no}: {e}")

    return pages_dict, page_images


# ---------------------------------------------------------------------------
# Table parsing utilities (replicated from backend/modules/ingestion.py)
# ---------------------------------------------------------------------------


def _detect_table_type(header_row: str) -> str:
    h = header_row.lower()
    if any(kw in h for kw in ("fault", "trouble", "problem", "malfunction", "symptom")):
        return "TROUBLESHOOTING"
    if any(kw in h for kw in ("part", "item", "quantity", "qty")):
        return "PARTS LIST"
    if any(kw in h for kw in ("spec", "limit", "tolerance", "torque", "dimension")):
        return "SPECIFICATIONS"
    return "TABLE"


def _parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]] | None:
    lines = [ln.strip() for ln in markdown.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    header_idx = None
    sep_idx = None
    for i, ln in enumerate(lines):
        if "|" not in ln:
            continue
        if i + 1 < len(lines) and re.match(r"^[\s|:\-]+$", lines[i + 1]):
            header_idx = i
            sep_idx = i + 1
            break

    if header_idx is None:
        return None

    def _split_row(row: str) -> list[str]:
        cells = row.split("|")
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        return [c.strip() for c in cells]

    col_names = _split_row(lines[header_idx])
    if not col_names:
        return None

    data_rows = []
    for ln in lines[sep_idx + 1:]:
        if "|" not in ln:
            continue
        if re.match(r"^[\s|:\-]+$", ln):
            continue
        cells = _split_row(ln)
        while len(cells) < len(col_names):
            cells.append("")
        data_rows.append(cells[:len(col_names)])

    return col_names, data_rows


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def _chunk_context_aware(
    pages_dict: dict,
    *,
    table_parent_child: bool,
    header_propagation: bool,
    caption_folding: bool,
) -> list[dict]:
    """Context-aware chunking with toggleable sub-innovations.

    Mirrors backend/_build_page_leaves but each sub-innovation can be disabled.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    leaves: list[dict] = []
    sequence_index = 0

    for page_no in sorted(pages_dict.keys(), key=int):
        content = pages_dict[page_no]
        text_blocks = content.get("text_blocks", []) or []
        tables = content.get("tables", []) or []
        images = content.get("images", []) or []
        section_header = content.get("section_header")

        # --- Prose chunks ---
        body_parts: list[str] = []
        text_body = "\n\n".join(b for b in text_blocks if b).strip()
        if text_body:
            body_parts.append(text_body)

        if caption_folding and images:
            caption_lines = [
                f"- {img['path']}: {img.get('caption', '')}".strip()
                for img in images
            ]
            body_parts.append("Diagrams on this page:\n" + "\n".join(caption_lines))

        if not body_parts and not tables:
            body_parts.append("(no extractable content on this page)")

        if body_parts:
            if header_propagation and section_header:
                leaf_text = f"[{section_header} | Page {page_no}]\n\n" + "\n\n".join(body_parts)
            else:
                leaf_text = f"[Page {page_no}]\n\n" + "\n\n".join(body_parts)

            page_chunks = (
                splitter.split_text(leaf_text) if len(leaf_text) > _CHUNK_SIZE else [leaf_text]
            )
            for chunk_text in page_chunks:
                leaves.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "level": 0,
                    "pages": [page_no],
                    "page_start": page_no,
                    "page_end": page_no,
                    "sequence_index": sequence_index,
                    "section_header": section_header if header_propagation else None,
                    "images": images if caption_folding else [],
                })
                sequence_index += 1

        # --- Table chunks ---
        for table_entry in tables:
            table_md = table_entry["markdown"]
            table_section = table_entry.get("section_header") or section_header

            if not table_parent_child:
                # No parent-child: treat table as prose
                if header_propagation and table_section:
                    fallback_text = f"[{table_section} | Page {page_no}]\n\n[TABLE]:\n{table_md}"
                else:
                    fallback_text = f"[Page {page_no}]\n\n[TABLE]:\n{table_md}"

                fallback_chunks = (
                    splitter.split_text(fallback_text)
                    if len(fallback_text) > _CHUNK_SIZE
                    else [fallback_text]
                )
                for chunk_text in fallback_chunks:
                    leaves.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "level": 0,
                        "pages": [page_no],
                        "page_start": page_no,
                        "page_end": page_no,
                        "sequence_index": sequence_index,
                        "section_header": table_section if header_propagation else None,
                        "images": [],
                    })
                    sequence_index += 1
                continue

            # Parent-child table chunking
            parsed = _parse_markdown_table(table_md)
            if parsed is None:
                # Unparseable: fall back to prose
                if header_propagation and table_section:
                    fallback_text = f"[{table_section} | Page {page_no}]\n\n[TABLE]:\n{table_md}"
                else:
                    fallback_text = f"[Page {page_no}]\n\n[TABLE]:\n{table_md}"
                fallback_chunks = (
                    splitter.split_text(fallback_text)
                    if len(fallback_text) > _CHUNK_SIZE
                    else [fallback_text]
                )
                for chunk_text in fallback_chunks:
                    leaves.append({
                        "id": str(uuid.uuid4()),
                        "text": chunk_text,
                        "level": 0,
                        "pages": [page_no],
                        "page_start": page_no,
                        "page_end": page_no,
                        "sequence_index": sequence_index,
                        "section_header": table_section if header_propagation else None,
                        "images": [],
                    })
                    sequence_index += 1
                continue

            col_names, data_rows = parsed
            header_row_text = " | ".join(col_names)
            table_type = _detect_table_type(header_row_text)

            # Parent chunk (not embedded, used for LLM context)
            parent_id = str(uuid.uuid4())
            if header_propagation and table_section:
                parent_text = f"[{table_section} | Page {page_no}]\n\n{table_md}"
            else:
                parent_text = f"[Page {page_no}]\n\n{table_md}"

            leaves.append({
                "id": parent_id,
                "text": parent_text,
                "level": 0,
                "pages": [page_no],
                "page_start": page_no,
                "page_end": page_no,
                "sequence_index": sequence_index,
                "section_header": table_section if header_propagation else None,
                "images": [],
                "is_parent": True,
            })
            sequence_index += 1

            # Child chunks (one per row, embedded)
            section_label = table_section or f"Page {page_no}"
            for row_cells in data_rows:
                pairs = [
                    f"{col}: {val}"
                    for col, val in zip(col_names, row_cells)
                    if val.strip()
                ]
                if not pairs:
                    continue
                row_text = " | ".join(pairs)
                child_text = f"{table_type} — {section_label}: {row_text}"
                leaves.append({
                    "id": str(uuid.uuid4()),
                    "text": child_text,
                    "level": 0,
                    "pages": [page_no],
                    "page_start": page_no,
                    "page_end": page_no,
                    "sequence_index": None,
                    "section_header": table_section if header_propagation else None,
                    "images": [],
                    "parent_id": parent_id,
                })

    return leaves


def _chunk_naive(pages_dict: dict) -> list[dict]:
    """Naive fixed-size chunking: concatenate all text, split uniformly."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )

    # Track (text_part, page_no) so we can resolve page provenance after splitting
    tagged_parts: list[tuple[str, int]] = []
    for page_no in sorted(pages_dict.keys(), key=int):
        content = pages_dict[page_no]
        text_blocks = content.get("text_blocks", []) or []
        tables = content.get("tables", []) or []
        for block in text_blocks:
            if block:
                tagged_parts.append((block, int(page_no)))
        for table_entry in tables:
            tagged_parts.append((table_entry["markdown"], int(page_no)))

    full_text = "\n\n".join(part for part, _ in tagged_parts)
    if not full_text.strip():
        full_text = "(no extractable content)"

    # Build a character-offset → page mapping
    char_to_page: list[tuple[int, int, int]] = []  # (start, end, page_no)
    offset = 0
    for i, (part, page_no) in enumerate(tagged_parts):
        start = offset
        end = offset + len(part)
        char_to_page.append((start, end, page_no))
        offset = end + 2  # +2 for "\n\n" separator

    # Resolve page provenance by searching sequentially through full_text.
    # We advance search_start after each match so overlapping chunks that
    # share text with earlier chunks resolve to the correct later position.
    chunks = splitter.split_text(full_text)
    leaves = []
    search_start = 0
    for i, chunk_text in enumerate(chunks):
        idx = full_text.find(chunk_text, search_start)
        if idx < 0:
            # Overlap may have shifted — try from beginning as fallback
            idx = full_text.find(chunk_text)
        if idx >= 0:
            chunk_end = idx + len(chunk_text)
            pages = sorted({
                pg for (s, e, pg) in char_to_page
                if s < chunk_end and e > idx
            })
            # Advance past start of this chunk (not past end, since chunks overlap)
            search_start = idx + 1
        else:
            pages = []
        leaves.append({
            "id": str(uuid.uuid4()),
            "text": chunk_text,
            "level": 0,
            "pages": pages,
            "page_start": pages[0] if pages else 0,
            "page_end": pages[-1] if pages else 0,
            "sequence_index": i,
            "section_header": None,
            "images": [],
        })
    return leaves


def _chunk_semantic(pages_dict: dict) -> list[dict]:
    """Semantic chunking: cosine-similarity boundary detection on sentence embeddings."""
    from modules.embeddings import embed_batch

    # Collect all sentences with page provenance
    sentence_page: list[tuple[str, int]] = []  # (sentence, page_no)
    for page_no in sorted(pages_dict.keys(), key=int):
        content = pages_dict[page_no]
        text_blocks = content.get("text_blocks", []) or []
        tables = content.get("tables", []) or []
        for block in text_blocks:
            if block:
                for sent in re.split(r"(?<=[.!?])\s+", block):
                    if sent.strip():
                        sentence_page.append((sent.strip(), int(page_no)))
        for table_entry in tables:
            sentence_page.append((table_entry["markdown"], int(page_no)))

    if not sentence_page:
        sentence_page = [("(no extractable content)", 0)]

    sentences = [s for s, _ in sentence_page]

    # Embed all sentences
    embeddings = embed_batch(sentences)
    valid_indices = [
        i for i, emb in enumerate(embeddings) if emb is not None
    ]
    if not valid_indices:
        logger.warning("Semantic chunking: all embeddings failed, falling back to naive")
        return _chunk_naive(pages_dict)

    valid_sents = [sentences[i] for i in valid_indices]
    valid_pages = [sentence_page[i][1] for i in valid_indices]
    valid_embs = [embeddings[i] for i in valid_indices]
    emb_array = np.array(valid_embs, dtype=np.float32)

    # Compute cosine similarities between consecutive sentences
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = emb_array / norms

    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

    # Find boundaries where similarity drops below threshold
    threshold = float(np.mean(similarities) - np.std(similarities))
    threshold = max(threshold, 0.3)

    # Build chunks by grouping consecutive sentences between boundaries
    # Track which pages each chunk spans
    chunk_groups: list[list[int]] = []  # list of [sentence_indices]
    current_group: list[int] = [0]

    for i in range(len(similarities)):
        current_text = " ".join(valid_sents[j] for j in current_group)
        if similarities[i] < threshold and len(current_text) > 200:
            chunk_groups.append(current_group)
            current_group = [i + 1]
        else:
            current_group.append(i + 1)

    if current_group:
        chunk_groups.append(current_group)

    # Post-process: split oversized chunks (pages from the group apply to all sub-chunks)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    leaves = []
    seq = 0
    for group in chunk_groups:
        text = " ".join(valid_sents[j] for j in group)
        pages = sorted({valid_pages[j] for j in group})

        if len(text) > _CHUNK_SIZE:
            sub_chunks = splitter.split_text(text)
        else:
            sub_chunks = [text]

        for chunk_text in sub_chunks:
            leaves.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "level": 0,
                "pages": pages,
                "page_start": pages[0] if pages else 0,
                "page_end": pages[-1] if pages else 0,
                "sequence_index": seq,
                "section_header": None,
                "images": [],
            })
            seq += 1

    return leaves


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def _embed_leaves(
    leaves: list[dict],
    page_images: dict[int, list[bytes]],
    multimodal: bool,
) -> list[dict]:
    """Embed all leaf nodes. Skips is_parent nodes (not embedded).

    Args:
        leaves: Leaf node dicts.
        page_images: {page_no: [png_bytes]} for multimodal embedding.
        multimodal: If True, include images in embedding calls.

    Returns:
        Leaf nodes with "embedding" field populated (np.ndarray or None).
    """
    from modules.embeddings import embed, embed_batch

    embeddable = [n for n in leaves if not n.get("is_parent")]
    non_embeddable = [n for n in leaves if n.get("is_parent")]

    if not multimodal:
        # Text-only batch embedding
        texts = [n["text"] for n in embeddable]
        embeddings = embed_batch(texts)
        for node, emb in zip(embeddable, embeddings):
            if emb is not None:
                node["embedding"] = np.array(emb, dtype=np.float32)
            else:
                node["embedding"] = None
    else:
        # Multimodal: nodes with images get per-node embed() calls,
        # text-only nodes get batched
        text_only_nodes = []
        image_nodes = []

        for node in embeddable:
            pages = node.get("pages", [])
            has_images = any(page_images.get(p) for p in pages)
            if has_images:
                image_nodes.append(node)
            else:
                text_only_nodes.append(node)

        # Batch text-only
        if text_only_nodes:
            texts = [n["text"] for n in text_only_nodes]
            embeddings = embed_batch(texts)
            for node, emb in zip(text_only_nodes, embeddings):
                if emb is not None:
                    node["embedding"] = np.array(emb, dtype=np.float32)
                else:
                    node["embedding"] = None

        # Per-node multimodal embedding
        def _embed_with_images(node: dict) -> tuple[dict, np.ndarray | None]:
            pages = node.get("pages", [])
            img_bytes_list = []
            for p in pages:
                img_bytes_list.extend(page_images.get(p, []))
            # Cap at 4 images per embedding call (matching backend)
            img_bytes_list = img_bytes_list[:4]
            try:
                emb = embed(node["text"], img_bytes_list)
                if emb is not None:
                    return node, np.array(emb, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Multimodal embed failed for node {node['id']}: {e}")
            # Fallback to text-only
            try:
                emb = embed(node["text"], [])
                if emb is not None:
                    return node, np.array(emb, dtype=np.float32)
            except Exception:
                pass
            return node, None

        if image_nodes:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(_embed_with_images, node): node
                    for node in image_nodes
                }
                for fut in as_completed(futures):
                    node, emb = fut.result()
                    node["embedding"] = emb

    # Mark non-embeddable parent nodes
    for node in non_embeddable:
        node["embedding"] = None

    return leaves


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_tree_for_ablation(
    docling_output: dict,
    ablation_config: dict,
    original_filename: str,
    cache_key: str | None = None,
) -> tuple[bytes, dict]:
    """Build a RAPTOR tree (or flat index) for a single ablation configuration.

    Args:
        docling_output: Serialized DoclingDocument dict.
        ablation_config: One entry from ABLATION_CASES.
        original_filename: Source document filename for metadata.
        cache_key: Document content hash, passed to CR for checkpointing.

    Returns:
        (pickled_tree_bytes, tree_stats_dict)
    """
    label = ablation_config["label"]
    logger.info(f"[{label}] Starting tree build...")

    # Step 1: Extract pages and images from Docling output
    pages_dict, page_images = _extract_pages_from_docling(docling_output)
    logger.info(f"[{label}] Extracted {len(pages_dict)} pages")

    # Step 2: Chunk based on config
    chunking_strategy = ablation_config.get("chunking_strategy")
    table_pc = ablation_config.get("table_parent_child", False)
    header_prop = ablation_config.get("header_propagation", False)
    caption_fold = ablation_config.get("caption_folding", False)

    cr_log = None
    if chunking_strategy == "semantic":
        leaves = _chunk_semantic(pages_dict)
    elif ablation_config.get("contextual_retrieval"):
        from ingestion.contextual_retrieval import build_cr_chunks
        leaves, cr_log = build_cr_chunks(pages_dict, cache_key=cache_key)
    elif not table_pc and not header_prop and not caption_fold and chunking_strategy is None:
        # All context-aware features off and no semantic strategy → naive chunking
        # But only for baseline_naive_chunking (no_context_aware case)
        # Check if this is truly the "no context aware" case
        leaves = _chunk_naive(pages_dict)
    else:
        leaves = _chunk_context_aware(
            pages_dict,
            table_parent_child=table_pc,
            header_propagation=header_prop,
            caption_folding=caption_fold,
        )

    logger.info(f"[{label}] Chunked into {len(leaves)} leaf nodes")

    # Step 3: Embed
    multimodal = ablation_config.get("multimodal_embedding", True)
    leaves = _embed_leaves(leaves, page_images, multimodal=multimodal)

    # Filter out nodes that failed embedding (except parents which are intentionally unembedded)
    embedded_leaves = [
        n for n in leaves
        if n.get("embedding") is not None or n.get("is_parent")
    ]
    failed_count = len(leaves) - len(embedded_leaves)
    embeddable_count = sum(1 for n in leaves if not n.get("is_parent"))
    embedded_count = sum(1 for n in leaves if n.get("embedding") is not None)

    if embeddable_count > 0:
        success_rate = embedded_count / embeddable_count
        logger.info(
            f"[{label}] Embedding: {embedded_count}/{embeddable_count} succeeded "
            f"({success_rate:.0%})"
        )
        if embedded_count == 0:
            raise RuntimeError(
                f"[{label}] All {embeddable_count} embedding calls failed. "
                f"Chunking produced {len(leaves)} nodes but none could be embedded. "
                f"Check API quota/credentials."
            )
        if success_rate < 0.5:
            logger.warning(
                f"[{label}] Embedding success rate critically low ({success_rate:.0%}). "
                f"Tree may be incomplete."
            )
    elif failed_count > 0:
        logger.warning(f"[{label}] {failed_count} nodes failed embedding, excluded from tree")

    # Step 4: Build RAPTOR tree or flat index
    build_raptor = ablation_config.get("raptor_tree", True)

    # Prepare nodes for RAPTOR builder (needs "text" and "embedding" keys)
    raptor_input = [n for n in embedded_leaves if n.get("embedding") is not None]

    if build_raptor and len(raptor_input) > 1:
        from benchmarks.raptor_tree import build_raptor_tree

        logger.info(f"[{label}] Building RAPTOR tree from {len(raptor_input)} embeddable nodes...")
        all_nodes = build_raptor_tree(raptor_input, max_levels=3)
    else:
        all_nodes = raptor_input
        if not build_raptor:
            logger.info(f"[{label}] Flat index mode — skipping RAPTOR tree construction")

    # Add back parent nodes (unembedded, for context retrieval)
    parent_nodes = [n for n in embedded_leaves if n.get("is_parent")]
    all_nodes.extend(parent_nodes)

    # Guard: tree must not be empty if chunking produced content
    if len(all_nodes) == 0 and len(leaves) > 0:
        raise RuntimeError(
            f"[{label}] Tree is empty despite {len(leaves)} chunks produced during "
            f"chunking. All embeddings likely failed — check API quota/credentials."
        )

    # Step 5: Compute stats and serialize
    num_leaf = sum(1 for n in all_nodes if n.get("level", 0) == 0)
    num_summary = sum(1 for n in all_nodes if n.get("level", 0) > 0)
    levels = set(n.get("level", 0) for n in all_nodes)
    num_levels = max(levels) + 1 if levels else 0
    num_image_nodes = sum(1 for n in all_nodes if n.get("images"))
    num_table_chunks = sum(1 for n in all_nodes if n.get("parent_id") or n.get("is_parent"))

    tree_stats = {
        "num_leaf_nodes": num_leaf,
        "num_summary_nodes": num_summary,
        "num_levels": num_levels,
        "num_image_nodes": num_image_nodes,
        "num_table_chunks": num_table_chunks,
    }

    if cr_log is not None:
        tree_stats["cr_context_log"] = cr_log

    logger.info(
        f"[{label}] Tree complete: {num_leaf} leaves, "
        f"{num_summary} summaries, {num_levels} levels"
    )

    tree_bytes = pickle.dumps(all_nodes, protocol=pickle.HIGHEST_PROTOCOL)
    return tree_bytes, tree_stats
