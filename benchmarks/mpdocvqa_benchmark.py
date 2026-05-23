"""
MP-DocVQA benchmark runner.

Dataset: lmms-lab/MP-DocVQA (val split: 5,190 questions, ~927 documents)
Primary metric: ANLS (Average Normalized Levenshtein Similarity)
Secondary metric: Page Accuracy (did retrieval surface the answer page?)

Two evaluation modes controlled by doc_processing config key:
  Mode A (ocr_text_only):
    Per-page OCR via Gemini Flash → concatenate with page markers → split into
    flat chunks (1600/200) → text-only embedding → flat cosine retrieval

  Mode C (multimodal_page):
    Per-page OCR + caption + table detection → multimodal page-level leaves
    + table parent-child chunking → RAPTOR tree → collapsed retrieval with
    parent expansion

All intermediate results cached to RAPTOR-evaluation/datasets/mpdocvqa_cache/:
  ocr/{doc_id}_p{page_idx}.txt           — OCR text per page
  captions/{doc_id}_p{page_idx}.txt      — Caption per page
  table_detection/{doc_id}_p{page_idx}.txt  — YES/NO
  tables/{doc_id}_p{page_idx}.md         — Table markdown (if detected)
  page_images/{doc_id}_p{page_idx}.png   — Saved PIL images
  embeddings/{doc_id}_{mode}.npz         — Leaf-level chunk embeddings
  trees/{doc_id}_{mode}.npz             — Full RAPTOR tree (Mode C only)

External comparison target (López et al., 2025):
  RAG-VT5 base (text RAG, frozen embed, no reranker): 58.23% ANLS
"""

import ast
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from benchmarks.base_benchmark import BaseBenchmark
from metrics import anls_score, score_with_multiple_gts

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
_EMBED_DIM = 3072
_EMBED_WORKERS = 5

# Cache subdirectory names
_OCR_DIR = "ocr"
_CAPTION_DIR = "captions"
_TABLE_DETECT_DIR = "table_detection"
_TABLE_MD_DIR = "tables"
_PAGE_IMAGE_DIR = "page_images"
_EMBED_DIR = "embeddings"
_TREE_DIR = "trees"

# Gemini prompts for page processing
_OCR_PROMPT = (
    "Extract all visible text from this document image exactly as it appears. "
    "Preserve layout structure. Return only the extracted text."
)
_CAPTION_PROMPT = (
    "Describe the content, structure, and any tables/figures/charts in this "
    "document page in 2-3 sentences."
)
_TABLE_DETECT_PROMPT = (
    "Does this document page contain a data table (rows and columns of structured data)? "
    "Reply with only YES or NO."
)
_TABLE_EXTRACT_PROMPT = (
    "Extract the table from this page as a markdown table. "
    "If there are multiple tables, extract all of them separated by blank lines."
)

_QA_PROMPT = """\
You are answering questions about a multi-page document.
Based on the provided context from the document, give a concise answer.
If the answer is a specific value, name, date, or number, extract it exactly as it appears.
If you cannot determine the answer from the context, say "unanswerable".

Context:
{retrieved_chunks}

Question: {question}

Answer:"""


def _cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors."""
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-10:
        return np.zeros(matrix.shape[0])
    q_hat = query / q_norm
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms = np.maximum(m_norms, 1e-10)
    m_hat = matrix / m_norms
    return m_hat @ q_hat


def _parse_markdown_table_rows(markdown: str) -> list[str]:
    """Extract data rows from a markdown table, skipping separator lines."""
    rows: list[str] = []
    for line in markdown.split("\n"):
        if line.count("|") < 2:
            continue
        # Skip separator rows (e.g. |---|---| or |:--:|)
        if re.match(r"^\s*\|[\s|:\-=]+\|\s*$", line):
            continue
        stripped = line.strip().strip("|").strip()
        if stripped:
            rows.append(stripped)
    return rows


def _gemini_image_call(image_bytes: bytes, prompt: str, max_output_tokens: int = 2048) -> str:
    """Call Gemini Flash with a single image + text prompt. Returns stripped response text."""
    from core.config import generate_with_retry, MODEL_FLASH
    from google.genai import types as genai_types

    contents = [
        genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        genai_types.Part.from_text(text=prompt),
    ]
    try:
        response = generate_with_retry(
            model=MODEL_FLASH,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            ),
        )
        return response.text.strip() if response.text else ""
    except Exception as exc:
        logger.error(f"Gemini image call failed: {exc}")
        return ""


class MpDocVqaBenchmark(BaseBenchmark):
    """MP-DocVQA benchmark with per-document retrieval and full caching.

    Mode A (ocr_text_only): OCR-only text, flat chunks, text-only embedding, flat retrieval.
    Mode C (multimodal_page): multimodal page leaves, table P-C, RAPTOR tree, collapsed retrieval.

    Per-document retrieval matches the evaluation setup from López et al., 2025,
    isolating comprehension quality from corpus-level retrieval noise.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self._doc_indices: dict[str, dict] = {}
        self._cache_root: Optional[Path] = None
        self._top_k: int = self.config.get("top_k", DEFAULT_TOP_K)
        self._sample_size: Optional[int] = self.config.get("sample_size")
        self._doc_processing: str = self.config.get("doc_processing", "ocr_text_only")
        self._build_raptor: bool = self.config.get("build_raptor_tree", False)
        self._rebuild_trees: bool = self.config.get("rebuild_trees", False)
        self._use_table_pc: bool = self.config.get("use_table_parent_child", False)
        self._use_expansion: bool = self.config.get("use_retrieval_expansion", False)
        self._use_multimodal: bool = self.config.get("use_multimodal_embed", False)
        self._shard_id: int = self.config.get("shard_id", 0)
        self._num_shards: int = self.config.get("num_shards", 1)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        # Page accuracies tracked per-question, in sync with predictions list
        self._page_accuracies: list[float] = []

    def _warn_unimplemented_toggles(self) -> None:
        # All ablation toggles are implemented for MP-DocVQA
        pass

    # ------------------------------------------------------------------
    # Checkpoint override — includes page_accuracies alongside predictions
    # ------------------------------------------------------------------

    def _load_checkpoint(
        self, checkpoint_path: Optional[Path]
    ) -> tuple[list[str], list[list[str]], int]:
        if checkpoint_path is None or not checkpoint_path.exists():
            return [], [], 0
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            predictions = data.get("predictions", [])
            ground_truths = data.get("ground_truths", [])
            completed = data.get("completed_index", 0)
            self._page_accuracies = data.get("page_accuracies", [])[:completed]
            logger.info(f"Resumed from checkpoint at index {completed}/{len(self.dataset or [])}")
            return predictions, ground_truths, completed
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Corrupt checkpoint, starting fresh: {e}")
            return [], [], 0

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        predictions: list[str],
        ground_truths: list[list[str]],
        completed_index: int,
    ) -> None:
        data = {
            "completed_index": completed_index,
            "predictions": predictions,
            "ground_truths": ground_truths,
            "page_accuracies": self._page_accuracies,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, checkpoint_path)

    # ------------------------------------------------------------------
    # Dataset loading — group by doc, process each doc, build QA list
    # ------------------------------------------------------------------

    def load_dataset(self, data_dir: str = "datasets") -> None:
        from datasets import load_dataset as hf_load_dataset

        self._cache_root = Path(data_dir) / "mpdocvqa_cache"
        self._cache_root.mkdir(parents=True, exist_ok=True)
        for subdir in (
            _OCR_DIR, _CAPTION_DIR, _TABLE_DETECT_DIR, _TABLE_MD_DIR,
            _PAGE_IMAGE_DIR, _EMBED_DIR, _TREE_DIR,
        ):
            (self._cache_root / subdir).mkdir(exist_ok=True)

        logger.info("Loading MP-DocVQA val split from HuggingFace (streaming)...")
        # select_columns avoids downloading/decoding the 20 image columns entirely;
        # remove_columns still reads them before discarding, which is extremely slow.
        _METADATA_COLS = ["doc_id", "page_ids", "questionId", "question", "answers", "answer_page_idx"]
        ds = hf_load_dataset("lmms-lab/MP-DocVQA", split="val", streaming=True)
        ds = ds.select_columns(_METADATA_COLS)

        logger.info("Streaming dataset — grouping questions by document (metadata only)...")
        doc_map: dict[str, dict] = {}
        for sample in ds:
            doc_id: str = sample["doc_id"]
            if doc_id not in doc_map:
                if self._sample_size and len(doc_map) >= self._sample_size:
                    break
                doc_map[doc_id] = {
                    "page_ids": sample["page_ids"],
                    "questions": [],
                }
                if len(doc_map) % 50 == 0:
                    total_q = sum(len(d["questions"]) for d in doc_map.values())
                    print(f"  Streamed {len(doc_map)} unique docs, {total_q} questions...", end="\r")
            raw_answers = sample["answers"]
            answers = ast.literal_eval(raw_answers) if isinstance(raw_answers, str) else raw_answers
            doc_map[doc_id]["questions"].append({
                "questionId": sample["questionId"],
                "question": sample["question"],
                "answers": answers,
                "answer_page_idx": int(sample["answer_page_idx"]),
                "page_ids": sample["page_ids"],
            })

        doc_ids = list(doc_map.keys())
        if self._sample_size:
            logger.info(f"Sample size: streamed {len(doc_ids)} documents")

        if self._num_shards > 1:
            total_docs_full = len(doc_ids)
            shard_size = (total_docs_full + self._num_shards - 1) // self._num_shards
            start = self._shard_id * shard_size
            doc_ids = doc_ids[start : start + shard_size]
            logger.info(
                f"Shard {self._shard_id + 1}/{self._num_shards}: processing docs "
                f"{start}–{start + len(doc_ids) - 1} ({len(doc_ids)} of {total_docs_full} total)"
            )

        total_docs = len(doc_ids)
        logger.info(f"Processing {total_docs} documents ({self._doc_processing} mode)...")

        # Aggregate processing statistics
        agg: dict = {
            "total_docs": 0,
            "total_pages": 0,
            "docs_with_tables": 0,
            "total_table_children": 0,
            "total_raptor_summary": 0,
        }

        self.dataset = []

        for doc_idx, doc_id in enumerate(doc_ids):
            doc_data = doc_map[doc_id]
            page_ids: list[str] = doc_data["page_ids"]
            n_pages = len(page_ids)

            print(
                f"\nProcessing document {doc_idx + 1}/{total_docs} ({doc_id}): {n_pages} pages"
            )

            try:
                # Pass 2: fetch page images for this document on demand (cached after first download)
                self._load_document_images(doc_id, n_pages)

                if self._doc_processing == "multimodal_page":
                    self._process_doc_mode_c(doc_id, page_ids, agg)
                else:
                    self._process_doc_mode_a(doc_id, page_ids, agg)
            except Exception as exc:
                logger.error(
                    f"Failed to process document {doc_id}: {exc}", exc_info=True
                )
                continue

            agg["total_docs"] += 1
            agg["total_pages"] += n_pages

            for q in doc_data["questions"]:
                if not q["answers"]:
                    continue
                self.dataset.append({
                    "question": q["question"],
                    "answers": q["answers"],
                    "kwargs": {
                        "doc_id": doc_id,
                        "answer_page_idx": q["answer_page_idx"],
                        "page_ids": q["page_ids"],
                    },
                })

        # Final summary
        print(f"\n{'=' * 60}")
        print("MP-DocVQA Document Processing Stats:")
        print(f"  Total documents:        {agg['total_docs']}")
        avg_pages = agg["total_pages"] / max(agg["total_docs"], 1)
        print(f"  Avg pages/document:     {avg_pages:.1f}")
        print(f"  Documents with tables:  {agg['docs_with_tables']} ({agg['docs_with_tables'] / max(agg['total_docs'], 1) * 100:.1f}%)")
        if self._doc_processing == "multimodal_page":
            print(f"  Total table row children: {agg['total_table_children']}")
            print(f"  Total RAPTOR summary nodes: {agg['total_raptor_summary']}")
        print(f"  Total questions loaded: {len(self.dataset)}")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Mode A: OCR → flat chunks per page → text-only embed → flat retrieval
    # ------------------------------------------------------------------

    def _process_doc_mode_a(
        self, doc_id: str, page_ids: list[str], agg: dict
    ) -> None:
        """Build Mode A index: OCR-only flat chunks, text-only embeddings."""
        mode_suffix = "ocr_text_only"
        embed_cache = self._cache_root / _EMBED_DIR / f"{doc_id}_{mode_suffix}.npz"

        # Load from cache if available
        if embed_cache.exists():
            try:
                data = np.load(embed_cache, allow_pickle=True)
                chunks = data["chunks"].tolist()
                page_nos = data["page_nos"].tolist()
                emb_matrix = data["embeddings"]
                if len(chunks) == len(emb_matrix) == len(page_nos):
                    self._doc_indices[doc_id] = {
                        "chunks": chunks,
                        "chunk_page_nos": page_nos,
                        "chunk_parent_ids": [""] * len(chunks),
                        "embeddings": emb_matrix,
                        "tree_nodes": None,
                        "parent_store": {},
                    }
                    print(f"  Embeddings: [cached] {len(chunks)} flat chunks")
                    return
            except Exception as exc:
                logger.warning(f"Embed cache load failed for {doc_id}: {exc}")

        # OCR all pages (cached per page), split per page to preserve page_no attribution
        ocr_results: list[tuple[str, str]] = []  # (page_id, ocr_text)
        cached_count = 0
        new_count = 0
        for page_idx, page_id in enumerate(page_ids):
            cache_path = self._cache_root / _OCR_DIR / f"{doc_id}_p{page_idx}.txt"
            is_cached = cache_path.exists()
            ocr_text = self._get_or_compute_ocr(doc_id, page_idx)
            if is_cached:
                cached_count += 1
            else:
                new_count += 1
            if ocr_text:
                ocr_results.append((page_id, ocr_text))

        print(f"  OCR: {cached_count} cached | {new_count} new")

        if not ocr_results:
            logger.warning(f"No OCR text for document {doc_id} — skipping")
            self._doc_indices[doc_id] = {
                "chunks": [], "chunk_page_nos": [],
                "chunk_parent_ids": [], "embeddings": np.zeros((0, _EMBED_DIM), dtype=np.float32),
                "tree_nodes": None, "parent_store": {},
            }
            return

        # Split per page, tag chunks with their page_id
        chunks: list[str] = []
        chunk_page_nos: list[str] = []
        for page_id, ocr_text in ocr_results:
            page_header = f"--- Page {page_ids.index(page_id) + 1} ---\n\n"
            for chunk in self._splitter.split_text(page_header + ocr_text):
                chunks.append(chunk)
                chunk_page_nos.append(page_id)

        print(f"  Chunks: {len(chunks)} flat text chunks")

        # Text-only batch embedding
        from modules.embeddings import embed_batch
        t0 = time.time()
        embeddings = embed_batch(chunks)
        elapsed = time.time() - t0

        emb_matrix = np.zeros((len(chunks), _EMBED_DIM), dtype=np.float32)
        ok_count = 0
        for i, emb in enumerate(embeddings):
            if emb is not None:
                emb_matrix[i] = emb
                ok_count += 1

        print(f"  Embeddings: {ok_count}/{len(chunks)} text-only in {elapsed:.1f}s")

        try:
            np.savez_compressed(
                embed_cache,
                embeddings=emb_matrix,
                chunks=np.array(chunks, dtype=object),
                page_nos=np.array(chunk_page_nos, dtype=object),
            )
        except Exception as exc:
            logger.warning(f"Failed to save embedding cache for {doc_id}: {exc}")

        self._doc_indices[doc_id] = {
            "chunks": chunks,
            "chunk_page_nos": chunk_page_nos,
            "chunk_parent_ids": [""] * len(chunks),
            "embeddings": emb_matrix,
            "tree_nodes": None,
            "parent_store": {},
        }

    # ------------------------------------------------------------------
    # Mode C: multimodal page leaves + table P-C + RAPTOR tree
    # ------------------------------------------------------------------

    def _process_doc_mode_c(
        self, doc_id: str, page_ids: list[str], agg: dict
    ) -> None:
        """Build Mode C index: multimodal embeddings, table P-C, RAPTOR tree."""
        mode_suffix = "multimodal_page"
        embed_cache = self._cache_root / _EMBED_DIR / f"{doc_id}_{mode_suffix}.npz"
        tree_cache = self._cache_root / _TREE_DIR / f"{doc_id}_{mode_suffix}.npz"

        # --- Try loading full tree cache first ---
        if tree_cache.exists() and not self._rebuild_trees:
            result = self._load_tree_cache(doc_id, tree_cache)
            if result is not None:
                n_leaves = sum(1 for n in result["tree_nodes"] if n["level"] == 0)
                n_summary = sum(1 for n in result["tree_nodes"] if n["level"] > 0)
                n_levels = max((n["level"] for n in result["tree_nodes"]), default=0)
                agg["total_raptor_summary"] += n_summary
                self._doc_indices[doc_id] = result
                print(
                    f"  Tree: [cached] {n_leaves} leaves + {n_summary} summary nodes "
                    f"({n_levels} levels)"
                )
                return

        # --- Try loading leaf embeddings cache ---
        leaf_data: Optional[dict] = None
        if embed_cache.exists():
            leaf_data = self._load_embed_cache(doc_id, embed_cache)
            if leaf_data is not None:
                print(f"  Embeddings: [cached] {len(leaf_data['chunks'])} leaf nodes")

        if leaf_data is None:
            # Build chunks and embeddings from scratch
            chunks, page_nos, parent_ids, parent_texts, has_table, n_table_children = (
                self._build_mode_c_chunks(doc_id, page_ids)
            )
            if has_table:
                agg["docs_with_tables"] += 1
            agg["total_table_children"] += n_table_children

            emb_matrix = self._embed_mode_c_chunks(doc_id, chunks, page_nos, parent_ids, page_ids)

            leaf_data = {
                "chunks": chunks,
                "chunk_page_nos": page_nos,
                "chunk_parent_ids": parent_ids,
                "embeddings": emb_matrix,
                "parent_store": parent_texts,
            }
            self._save_embed_cache(embed_cache, leaf_data)

        chunks = leaf_data["chunks"]
        page_nos = leaf_data["chunk_page_nos"]
        parent_ids = leaf_data["chunk_parent_ids"]
        emb_matrix = leaf_data["embeddings"]
        parent_texts = leaf_data["parent_store"]

        # Build RAPTOR tree from valid leaf embeddings
        leaf_nodes: list[dict] = []
        for j in range(len(chunks)):
            if np.any(emb_matrix[j]):
                leaf_nodes.append({
                    "text": chunks[j],
                    "embedding": emb_matrix[j].copy(),
                    "level": 0,
                    "parent_id": parent_ids[j],
                    "page_no": page_nos[j],
                })

        if not leaf_nodes:
            logger.warning(f"No valid leaf embeddings for {doc_id}")
            self._doc_indices[doc_id] = {
                "chunks": chunks,
                "chunk_page_nos": page_nos,
                "chunk_parent_ids": parent_ids,
                "embeddings": emb_matrix,
                "tree_nodes": [],
                "parent_store": parent_texts,
            }
            return

        from benchmarks.raptor_tree import build_raptor_tree
        t0 = time.time()
        tree_nodes = build_raptor_tree(leaf_nodes)
        elapsed = time.time() - t0

        n_summary = sum(1 for n in tree_nodes if n["level"] > 0)
        n_levels = max((n["level"] for n in tree_nodes), default=0)
        agg["total_raptor_summary"] += n_summary
        print(
            f"  Tree: {len(leaf_nodes)} leaves + {n_summary} summary nodes "
            f"({n_levels} levels) [{elapsed:.1f}s]"
        )

        self._save_tree_cache(tree_cache, tree_nodes, parent_texts)
        self._doc_indices[doc_id] = {
            "chunks": chunks,
            "chunk_page_nos": page_nos,
            "chunk_parent_ids": parent_ids,
            "embeddings": emb_matrix,
            "tree_nodes": tree_nodes,
            "parent_store": parent_texts,
        }

    def _build_mode_c_chunks(
        self, doc_id: str, page_ids: list[str]
    ) -> tuple[list[str], list[str], list[str], dict[str, str], bool, int]:
        """Build embeddable leaf chunks for Mode C (without embedding).

        Returns: (chunks, page_nos, parent_ids, parent_texts, has_table, n_table_children)
        """
        chunks: list[str] = []
        chunk_page_nos: list[str] = []
        chunk_parent_ids: list[str] = []
        parent_texts: dict[str, str] = {}
        doc_has_table = False
        n_table_children = 0

        ocr_cached = new_ocr = 0
        cap_cached = new_cap = 0
        tables_detected = 0

        for page_idx, page_id in enumerate(page_ids):
            img_path = self._cache_root / _PAGE_IMAGE_DIR / f"{doc_id}_p{page_idx}.png"
            if not img_path.exists():
                logger.warning(f"Page image missing: {img_path}, skipping page {page_idx}")
                continue

            with open(img_path, "rb") as fh:
                image_bytes = fh.read()

            # OCR (cached)
            ocr_cache = self._cache_root / _OCR_DIR / f"{doc_id}_p{page_idx}.txt"
            is_ocr_cached = ocr_cache.exists()
            ocr_text = self._get_or_compute_ocr(doc_id, page_idx)
            if is_ocr_cached:
                ocr_cached += 1
            else:
                new_ocr += 1

            # Caption (cached)
            cap_cache = self._cache_root / _CAPTION_DIR / f"{doc_id}_p{page_idx}.txt"
            is_cap_cached = cap_cache.exists()
            caption = self._get_or_compute_caption(doc_id, page_idx, image_bytes)
            if is_cap_cached:
                cap_cached += 1
            else:
                new_cap += 1

            # Table detection + extraction (cached)
            has_table_on_page = self._get_or_compute_table_detect(doc_id, page_idx, image_bytes)
            if has_table_on_page:
                tables_detected += 1
                doc_has_table = True
                table_md = self._get_or_compute_table_extract(doc_id, page_idx, image_bytes)
                if table_md:
                    parent_id = f"{doc_id}_tbl_{page_idx:03d}"
                    parent_texts[parent_id] = table_md
                    for row in _parse_markdown_table_rows(table_md):
                        child_text = f"TABLE ROW — Table on page {page_idx + 1}: {row}"
                        chunks.append(child_text)
                        chunk_page_nos.append(page_id)
                        chunk_parent_ids.append(parent_id)
                        n_table_children += 1

            # Page-level leaf: OCR text + caption
            page_text = ocr_text
            if caption:
                page_text += f"\nDiagrams on this page:\n- {caption}"

            if not page_text.strip():
                continue

            # Split if over chunk_size
            if len(page_text) > CHUNK_SIZE:
                sub_chunks = self._splitter.split_text(page_text)
            else:
                sub_chunks = [page_text]

            for sub in sub_chunks:
                chunks.append(sub)
                chunk_page_nos.append(page_id)
                chunk_parent_ids.append("")  # no parent for page leaves

        print(
            f"  OCR: {ocr_cached} cached | {new_ocr} new  "
            f"| Captions: {cap_cached} cached | {new_cap} new  "
            f"| Tables detected: {tables_detected}"
        )
        print(
            f"  Chunks: {len(chunks) - n_table_children} prose + "
            f"{n_table_children} table children + {len(parent_texts)} parents"
        )

        return chunks, chunk_page_nos, chunk_parent_ids, parent_texts, doc_has_table, n_table_children

    def _embed_mode_c_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        page_nos: list[str],
        parent_ids: list[str],
        page_ids: list[str],
    ) -> np.ndarray:
        """Embed Mode C leaf chunks.

        Page leaves: multimodal embed (text + page image) when use_multimodal is True.
        Table row children: text-only embed (parent_id != "").
        """
        from modules.embeddings import embed, embed_batch

        # Preload image bytes per page_id
        page_image_bytes: dict[str, Optional[bytes]] = {}
        for page_idx, page_id in enumerate(page_ids):
            img_path = self._cache_root / _PAGE_IMAGE_DIR / f"{doc_id}_p{page_idx}.png"
            if img_path.exists():
                try:
                    with open(img_path, "rb") as fh:
                        page_image_bytes[page_id] = fh.read()
                except Exception:
                    page_image_bytes[page_id] = None
            else:
                page_image_bytes[page_id] = None

        # Separate table children (text-only batch) from page leaves (per-call)
        table_child_indices: list[int] = []
        page_leaf_indices: list[int] = []
        for i, pid in enumerate(parent_ids):
            if pid:
                table_child_indices.append(i)
            else:
                page_leaf_indices.append(i)

        emb_matrix = np.zeros((len(chunks), _EMBED_DIM), dtype=np.float32)
        ok_count = 0
        t0 = time.time()

        # Batch-embed table children (text-only)
        if table_child_indices:
            child_texts = [chunks[i] for i in table_child_indices]
            child_embs = embed_batch(child_texts)
            for j, (idx, emb) in enumerate(zip(table_child_indices, child_embs)):
                if emb is not None:
                    emb_matrix[idx] = emb
                    ok_count += 1

        # Embed page leaves (multimodal or text-only per chunk)
        def _embed_one(idx: int) -> tuple[int, Optional[list[float]]]:
            chunk_text = chunks[idx]
            page_id = page_nos[idx]
            img_bytes = page_image_bytes.get(page_id)
            if self._use_multimodal and img_bytes:
                emb = embed(text=chunk_text, image_bytes_list=[img_bytes])
            else:
                emb = embed(text=chunk_text)
            return idx, emb

        with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
            futures = {pool.submit(_embed_one, idx): idx for idx in page_leaf_indices}
            for fut in as_completed(futures):
                idx, emb = fut.result()
                if emb is not None:
                    emb_matrix[idx] = emb
                    ok_count += 1

        elapsed = time.time() - t0
        print(f"  Embeddings: {ok_count}/{len(chunks)} in {elapsed:.1f}s")
        return emb_matrix

    # ------------------------------------------------------------------
    # Per-page Gemini calls with caching
    # ------------------------------------------------------------------

    def _get_or_compute_ocr(self, doc_id: str, page_idx: int) -> str:
        cache_path = self._cache_root / _OCR_DIR / f"{doc_id}_p{page_idx}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        img_path = self._cache_root / _PAGE_IMAGE_DIR / f"{doc_id}_p{page_idx}.png"
        if not img_path.exists():
            return ""
        with open(img_path, "rb") as fh:
            image_bytes = fh.read()
        text = _gemini_image_call(image_bytes, _OCR_PROMPT, max_output_tokens=4096)
        cache_path.write_text(text, encoding="utf-8")
        return text

    def _get_or_compute_caption(
        self, doc_id: str, page_idx: int, image_bytes: bytes
    ) -> str:
        cache_path = self._cache_root / _CAPTION_DIR / f"{doc_id}_p{page_idx}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        text = _gemini_image_call(image_bytes, _CAPTION_PROMPT, max_output_tokens=256)
        cache_path.write_text(text, encoding="utf-8")
        return text

    def _get_or_compute_table_detect(
        self, doc_id: str, page_idx: int, image_bytes: bytes
    ) -> bool:
        cache_path = self._cache_root / _TABLE_DETECT_DIR / f"{doc_id}_p{page_idx}.txt"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8").strip().upper() == "YES"
        response = _gemini_image_call(image_bytes, _TABLE_DETECT_PROMPT, max_output_tokens=8)
        result = response.strip().upper()
        # Normalize to YES or NO
        is_yes = result.startswith("YES")
        cache_path.write_text("YES" if is_yes else "NO", encoding="utf-8")
        return is_yes

    def _get_or_compute_table_extract(
        self, doc_id: str, page_idx: int, image_bytes: bytes
    ) -> str:
        cache_path = self._cache_root / _TABLE_MD_DIR / f"{doc_id}_p{page_idx}.md"
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")
        text = _gemini_image_call(image_bytes, _TABLE_EXTRACT_PROMPT, max_output_tokens=2048)
        cache_path.write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------------
    # Cache save/load helpers for embedding and tree data
    # ------------------------------------------------------------------

    def _load_document_images(self, doc_id: str, num_pages: int) -> None:
        """Fetch and cache page images for a single document.

        Streams the HF dataset (with images) until the target document is found,
        then writes each page image to disk.  Skips entirely when all pages are
        already cached.  O(N) over the stream on a cold cache; O(1) on warm cache.
        """
        from datasets import load_dataset as hf_load_dataset

        cache_dir = self._cache_root / _PAGE_IMAGE_DIR
        if all((cache_dir / f"{doc_id}_p{i}.png").exists() for i in range(num_pages)):
            return

        logger.info(f"Downloading page images for {doc_id} ({num_pages} pages)...")
        ds = hf_load_dataset("lmms-lab/MP-DocVQA", split="val", streaming=True)
        for sample in ds:
            if sample["doc_id"] != doc_id:
                continue
            for i in range(1, 21):
                img = sample.get(f"image_{i}")
                if img is None:
                    break
                img_path = cache_dir / f"{doc_id}_p{i - 1}.png"
                if not img_path.exists():
                    try:
                        img.save(str(img_path))
                    except Exception as exc:
                        logger.warning(f"Failed to save page image {img_path}: {exc}")
            break  # First occurrence of this doc contains all pages

    # ------------------------------------------------------------------

    def _save_embed_cache(self, cache_path: Path, leaf_data: dict) -> None:
        """Save leaf-level chunk data (without tree) to npz."""
        parent_texts: dict = leaf_data["parent_store"]
        keys = list(parent_texts.keys())
        vals = [parent_texts[k] for k in keys]
        try:
            np.savez_compressed(
                cache_path,
                embeddings=leaf_data["embeddings"],
                chunks=np.array(leaf_data["chunks"], dtype=object),
                page_nos=np.array(leaf_data["chunk_page_nos"], dtype=object),
                parent_ids=np.array(leaf_data["chunk_parent_ids"], dtype=object),
                parent_text_keys=np.array(keys, dtype=object),
                parent_text_vals=np.array(vals, dtype=object),
            )
        except Exception as exc:
            logger.warning(f"Failed to save embedding cache at {cache_path}: {exc}")

    def _load_embed_cache(self, doc_id: str, cache_path: Path) -> Optional[dict]:
        """Load leaf-level chunk data from npz. Returns None on failure."""
        try:
            data = np.load(cache_path, allow_pickle=True)
            chunks = data["chunks"].tolist()
            page_nos = data["page_nos"].tolist()
            parent_ids = data["parent_ids"].tolist()
            emb_matrix = data["embeddings"]
            if len(chunks) != len(emb_matrix) or len(chunks) != len(page_nos):
                logger.warning(f"Shape mismatch in embed cache for {doc_id}")
                return None
            keys = data["parent_text_keys"].tolist() if "parent_text_keys" in data.files else []
            vals = data["parent_text_vals"].tolist() if "parent_text_vals" in data.files else []
            return {
                "chunks": chunks,
                "chunk_page_nos": page_nos,
                "chunk_parent_ids": parent_ids,
                "embeddings": emb_matrix,
                "parent_store": dict(zip(keys, vals)),
            }
        except Exception as exc:
            logger.warning(f"Embed cache load failed for {doc_id}: {exc}")
            return None

    def _save_tree_cache(
        self, cache_path: Path, tree_nodes: list[dict], parent_texts: dict[str, str]
    ) -> None:
        """Save full RAPTOR tree + parent texts to npz."""
        keys = list(parent_texts.keys())
        vals = [parent_texts[k] for k in keys]
        try:
            np.savez_compressed(
                cache_path,
                texts=np.array([n["text"] for n in tree_nodes], dtype=object),
                embeddings=np.array(
                    [n["embedding"] for n in tree_nodes], dtype=np.float32
                ),
                levels=np.array([n["level"] for n in tree_nodes], dtype=np.int32),
                parent_ids=np.array(
                    [n.get("parent_id", "") for n in tree_nodes], dtype=object
                ),
                page_nos=np.array(
                    [n.get("page_no", "") for n in tree_nodes], dtype=object
                ),
                parent_text_keys=np.array(keys, dtype=object),
                parent_text_vals=np.array(vals, dtype=object),
            )
        except Exception as exc:
            logger.warning(f"Failed to save tree cache at {cache_path}: {exc}")

    def _load_tree_cache(self, doc_id: str, cache_path: Path) -> Optional[dict]:
        """Load full RAPTOR tree index from npz. Returns None on failure."""
        try:
            data = np.load(cache_path, allow_pickle=True)
            texts = data["texts"].tolist()
            embeddings = data["embeddings"]
            levels = data["levels"].tolist()
            parent_ids = data["parent_ids"].tolist()
            page_nos = data["page_nos"].tolist()
            if not (len(texts) == len(embeddings) == len(levels)):
                return None

            tree_nodes = [
                {
                    "text": texts[j],
                    "embedding": embeddings[j],
                    "level": int(levels[j]),
                    "parent_id": parent_ids[j] if j < len(parent_ids) else "",
                    "page_no": page_nos[j] if j < len(page_nos) else "",
                }
                for j in range(len(texts))
            ]

            keys = data["parent_text_keys"].tolist() if "parent_text_keys" in data.files else []
            vals = data["parent_text_vals"].tolist() if "parent_text_vals" in data.files else []

            # Reconstruct leaf-level chunk arrays from tree_nodes (level==0)
            leaf_nodes = [n for n in tree_nodes if n["level"] == 0]

            return {
                "chunks": [n["text"] for n in leaf_nodes],
                "chunk_page_nos": [n["page_no"] for n in leaf_nodes],
                "chunk_parent_ids": [n["parent_id"] for n in leaf_nodes],
                "embeddings": np.array([n["embedding"] for n in leaf_nodes], dtype=np.float32),
                "tree_nodes": tree_nodes,
                "parent_store": dict(zip(keys, vals)),
            }
        except Exception as exc:
            logger.warning(f"Tree cache load failed for {doc_id}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Retrieval — per-document cosine search
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, doc_id: str) -> list[dict]:
        """Retrieve top-K chunks from the document's index.

        Returns list of dicts with keys: text, page_no, parent_id, level.
        For Mode C with RAPTOR: searches all tree levels (collapsed).
        For Mode A: searches leaf chunks only.
        """
        from modules.embeddings import embed

        query_emb = embed(text=question)
        if query_emb is None:
            logger.error(f"Failed to embed question: {question[:80]}")
            return []

        query_vec = np.array(query_emb, dtype=np.float32)
        index = self._doc_indices.get(doc_id)
        if index is None:
            logger.error(f"No index found for document {doc_id}")
            return []

        # Collapsed retrieval: search all tree levels if tree exists
        if self._build_raptor and self.retrieval_mode == "collapsed":
            tree_nodes = index.get("tree_nodes") or []
            if tree_nodes:
                all_embs = np.array(
                    [n["embedding"] for n in tree_nodes], dtype=np.float32
                )
                scores = _cosine_similarity_batch(query_vec, all_embs)
                top_indices = np.argsort(scores)[::-1][: self._top_k]
                return [
                    {
                        "text": tree_nodes[i]["text"],
                        "page_no": tree_nodes[i].get("page_no", ""),
                        "parent_id": tree_nodes[i].get("parent_id", ""),
                        "level": tree_nodes[i].get("level", 0),
                    }
                    for i in top_indices
                ]
            logger.warning(f"No tree nodes for {doc_id}, falling back to flat retrieval")

        # Flat retrieval: search leaf chunks only
        emb_matrix = index.get("embeddings")
        chunks = index.get("chunks", [])
        page_nos = index.get("chunk_page_nos", [])
        parent_ids = index.get("chunk_parent_ids", [])

        if emb_matrix is None or len(chunks) == 0:
            return []

        scores = _cosine_similarity_batch(query_vec, emb_matrix)
        top_indices = np.argsort(scores)[::-1][: self._top_k]
        return [
            {
                "text": chunks[i],
                "page_no": page_nos[i] if i < len(page_nos) else "",
                "parent_id": parent_ids[i] if i < len(parent_ids) else "",
                "level": 0,
            }
            for i in top_indices
        ]

    def _expand_parents(self, retrieved: list[dict], doc_id: str) -> list[str]:
        """Swap table row child hits for their full parent table text.

        Multiple children from the same parent are deduplicated.
        Non-child nodes pass through unchanged.
        """
        parent_store = self._doc_indices.get(doc_id, {}).get("parent_store", {})
        if not parent_store:
            return [r["text"] for r in retrieved]

        expanded: list[str] = []
        seen_parents: set[str] = set()
        for node in retrieved:
            pid = node.get("parent_id", "")
            if pid and pid in parent_store:
                if pid not in seen_parents:
                    expanded.append(parent_store[pid])
                    seen_parents.add(pid)
            else:
                expanded.append(node["text"])
        return expanded

    def _compute_page_accuracy(
        self, retrieved: list[dict], answer_page_idx: int, page_ids: list[str]
    ) -> float:
        """1.0 if any retrieved chunk came from the answer page, else 0.0."""
        if not page_ids or answer_page_idx < 0 or answer_page_idx >= len(page_ids):
            return 0.0
        correct_page = page_ids[answer_page_idx]
        hit = any(r.get("page_no") == correct_page for r in retrieved)
        return 1.0 if hit else 0.0

    # ------------------------------------------------------------------
    # run_single and evaluate
    # ------------------------------------------------------------------

    def run_single(
        self,
        question: str,
        doc_id: str = "",
        answer_page_idx: int = 0,
        page_ids: Optional[list] = None,
        **kwargs,
    ) -> str:
        """Retrieve context, compute page accuracy, generate answer."""
        from core.config import generate_with_retry, MODEL_FLASH
        from google.genai import types as genai_types

        retrieved = self._retrieve(question, doc_id)

        # Track page accuracy (side-effect, consumed by evaluate())
        pa = self._compute_page_accuracy(retrieved, answer_page_idx, page_ids or [])
        self._page_accuracies.append(pa)

        if not retrieved:
            return "unanswerable"

        # Parent expansion for table row children
        if self._use_expansion:
            texts = self._expand_parents(retrieved, doc_id)
        else:
            texts = [r["text"] for r in retrieved]

        context = "\n\n---\n\n".join(texts)
        prompt = _QA_PROMPT.format(retrieved_chunks=context, question=question)

        try:
            response = generate_with_retry(
                model=MODEL_FLASH,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip() if response.text else "unanswerable"
        except Exception as exc:
            logger.error(f"Generation failed for '{question[:60]}': {exc}")
            return "unanswerable"

    def evaluate(
        self,
        predictions: list[str],
        ground_truths: list[list[str]],
    ) -> dict[str, float]:
        anls_scores = [
            score_with_multiple_gts(pred, gts, anls_score)
            for pred, gts in zip(predictions, ground_truths)
        ]
        avg_anls = sum(anls_scores) / len(anls_scores) if anls_scores else 0.0

        if self._page_accuracies and len(self._page_accuracies) == len(predictions):
            page_acc = sum(self._page_accuracies) / len(self._page_accuracies)
        else:
            page_acc = 0.0
            if self._page_accuracies:
                logger.warning(
                    f"page_accuracies length ({len(self._page_accuracies)}) "
                    f"doesn't match predictions ({len(predictions)}) — skipping page_accuracy"
                )

        return {
            "anls": avg_anls,
            "page_accuracy": page_acc,
        }
