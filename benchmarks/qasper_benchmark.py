"""
QASPER benchmark runner.

Dataset: allenai/qasper v0.3 (downloaded from AllenAI S3)
Primary metric: Token-level F1

Architecture:
  - Each QASPER paper is chunked and embedded independently
  - Leaf embeddings are cached to disk (datasets/qasper_embeddings/) for re-runs
  - When build_raptor_tree=true: RAPTOR trees are built per paper and cached
    to datasets/qasper_trees/. Trees are built eagerly during load_dataset so
    the full 44-minute build runs once; subsequent runs load from cache.
  - For each question: embed query → cosine search → LLM answer
    - retrieval_mode=flat: searches leaf nodes (level=0) only
    - retrieval_mode=collapsed: searches all RAPTOR levels (leaves + summaries)

Row C (use_table_parent_child=true + use_retrieval_expansion=true):
  - Detects markdown-style tables in paper text
  - Removes table spans from prose before chunking (avoids double-counting)
  - Table rows become "child" leaf nodes; full table text stored as "parent"
  - At retrieval time: child hits are swapped for their full parent table text
  - Separate embedding + tree caches to preserve Row B caches

Downloads train+dev and test tarballs from AllenAI S3 (auto-cached).
Default split: dev (has ground truth answers; test also has answers in v0.3).
"""

import json
import logging
import os
import re
import ssl
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from benchmarks.base_benchmark import BaseBenchmark
from metrics import f1_token_score, exact_match, score_with_multiple_gts

logger = logging.getLogger(__name__)

# AllenAI S3 URLs for QASPER v0.3
_TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5

# Separate cache dirs for Row C to preserve Row B caches
_EMBED_CACHE_DIR = "qasper_embeddings"
_TREE_CACHE_DIR = "qasper_trees"
_EMBED_CACHE_DIR_MM = "qasper_embeddings_multimodal"
_TREE_CACHE_DIR_MM = "qasper_trees_multimodal"

# Map split names to archive URLs and expected JSON filenames
_SPLIT_MAP = {
    "train": (_TRAIN_DEV_URL, "qasper-train-v0.3.json"),
    "dev": (_TRAIN_DEV_URL, "qasper-dev-v0.3.json"),
    "validation": (_TRAIN_DEV_URL, "qasper-dev-v0.3.json"),  # alias
    "test": (_TEST_URL, "qasper-test-v0.3.json"),
}


def _download_and_extract(url: str, cache_dir: Path) -> Path:
    """Download a tarball and extract to cache_dir if not already present."""
    tarball_name = url.rsplit("/", 1)[-1]
    tarball_path = cache_dir / tarball_name
    extract_dir = cache_dir / tarball_name.replace(".tgz", "")

    if extract_dir.exists():
        return extract_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not tarball_path.exists():
        logger.info(f"Downloading {url}...")
        # Handle macOS SSL cert issues by trying certifi first
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl.create_default_context()
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_ctx) as response:
            with open(tarball_path, "wb") as out_file:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    out_file.write(chunk)
        logger.info(f"Downloaded to {tarball_path}")

    logger.info(f"Extracting {tarball_path}...")
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    return extract_dir


def _load_qasper_json(data_dir: str, split: str) -> dict:
    """Download and load a QASPER split as a dict keyed by paper ID."""
    if split not in _SPLIT_MAP:
        raise ValueError(
            f"Unknown split '{split}'. Valid: {list(_SPLIT_MAP.keys())}"
        )

    url, json_filename = _SPLIT_MAP[split]
    cache_dir = Path(data_dir) / "qasper_raw"
    extract_dir = _download_and_extract(url, cache_dir)

    # Find the JSON file in extracted directory
    json_path = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == json_filename:
                json_path = Path(root) / f
                break
        if json_path:
            break

    if json_path is None:
        raise FileNotFoundError(
            f"Could not find {json_filename} in {extract_dir}"
        )

    with open(json_path) as f:
        return json.load(f)


def _extract_paper_text(paper: dict) -> str:
    """Extract concatenated full text from a QASPER paper.

    QASPER JSON format:
        full_text: [{"section_name": str, "paragraphs": [str, ...]}, ...]
    """
    sections: list[str] = []

    abstract = paper.get("abstract", "")
    if abstract:
        sections.append(f"Abstract\n{abstract}")

    for section in paper.get("full_text", []):
        sec_name = section.get("section_name", "")
        paragraphs = section.get("paragraphs", [])
        if not paragraphs:
            continue
        section_text = "\n".join(p for p in paragraphs if p)
        if section_text.strip():
            header = sec_name if sec_name else "Untitled Section"
            sections.append(f"{header}\n{section_text}")

    return "\n\n".join(sections)


def _extract_ground_truths(qa: dict) -> list[str]:
    """Extract all valid ground truth answers from a QASPER question.

    QASPER JSON format:
        answers: [{"answer": {"unanswerable": bool, "extractive_spans": [...],
                              "yes_no": null/bool, "free_form_answer": str,
                              "evidence": [...], "highlighted_evidence": [...]},
                   "annotation_id": str, "worker_id": str}, ...]

    Answer type priority: unanswerable > yes_no > extractive_spans > free_form_answer.
    Multiple annotators produce multiple valid GTs. Evaluation takes max score.
    """
    gts: list[str] = []
    for annotator in qa.get("answers", []):
        answer = annotator.get("answer", {})
        gt = _parse_single_answer(answer)
        if gt:
            gts.append(gt)
    return gts


def _parse_single_answer(answer: dict) -> str | None:
    """Parse one annotator's answer into a ground truth string."""
    if not isinstance(answer, dict):
        return None

    if answer.get("unanswerable"):
        return "unanswerable"

    yes_no = answer.get("yes_no")
    if yes_no is not None:
        return "yes" if yes_no else "no"

    spans = answer.get("extractive_spans", [])
    if spans and isinstance(spans, list):
        joined = " ".join(s for s in spans if s)
        if joined.strip():
            return joined

    ff = answer.get("free_form_answer", "")
    if ff and isinstance(ff, str) and ff.strip():
        return ff

    return None


def _cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors.

    Args:
        query: (D,) query embedding
        matrix: (N, D) chunk embeddings

    Returns:
        (N,) cosine similarity scores
    """
    q_norm = np.linalg.norm(query)
    if q_norm < 1e-10:
        return np.zeros(matrix.shape[0])
    q_hat = query / q_norm

    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms = np.maximum(m_norms, 1e-10)
    m_hat = matrix / m_norms

    return m_hat @ q_hat


def detect_text_tables(text: str) -> list[dict]:
    """Detect markdown-style table structures in plain text.

    Scans for runs of lines with 2+ pipe characters ('|'), which indicates a
    pipe-delimited table. Looks backwards from each table block for a
    "Table N:" caption line (up to 5 lines back).

    Returns list of dicts:
        header:     str  — caption text or "Table at line N"
        full_text:  str  — caption + all table lines joined with \\n
        row_lines:  list[str] — non-separator rows, used to generate child chunks
        start_line: int  — first line index to remove (caption if found, else first table line)
        end_line:   int  — last table line index to remove (inclusive, 0-based)
    """
    lines = text.split("\n")
    tables: list[dict] = []
    i = 0

    while i < len(lines):
        if lines[i].count("|") < 2:
            i += 1
            continue

        # Collect the run of consecutive pipe-delimited lines
        table_start = i
        table_lines: list[str] = []
        while i < len(lines) and lines[i].count("|") >= 2:
            table_lines.append(lines[i])
            i += 1

        # Require at least 2 lines to qualify as a table
        if len(table_lines) < 2:
            continue

        # Look backwards for a "Table N:" / "Tab. N." caption
        caption = ""
        caption_line_idx = table_start
        for back in range(1, min(6, table_start + 1)):
            candidate = lines[table_start - back].strip()
            if re.match(r"^(Table|Tab\.?)\s*\d+[:\.]?\s*", candidate, re.IGNORECASE):
                caption = candidate
                caption_line_idx = table_start - back
                break

        # Skip pure separator rows (e.g. |---|---| or |===|) for child generation
        row_lines = [
            ln for ln in table_lines
            if not re.match(r"^[\s|:\-=]+$", ln)
        ]
        if not row_lines:
            continue

        header = caption if caption else f"Table at line {table_start + 1}"
        full_table_text = "\n".join(([caption] if caption else []) + table_lines)

        tables.append({
            "header": header,
            "full_text": full_table_text,
            "row_lines": row_lines,
            "start_line": caption_line_idx,
            "end_line": i - 1,  # inclusive last pipe-line index
        })

    return tables


def _remove_table_spans(text: str, tables: list[dict]) -> str:
    """Remove detected table lines (and their caption lines) from text.

    Prevents table content from being double-counted — once as full prose
    chunks and again as parent-child rows.
    """
    if not tables:
        return text
    lines = text.split("\n")
    remove: set[int] = set()
    for t in tables:
        for idx in range(t["start_line"], t["end_line"] + 1):
            remove.add(idx)
    return "\n".join(ln for idx, ln in enumerate(lines) if idx not in remove)


QA_GENERATION_PROMPT = """\
You are answering questions about a research paper based on retrieved passages.

Rules:
- Answer concisely and directly based ONLY on the provided context.
- For yes/no questions, respond with just "yes" or "no".
- If the answer cannot be determined from the context, respond with exactly "unanswerable".
- Do not add explanations or caveats — just the answer.

Context:
{context}

Question: {question}

Answer:"""


class QasperBenchmark(BaseBenchmark):
    """QASPER benchmark with in-memory per-paper chunk retrieval.

    Each paper's text is chunked and embedded once (cached to disk).
    When build_raptor_tree=true, RAPTOR trees are built eagerly during
    load_dataset and cached separately so subsequent runs skip the ~44-min
    build phase.

    Questions are answered by retrieving top-K chunks from the paper
    and generating an answer with Gemini Flash.

    Row C extensions (use_table_parent_child + use_retrieval_expansion):
      - Tables in paper text are detected and extracted before prose chunking
      - Each table gets a parent node (full table text, not embedded) and
        child nodes (individual rows, embedded and searchable)
      - At retrieval time, child hits are swapped for their full parent text
      - Separate embedding and tree caches avoid overwriting Row B caches
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)
        self._paper_indices: dict[str, dict] = {}
        self._cache_dir: Path | None = None
        self._tree_cache_dir: Path | None = None
        self._top_k: int = self.config.get("top_k", DEFAULT_TOP_K)
        self._sample_size: int | None = self.config.get("sample_size")
        self._build_raptor: bool = self.config.get("build_raptor_tree", False)
        self._rebuild_trees: bool = self.config.get("rebuild_trees", False)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        # Row C flags
        self._use_table_pc: bool = self.config.get("use_table_parent_child", False)
        self._use_expansion: bool = self.config.get("use_retrieval_expansion", False)
        # parent_id → full table text, keyed by paper_id
        self._parent_stores: dict[str, dict[str, str]] = {}
        # Aggregate table detection stats (populated during load_dataset)
        self._table_agg: dict = {
            "papers_with_tables": 0,
            "total_tables": 0,
            "total_rows": 0,
            "papers_total": 0,
            "questions_with_tables": 0,
            "questions_total": 0,
        }

    def _warn_unimplemented_toggles(self) -> None:
        # QASPER implements use_table_parent_child and use_retrieval_expansion.
        # Only warn about use_multimodal_embed when it's True (images unsupported).
        mm_embed = self.config.get("use_multimodal_embed")
        if mm_embed is True:
            logger.warning(
                "Toggle 'use_multimodal_embed' set to True, but QASPER has no image "
                "bytes — multimodal embedding is not applied."
            )

    def load_dataset(self, data_dir: str = "datasets") -> None:
        # Choose cache dirs: multimodal path uses separate dirs to preserve Row B
        embed_dir = _EMBED_CACHE_DIR_MM if self._use_table_pc else _EMBED_CACHE_DIR
        tree_dir = _TREE_CACHE_DIR_MM if self._use_table_pc else _TREE_CACHE_DIR

        self._cache_dir = Path(data_dir) / embed_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        split = self.config.get("split", "dev")
        logger.info(f"Loading QASPER ({split} split)...")
        papers_dict = _load_qasper_json(data_dir, split)

        paper_ids = list(papers_dict.keys())
        if self._sample_size:
            paper_ids = paper_ids[: self._sample_size]

        logger.info(f"Processing {len(paper_ids)} papers...")
        if self._use_table_pc:
            logger.info("Table parent-child chunking: ENABLED")

        self.dataset = []
        skipped_no_answer = 0
        skipped_no_text = 0

        for paper_id in paper_ids:
            paper = papers_dict[paper_id]
            title = paper.get("title", "")
            full_text = _extract_paper_text(paper)

            if not full_text.strip():
                skipped_no_text += 1
                continue

            if self._use_table_pc:
                chunks, chunk_parent_ids, parent_store, stats = (
                    self._build_multimodal_chunks(paper_id, full_text)
                )
                self._parent_stores[paper_id] = parent_store
                # Per-paper diagnostic
                print(
                    f"Paper {paper_id}: {stats['n_tables']} tables detected, "
                    f"{stats['n_table_rows']} total rows"
                )
                print(
                    f"  → {stats['n_prose_chunks']} prose chunks + "
                    f"{stats['n_table_children']} table children + "
                    f"{stats['n_parents']} parents"
                )
                if stats["n_tables"] > 0:
                    self._table_agg["papers_with_tables"] += 1
                self._table_agg["total_tables"] += stats["n_tables"]
                self._table_agg["total_rows"] += stats["n_table_rows"]
            else:
                chunks = self._splitter.split_text(full_text)
                chunk_parent_ids = [""] * len(chunks)

            self._paper_indices[paper_id] = {
                "title": title,
                "chunks": chunks,
                "chunk_parent_ids": chunk_parent_ids,
                "embeddings": None,
                "tree_nodes": None,
            }
            self._table_agg["papers_total"] += 1

            has_table = bool(
                self._use_table_pc
                and self._parent_stores.get(paper_id)
            )
            paper_qa_count = 0
            for qa in paper.get("qas", []):
                question = qa.get("question", "")
                if not question:
                    continue

                gts = _extract_ground_truths(qa)
                if not gts:
                    skipped_no_answer += 1
                    continue

                self.dataset.append({
                    "question": question,
                    "answers": gts,
                    "kwargs": {"paper_id": paper_id},
                })
                paper_qa_count += 1

            self._table_agg["questions_total"] += paper_qa_count
            if has_table:
                self._table_agg["questions_with_tables"] += paper_qa_count

        logger.info(
            f"Loaded {len(self.dataset)} questions from "
            f"{len(self._paper_indices)} papers "
            f"(skipped: {skipped_no_answer} no-answer, {skipped_no_text} no-text)"
        )

        if self._use_table_pc:
            self._print_table_detection_summary()

        # Eagerly build RAPTOR trees so the long build runs once before retrieval
        if self._build_raptor:
            self._tree_cache_dir = Path(data_dir) / tree_dir
            self._tree_cache_dir.mkdir(parents=True, exist_ok=True)
            self._prebuild_raptor_trees(list(self._paper_indices.keys()))

    # ------------------------------------------------------------------
    # Row C: table parent-child chunking
    # ------------------------------------------------------------------

    def _build_multimodal_chunks(
        self,
        paper_id: str,
        full_text: str,
    ) -> tuple[list[str], list[str], dict[str, str], dict]:
        """Detect tables, extract parent-child structure, chunk remaining prose.

        Returns:
            chunks:           list[str]       — texts of embeddable nodes (prose + children)
            chunk_parent_ids: list[str]       — "" for prose, parent_id for table row children
            parent_store:     dict[str, str]  — {parent_id: full_table_text}
            stats:            dict            — table detection counts for logging
        """
        tables = detect_text_tables(full_text)
        remaining_text = _remove_table_spans(full_text, tables)

        prose_chunks = self._splitter.split_text(remaining_text)
        chunks: list[str] = list(prose_chunks)
        chunk_parent_ids: list[str] = [""] * len(prose_chunks)
        parent_store: dict[str, str] = {}
        total_rows = 0

        for table_idx, table in enumerate(tables):
            # Deterministic parent_id so cache-loaded parent_ids stay valid across runs
            parent_id = f"{paper_id}_tbl_{table_idx:03d}"
            parent_store[parent_id] = table["full_text"]

            header = table["header"]
            for row in table["row_lines"]:
                row_text = row.strip().strip("|").strip()
                if not row_text:
                    continue
                child_text = f"TABLE ROW — {header}: {row_text}"
                chunks.append(child_text)
                chunk_parent_ids.append(parent_id)
                total_rows += 1

        stats = {
            "n_tables": len(tables),
            "n_table_rows": total_rows,
            "n_prose_chunks": len(prose_chunks),
            "n_table_children": len(chunks) - len(prose_chunks),
            "n_parents": len(tables),
        }
        return chunks, chunk_parent_ids, parent_store, stats

    def _print_table_detection_summary(self) -> None:
        """Print aggregate table detection stats after load_dataset completes."""
        agg = self._table_agg
        n_total = agg["papers_total"]
        n_with = agg["papers_with_tables"]
        pct = (n_with / n_total * 100) if n_total else 0.0
        avg_rows = (
            agg["total_rows"] / agg["total_tables"]
            if agg["total_tables"] > 0
            else 0.0
        )
        print(f"\n{'=' * 60}")
        print("Table Detection Summary:")
        print(f"  Papers with tables:  {n_with} / {n_total} ({pct:.1f}%)")
        print(f"  Total tables found:  {agg['total_tables']}")
        print(f"  Total table rows:    {agg['total_rows']}")
        print(f"  Avg rows per table:  {avg_rows:.1f}")
        print(
            f"  Papers where table P-C chunking changed chunk set: "
            f"{n_with} / {n_total}"
        )
        print("=" * 60 + "\n")

    def print_table_detection_impact(self) -> None:
        """Print post-run impact summary (called by run_eval.py)."""
        if not self._use_table_pc:
            return
        agg = self._table_agg
        print("Table Detection Impact:")
        print(
            f"  Papers with tables:              "
            f"{agg['papers_with_tables']}/{agg['papers_total']}"
        )
        print(
            f"  Questions affected by table P-C: "
            f"{agg['questions_with_tables']}/{agg['questions_total']}"
        )

    # ------------------------------------------------------------------
    # Leaf embedding (unchanged from Row B)
    # ------------------------------------------------------------------

    def _ensure_index(self, paper_id: str) -> dict:
        """Lazy-load or compute leaf embeddings for a paper's chunks.

        Embeddings are cached to disk as .npz files keyed by paper_id.
        For Row C, self._cache_dir points to the multimodal directory so
        Row B caches are never touched.
        """
        index = self._paper_indices[paper_id]
        if index["embeddings"] is not None:
            return index

        cache_file = self._cache_dir / f"{paper_id}.npz"
        chunks = index["chunks"]

        if cache_file.exists():
            try:
                data = np.load(cache_file)
                cached_embs = data["embeddings"]
                if len(cached_embs) == len(chunks):
                    index["embeddings"] = cached_embs
                    logger.debug(f"Loaded cached embeddings for paper {paper_id}")
                    return index
                logger.warning(
                    f"Cache shape mismatch for {paper_id}: "
                    f"{len(cached_embs)} cached vs {len(chunks)} chunks, re-embedding"
                )
            except Exception as e:
                logger.warning(f"Cache load failed for {paper_id}: {e}")

        from modules.embeddings import embed_batch

        logger.info(
            f"Embedding {len(chunks)} chunks for paper {paper_id} "
            f"({index['title'][:60]})"
        )
        t0 = time.time()
        embeddings = embed_batch(chunks)
        elapsed = time.time() - t0

        dim = 3072
        emb_matrix = np.zeros((len(chunks), dim), dtype=np.float32)
        ok_count = 0
        for i, emb in enumerate(embeddings):
            if emb is not None:
                emb_matrix[i] = emb
                ok_count += 1

        logger.info(
            f"Embedded paper {paper_id}: {ok_count}/{len(chunks)} in {elapsed:.1f}s"
        )

        np.savez_compressed(cache_file, embeddings=emb_matrix)
        index["embeddings"] = emb_matrix
        return index

    # ------------------------------------------------------------------
    # RAPTOR tree — eager pre-build during load_dataset
    # ------------------------------------------------------------------

    def _prebuild_raptor_trees(self, paper_ids: list[str]) -> None:
        """Build or load RAPTOR trees for all papers before retrieval begins.

        For Row C: leaf nodes include table row children (with parent_ids),
        and the tree cache stores parent_ids alongside texts/embeddings/levels.
        Row B's qasper_trees/ cache is never touched (separate dir used).

        Prints per-paper progress, running stats every 20 papers, and a
        final summary with level distribution.
        """
        from benchmarks.raptor_tree import build_raptor_tree

        total = len(paper_ids)
        total_summary_nodes = 0
        level_dist: dict[int, int] = {}

        print(
            f"\n[RAPTOR] Starting tree construction for {total} papers "
            f"(cache: {self._tree_cache_dir})\n"
        )

        for i, paper_id in enumerate(paper_ids, 1):
            index = self._paper_indices[paper_id]
            tree_cache_file = self._tree_cache_dir / f"{paper_id}.npz"

            # ---- Try to load from cache ----
            if tree_cache_file.exists() and not self._rebuild_trees:
                try:
                    data = np.load(tree_cache_file, allow_pickle=True)
                    texts = data["texts"].tolist()
                    embeddings = data["embeddings"]
                    levels = data["levels"].tolist()
                    # parent_ids absent in old Row B caches — default to ""
                    if "parent_ids" in data.files:
                        parent_ids = data["parent_ids"].tolist()
                    else:
                        parent_ids = [""] * len(texts)

                    if len(texts) == len(embeddings) == len(levels):
                        tree_nodes = [
                            {
                                "text": texts[j],
                                "embedding": embeddings[j],
                                "level": int(levels[j]),
                                "parent_id": parent_ids[j] if j < len(parent_ids) else "",
                            }
                            for j in range(len(texts))
                        ]
                        index["tree_nodes"] = tree_nodes

                        n_leaves = sum(1 for lv in levels if lv == 0)
                        n_summary = len(tree_nodes) - n_leaves
                        n_levels = max(levels) if levels else 0
                        total_summary_nodes += n_summary
                        for lv, cnt in zip(*np.unique(levels, return_counts=True)):
                            level_dist[int(lv)] = level_dist.get(int(lv), 0) + int(cnt)

                        print(
                            f"[RAPTOR] Paper {i}/{total} ({paper_id}): "
                            f"{n_leaves} chunks → {n_summary} summary nodes "
                            f"({n_levels} levels) [cached]"
                        )
                        if i % 20 == 0:
                            print(
                                f"[RAPTOR] Running total after {i}/{total} papers: "
                                f"{total_summary_nodes} summary nodes\n"
                            )
                        continue
                except Exception as exc:
                    logger.warning(
                        f"Tree cache corrupt for {paper_id}: {exc} — rebuilding"
                    )

            # ---- Ensure leaf embeddings are ready ----
            self._ensure_index(paper_id)
            chunks = index["chunks"]
            emb_matrix = index["embeddings"]
            chunk_parent_ids = index.get("chunk_parent_ids", [""] * len(chunks))

            # Build leaf_nodes, including parent_id for table row children
            leaf_nodes = [
                {
                    "text": chunks[j],
                    "embedding": emb_matrix[j].copy(),
                    "level": 0,
                    "parent_id": chunk_parent_ids[j] if j < len(chunk_parent_ids) else "",
                }
                for j in range(len(chunks))
                if np.any(emb_matrix[j])
            ]

            if not leaf_nodes:
                logger.warning(
                    f"[RAPTOR] {paper_id}: no valid leaf embeddings — skipping tree build"
                )
                index["tree_nodes"] = []
                print(
                    f"[RAPTOR] Paper {i}/{total} ({paper_id}): "
                    f"0 chunks → 0 summary nodes (skipped — no valid embeddings)"
                )
                if i % 20 == 0:
                    print(
                        f"[RAPTOR] Running total after {i}/{total} papers: "
                        f"{total_summary_nodes} summary nodes\n"
                    )
                continue

            # ---- Build tree ----
            t0 = time.time()
            tree_nodes = build_raptor_tree(leaf_nodes)
            elapsed = time.time() - t0

            n_summary = sum(1 for n in tree_nodes if n["level"] > 0)
            n_levels = max(n["level"] for n in tree_nodes) if tree_nodes else 0
            total_summary_nodes += n_summary
            for n in tree_nodes:
                lv = n["level"]
                level_dist[lv] = level_dist.get(lv, 0) + 1

            print(
                f"[RAPTOR] Paper {i}/{total} ({paper_id}): "
                f"{len(leaf_nodes)} chunks → {n_summary} summary nodes "
                f"({n_levels} levels) [{elapsed:.1f}s]"
            )
            if i % 20 == 0:
                print(
                    f"[RAPTOR] Running total after {i}/{total} papers: "
                    f"{total_summary_nodes} summary nodes\n"
                )

            # ---- Save tree cache (with parent_ids for Row C) ----
            try:
                texts_arr = np.array([n["text"] for n in tree_nodes], dtype=object)
                embs_arr = np.array(
                    [n["embedding"] for n in tree_nodes], dtype=np.float32
                )
                levels_arr = np.array(
                    [n["level"] for n in tree_nodes], dtype=np.int32
                )
                parent_ids_arr = np.array(
                    [n.get("parent_id", "") for n in tree_nodes], dtype=object
                )
                np.savez_compressed(
                    tree_cache_file,
                    texts=texts_arr,
                    embeddings=embs_arr,
                    levels=levels_arr,
                    parent_ids=parent_ids_arr,
                )
            except Exception as exc:
                logger.warning(f"Failed to save tree cache for {paper_id}: {exc}")

            index["tree_nodes"] = tree_nodes

        # ---- Final summary ----
        print(f"\n{'=' * 60}")
        print("[RAPTOR] Tree construction complete")
        print(f"  Papers processed:    {total}")
        print(f"  Total summary nodes: {total_summary_nodes}")
        print("  Level distribution:")
        for lv in sorted(level_dist.keys()):
            label = "leaves (level 0)" if lv == 0 else f"summaries level {lv}"
            print(f"    Level {lv} ({label}): {level_dist[lv]} nodes")
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, paper_id: str) -> list[str]:
        """Retrieve top-K texts for a question.

        - retrieval_mode=collapsed + build_raptor_tree=true:
            searches all RAPTOR levels (leaves + summary nodes)
        - retrieval_mode=flat OR no tree built:
            searches leaf nodes only

        When use_retrieval_expansion=true: table row child hits are swapped
        for their full parent table text (deduplicating same-parent hits).
        """
        from modules.embeddings import embed

        query_emb = embed(text=question)
        if query_emb is None:
            logger.error(f"Failed to embed question: {question[:80]}")
            return []

        query_vec = np.array(query_emb, dtype=np.float32)

        if self._build_raptor and self.retrieval_mode == "collapsed":
            index = self._paper_indices[paper_id]
            tree_nodes = index.get("tree_nodes") or []

            if tree_nodes:
                all_embs = np.array(
                    [n["embedding"] for n in tree_nodes], dtype=np.float32
                )
                scores = _cosine_similarity_batch(query_vec, all_embs)
                top_indices = np.argsort(scores)[::-1][: self._top_k]
                retrieved = [tree_nodes[i] for i in top_indices]
                if self._use_expansion:
                    return self._expand_parents(retrieved, paper_id)
                return [n["text"] for n in retrieved]

            # Tree empty / failed — fall through to flat retrieval
            logger.warning(
                f"[RAPTOR] No tree nodes for {paper_id}, falling back to flat retrieval"
            )

        # Flat retrieval (leaf nodes only)
        index = self._ensure_index(paper_id)
        scores = _cosine_similarity_batch(query_vec, index["embeddings"])
        top_indices = np.argsort(scores)[::-1][: self._top_k]
        if self._use_expansion and self._use_table_pc:
            chunk_pids = index.get("chunk_parent_ids", [""] * len(index["chunks"]))
            retrieved = [
                {"text": index["chunks"][k], "parent_id": chunk_pids[k]}
                for k in top_indices
            ]
            return self._expand_parents(retrieved, paper_id)
        return [index["chunks"][k] for k in top_indices]

    def _expand_parents(
        self, retrieved: list[dict], paper_id: str
    ) -> list[str]:
        """Swap table row child hits for their full parent table text.

        Multiple children from the same parent are deduplicated — the parent
        text appears once. Non-child nodes pass through unchanged.
        """
        parent_store = self._parent_stores.get(paper_id, {})
        if not parent_store:
            return [n["text"] for n in retrieved]

        expanded: list[str] = []
        seen_parents: set[str] = set()

        for node in retrieved:
            pid = node.get("parent_id", "")
            if pid and pid in parent_store:
                if pid not in seen_parents:
                    expanded.append(parent_store[pid])
                    seen_parents.add(pid)
                # Drop duplicate child rows from the same parent
            else:
                expanded.append(node["text"])

        return expanded

    def run_single(self, question: str, paper_id: str = "", **kwargs) -> str:
        """Retrieve context chunks, generate answer with Gemini Flash."""
        from core.config import generate_with_retry, MODEL_FLASH
        from google.genai import types as genai_types

        chunks = self._retrieve(question, paper_id)
        if not chunks:
            return "unanswerable"

        context = "\n\n---\n\n".join(chunks)
        prompt = QA_GENERATION_PROMPT.format(context=context, question=question)

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
        except Exception as e:
            logger.error(f"Generation failed for question '{question[:60]}': {e}")
            return "unanswerable"

    def evaluate(
        self,
        predictions: list[str],
        ground_truths: list[list[str]],
    ) -> dict[str, float]:
        f1_scores = [
            score_with_multiple_gts(pred, gts, f1_token_score)
            for pred, gts in zip(predictions, ground_truths)
        ]
        em_scores = [
            score_with_multiple_gts(pred, gts, exact_match)
            for pred, gts in zip(predictions, ground_truths)
        ]
        return {
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        }
