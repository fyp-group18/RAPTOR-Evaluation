"""Export production RAPTOR tree from the main project's database.

Reads DocumentChunkMultimodal rows for a given document and serializes them
as a pickle file compatible with the evaluation harness.

Usage:
    # List available documents
    python -m ingestion.export_production_tree --list-documents

    # Export a specific document
    python -m ingestion.export_production_tree --document-id 42 --output tree.pkl

    # Export without embeddings (smaller file)
    python -m ingestion.export_production_tree --document-id 42 --skip-embeddings

Requires the main project (raptor/backend) on PYTHONPATH so that
`core.models` and `core.database` are importable.
"""

from __future__ import annotations

import argparse
import base64
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import UUID


# ---------------------------------------------------------------------------
# Import external model + session (READ-ONLY, no modifications)
# ---------------------------------------------------------------------------

def _import_model_and_session(db_url: str | None = None):
    """Import DocumentChunkMultimodal and a session factory from the main project.

    If db_url is provided, builds a standalone engine/session instead of using
    the main project's SessionLocal (which requires DATABASE_URL at import time).
    """
    if db_url:
        import os
        os.environ.setdefault("DATABASE_URL", db_url)

    def _do_import():
        # Import order matters: core.database must be imported first so that
        # Base is defined before core.models tries to import it.
        # core.database.init_db() (called at module load) imports core.models,
        # but Base is already bound by that point — no circular issue.
        from core.database import SessionLocal  # noqa: F811
        from core.models import DocumentChunkMultimodal  # noqa: F811
        return DocumentChunkMultimodal, SessionLocal

    try:
        return _do_import()
    except (ImportError, ValueError) as exc:
        # Try adding the known sibling project path
        backend_path = str(Path(__file__).resolve().parent.parent.parent / "raptor" / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
            try:
                return _do_import()
            except (ImportError, ValueError) as inner_exc:
                print(
                    f"ERROR: Could not import DocumentChunkMultimodal from core.models.\n"
                    f"  Ensure the main project is on PYTHONPATH.\n"
                    f"  Tried: {backend_path}\n"
                    f"  Error: {inner_exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            print(
                f"ERROR: Could not import DocumentChunkMultimodal from core.models.\n"
                f"  Ensure the main project is on PYTHONPATH.\n"
                f"  Error: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    return DocumentChunkMultimodal, SessionLocal


# ---------------------------------------------------------------------------
# Deserialization helpers (duplicated from main project to avoid runtime
# dependency on model methods)
# ---------------------------------------------------------------------------

def _deserialize_embedding(raw) -> list[float] | None:
    """Convert pgvector/numpy embedding to a plain list of floats."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    try:
        import numpy as np
        if isinstance(raw, np.ndarray):
            return raw.tolist()
    except ImportError:
        pass
    # pgvector returns a custom type with __iter__
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


def _serialize_value(val):
    """Convert a single column value to a pickle-safe Python type."""
    if val is None:
        return None
    if isinstance(val, UUID):
        return str(val)
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, bytes):
        return base64.b64encode(val).decode("ascii")
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, (list, dict)):
        return val
    # Enum types
    if hasattr(val, "value"):
        return val.value
    # pgvector / numpy array
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.tolist()
    except ImportError:
        pass
    # Fallback: try iteration (pgvector)
    try:
        return [float(x) for x in val]
    except (TypeError, ValueError):
        return str(val)


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

def list_documents(SessionLocal, Model):
    """Print all distinct document_ids with chunk counts."""
    from sqlalchemy import func

    with SessionLocal() as db:
        rows = (
            db.query(Model.document_id, func.count(Model.id).label("chunks"))
            .group_by(Model.document_id)
            .order_by(Model.document_id)
            .all()
        )

    if not rows:
        print("No documents found in database.")
        return

    print(f"{'Document ID':<14} {'Chunks':<10}")
    print(f"{'-' * 14} {'-' * 10}")
    for doc_id, count in rows:
        print(f"{doc_id:<14} {count:<10}")
    print(f"\nTotal: {len(rows)} documents, {sum(r[1] for r in rows)} chunks")


def export_document(SessionLocal, Model, document_id: int, output: str, skip_embeddings: bool):
    """Export all chunks for a document to a pickle file."""
    from sqlalchemy import inspect as sa_inspect

    # Enumerate columns dynamically
    mapper = sa_inspect(Model)
    columns = [c.key for c in mapper.column_attrs]

    with SessionLocal() as db:
        rows = db.query(Model).filter(Model.document_id == document_id).all()

        if not rows:
            print(f"ERROR: No chunks found for document_id={document_id}", file=sys.stderr)
            sys.exit(1)

        nodes = []
        for row in rows:
            node = {}
            for col in columns:
                val = getattr(row, col)
                if col == "embedding":
                    if skip_embeddings:
                        node[col] = None
                    else:
                        node[col] = _deserialize_embedding(val)
                elif col == "tsv":
                    # TSVECTOR is not pickle-safe; store as string repr
                    node[col] = str(val) if val is not None else None
                else:
                    node[col] = _serialize_value(val)
            nodes.append(node)

    # Write pickle
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    # --- Summary ---
    level_counts = Counter(n.get("level") for n in nodes)
    has_embedding = sum(1 for n in nodes if n.get("embedding") is not None)
    has_images = sum(1 for n in nodes if n.get("images"))
    has_pages = sum(1 for n in nodes if n.get("pages"))

    print(f"\nExported {len(nodes)} nodes to {out_path}")
    print(f"\n  By level:")
    for lvl in sorted(level_counts):
        print(f"    Level {lvl}: {level_counts[lvl]}")
    print(f"\n  With embeddings: {has_embedding}")
    print(f"  With images: {has_images}")
    print(f"  With pages populated: {has_pages}")

    # Sample node
    sample = nodes[0].copy()
    if sample.get("embedding") and len(sample["embedding"]) > 5:
        sample["embedding"] = sample["embedding"][:5] + ["..."]
    print(f"\n  Sample node (first):")
    for k, v in sample.items():
        display = repr(v)
        if len(display) > 120:
            display = display[:117] + "..."
        print(f"    {k}: {display}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export production RAPTOR tree from the main project database."
    )
    parser.add_argument("--document-id", type=int, help="Export chunks for this document ID")
    parser.add_argument("--list-documents", action="store_true", help="List all documents with chunk counts")
    parser.add_argument("--output", default="production_tree.pkl", help="Output pickle path (default: production_tree.pkl)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Replace embeddings with None to reduce file size")
    parser.add_argument("--db-url", default=None, help="Override DATABASE_URL (falls back to env var or project config)")

    args = parser.parse_args()

    if not args.list_documents and args.document_id is None:
        parser.error("Either --document-id or --list-documents is required")

    Model, SessionLocal = _import_model_and_session(args.db_url)

    if args.list_documents:
        list_documents(SessionLocal, Model)
    else:
        export_document(SessionLocal, Model, args.document_id, args.output, args.skip_embeddings)


if __name__ == "__main__":
    main()
