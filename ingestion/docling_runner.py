"""Docling document parsing.

Runs Docling on a PDF file and returns the full DoclingDocument serialized
as a dict (via Pydantic's export_to_dict). The caller handles GCS caching.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def run_docling(file_path: Path) -> dict:
    """Parse a document with Docling and return the serialized DoclingDocument.

    Lazy-imports Docling to avoid pulling in torch/transformers unless needed.

    Args:
        file_path: Path to the PDF document.

    Returns:
        Dict representation of the DoclingDocument (via export_to_dict).
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    logger.info(f"Running Docling on {file_path.name}...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_picture_images = True
    pipeline_options.do_table_structure = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = doc_converter.convert(str(file_path))
    doc = result.document

    logger.info(
        f"Docling complete: {file_path.name} "
        f"({len(list(doc.iterate_items()))} items extracted)"
    )

    return doc.export_to_dict()


def get_docling_version() -> str:
    """Return the installed docling version string."""
    import docling

    return docling.__version__
