"""Ablation configurations for the RAPTOR evaluation pipeline.

Each configuration controls which context-aware chunking sub-innovations
are active during tree construction.
"""

from __future__ import annotations

ABLATION_CASES: dict[str, dict] = {
    "full": {
        "label": "full_context_aware",
        "table_parent_child": True,
        "header_propagation": True,
        "caption_folding": True,
        "multimodal_embedding": True,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
    },
    "no_table_pc": {
        "label": "no_table_parent_child",
        "table_parent_child": False,
        "header_propagation": True,
        "caption_folding": True,
        "multimodal_embedding": True,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
    },
    "no_header_prop": {
        "label": "no_header_propagation",
        "table_parent_child": True,
        "header_propagation": False,
        "caption_folding": True,
        "multimodal_embedding": True,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
    },
    "no_caption_fold": {
        "label": "no_caption_folding",
        "table_parent_child": True,
        "header_propagation": True,
        "caption_folding": False,
        "multimodal_embedding": True,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
    },
    "no_context_aware": {
        "label": "baseline_naive_chunking",
        "table_parent_child": False,
        "header_propagation": False,
        "caption_folding": False,
        "multimodal_embedding": True,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
    },
    "semantic_chunking": {
        "label": "semantic_chunking_baseline",
        "table_parent_child": False,
        "header_propagation": False,
        "caption_folding": False,
        "multimodal_embedding": True,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
        "chunking_strategy": "semantic",
    },
    "flat_retrieval": {
        "label": "flat_no_raptor",
        "table_parent_child": True,
        "header_propagation": True,
        "caption_folding": True,
        "multimodal_embedding": True,
        "raptor_tree": False,
        "retrieval_mode": "flat",
    },
    "text_only_raptor": {
        "label": "original_raptor_text_only",
        "table_parent_child": False,
        "header_propagation": False,
        "caption_folding": False,
        "multimodal_embedding": False,
        "raptor_tree": True,
        "retrieval_mode": "collapsed",
    },
}

ALL_ABLATION_NAMES = list(ABLATION_CASES.keys())


def validate_ablation_selection(names: list[str]) -> list[tuple[str, dict]]:
    """Resolve ablation names to (name, config) pairs.

    Raises ValueError for unknown names.
    """
    unknown = [n for n in names if n not in ABLATION_CASES]
    if unknown:
        raise ValueError(
            f"Unknown ablation(s): {unknown}. "
            f"Valid names: {ALL_ABLATION_NAMES}"
        )
    return [(n, ABLATION_CASES[n]) for n in names]
