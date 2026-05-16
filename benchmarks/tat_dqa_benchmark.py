"""
TAT-DQA benchmark runner.

Dataset: TAT-DQA (from original paper repository)
  Source: https://github.com/NExTplusplus/TAT-DQA
Primary metrics: F1, Exact Match
"""

import logging

from benchmarks.base_benchmark import BaseBenchmark
from metrics import f1_token_score, exact_match, score_with_multiple_gts

logger = logging.getLogger(__name__)


class TatDqaBenchmark(BaseBenchmark):
    def load_dataset(self, data_dir: str = "datasets") -> None:
        # TAT-DQA is not on HuggingFace in a standard format.
        # Download from the original repo and place in datasets/ directory.
        raise NotImplementedError(
            "TAT-DQA dataset must be downloaded manually from "
            "https://github.com/NExTplusplus/TAT-DQA and placed in the "
            f"'{data_dir}/' directory. See README.md for instructions."
        )

    def run_single(self, question: str, **kwargs) -> str:
        # TODO: Call into main system's retrieval + generation pipeline
        raise NotImplementedError(
            "run_single() requires integration with the main backend pipeline. "
            "Implement by calling multimodal_semantic_search() and the generation LLM."
        )

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
