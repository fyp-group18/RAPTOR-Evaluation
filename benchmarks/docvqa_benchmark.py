"""
DocVQA benchmark runner.

Dataset: lmms-lab/DocVQA (HuggingFace)
Primary metric: ANLS (Average Normalized Levenshtein Similarity)
"""

import logging

from benchmarks.base_benchmark import BaseBenchmark
from metrics import anls_score, score_with_multiple_gts

logger = logging.getLogger(__name__)

HUGGINGFACE_DATASET_ID = "lmms-lab/DocVQA"


class DocvqaBenchmark(BaseBenchmark):
    def load_dataset(self, data_dir: str = "datasets") -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Install the 'datasets' package: pip install datasets"
            )

        raw = load_dataset(HUGGINGFACE_DATASET_ID, split="test", cache_dir=data_dir)
        self.dataset = []
        for item in raw:
            self.dataset.append({
                "question": item.get("question", ""),
                "answers": item.get("answers", []),
                "kwargs": {"image": item.get("image")},
            })
        logger.info(f"Loaded DocVQA dataset: {len(self.dataset)} questions")

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
        anls_scores = [
            score_with_multiple_gts(pred, gts, anls_score)
            for pred, gts in zip(predictions, ground_truths)
        ]
        return {
            "anls": sum(anls_scores) / len(anls_scores) if anls_scores else 0.0,
        }
