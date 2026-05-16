"""
Abstract base class for benchmark runners.

Each benchmark subclass implements:
  - load_dataset(): download/load the dataset into self.dataset
  - run_single(question, context_docs) -> prediction: run one QA pair through the system
  - evaluate(predictions, ground_truths) -> dict of metric scores

Includes checkpoint/resume support for long-running benchmarks.
"""

import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Ablation toggles that require main codebase support
_ABLATION_TOGGLES = ("use_multimodal_embed", "use_table_parent_child", "use_retrieval_expansion")


class BaseBenchmark(ABC):
    """Base class for all RAPTOR evaluation benchmarks."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.retrieval_mode: str = self.config.get("retrieval_mode", "collapsed")
        self.description: str = self.config.get("description", "")
        self.dataset = None
        self._warn_unimplemented_toggles()

    def _warn_unimplemented_toggles(self) -> None:
        for toggle in _ABLATION_TOGGLES:
            value = self.config.get(toggle)
            if value is not None and value is not True:
                logger.warning(
                    f"Toggle '{toggle}' set to {value} in config, but not yet "
                    f"implemented in main codebase — running with default behavior."
                )

    @abstractmethod
    def load_dataset(self, data_dir: str = "datasets") -> None:
        """Download or load the dataset into self.dataset."""

    @abstractmethod
    def run_single(self, question: str, **kwargs) -> str:
        """Run one QA pair through the retrieval + generation pipeline.

        Returns the predicted answer string.
        """

    @abstractmethod
    def evaluate(
        self,
        predictions: list[str],
        ground_truths: list[list[str]],
    ) -> dict[str, float]:
        """Compute metric scores over all prediction/ground-truth pairs.

        Args:
            predictions: list of predicted answer strings
            ground_truths: list of lists of valid ground truth answers
                           (multiple GTs per question)

        Returns:
            Dict mapping metric name to average score.
        """

    def run_all(self, output_path: str | None = None) -> dict[str, float]:
        """Run the full benchmark with checkpoint/resume support.

        Saves a checkpoint after each prediction so the run can resume
        after crashes. The checkpoint file is co-located with the output.
        """
        if self.dataset is None:
            raise RuntimeError("Call load_dataset() before run_all()")

        checkpoint_path = None
        if output_path:
            # Resolve to absolute so the path stays valid regardless of CWD changes
            checkpoint_path = Path(output_path).resolve().with_suffix(".checkpoint.json")

        # Resume from checkpoint if available
        predictions, ground_truths, start_idx = self._load_checkpoint(checkpoint_path)

        if start_idx > 0:
            logger.info(
                f"Resumed from checkpoint at index {start_idx}/{len(self.dataset)}"
            )

        total = len(self.dataset)
        t_start = time.time()

        for i in range(start_idx, total):
            item = self.dataset[i]
            question = item["question"]
            gts = item["answers"]

            elapsed = time.time() - t_start
            rate = (i - start_idx + 1) / max(elapsed, 0.1)
            remaining = (total - i - 1) / max(rate, 0.001)
            logger.info(
                f"[{i + 1}/{total}] ({rate:.1f} q/s, ~{remaining / 60:.0f}m left) "
                f"{question[:80]}..."
            )

            pred = self.run_single(question, **item.get("kwargs", {}))
            predictions.append(pred)
            ground_truths.append(gts)

            # Checkpoint every prediction
            if checkpoint_path:
                self._save_checkpoint(
                    checkpoint_path, predictions, ground_truths, i + 1
                )

        scores = self.evaluate(predictions, ground_truths)

        if output_path:
            self._save_results(output_path, predictions, ground_truths, scores)
            # Clean up checkpoint on successful completion
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Checkpoint removed after successful completion")

        return scores

    def _load_checkpoint(
        self, checkpoint_path: Path | None
    ) -> tuple[list[str], list[list[str]], int]:
        """Load predictions from a checkpoint file if it exists."""
        if checkpoint_path is None or not checkpoint_path.exists():
            return [], [], 0

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            predictions = data.get("predictions", [])
            ground_truths = data.get("ground_truths", [])
            completed = data.get("completed_index", 0)
            logger.info(
                f"Found checkpoint: {completed} items completed at {checkpoint_path}"
            )
            return predictions, ground_truths, completed
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupt checkpoint at {checkpoint_path}: {e}")
            return [], [], 0

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        predictions: list[str],
        ground_truths: list[list[str]],
        completed_index: int,
    ) -> None:
        """Save current progress to a checkpoint file."""
        data = {
            "completed_index": completed_index,
            "predictions": predictions,
            "ground_truths": ground_truths,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, checkpoint_path)

    def _save_results(
        self,
        output_path: str,
        predictions: list[str],
        ground_truths: list[list[str]],
        scores: dict[str, float],
    ) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "prediction", "ground_truths", *scores.keys()])
            for i, (pred, gts) in enumerate(zip(predictions, ground_truths)):
                writer.writerow([i, pred, "|".join(gts)])
            # Summary row
            writer.writerow(["AVERAGE", "", "", *scores.values()])
        logger.info(f"Results saved to {output_path}")
        path = Path(output_path)
        summary_path = path.with_suffix(".summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Config: {self.description}\n")
            f.write(f"Retrieval mode: {self.retrieval_mode}\n")
            f.write(f"Samples: {len(predictions)}\n")
            for metric, value in scores.items():
                f.write(f"{metric}: {value:.4f}\n")
        logger.info(f"Summary saved to {summary_path}")
