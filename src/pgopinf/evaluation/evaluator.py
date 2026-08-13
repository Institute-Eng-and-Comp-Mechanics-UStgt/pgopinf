from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.results import MetricOutput
from pgopinf.specs.evaluation.evaluation import EvaluationSpec

logger = logging.getLogger(__name__)


@dataclass
class MetricEvaluator:
    """Evaluate configured metrics for one experiment run."""

    def evaluate(
        self, *, eval_ctx: EvaluationContext, evaluation_spec: EvaluationSpec
    ) -> list[MetricOutput]:
        """Compute all metrics in an evaluation specification.

        Parameters
        ----------
        eval_ctx : EvaluationContext
            Runtime context containing systems, datasets, and run metadata.
        evaluation_spec : EvaluationSpec
            Specification containing metric configurations.

        Returns
        -------
        list of MetricOutput
            Metric summaries and artifacts, enriched with experiment metadata.
        """
        outputs: list[MetricOutput] = []

        logger.info(f"Computing metrics for run {eval_ctx.run_id}...")
        for metric_spec in evaluation_spec.metrics:
            metric = metric_spec.build()
            out = metric.compute(eval_ctx)

            # enrich summary rows with common experiment info
            enriched = []
            for row in out.summaries:
                row = dict(row)
                row["experiment_name"] = eval_ctx.experiment_name
                row["run_id"] = eval_ctx.run_id
                enriched.append(row)

            outputs.append(
                MetricOutput(
                    summaries=enriched,
                    artifacts=out.artifacts,
                )
            )

        return outputs
