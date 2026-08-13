from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.results import (
    ArrayPlotResult,
    MetricOutput,
)
from pgopinf.specs.utils.utils import _as_tuple


@dataclass
class SingularValueDecayMetric:
    """Plot singular value decay of a full basis artifact."""

    name: str = "singular_value_decay"

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Build a singular-value decay plot artifact.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing the reduction result.

        Returns
        -------
        MetricOutput or None
            Array plot artifact if singular values are available; otherwise
            ``None``.
        """

        if hasattr(ctx.reduction_result.full_basis, "S"):
            S = np.asarray(ctx.reduction_result.full_basis.S)
            r = np.arange(1, len(S) + 1)
            y = np.empty((r.shape[0], 1, 1), dtype=float)
            y[:, 0, 0] = S
        else:
            return None

        meta = {}
        artifact = ArrayPlotResult(
            name=self.name,
            x=r,
            y=y,
            x_label="reduced order r",
            y_labels=_as_tuple("singular value"),
            line_labels=(),
            subplot_titles=_as_tuple("singular value decay over r"),
            meta=meta,
        )

        summaries = []

        return MetricOutput(
            summaries=summaries,
            artifacts=[artifact],
        )
