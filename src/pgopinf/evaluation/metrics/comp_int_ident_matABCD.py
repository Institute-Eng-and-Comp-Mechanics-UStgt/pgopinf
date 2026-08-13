from dataclasses import dataclass
import numpy as np
import scipy

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.results import (
    MetricOutput,
    ScalarMetricResult,
)


@dataclass
class CompIntIdentMatABCDMetric:
    """Compare intrusive and identified ``A``, ``B``, ``C``, and ``D`` matrices."""

    name: str = "comp_int_ident_matABCD"
    relative: bool = True
    rel_threshold: float = 1e-8

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Compute matrix-wise Frobenius norm differences.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing intrusive and identified systems.

        Returns
        -------
        MetricOutput
            Scalar artifacts for the ``A``, ``B``, ``C``, and ``D`` matrix
            differences.
        """

        intrusive_system = ctx.reduction_result.reduced_system
        identified_system = ctx.identification_result.system_identified

        if identified_system.issparse or intrusive_system.issparse:
            frobenius_norm_function = lambda x: scipy.sparse.linalg.norm(x, "fro")
        else:
            frobenius_norm_function = lambda x: np.linalg.norm(x, "fro")

        artifacts = []
        for matrix in ["A", "B", "C", "D"]:
            mat_intrusive = getattr(intrusive_system, matrix)
            mat_id = getattr(identified_system, matrix)
            diff = mat_intrusive - mat_id
            if self.relative:
                norm_diff = frobenius_norm_function(diff) / np.maximum(
                    frobenius_norm_function(mat_intrusive), self.rel_threshold
                )
            else:
                norm_diff = frobenius_norm_function(diff)

            artifacts.append(
                ScalarMetricResult(
                    name=f"comp_int_ident_mat{matrix}",
                    value=norm_diff,
                    unit="-",
                    meta={
                        "relative": self.relative,
                        "rel_threshold": self.rel_threshold,
                    },
                )
            )

        summaries = []

        return MetricOutput(
            summaries=summaries,
            artifacts=artifacts,
        )
