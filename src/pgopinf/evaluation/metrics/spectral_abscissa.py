from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.subroutines.hinf import hinf_error
from pgopinf.evaluation.results import (
    MetricOutput,
    ScalarMetricResult,
)


@dataclass
class SpectralAbscissaMetric:
    """Compute the spectral abscissa of a selected system."""

    target: str = "identified"  # "identified" | "intrusive" | "original"
    name: str = "spectral_abscissa"
    tol: float = 1e-10
    stabilized: bool = False

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Compute the largest real part of system eigenvalues.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing candidate systems.

        Returns
        -------
        MetricOutput
            Scalar spectral-abscissa artifact and summary row.
        """

        if self.target == "identified":
            if ctx.dataset_artifact_identified is None:
                raise ValueError("No identified dataset artifact available.")
            system = ctx.identification_result.system_identified
        elif self.target == "intrusive":
            if ctx.dataset_artifact_intrusive is None:
                raise ValueError("No intrusive dataset artifact available.")
            system = ctx.reduction_result.reduced_system
        elif self.target == "original":
            system = ctx.original_system
        else:
            raise ValueError(f"Unknown target {self.target!r}")

        if self.stabilized:
            system = system.stable_decomposition()

        eigvals = system.eigvals()
        spectral_abscissa = np.max(np.real(eigvals))

        artifact = []
        artifact.append(
            ScalarMetricResult(
                name=self.name,
                value=spectral_abscissa,
                unit="-",
                meta={
                    "target": self.target,
                    "tol": self.tol,
                    "stabilized": self.stabilized,
                },
            )
        )

        summaries = [
            {
                "metric": self.name,
                "value": spectral_abscissa,
                "unit": "-",
                "target": self.target,
                "tol": self.tol,
                "stabilized": self.stabilized,
            }
        ]

        return MetricOutput(
            summaries=summaries,
            artifacts=artifact,
        )
