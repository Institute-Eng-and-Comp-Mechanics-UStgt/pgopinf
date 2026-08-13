from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.subroutines.hinf import (
    hinf_error,
    hinf_norm,
)
from pgopinf.evaluation.results import (
    MetricOutput,
    ScalarMetricResult,
)

logger = logging.getLogger(__name__)


@dataclass
class HInfErrorMetric:
    """Compute H-infinity error against the original system."""

    target: str = "identified"
    name: str = "hinf_error"
    tol: float = 1e-10
    stabilized: bool = False
    use_minimal_realization: bool = False
    relative: bool = True

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Compute the H-infinity error for a comparison system.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing original, intrusive, and identified
            systems.

        Returns
        -------
        MetricOutput
            Scalar H-infinity error and spectral-abscissa artifacts.
        """

        # use (precomputed) minimal realization if available
        if self.use_minimal_realization:
            if (
                ctx.system_analysis is not None
                and "minimal_realization" in ctx.system_analysis.values
            ):
                ref_system = ctx.system_analysis.values["minimal_realization"]
            else:
                logger.info(
                    "No precomputed minimal realization available. Consider using system analysis task MinimalRealizationSpec. Computing on the fly..."
                )
                ref_system = ctx.original_system.minimal_realization(trunc_tol=1e-10)
        else:
            # original system
            ref_system = ctx.original_system

        if self.target == "identified":
            if ctx.dataset_artifact_identified is None:
                raise ValueError("No identified dataset artifact available.")
            cmp_system = ctx.identification_result.system_identified
        elif self.target == "intrusive":
            if ctx.dataset_artifact_intrusive is None:
                raise ValueError("No intrusive dataset artifact available.")
            cmp_system = ctx.reduction_result.reduced_system
        else:
            raise ValueError(f"Unknown target {self.target!r}")

        if self.stabilized:
            cmp_system = cmp_system.stable_decomposition()

        # use precomputed eigenvalues if available (to avoid recomputation in hinf_error)
        if (
            ctx.system_analysis is not None
            and "eigenvalues" in ctx.system_analysis.values
        ):
            first_system_eigvals = ctx.system_analysis.values["eigenvalues"]
        else:
            first_system_eigvals = None

        hinf_error_value, ref_system_spectral_abscissa, cmp_system_spectral_abscissa = (
            hinf_error(
                first_system=ref_system,
                second_system=cmp_system,
                tol=self.tol,
                return_spectral_abscissas=True,
                first_system_eigvals=first_system_eigvals,
            )
        )

        if self.relative:
            if (
                ctx.system_analysis is not None
                and "hinf_norm" in ctx.system_analysis.values
            ):
                hinf_error_value = (
                    hinf_error_value / ctx.system_analysis.values["hinf_norm"]
                )
            else:
                logger.info(
                    "No precomputed hinf norm available. Consider using system analysis task HInfNormSpec. Calculating Hinf norm on the fly...",
                    extra={"once": True},
                )
                hinf_error_value = hinf_error_value / hinf_norm(first_system=ref_system)

        artifact = []
        artifact.append(
            ScalarMetricResult(
                name=self.name,
                value=hinf_error_value,
                unit="-",
                meta={
                    "target": self.target,
                    "tol": self.tol,
                    "stabilized": self.stabilized,
                    "relative": self.relative,
                },
            )
        )
        artifact.append(
            ScalarMetricResult(
                name=f"spectral_abscissa_orig",
                value=ref_system_spectral_abscissa,
                unit="-",
                meta={},
            )
        )
        artifact.append(
            ScalarMetricResult(
                name=f"spectral_abscissa_cmp",
                value=cmp_system_spectral_abscissa,
                unit="-",
                meta={
                    "target": self.target,
                    "stabilized": self.stabilized,
                },
            )
        )

        summaries = [
            {
                "metric": self.name,
                "value": hinf_error_value,
                "unit": "-",
                "target": self.target,
                "tol": self.tol,
                "stabilized": self.stabilized,
                "relative": self.relative,
            }
        ]

        return MetricOutput(
            summaries=summaries,
            artifacts=artifact,
        )
