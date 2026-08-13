from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.subroutines.convert_data import (
    convert_to_sample_format,
)
from pgopinf.evaluation.metrics.subroutines.error_time_trajectories import (
    error_time_trajectories,
)
from pgopinf.evaluation.metrics.subroutines.pick_trajectories import (
    pick_trajectories,
)
from pgopinf.evaluation.results import (
    ArrayPlotResult,
    MetricOutput,
    ScalarMetricResult,
    SeriesMetricResult,
)


@dataclass
class InputValuesOverTimeMetric:
    """Create input trajectory plots for a dataset split."""

    split: str = "TEST"
    name: str = "input_values_over_time"

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Build input trajectory plot artifacts.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing the reference dataset.

        Returns
        -------
        MetricOutput
            Array plot artifact for selected input trajectories.
        """
        ref_ds = ctx.dataset_artifact.data

        ref = getattr(ref_ds, self.split)

        U_ref = np.asarray(ref.U)  # (n, n_t, n_sim)

        t = np.asarray(ref.time.t)

        # picking parameters
        idx_type = "rand"
        max_size = 4

        # %% Prepare time trajectory data
        U_ref, U_ref_idx = pick_trajectories(
            U_ref,
            idx_type=idx_type,
            max_size=max_size,
            return_idx=True,
        )

        # stack simulations one after another
        U_ref = convert_to_sample_format(U_ref)  # (n, n_t, n_sim) -> (n, n_t * n_sim)

        # %% Prepare data for plotting
        # ArrayPlotResult expects y as (n_t, n_lines, n_subplots)
        # Plot each input in a separate subplot, with lines for reference and comparison
        n, n_s = U_ref.shape
        t_traj = np.tile(t, n_s // t.shape[0])
        y_traj = np.empty((n_s, 1, n), dtype=float)
        for i in range(n):
            y_traj[:, 0, i] = U_ref[i, :]

        artifact = []
        artifact.append(
            ArrayPlotResult(
                name="input_values_over_time",
                x=t_traj,
                y=y_traj,
                x_label="time",
                y_labels=tuple(f"u_{U_ref_idx[ix]}" for ix in range(n)),
                line_labels=("u",),
                subplot_titles=tuple(f"Input {U_ref_idx[ix]}" for ix in range(n)),
                meta={"split": self.split},
            )
        )

        summaries = []

        return MetricOutput(
            summaries=summaries,
            artifacts=artifact,
        )
