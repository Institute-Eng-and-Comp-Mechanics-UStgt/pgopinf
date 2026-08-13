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
class OutputValuesOverTimeMetric:
    """Compare output trajectories over time."""

    split: str = "TEST"
    target: str = "identified"
    relative: bool = True
    name: str = "output_values_over_time"

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Build output comparison and error artifacts.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing reference and comparison datasets.

        Returns
        -------
        MetricOutput
            Array plot artifacts and scalar output-error summaries.
        """
        ref_ds = ctx.dataset_artifact.data

        if self.target == "identified":
            if ctx.dataset_artifact_identified is None:
                raise ValueError("No identified dataset artifact available.")
            cmp_ds = ctx.dataset_artifact_identified.data
        elif self.target == "intrusive":
            if ctx.dataset_artifact_intrusive is None:
                raise ValueError("No intrusive dataset artifact available.")
            cmp_ds = ctx.dataset_artifact_intrusive.data
        else:
            raise ValueError(f"Unknown target {self.target!r}")

        ref = getattr(ref_ds, self.split)
        cmp = getattr(cmp_ds, self.split)

        Y_ref = np.asarray(ref.Y)  # (n, n_t, n_sim)
        Y_cmp = np.asarray(cmp.Y)  # (n, n_t, n_sim)

        t = np.asarray(ref.time.t)

        # picking parameters
        idx_type = "rand"
        max_size = 4

        # %% compute error trajectories for all outputs and simulations, and pick a few to plot
        error_dict = error_time_trajectories(
            Y_ref,
            Y_cmp,
            relative=self.relative,
            rel_error_type="mean_over_time",
        )

        error_value, error_idx = pick_trajectories(
            error_dict["error_value"],
            idx_type=idx_type,
            max_size=max_size,
            return_idx=True,
        )

        # stack simulations one after another
        error_value = convert_to_sample_format(
            error_value
        )  # (n, n_t, n_sim) -> (n, n_t * n_sim)

        # %% Prepare time trajectory data
        Y_ref, Y_ref_idx = pick_trajectories(
            Y_ref,
            idx_type=idx_type,
            max_size=max_size,
            return_idx=True,
        )
        Y_cmp, Y_cmp_idx = pick_trajectories(
            Y_cmp,
            idx_type=idx_type,
            max_size=max_size,
            return_idx=True,
        )

        assert np.allclose(Y_ref_idx, error_idx) and np.allclose(
            error_idx, Y_cmp_idx
        ), "Trajectory picking should be consistent"

        # stack simulations one after another
        Y_ref = convert_to_sample_format(Y_ref)  # (n, n_t, n_sim) -> (n, n_t * n_sim)
        Y_cmp = convert_to_sample_format(Y_cmp)  # (n, n_t, n_sim) -> (n, n_t * n_sim)

        # %% Prepare data for plotting
        # ArrayPlotResult expects y as (n_t, n_lines, n_subplots)
        # Plot each output in a separate subplot, with lines for reference and comparison
        n, n_s = Y_ref.shape
        t_traj = np.tile(t, n_s // t.shape[0])
        y_traj = np.empty((n_s, 2, n), dtype=float)
        for i in range(n):
            y_traj[:, 0, i] = Y_ref[i, :]
            y_traj[:, 1, i] = Y_cmp[i, :]

        artifact = []
        artifact.append(
            ArrayPlotResult(
                name="output_vs_original_over_time",
                x=t_traj,
                y=y_traj,
                x_label="time",
                y_labels=tuple(f"y_{Y_ref_idx[ix]}" for ix in range(n)),
                line_labels=("original", self.target),
                subplot_titles=tuple(f"Output {Y_ref_idx[ix]}" for ix in range(n)),
                meta={"split": self.split, "target": self.target},
                linestyles=("-", "--"),
            )
        )

        # error plots - plot all error into one subplot
        y_error = np.empty((n_s, n, 1), dtype=float)
        for i in range(n):
            y_error[:, i, 0] = error_value[i, :]

        if self.relative:
            error_label_prefix = "rel."
        else:
            error_label_prefix = "abs."
        artifact.append(
            ArrayPlotResult(
                name="error_output_vs_original_over_time",
                x=t_traj,
                y=y_error,
                x_label="time",
                y_labels=(f"{error_label_prefix} error",),
                line_labels=tuple(f"y_{error_idx[ix]}" for ix in range(n)),
                subplot_titles=(f"{error_label_prefix} error {self.split} outputs",),
                meta={
                    "split": self.split,
                    "target": self.target,
                    "relative": self.relative,
                },
            )
        )
        artifact.append(
            ScalarMetricResult(
                name=f"mean_error_output",
                value=error_dict["overall_mean_error"],
                unit="-",
                meta={
                    "split": self.split,
                    "target": self.target,
                    "relative": self.relative,
                },
            )
        )
        artifact.append(
            ScalarMetricResult(
                name=f"max_error_output",
                value=error_dict["overall_max_error"],
                unit="-",
                meta={
                    "split": self.split,
                    "target": self.target,
                    "relative": self.relative,
                },
            )
        )

        summaries = [
            {
                "metric": f"max_error_output",
                "split": self.split,
                "target": self.target,
                "value": error_dict["overall_max_error"],
                "unit": "-",
            },
            {
                "metric": f"mean_error_output",
                "split": self.split,
                "target": self.target,
                "value": error_dict["overall_mean_error"],
                "unit": "-",
            },
        ]

        return MetricOutput(
            summaries=summaries,
            artifacts=artifact,
        )
