from __future__ import annotations

from pathlib import Path
import pandas as pd
import shutil

from pgopinf.evaluation.reporting.render_scalar import ScalarRenderer
from pgopinf.evaluation.results import (
    ArrayPlotResult,
    MetricOutput,
    ScalarMetricResult,
    SeriesMetricResult,
    TableMetricResult,
)
from pgopinf.evaluation.reporting.render_series import (
    ArrayPlotRenderer,
    SeriesRenderer,
)
from pgopinf.io.results_path import RunResultsPath
from pgopinf.specs.evaluation.reporting import ReportingSpec


class ReportWriter:
    """Write metric summaries and artifacts for one run."""

    def __init__(self, reporting_spec: ReportingSpec):
        """Initialize the report writer.

        Parameters
        ----------
        reporting_spec : ReportingSpec
            Reporting configuration controlling summary and artifact outputs.
        """
        self.reporting_spec = reporting_spec
        self.series_renderer = SeriesRenderer()
        self.array_plot_renderer = ArrayPlotRenderer()
        self.scalar_renderer = ScalarRenderer()

    def write(
        self,
        *,
        run_paths: RunResultsPath,
        metric_outputs: list[MetricOutput],
    ) -> None:
        """Write all metric outputs for a run.

        Parameters
        ----------
        run_paths : RunResultsPath
            Run-specific output directories and file paths.
        metric_outputs : list of MetricOutput
            Metric summaries and artifacts to write.
        """

        if run_paths.evaluation_dir.exists():
            # remove old run evaluation results because some results get appended
            shutil.rmtree(run_paths.evaluation_dir)
        run_paths.evaluation_dir.mkdir(parents=True, exist_ok=True)

        # 1) collect summary rows
        summary_rows: list[dict] = []
        for mo in metric_outputs:
            summary_rows.extend(mo.summaries)

        if self.reporting_spec.save_summaries_csv and summary_rows:
            df = pd.DataFrame(summary_rows)
            df.to_csv(run_paths.metric_summaries_csv, index=False)

        # 2) render artifacts
        for mo in metric_outputs:
            for art in mo.artifacts:
                self._render_artifact(
                    artifact=art,
                    run_paths=run_paths,
                )

    def _render_artifact(self, *, artifact, run_paths: RunResultsPath) -> None:
        outputs = self.reporting_spec.default_outputs

        if isinstance(artifact, SeriesMetricResult):
            self.series_renderer.render(
                result=artifact,
                out_dir=run_paths.series_dir,
                outputs=outputs,
            )
            return
        elif isinstance(artifact, ArrayPlotResult):
            self.array_plot_renderer.render(
                result=artifact,
                out_dir=run_paths.array_plots_dir,
                outputs=outputs,
            )
        elif isinstance(artifact, ScalarMetricResult):
            self.scalar_renderer.render(
                result=artifact,
                out_dir=run_paths.scalars_dir,
                outputs=outputs,
            )

        else:
            raise ValueError(f"Unsupported artifact type: {type(artifact)}")

        # later:
        # if isinstance(artifact, TableMetricResult): ...
