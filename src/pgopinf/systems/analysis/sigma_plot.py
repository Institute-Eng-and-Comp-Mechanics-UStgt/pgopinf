from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass(frozen=True)
class SigmaPlotArtifact:
    """Reference to a stored singular-value plot.

    Attributes
    ----------
    png_path : str
        Path to the saved PNG file.
    """

    png_path: str


@dataclass
class SigmaPlotTask:
    """System-analysis task that creates a singular-value plot."""

    @property
    def task_kind(self) -> str:
        """Task kind label.

        Returns
        -------
        str
            The string ``"sigma_plot"``.
        """
        return "sigma_plot"

    def compute(self, *, system: PHSystem | LTISystem) -> np.ndarray:
        """Compute a singular-value plot figure.

        Parameters
        ----------
        system : PHSystem or LTISystem
            System to analyze.

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the singular-value plot.
        """
        sigma_plot_figure = system.singular_value_plot(
            "sigma_plot.png", return_only_figure=True
        )

        return sigma_plot_figure

    def save_result(self, *, task_dir: Path, value) -> None:
        """
        Save the figure to disk instead of pickling it.
        """
        png_path = task_dir / "sigma_plot.png"
        value.savefig(png_path)

        # close the figure if it is a matplotlib figure
        try:
            import matplotlib.pyplot as plt

            plt.close(value)
        except Exception:
            pass

    def load_result(self, *, task_dir: Path) -> SigmaPlotArtifact:
        """
        Return a lightweight reference to the saved plot.
        """
        png_path = task_dir / "sigma_plot.png"
        return SigmaPlotArtifact(png_path=str(png_path))
