from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pgopinf.evaluation.results import (
    ArrayPlotResult,
    SeriesMetricResult,
)
from pgopinf.specs.evaluation.reporting import OutputSpec


class SeriesRenderer:
    """Renderer for one-dimensional series metric artifacts."""

    def render(
        self,
        *,
        result: SeriesMetricResult,
        out_dir: Path,
        outputs: OutputSpec,
    ) -> None:
        """Render a series result to the requested output formats.

        Parameters
        ----------
        result : SeriesMetricResult
            Series result to render.
        out_dir : Path
            Output directory.
        outputs : OutputSpec
            Output switches for CSV, PNG, and PDF files.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = self._make_stem(result)

        if outputs.csv:
            self.to_csv(result=result, path=out_dir / f"{stem}.csv")

        if outputs.png:
            self.to_png(result=result, path=out_dir / f"{stem}.png")

        if outputs.pdf:
            self.to_pdf(result=result, path=out_dir / f"{stem}.pdf")

    def to_csv(self, *, result: SeriesMetricResult, path: Path) -> None:
        """Write a series result to CSV.

        Parameters
        ----------
        result : SeriesMetricResult
            Series result to write.
        path : Path
            Destination CSV path.
        """
        df = pd.DataFrame(
            {
                result.x_label: np.asarray(result.x),
                result.y_label: np.asarray(result.y),
            }
        )
        df.to_csv(path, index=False)

    def to_png(self, *, result: SeriesMetricResult, path: Path) -> None:
        """Write a series plot to PNG.

        Parameters
        ----------
        result : SeriesMetricResult
            Series result to plot.
        path : Path
            Destination PNG path.
        """
        plt.figure()
        plt.plot(result.x, result.y)
        plt.xlabel(result.x_label)
        plt.ylabel(result.y_label)
        plt.title(result.name)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def to_pdf(self, *, result: SeriesMetricResult, path: Path) -> None:
        """Write a series plot to PDF.

        Parameters
        ----------
        result : SeriesMetricResult
            Series result to plot.
        path : Path
            Destination PDF path.
        """
        plt.figure()
        plt.plot(result.x, result.y)
        plt.xlabel(result.x_label)
        plt.ylabel(result.y_label)
        plt.title(result.name)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _make_stem(self, result: SeriesMetricResult) -> str:
        parts = [result.name]
        for key in ("split", "target"):
            if key in result.meta:
                parts.append(str(result.meta[key]))
        return "_".join(parts)


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ArrayPlotRenderer:
    """Renderer for multi-line, multi-subplot array plot artifacts."""

    def render(
        self,
        *,
        result: ArrayPlotResult,
        out_dir: Path,
        outputs: OutputSpec,
    ):
        """Render an array plot result to requested output formats.

        Parameters
        ----------
        result : ArrayPlotResult
            Array plot artifact to render.
        out_dir : Path
            Output directory.
        outputs : OutputSpec
            Output switches for CSV, JSON, PNG, and PDF files.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = self._stem(result)

        if outputs.csv:
            self.to_csv(result=result, path=out_dir / f"{stem}.csv")
            self.to_json(result=result, path=out_dir / f"{stem}.json")
        if outputs.png:
            self.to_png(result=result, path=out_dir / f"{stem}.png")
        if outputs.pdf:
            self.to_pdf(result=result, path=out_dir / f"{stem}.pdf")

        self._update_index(out_dir=out_dir, result=result, stem=stem)

    def to_csv(self, *, result: ArrayPlotResult, path: Path):
        """Write array plot values to CSV.

        Parameters
        ----------
        result : ArrayPlotResult
            Array plot artifact.
        path : Path
            Destination CSV path.
        """
        x = np.asarray(result.x)
        y = np.asarray(result.y)
        n_x, n_lines, n_subplots = y.shape

        data = {result.x_label: x}
        single_subplot = n_subplots == 1

        for i_subplot in range(n_subplots):
            subplot_name = (
                result.subplot_titles[i_subplot]
                if result.subplot_titles is not None
                else f"subplot{i_subplot}"
            )
            for i_line in range(n_lines):
                line_name = (
                    result.line_labels[i_line]
                    if result.line_labels
                    else f"line{i_line}"
                )

                if single_subplot:
                    col_name = line_name
                else:
                    col_name = f"{subplot_name}__{line_name}"

                data[col_name] = y[:, i_line, i_subplot]

        pd.DataFrame(data).to_csv(path, index=False)

    def to_png(self, *, result: ArrayPlotResult, path: Path, replace_inf: bool = True):
        """Write an array plot to PNG.

        Parameters
        ----------
        result : ArrayPlotResult
            Array plot artifact.
        path : Path
            Destination PNG path.
        replace_inf : bool, optional
            If ``True``, replace infinite values before plotting.
        """
        x = np.asarray(result.x)
        y = np.asarray(result.y)
        n_x, n_lines, n_subplots = y.shape

        # check if inf in array and replace
        if replace_inf and np.isinf(y).any():
            replace_num = 1e10  # large number to replace inf values for plotting
            y = np.where(np.isinf(y), replace_num, y)

        fig, axes = plt.subplots(n_subplots, 1, sharex=True, squeeze=False)
        axes = axes[:, 0]

        for i_subplot, ax in enumerate(axes):
            y_subplot = y[:, :, i_subplot]

            # set all values above threshold to threshold since it is assumed that these plots are unstable
            stability_threshold = (
                1e11  # large number to replace unstable values for plotting
            )
            y_subplot = np.where(
                y_subplot > stability_threshold, stability_threshold, y_subplot
            )
            y_subplot = np.where(
                y_subplot < -stability_threshold, -stability_threshold, y_subplot
            )

            if np.isfinite(y_subplot).all():
                # detect scale for each subplot and set yscale to log if values span several orders of magnitude
                y_min = np.min(y_subplot)
                y_max = np.max(y_subplot)
                if y_min > 0 and y_max / y_min > 100:
                    ax.set_yscale("log")
            for i_line in range(n_lines):
                label = (
                    result.line_labels[i_line]
                    if result.line_labels
                    else f"line{i_line}"
                )
                style_kwargs = self._line_style_kwargs(result, i_line)
                ax.plot(x, y_subplot[:, i_line], label=label, **style_kwargs)

            if result.subplot_titles is not None:
                ax.set_title(result.subplot_titles[i_subplot])
            if result.y_labels is not None:
                ax.set_ylabel(result.y_labels[i_subplot])

            ax.legend()

        axes[-1].set_xlabel(result.x_label)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def to_pdf(self, *, result: ArrayPlotResult, path: Path):
        """Write an array plot to PDF.

        Parameters
        ----------
        result : ArrayPlotResult
            Array plot artifact.
        path : Path
            Destination PDF path.
        """
        self.to_png(result=result, path=path)

    def _stem(self, result: ArrayPlotResult) -> str:
        parts = [result.name]
        for key in ("split", "target", "simulation_mode"):
            if key in result.meta:
                parts.append(str(result.meta[key]))
        if "simulation_index" in result.meta:
            parts.append(f"sim{result.meta['simulation_index']}")
        return "_".join(parts)

    def _update_index(
        self, *, out_dir: Path, result: ArrayPlotResult, stem: str
    ) -> None:
        index_path = out_dir / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = []

        entry = {
            "name": result.name,
            "stem": stem,
            "meta": result.meta,
        }

        index = [e for e in index if e.get("stem") != stem]
        index.append(entry)

        index_path.write_text(
            json.dumps(index, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _line_style_kwargs(self, result: ArrayPlotResult, i_line: int) -> dict:
        mode = result.line_modes[i_line] if result.line_modes is not None else "line"
        marker = result.markers[i_line] if result.markers is not None else None
        linestyle = result.linestyles[i_line] if result.linestyles is not None else "-"
        markevery = result.markevery[i_line] if result.markevery is not None else None
        color = (
            result.colors[i_line]
            if getattr(result, "colors", None) is not None
            else None
        )

        kwargs: dict = {}

        if mode == "line":
            kwargs["linestyle"] = linestyle if linestyle is not None else "-"
            if marker is not None:
                kwargs["marker"] = marker
            if markevery is not None:
                kwargs["markevery"] = markevery

        elif mode == "line+marker":
            kwargs["linestyle"] = linestyle if linestyle is not None else "-"
            kwargs["marker"] = marker if marker is not None else "o"
            if markevery is not None:
                kwargs["markevery"] = markevery

        elif mode == "marker":
            kwargs["linestyle"] = "None"
            kwargs["marker"] = marker if marker is not None else "o"
            if markevery is not None:
                kwargs["markevery"] = markevery

        else:
            raise ValueError(
                f"Unknown line mode {mode!r}. Valid modes: 'line', 'line+marker', 'marker'."
            )

        if color is not None:
            kwargs["color"] = color

        return kwargs

    def to_json(self, *, result: ArrayPlotResult, path: Path):
        """Write array plot metadata to JSON.

        Parameters
        ----------
        result : ArrayPlotResult
            Array plot artifact.
        path : Path
            Destination JSON path.
        """
        y = np.asarray(result.y)
        payload = {
            "name": result.name,
            "x_label": result.x_label,
            "line_labels": list(result.line_labels),
            "subplot_titles": (
                list(result.subplot_titles)
                if result.subplot_titles is not None
                else None
            ),
            "y_labels": list(result.y_labels) if result.y_labels is not None else None,
            "n_lines": int(y.shape[1]),
            "n_subplots": int(y.shape[2]),
            "meta": result.meta,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
