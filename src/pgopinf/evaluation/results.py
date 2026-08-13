from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ScalarMetricResult:
    """Scalar metric artifact.

    Attributes
    ----------
    name : str
        Metric name.
    value : float
        Scalar value.
    unit : str, optional
        Physical or normalized unit.
    meta : dict of str to Any
        Additional metadata used for reporting.
    """

    name: str
    value: float
    unit: str = "-"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesMetricResult:
    """One-dimensional series metric artifact.

    Attributes
    ----------
    name : str
        Metric name.
    x : ndarray
        Horizontal-axis values.
    y : ndarray
        Series values.
    x_label : str, optional
        Label for ``x``.
    y_label : str, optional
        Label for ``y``.
    unit : str, optional
        Physical or normalized unit.
    meta : dict of str to Any
        Additional metadata used for reporting.
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str = "x"
    y_label: str = "y"
    unit: str = "-"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TableMetricResult:
    """Tabular metric artifact.

    Attributes
    ----------
    name : str
        Metric name.
    columns : tuple of str
        Column names.
    values : ndarray
        Table values.
    meta : dict of str to Any
        Additional metadata used for reporting.
    """

    name: str
    columns: tuple[str, ...]
    values: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricOutput:
    """
    summaries:
        flat rows for aggregation / CSV tables / studies

    artifacts:
        richer objects for plotting/export
    """

    summaries: list[dict[str, Any]]
    artifacts: list[Any]


@dataclass(frozen=True)
class ArrayPlotResult:
    """
    Generic plotting/export artifact.

    x: shape (n_x,)
    y: shape (n_x, n_lines, n_subplots)

    Typical interpretation:
      - x is time
      - each subplot is one output/state/channel
      - each line is one source (original, identified, intrusive, ...)
    """

    name: str
    x: np.ndarray
    y: np.ndarray

    x_label: str = "x"
    y_labels: tuple[str, ...] | None = None  # len = n_subplots
    line_labels: tuple[str, ...] = ()  # len = n_lines
    subplot_titles: tuple[str, ...] | None = None  # len = n_subplots

    # Optional per-line styling
    # len must be n_lines if provided
    markers: tuple[str | None, ...] | None = None
    linestyles: tuple[str | None, ...] | None = None
    line_modes: tuple[str, ...] | None = None  # line | line+marker | marker
    markevery: tuple[int | None, ...] | None = None
    colors: tuple[str | None, ...] | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        x = np.asarray(self.x)
        y = np.asarray(self.y)

        if x.ndim != 1:
            raise ValueError(f"x must be 1D, got shape {x.shape}")
        if y.ndim != 3:
            raise ValueError(
                f"y must have shape (n_x, n_lines, n_subplots), got {y.shape}"
            )
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"x length {x.shape[0]} does not match y first dim {y.shape[0]}"
            )

        n_lines = y.shape[1]
        n_subplots = y.shape[2]

        if self.line_labels and len(self.line_labels) != n_lines:
            raise ValueError(
                f"line_labels has length {len(self.line_labels)} but n_lines={n_lines}"
            )
        if self.y_labels is not None and len(self.y_labels) != n_subplots:
            raise ValueError(
                f"y_labels has length {len(self.y_labels)} but n_subplots={n_subplots}"
            )
        if self.subplot_titles is not None and len(self.subplot_titles) != n_subplots:
            raise ValueError(
                f"subplot_titles has length {len(self.subplot_titles)} but n_subplots={n_subplots}"
            )
        if self.markers is not None and len(self.markers) != n_lines:
            raise ValueError(
                f"markers must have length {n_lines}, got {len(self.markers)}"
            )
        if self.linestyles is not None and len(self.linestyles) != n_lines:
            raise ValueError(
                f"linestyles must have length {n_lines}, got {len(self.linestyles)}"
            )
        if self.line_modes is not None and len(self.line_modes) != n_lines:
            raise ValueError(
                f"line_modes must have length {n_lines}, got {len(self.line_modes)}"
            )
        if self.markevery is not None and len(self.markevery) != n_lines:
            raise ValueError(
                f"markevery must have length {n_lines}, got {len(self.markevery)}"
            )
        if self.colors is not None and len(self.colors) != n_lines:
            raise ValueError(
                f"colors must have length {n_lines}, got {len(self.colors)}"
            )
