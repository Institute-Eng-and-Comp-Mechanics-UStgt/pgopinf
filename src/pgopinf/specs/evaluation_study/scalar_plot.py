from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation_study.base import (
    register_study_evaluation,
)
from pgopinf.specs.utils.utils import _as_tuple


def _normalize_mode_map(
    mode_map: Mapping[Any, str] | None,
) -> dict[tuple[Any, ...], str]:
    if not mode_map:
        return {}

    out: dict[tuple[Any, ...], str] = {}
    for k, v in mode_map.items():
        if isinstance(k, tuple):
            out[k] = v
        elif isinstance(k, str):
            try:
                parsed = ast.literal_eval(k)
            except (SyntaxError, ValueError):
                parsed = k
            out[parsed if isinstance(parsed, tuple) else (parsed,)] = v
        else:
            out[(k,)] = v
    return out


@register_study_evaluation("scalar_plot")
@dataclass(frozen=True)
class ScalarPlotSpec(SpecBase):
    """
    General study-level plot for scalar metrics.

    ----------
    Parameters
    ----------

    metrics : tuple[str]
        Names of scalar metrics to include in the plot. Only rows with
        `metric` matching one of these names are used.

        Examples:
            ("hinf_error",)
            ("hinf_error", "l2_error")

    x : str
        Dotted path specifying the parameter used for the x-axis.

        This path is resolved on the ExperimentSpec of each run.

        Examples:
            "reduction.r"
            "identification.lambda_reg"
            "seed"

    line_by : tuple[str]
        Columns used to define individual lines in the plot.

        Each unique combination of these columns produces one line.

        Typical choices include:

            "variant"
                Study variant name (e.g. pod_opinf, modal_opinf)

            "target"
                e.g. identified vs intrusive

            "metric"
                when plotting multiple metrics in the same plot

    panel_by : tuple[str]
        Columns used to split the plot into multiple subplots.

        Each unique combination of these columns produces one subplot.

        Typical choices include:

            "metric"
                separate subplot per metric

            "target"
                separate subplot for identified vs intrusive results

    filters : dict[str, Any]
        Optional row filters applied before plotting.

        Example:

            filters={"split": "TEST", "target": "identified"}

        Only rows satisfying these conditions are included.

    title : str | None
        Plot title.

    x_label : str | None
        Label for the x-axis. If None, the `x` parameter path is used.

    y_labels : tuple[str] | None
        Optional y-axis labels per subplot.

    --------
    Examples
    --------
    - one metric, different variants over r:
        metrics=("hinf_error",)
        x="reduction.r"
        line_by=("variant",)

    - identified and intrusive in same plot:
        metrics=("hinf_error",)
        x="reduction.r"
        line_by=("variant", "target")

    - multiple metrics in separate panels:
        metrics=("hinf_error", "l2_error")
        x="reduction.r"
        line_by=("variant",)
        panel_by=("metric",)
    """

    kind: Literal["scalar_plot"] = "scalar_plot"

    metrics: tuple[str, ...] = ("hinf_error",)
    x: str = "reduction.r"

    line_by: tuple[str, ...] = ("variant",)
    panel_by: tuple[str, ...] = ()

    filters: dict[str, Any] = field(default_factory=dict)

    title: str | None = None
    x_label: str | None = None
    y_labels: tuple[str, ...] | None = None

    # How to assign line modes to final lines.
    #
    # Examples:
    #  1) all variants of one metric share the same style:
    #   metrics=("hinf_error", "comp_int_ident_matA"),
    #   line_by=("metric", "variant"),
    #   line_mode_by=("metric",),
    #   line_mode_map={
    #       ("hinf_error",): "line+marker",
    #       ("comp_int_ident_matA",): "marker",
    #       },
    #  2) line mode depends on metric and variant
    #   metrics=("hinf_error", "comp_int_ident_matA"),
    #   line_by=("metric", "variant"),
    #   line_mode_by=("metric", "variant"),
    #   line_mode_map={
    #         ("hinf_error", "pod_opinf"): "line+marker",
    #         ("hinf_error", "modal_opinf"): "line",
    #         ("comp_int_ident_matA", "pod_opinf"): "marker",
    #         ("comp_int_ident_matA", "modal_opinf"): "marker",
    #     },
    #  3) line mode keyed by final label
    #       metrics=("hinf_error",),
    #       line_mode_by=("__line_label__",),
    #       line_mode_map={
    #           ("pod_opinf | identified",): "line+marker",
    #           ("pod_opinf | intrusive",): "marker",
    #           ("modal_opinf | identified",): "line",
    #           ("modal_opinf | intrusive",): "marker",
    #           },
    # line_mode_map maps keys of that form to a mode:
    #   "line", "line+marker", "marker"
    line_mode_by: tuple[str, ...] | None = None
    line_mode_map: dict[tuple[Any, ...], str] = field(default_factory=dict)

    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "metrics", _as_tuple(self.metrics))
        object.__setattr__(self, "line_by", _as_tuple(self.line_by))
        object.__setattr__(self, "panel_by", _as_tuple(self.panel_by))

        if self.line_mode_by is not None:
            object.__setattr__(self, "line_mode_by", _as_tuple(self.line_mode_by))
        object.__setattr__(
            self,
            "line_mode_map",
            _normalize_mode_map(self.line_mode_map),
        )

        if self.y_labels is not None:
            object.__setattr__(self, "y_labels", _as_tuple(self.y_labels))

        if not self.name:
            # if title is not provided, use kind as name for memoization and reporting
            (
                object.__setattr__(self, "name", self.kind)
                if self.title is None
                else object.__setattr__(self, "name", self.title)
            )

    def build(self):
        """Build the scalar-plot study evaluator."""
        from pgopinf.study.evaluator import ScalarPlotEvaluator

        return ScalarPlotEvaluator(
            metrics=self.metrics,
            x=self.x,
            line_by=self.line_by,
            panel_by=self.panel_by,
            filters=self.filters,
            title=self.title,
            x_label=self.x_label,
            y_labels=self.y_labels,
            line_mode_by=self.line_mode_by,
            line_mode_map=self.line_mode_map,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScalarPlotSpec":
        """Deserialize a scalar-plot study spec."""
        return cls(
            metrics=_as_tuple(d.get("metrics", ("hinf_error",))),
            x=str(d.get("x", "reduction.r")),
            line_by=_as_tuple(d.get("line_by", ("variant",))),
            panel_by=_as_tuple(d.get("panel_by", ())),
            filters=dict(d.get("filters", {})),
            title=d.get("title"),
            x_label=d.get("x_label"),
            y_labels=(
                _as_tuple(d["y_labels"]) if d.get("y_labels") is not None else None
            ),
            line_mode_by=(
                _as_tuple(d["line_mode_by"])
                if d.get("line_mode_by") is not None
                else None
            ),
            line_mode_map=(
                _normalize_mode_map(d["line_mode_map"])
                if d.get("line_mode_map") is not None
                else None
            ),
        )
