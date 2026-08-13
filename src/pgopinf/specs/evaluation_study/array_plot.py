from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation_study.base import (
    register_study_evaluation,
)
from pgopinf.specs.utils.utils import _as_tuple


def _normalize_map(
    value_map: Mapping[Any, Any] | None,
) -> dict[tuple[Any, ...], Any]:
    if not value_map:
        return {}

    out: dict[tuple[Any, ...], Any] = {}
    for k, v in value_map.items():
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


@register_study_evaluation("array_plot")
@dataclass(frozen=True)
class ArrayPlotStudySpec(SpecBase):
    """
    General study-level plot for run-level ArrayPlot artifacts.

    Parameters
    ----------
    artifact_name : str
        Logical name of the run-level array plot artifact.
        Example:
            "error_output_vs_original_over_time"

    artifact_filters : dict[str, Any]
        Filters on run-level artifact metadata, not on the run spec.

        Example:
            {"split": "TRAIN"}
            {"split": "TRAIN", "target": "identified"}

        Important:
            This may match multiple artifacts per run.
            Example:
                target="identified" and target="intrusive"
            can both be included if only split is filtered.

    subplot_indices : tuple[int, ...]
        Indices of the subplots from the run-level array plot to include.

    line_labels : tuple[str, ...]
        Which line labels from the run-level plot should be extracted.
        Empty means all line labels from the artifact.

    variants : tuple[str, ...]
        Which study variants to include.
        Empty means all variants.

    line_by : tuple[str, ...]
        Defines how selected trajectories are grouped into lines
        in the study plot.

        Supported entries:
            "variant"
            "source_line"
            "spec:<dotted.path>"
            "artifact:<meta_key>"

        Examples:
            ("variant",)
            ("variant", "artifact:target")
            ("variant", "spec:reduction.r")
            ("variant", "artifact:target", "source_line")

    filters : dict[str, Any]
        Filters on the run spec.
        Example:
            {"reduction.r": 10}

    title : str | None
        Plot title.

    name : str | None
        Internal name for memoization/reporting.

    line_mode_by / line_mode_map
        Same semantics as in ScalarPlotSpec.
        Supported modes:
            "line", "line+marker", "marker"

    linestyle_by / linestyle_map
        Line style formatting keyed analogously to line_mode_by/map.
    """

    kind: Literal["array_plot"] = "array_plot"

    artifact_name: str = "output_vs_original_over_time"
    artifact_filters: dict[str, Any] = field(default_factory=dict)

    subplot_indices: tuple[int, ...] = (0,)
    line_labels: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()

    line_by: tuple[str, ...] = ("variant",)
    filters: dict[str, Any] = field(default_factory=dict)

    title: str | None = None
    name: str | None = None

    line_mode_by: tuple[str, ...] | None = None
    line_mode_map: dict[tuple[Any, ...], str] = field(default_factory=dict)

    linestyle_by: tuple[str, ...] | None = None
    linestyle_map: dict[tuple[Any, ...], str | None] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "subplot_indices", tuple(self.subplot_indices))
        object.__setattr__(self, "line_labels", _as_tuple(self.line_labels))
        object.__setattr__(self, "variants", _as_tuple(self.variants))
        object.__setattr__(self, "line_by", _as_tuple(self.line_by))

        if self.line_mode_by is not None:
            object.__setattr__(self, "line_mode_by", _as_tuple(self.line_mode_by))
        if self.linestyle_by is not None:
            object.__setattr__(self, "linestyle_by", _as_tuple(self.linestyle_by))

        object.__setattr__(self, "line_mode_map", _normalize_map(self.line_mode_map))
        object.__setattr__(self, "linestyle_map", _normalize_map(self.linestyle_map))

        if not self.name:
            object.__setattr__(
                self,
                "name",
                self.kind if self.title is None else self.title,
            )

    def build(self):
        """Build the array-plot study evaluator."""
        from pgopinf.study.array_plot_evaluator import (
            ArrayPlotStudyEvaluator,
        )

        return ArrayPlotStudyEvaluator(
            artifact_name=self.artifact_name,
            artifact_filters=self.artifact_filters,
            subplot_indices=self.subplot_indices,
            line_labels=self.line_labels,
            variants=self.variants,
            line_by=self.line_by,
            filters=self.filters,
            title=self.title,
            line_mode_by=self.line_mode_by,
            line_mode_map=self.line_mode_map,
            linestyle_by=self.linestyle_by,
            linestyle_map=self.linestyle_map,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ArrayPlotStudySpec":
        """Deserialize an array-plot study spec."""
        return cls(
            artifact_name=str(d.get("artifact_name", "output_vs_original_over_time")),
            artifact_filters=dict(d.get("artifact_filters", {})),
            subplot_indices=tuple(d.get("subplot_indices", (0,))),
            line_labels=_as_tuple(d.get("line_labels", ())),
            variants=_as_tuple(d.get("variants", ())),
            line_by=_as_tuple(d.get("line_by", ("variant",))),
            filters=dict(d.get("filters", {})),
            title=d.get("title"),
            name=d.get("name"),
            line_mode_by=(
                _as_tuple(d["line_mode_by"])
                if d.get("line_mode_by") is not None
                else None
            ),
            line_mode_map=_normalize_map(d.get("line_mode_map", {})),
            linestyle_by=(
                _as_tuple(d["linestyle_by"])
                if d.get("linestyle_by") is not None
                else None
            ),
            linestyle_map=_normalize_map(d.get("linestyle_map", {})),
        )
