from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pgopinf.evaluation.reporting.labels import (
    pretty_join,
    pretty_kv,
    pretty_label,
)
from pgopinf.evaluation.results import ArrayPlotResult
from pgopinf.io.results_path import ResultsPath, StudyResultsPath
from pgopinf.specs.utils.utils import _as_tuple
from pgopinf.study.result import StudyResult


def _get_dotted_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _label_from_cols(row: pd.Series, cols: tuple[str, ...]) -> str:
    if len(cols) == 0:
        return "value"
    return pretty_join([row[c] for c in cols])


def _line_key_from_row(row: pd.Series, cols: tuple[str, ...]) -> tuple[Any, ...]:
    if len(cols) == 0:
        return tuple()
    return tuple(row[c] for c in cols)


def _line_label_from_key(key: tuple[Any, ...]) -> str:
    if len(key) == 0:
        return "value"
    return pretty_join(list(key))


def _panel_key_from_row(row: pd.Series, cols: tuple[str, ...]) -> tuple[Any, ...]:
    if len(cols) == 0:
        return tuple()
    return tuple(row[c] for c in cols)


def _panel_label_from_key(key: tuple[Any, ...], cols: tuple[str, ...]) -> str:
    if len(cols) == 0:
        return "scalar"
    return " | ".join(pretty_kv(c, v) for c, v in zip(cols, key))


def _expand_metadata_json(df: pd.DataFrame) -> pd.DataFrame:
    if "metadata_json" not in df.columns:
        return df

    meta_rows = []
    for s in df["metadata_json"].fillna("{}"):
        try:
            meta_rows.append(json.loads(s))
        except Exception:
            meta_rows.append({})

    if not meta_rows:
        return df

    meta_df = pd.DataFrame(meta_rows)
    for col in meta_df.columns:
        if col not in df.columns:
            df[col] = meta_df[col]

    return df


def _unique_in_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# create protocol
class StudyEvaluator(Protocol):
    """Protocol for study-level evaluators."""

    def evaluate(self, study_result, paths: ResultsPath):
        """Evaluate a completed study."""
        ...


@dataclass
class ScalarPlotEvaluator:
    """Build a study-level array plot from scalar metric CSV files."""

    metrics: tuple[str, ...]
    x: str
    line_by: tuple[str, ...]
    panel_by: tuple[str, ...]
    filters: dict[str, Any]
    title: str | None = None
    x_label: str | None = None
    y_labels: tuple[str, ...] | None = None

    line_mode_by: tuple[str, ...] | None = None
    line_mode_map: dict[tuple[Any, ...], str] | None = None

    def evaluate(
        self, *, study_result: StudyResult, paths: ResultsPath
    ) -> tuple[pd.DataFrame, ArrayPlotResult]:
        """Evaluate scalar metric rows across study runs.

        Parameters
        ----------
        study_result : StudyResult
            Completed study result.
        paths : ResultsPath
            Result path manager used to locate run outputs.

        Returns
        -------
        tuple[pandas.DataFrame, ArrayPlotResult]
            Combined scalar table and plot artifact for rendering.
        """
        rows: list[dict[str, Any]] = []

        for run_ref in study_result.run_refs:
            run_paths = paths.for_run(
                experiment_name=run_ref.spec.name,
                ident_id=run_ref.ident_id,
            )
            scalar_csv = run_paths.scalar_metrics_csv
            if not scalar_csv.exists():
                continue

            df = pd.read_csv(scalar_csv)
            if df.empty:
                continue

            df = _expand_metadata_json(df)

            if "metric" not in df.columns:
                raise ValueError(f"{scalar_csv} does not contain a 'metric' column.")

            df = df[df["metric"].isin(self.metrics)]
            if df.empty:
                continue

            for key, value in self.filters.items():
                if key not in df.columns:
                    raise ValueError(
                        f"ScalarPlotEvaluator filter key {key!r} not found in scalar table columns "
                        f"{list(df.columns)} for file {scalar_csv}"
                    )
                df = df[df[key] == value]

            if df.empty:
                continue

            x_value = _get_dotted_attr(run_ref.spec, self.x)

            for _, row in df.iterrows():
                row_dict = dict(row)
                row_dict["variant"] = run_ref.variant_name
                row_dict["run_id"] = run_ref.ident_id
                row_dict["x_value"] = x_value
                rows.append(row_dict)

        raw_df = pd.DataFrame(rows)
        if raw_df.empty:
            return raw_df, self._empty_plot_result()

        plot_result = self._build_array_plot_result(raw_df)
        return raw_df, plot_result

    def _build_array_plot_result(self, df: pd.DataFrame) -> ArrayPlotResult:
        required_cols = {"x_value", "value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for scalar plot: {missing}")

        panel_keys = _unique_in_order(
            [_panel_key_from_row(row, self.panel_by) for _, row in df.iterrows()]
        )

        # line_labels_all = _unique_in_order(
        #     [_label_from_cols(row, self.line_by) for _, row in df.iterrows()]
        # )

        line_keys_all = _unique_in_order(
            [_line_key_from_row(row, self.line_by) for _, row in df.iterrows()]
        )

        # keep line labels in the order of self.metrics if line_by is ("metric",)
        if self.line_by == ("metric",):
            key_set = set(line_keys_all)
            line_keys_all = [(m,) for m in self.metrics if (m,) in key_set]

        line_modes = self._resolve_line_modes(df, line_keys_all)

        line_labels_all = [_line_label_from_key(key) for key in line_keys_all]

        x_values_sorted = np.array(sorted(df["x_value"].unique()))

        n_x = len(x_values_sorted)
        n_lines = len(line_labels_all)
        n_subplots = len(panel_keys) if len(panel_keys) > 0 else 1

        y = np.full((n_x, n_lines, n_subplots), np.nan)

        panel_index = {key: i for i, key in enumerate(panel_keys)}
        line_index = {key: i for i, key in enumerate(line_keys_all)}
        x_index = {v: i for i, v in enumerate(x_values_sorted)}

        for _, row in df.iterrows():
            pkey = _panel_key_from_row(row, self.panel_by)
            if len(self.panel_by) == 0:
                pidx = 0
            else:
                pidx = panel_index[pkey]

            lkey = _line_key_from_row(row, self.line_by)
            lidx = line_index[lkey]

            xidx = x_index[row["x_value"]]
            y[xidx, lidx, pidx] = row["value"]

        subplot_titles = None
        if len(self.panel_by) > 0:
            subplot_titles = tuple(
                _panel_label_from_key(key, self.panel_by) for key in panel_keys
            )
        elif self.title is not None:
            subplot_titles = (self.title,)

        if self.y_labels is not None:
            y_labels = self.y_labels
        else:
            if len(self.panel_by) > 0:
                y_labels = tuple("value" for _ in range(n_subplots))
            else:
                y_labels = ("value",)

        return ArrayPlotResult(
            name=self.title or "scalar_plot",
            x=x_values_sorted,
            y=y,
            x_label=self.x_label or pretty_label(self.x),
            y_labels=y_labels,
            line_labels=tuple(line_labels_all),
            subplot_titles=subplot_titles,
            line_modes=line_modes,
            meta={
                "study_plot_kind": "scalar_plot",
                "metrics": list(self.metrics),
                "x": self.x,
                "line_by": list(self.line_by),
                "panel_by": list(self.panel_by),
            },
        )

    def _resolve_line_modes(
        self,
        df: pd.DataFrame,
        line_keys_all: list[tuple[Any, ...]],
    ) -> list[str] | None:
        """
        Resolve line modes for the final ordered raw line keys.

        Semantics:
        - if line_mode_by is None or line_mode_map empty -> no explicit modes
        - if line_mode_by == ("__line_label__",) -> map by display label
        - otherwise -> map by tuple of row values from those columns
        """
        if self.line_mode_by is None or not self.line_mode_map:
            return None

        key_to_mode_key: dict[tuple[Any, ...], tuple[Any, ...]] = {}

        for _, row in df.iterrows():
            lkey = _line_key_from_row(row, self.line_by)

            if self.line_mode_by == ("__line_label__",):
                mode_key = (_line_label_from_key(lkey),)
            else:
                missing = [c for c in self.line_mode_by if c not in row.index]
                if missing:
                    raise ValueError(
                        f"line_mode_by columns {missing} are not present in scalar table columns "
                        f"{list(df.columns)}"
                    )
                mode_key = tuple(row[c] for c in self.line_mode_by)

            if lkey not in key_to_mode_key:
                key_to_mode_key[lkey] = mode_key

        out: list[str] = []
        for lkey in line_keys_all:
            if self.line_mode_by == ("__line_label__",):
                key = (_line_label_from_key(lkey),)
            else:
                key = key_to_mode_key.get(lkey, None)

            if key is None:
                out.append("line")
            else:
                out.append(self.line_mode_map.get(key, "line"))

        return out

    def _empty_plot_result(self) -> ArrayPlotResult:
        return ArrayPlotResult(
            name=self.title or "scalar_plot_empty",
            x=np.array([], dtype=float),
            y=np.zeros((0, 0, 1), dtype=float),
            x_label=self.x_label or self.x,
            y_labels=("value",),
            line_labels=(),
            subplot_titles=(self.title or "empty",),
            meta={"study_plot_kind": "scalar_plot", "empty": True},
        )
