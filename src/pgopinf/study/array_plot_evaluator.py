from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pgopinf.evaluation.results import ArrayPlotResult
from pgopinf.io.results_path import ResultsPath
from pgopinf.study.result import StudyResult


def _unique_in_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _get_dotted_attr(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


@dataclass
class ArrayPlotStudyEvaluator:
    """Build a study-level plot from run-level array plot artifacts."""

    artifact_name: str
    artifact_filters: dict[str, Any]

    subplot_indices: tuple[int, ...]
    line_labels: tuple[str, ...]
    variants: tuple[str, ...]
    line_by: tuple[str, ...]
    filters: dict[str, Any]
    title: str | None = None

    line_mode_by: tuple[str, ...] | None = None
    line_mode_map: dict[tuple[Any, ...], str] | None = None

    linestyle_by: tuple[str, ...] | None = None
    linestyle_map: dict[tuple[Any, ...], str | None] | None = None

    def evaluate(
        self,
        *,
        study_result: StudyResult,
        paths: ResultsPath,
    ) -> tuple[pd.DataFrame, ArrayPlotResult]:
        """Evaluate matching array plot artifacts across study runs.

        Parameters
        ----------
        study_result : StudyResult
            Completed study result.
        paths : ResultsPath
            Result path manager used to locate run outputs.

        Returns
        -------
        tuple[pandas.DataFrame, ArrayPlotResult]
            Raw extracted trajectory table and plot artifact for rendering.
        """
        rows: list[dict[str, Any]] = []
        needed_spec_paths = self._collect_needed_spec_paths()

        for run_ref in study_result.run_refs:
            if self.variants and run_ref.variant_name not in self.variants:
                continue

            if not self._run_matches_filters(run_ref.spec):
                continue

            run_paths = paths.for_run(
                experiment_name=run_ref.spec.name,
                ident_id=run_ref.ident_id,
            )

            matches = self._resolve_matching_artifacts(run_paths.array_plots_dir)
            if not matches:
                continue

            for match in matches:
                csv_path = match["csv_path"]
                json_path = match["json_path"]
                artifact_meta = match["artifact_meta"]

                df = pd.read_csv(csv_path)
                meta = json.loads(json_path.read_text(encoding="utf-8"))

                x_label = meta["x_label"]
                n_subplots = int(meta["n_subplots"])

                subplot_titles = meta.get("subplot_titles")
                if subplot_titles is None:
                    subplot_titles = [f"subplot{i}" for i in range(n_subplots)]

                source_line_labels = meta.get("line_labels")
                if source_line_labels is None:
                    source_line_labels = [
                        f"line{i}" for i in range(int(meta["n_lines"]))
                    ]

                x = np.asarray(df[x_label])

                for subplot_idx in self.subplot_indices:
                    subplot_title = subplot_titles[subplot_idx]

                    for source_line_label in source_line_labels:
                        if (
                            self.line_labels
                            and source_line_label not in self.line_labels
                        ):
                            continue

                        col_name = self._column_name(
                            subplot_title=subplot_title,
                            source_line_label=source_line_label,
                            single_subplot=(n_subplots == 1),
                        )
                        if col_name not in df.columns:
                            continue

                        row = {
                            "variant": run_ref.variant_name,
                            "run_id": run_ref.ident_id,
                            "subplot_index": subplot_idx,
                            "subplot_title": subplot_title,
                            "source_line": source_line_label,
                            "x_label": x_label,
                            "x": x,
                            "y": np.asarray(df[col_name]),
                        }

                        # expose run-spec fields requested by filters/line_by/styles
                        for spec_path in needed_spec_paths:
                            row[f"spec:{spec_path}"] = _get_dotted_attr(
                                run_ref.spec, spec_path
                            )

                        # expose artifact metadata fields requested by line_by/styles
                        for k, v in artifact_meta.items():
                            row[f"artifact:{k}"] = v

                        rows.append(row)

        raw_df = pd.DataFrame(rows)
        if raw_df.empty:
            return raw_df, self._empty_result()

        return raw_df, self._build_array_plot_result(raw_df)

    def _collect_needed_spec_paths(self) -> set[str]:
        out: set[str] = set()

        for path in self.filters.keys():
            out.add(path)

        for cols in (
            self.line_by,
            self.line_mode_by or (),
            self.linestyle_by or (),
        ):
            for key in cols:
                if key.startswith("spec:"):
                    out.add(key[len("spec:") :])

        return out

    def _run_matches_filters(self, spec) -> bool:
        for path, expected in self.filters.items():
            actual = _get_dotted_attr(spec, path)
            if actual != expected:
                return False
        return True

    def _resolve_matching_artifacts(
        self, array_plots_dir: Path
    ) -> list[dict[str, Any]]:
        index_path = array_plots_dir / "index.json"
        if not index_path.exists():
            return []

        index = json.loads(index_path.read_text(encoding="utf-8"))

        matches = []
        for entry in index:
            if entry.get("name") != self.artifact_name:
                continue

            meta = entry.get("meta", {})
            ok = True
            for k, v in self.artifact_filters.items():
                if meta.get(k) != v:
                    ok = False
                    break

            if not ok:
                continue

            stem = entry["stem"]
            csv_path = array_plots_dir / f"{stem}.csv"
            json_path = array_plots_dir / f"{stem}.json"

            if not csv_path.exists() or not json_path.exists():
                continue

            matches.append(
                {
                    "stem": stem,
                    "csv_path": csv_path,
                    "json_path": json_path,
                    "artifact_meta": meta,
                }
            )

        return matches

    def _column_name(
        self,
        *,
        subplot_title: str,
        source_line_label: str,
        single_subplot: bool,
    ) -> str:
        if single_subplot:
            return source_line_label
        return f"{subplot_title}__{source_line_label}"

    def _line_key(self, row: pd.Series) -> tuple[Any, ...]:
        out = []
        for c in self.line_by:
            if c == "variant":
                out.append(row["variant"])
            elif c == "source_line":
                out.append(row["source_line"])
            elif c.startswith("spec:"):
                out.append(row[c])
            elif c.startswith("artifact:"):
                out.append(row[c])
            else:
                raise ValueError(
                    f"Unsupported line_by entry {c!r}. "
                    "Allowed: 'variant', 'source_line', 'spec:<dotted.path>', 'artifact:<meta_key>'"
                )
        return tuple(out)

    def _style_key(self, row: pd.Series, cols: tuple[str, ...]) -> tuple[Any, ...]:
        out = []
        for c in cols:
            if c == "variant":
                out.append(row["variant"])
            elif c == "source_line":
                out.append(row["source_line"])
            elif c.startswith("spec:"):
                out.append(row[c])
            elif c.startswith("artifact:"):
                out.append(row[c])
            else:
                raise ValueError(
                    f"Unsupported style key entry {c!r}. "
                    "Allowed: 'variant', 'source_line', 'spec:<dotted.path>', 'artifact:<meta_key>'"
                )
        return tuple(out)

    def _line_label(self, key: tuple[Any, ...]) -> str:
        if len(key) == 0:
            return "value"
        return " | ".join(str(v) for v in key)

    def _resolve_style_values(
        self,
        df: pd.DataFrame,
        line_keys: list[tuple[Any, ...]],
        *,
        style_by: tuple[str, ...] | None,
        style_map: dict[tuple[Any, ...], Any] | None,
        default: Any,
    ) -> list[Any] | None:
        if style_by is None or not style_map:
            return None

        linekey_to_stylekey: dict[tuple[Any, ...], tuple[Any, ...]] = {}

        for _, row in df.iterrows():
            lkey = self._line_key(row)
            skey = self._style_key(row, style_by)
            if lkey not in linekey_to_stylekey:
                linekey_to_stylekey[lkey] = skey

        out = []
        for lkey in line_keys:
            skey = linekey_to_stylekey.get(lkey, None)
            if skey is None:
                out.append(default)
            else:
                out.append(style_map.get(skey, default))
        return out

    def _build_array_plot_result(self, df: pd.DataFrame) -> ArrayPlotResult:
        subplot_keys = _unique_in_order(
            [(row["subplot_index"], row["subplot_title"]) for _, row in df.iterrows()]
        )
        line_keys = _unique_in_order([self._line_key(row) for _, row in df.iterrows()])

        first_x = np.asarray(df.iloc[0]["x"])
        n_x = len(first_x)
        n_lines = len(line_keys)
        n_subplots = len(subplot_keys)

        y = np.full((n_x, n_lines, n_subplots), np.nan, dtype=float)

        subplot_index = {k: i for i, k in enumerate(subplot_keys)}
        line_index = {k: i for i, k in enumerate(line_keys)}

        for _, row in df.iterrows():
            pkey = (row["subplot_index"], row["subplot_title"])
            lkey = self._line_key(row)

            pidx = subplot_index[pkey]
            lidx = line_index[lkey]

            x = np.asarray(row["x"])
            yy = np.asarray(row["y"])

            if len(x) != n_x or not np.allclose(x, first_x):
                raise ValueError(
                    "Study array plot requires identical x grids across selected runs/artifacts."
                )

            y[:, lidx, pidx] = yy

        line_labels = tuple(self._line_label(k) for k in line_keys)
        subplot_titles = tuple(title for _, title in subplot_keys)

        line_modes = self._resolve_style_values(
            df,
            line_keys,
            style_by=self.line_mode_by,
            style_map=self.line_mode_map,
            default="line",
        )
        linestyles = self._resolve_style_values(
            df,
            line_keys,
            style_by=self.linestyle_by,
            style_map=self.linestyle_map,
            default="-",
        )

        return ArrayPlotResult(
            name=self.title or self.artifact_name,
            x=first_x,
            y=y,
            x_label=str(df.iloc[0]["x_label"]),
            y_labels=tuple("value" for _ in range(n_subplots)),
            line_labels=line_labels,
            subplot_titles=subplot_titles,
            line_modes=tuple(line_modes) if line_modes is not None else None,
            linestyles=tuple(linestyles) if linestyles is not None else None,
            meta={
                "study_plot_kind": "array_plot",
                "artifact_name": self.artifact_name,
                "artifact_filters": self.artifact_filters,
                "line_by": list(self.line_by),
                "filters": self.filters,
            },
        )

    def _empty_result(self) -> ArrayPlotResult:
        return ArrayPlotResult(
            name=self.title or f"{self.artifact_name}_empty",
            x=np.array([], dtype=float),
            y=np.zeros((0, 0, 1), dtype=float),
            x_label="x",
            y_labels=("value",),
            line_labels=(),
            subplot_titles=("empty",),
            meta={"study_plot_kind": "array_plot", "empty": True},
        )
