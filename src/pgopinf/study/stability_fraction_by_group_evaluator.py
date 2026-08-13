from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from pgopinf.io.results_path import ResultsPath
from pgopinf.study.result import StudyResult


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


def _get_group_value(run_ref, row: pd.Series, key: str):
    if key == "variant":
        return run_ref.variant_name
    if key in row.index:
        return row[key]
    raise ValueError(
        f"group_by key {key!r} not available. " f"Row columns are {list(row.index)}"
    )


@dataclass
class StabilityFractionByGroupEvaluator:
    """Aggregate stable and unstable run fractions by groups."""

    metric: str
    group_by: tuple[str, ...]
    filters: dict[str, Any]
    include_stable: bool = True
    include_unstable: bool = True
    title: str | None = None

    def evaluate(
        self,
        *,
        study_result: StudyResult,
        paths: ResultsPath,
    ) -> pd.DataFrame:
        """Compute stability fractions from scalar metric outputs.

        Parameters
        ----------
        study_result : StudyResult
            Completed study result.
        paths : ResultsPath
            Result path manager used to locate run outputs.

        Returns
        -------
        pandas.DataFrame
            Grouped stability-fraction table.
        """
        rows = []

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
            df = df[df["metric"] == self.metric]
            if df.empty:
                continue

            for key, value in self.filters.items():
                if key not in df.columns:
                    raise ValueError(
                        f"Filter key {key!r} not found in scalar table columns {list(df.columns)}"
                    )
                df = df[df[key] == value]

            if df.empty:
                continue

            for _, row in df.iterrows():
                value = row["value"]
                group_key = tuple(
                    _get_group_value(run_ref, row, key) for key in self.group_by
                )

                rows.append(
                    {
                        "run_id": run_ref.ident_id,
                        "metric": self.metric,
                        "group_key": group_key,
                        "is_stable": value < 0.0,
                    }
                )

        raw_df = pd.DataFrame(rows)
        if raw_df.empty:
            cols = list(self.group_by) + [
                "fraction_type",
                "count",
                "total",
                "fraction",
                "percentage",
            ]
            return pd.DataFrame(columns=cols)

        grouped = (
            raw_df.groupby("group_key", dropna=False)
            .agg(
                count=("run_id", "count"),
                n_stable=("is_stable", "sum"),
            )
            .reset_index()
        )

        grouped["n_unstable"] = grouped["count"] - grouped["n_stable"]

        out_rows = []

        for _, row in grouped.iterrows():
            group_key = row["group_key"]

            group_dict = {k: v for k, v in zip(self.group_by, group_key)}

            if self.include_stable:
                frac = row["n_stable"] / row["count"]
                out_rows.append(
                    {
                        **group_dict,
                        "fraction_type": "stable",
                        "count": int(row["n_stable"]),
                        "total": int(row["count"]),
                        "fraction": frac,
                        "percentage": 100.0 * frac,
                    }
                )

            if self.include_unstable:
                frac = row["n_unstable"] / row["count"]
                out_rows.append(
                    {
                        **group_dict,
                        "fraction_type": "unstable",
                        "count": int(row["n_unstable"]),
                        "total": int(row["count"]),
                        "fraction": frac,
                        "percentage": 100.0 * frac,
                    }
                )

        return pd.DataFrame(out_rows)
