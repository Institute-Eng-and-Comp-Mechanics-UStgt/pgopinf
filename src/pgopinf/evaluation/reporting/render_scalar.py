from __future__ import annotations

from pathlib import Path
import json
import logging
import pandas as pd

from typing import Any

from pgopinf.evaluation.results import ScalarMetricResult
from pgopinf.specs.evaluation.reporting import OutputSpec

logger = logging.getLogger(__name__)

COMMON_SCALAR_META_KEYS = ("target", "split", "relative")


class ScalarRenderer:
    """
    Scalar results are:
    - printed/logged
    - written as structured rows for later aggregation

    No png/pdf output.
    """

    def render(
        self,
        *,
        result: ScalarMetricResult,
        out_dir: Path,
        outputs: OutputSpec,
    ) -> None:
        """Render a scalar metric result.

        Parameters
        ----------
        result : ScalarMetricResult
            Scalar result to render.
        out_dir : Path
            Directory for scalar outputs.
        outputs : OutputSpec
            Output switches controlling logging and append-only files.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        row = self._to_row(result)

        if outputs.log_to_terminal:
            logger.info(self._format_console_line(row))

        if outputs.append_csv:
            self._append_csv(row=row, path=out_dir / "scalar_metrics.csv")

        if outputs.append_jsonl:
            self._append_jsonl(row=row, path=out_dir / "scalar_metrics.jsonl")

    @staticmethod
    def _to_row(result: ScalarMetricResult) -> dict[str, Any]:
        meta = dict(result.meta or {})

        row: dict[str, Any] = {
            "metric": result.name,
            "value": result.value,
            "unit": result.unit,
        }

        # extract a few common keys if present
        for key in COMMON_SCALAR_META_KEYS:
            row[key] = meta.pop(key, None)  # remove from meta if column is created

        # keep everything else in one structured column
        row["metadata_json"] = json.dumps(meta, sort_keys=True)

        return row

    def _format_console_line(self, row: dict) -> str:
        parts = [f"{row['metric']} = {row['value']:.6g}"]
        if row.get("unit", "-") != "-":
            parts[-1] += f" {row['unit']}"

        for key in ("summary", "split", "target", "field", "matrix", "r"):
            if key in row:
                parts.append(f"{key}={row[key]}")

        return " | ".join(parts)

    def _append_csv(self, *, row: dict, path: Path) -> None:
        df = pd.DataFrame([row])

        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def _append_jsonl(self, *, row: dict, path: Path) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
