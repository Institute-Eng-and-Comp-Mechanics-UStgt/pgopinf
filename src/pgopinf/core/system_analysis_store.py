from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pgopinf.specs.base import stable_id
from pgopinf.systems.analysis.artifact import SystemAnalysisBundle


class SystemAnalysisStore:
    """Store bundles of system-analysis task results."""

    def __init__(self, root: str | Path):
        """Initialize the system-analysis store.

        Parameters
        ----------
        root : str or Path
            Directory in which analysis bundles are stored.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def compute_id(self, *, system_id: str, analysis_spec) -> str:
        """Compute the stable identifier for a system-analysis bundle.

        Parameters
        ----------
        system_id : str
            Identifier of the analyzed system.
        analysis_spec
            Specification containing the analysis tasks.

        Returns
        -------
        str
            Stable analysis-bundle identifier.
        """
        key = {
            "artifact": "system_analysis",
            "artifact_format_version": 1,
            "system_id": system_id,
            "analysis_spec": analysis_spec.to_dict(),
        }
        return stable_id(key)

    def _dir(self, analysis_id: str) -> Path:
        return self.root / f"system_analysis_{analysis_id}"

    def exists(self, analysis_id: str) -> bool:
        """Return whether a system-analysis bundle exists.

        Parameters
        ----------
        analysis_id : str
            Analysis-bundle identifier.

        Returns
        -------
        bool
            ``True`` if metadata and value files are present.
        """
        d = self._dir(analysis_id)
        return (d / "meta.json").exists() and (d / "values.pkl").exists()

    def save(self, analysis_id: str, artifact: SystemAnalysisBundle) -> None:
        """Persist a system-analysis bundle.

        Parameters
        ----------
        analysis_id : str
            Analysis-bundle identifier.
        artifact : SystemAnalysisBundle
            Bundle containing task IDs, computed values, and metadata.
        """
        d = self._dir(analysis_id)
        d.mkdir(parents=True, exist_ok=True)

        (d / "meta.json").write_text(
            json.dumps(
                {
                    "task_ids": artifact.task_ids,
                    "meta": artifact.meta,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with open(d / "values.pkl", "wb") as f:
            pickle.dump(artifact.values, f)

    def load(self, analysis_id: str) -> SystemAnalysisBundle:
        """Load a system-analysis bundle.

        Parameters
        ----------
        analysis_id : str
            Analysis-bundle identifier.

        Returns
        -------
        SystemAnalysisBundle
            Stored task values and metadata.
        """
        d = self._dir(analysis_id)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        with open(d / "values.pkl", "rb") as f:
            values = pickle.load(f)
        return SystemAnalysisBundle(
            task_ids=meta.get("task_ids", {}),
            values=values,
            meta=meta.get("meta", meta),
        )

    def get_or_create(self, *, system_id: str, system, analysis_spec):
        """Load an analysis bundle or compute and cache the requested tasks.

        Parameters
        ----------
        system_id : str
            Identifier of the analyzed system.
        system
            System instance passed to each analysis task.
        analysis_spec
            Specification containing the tasks to compute.

        Returns
        -------
        tuple[str, SystemAnalysisBundle]
            Analysis-bundle identifier and bundle.

        Raises
        ------
        ValueError
            If two tasks produce the same result key.
        """
        analysis_id = self.compute_id(system_id=system_id, analysis_spec=analysis_spec)

        if self.exists(analysis_id):
            return analysis_id, self.load(analysis_id)

        values: dict[str, Any] = {}
        task_ids: dict[str, str] = {}
        for task_spec in analysis_spec.tasks:
            task = task_spec.build()
            key = task_spec.result_key()
            if key in values:
                raise ValueError(
                    f"Duplicate system-analysis result key {key!r}. "
                    "Use unique result_key() values."
                )
            values[key] = task.compute(system=system)
            task_ids[key] = stable_id(
                {
                    "system_id": system_id,
                    "task_spec": task_spec.to_dict(),
                }
            )

        artifact = SystemAnalysisBundle(
            task_ids=task_ids,
            values=values,
            meta={
                "system_id": system_id,
                "analysis_spec": analysis_spec.to_dict(),
            },
        )
        self.save(analysis_id, artifact)
        return analysis_id, artifact
