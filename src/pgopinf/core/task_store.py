from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib as mpl

from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.base import dataclass_to_dict, stable_id
from pgopinf.specs.system.base import SystemSpec
from pgopinf.specs.system_analysis.base import SystemAnalysisTaskSpec
from pgopinf.systems.analysis.artifact import SystemTaskArtifact
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem

logger = logging.getLogger(__name__)


def _spec_to_dict(spec):
    return spec.to_dict() if hasattr(spec, "to_dict") else dataclass_to_dict(spec)


class SystemAnalysisTaskStore:
    """Cache individual system-analysis task artifacts."""

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the task store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager that provides the task-cache root.
        force_recompute : bool, optional
            If ``True``, recompute tasks even when cached results exist.
        """
        self.root = paths.system_analysis_tasks
        self.force_recompute = force_recompute

    def compute_id(
        self, *, system_spec: SystemSpec, task_spec: SystemAnalysisTaskSpec
    ) -> str:
        """Compute the stable identifier for a system-analysis task.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the analyzed system.
        task_spec : SystemAnalysisTaskSpec
            Specification of the analysis task.

        Returns
        -------
        str
            Stable task identifier.
        """
        key = {
            "system_id": _spec_to_dict(system_spec),
            "task_spec": task_spec.to_dict(),
        }
        return stable_id(key)

    def _dir(self, system_spec: SystemSpec, task_spec: SystemAnalysisTaskSpec) -> Path:
        task_id = self.compute_id(system_spec=system_spec, task_spec=task_spec)
        return self.root / f"{system_spec.kind}" / f"{task_spec.kind}_{task_id}"

    def exists(
        self, system_spec: SystemSpec, task_spec: SystemAnalysisTaskSpec
    ) -> bool:
        """Return whether a task artifact exists.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the analyzed system.
        task_spec : SystemAnalysisTaskSpec
            Specification of the analysis task.

        Returns
        -------
        bool
            ``True`` if the task metadata and result storage are present.
        """
        d = self._dir(system_spec, task_spec)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("storage_mode") == "custom":
            return True
        return (d / "value.pkl").exists()

    def save(
        self,
        system_spec: SystemSpec,
        task_spec: SystemAnalysisTaskSpec,
        task: Any,
        artifact: SystemTaskArtifact,
    ) -> None:
        """Persist a system-analysis task artifact.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the analyzed system.
        task_spec : SystemAnalysisTaskSpec
            Specification of the analysis task.
        task : Any
            Task implementation. If it provides custom result I/O methods,
            those methods are used instead of pickle.
        artifact : SystemTaskArtifact
            Computed task result and metadata.
        """
        d = self._dir(system_spec, task_spec)
        d.mkdir(parents=True, exist_ok=True)

        meta = {
            "task_key": artifact.task_key,
            "task_kind": artifact.task_kind,
            "meta": artifact.meta,
            "storage_mode": (
                "custom"
                if hasattr(task, "save_result") and hasattr(task, "load_result")
                else "pickle"
            ),
        }
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        if hasattr(task, "save_result") and hasattr(task, "load_result"):
            task.save_result(task_dir=d, value=artifact.value)
        else:
            with open(d / "value.pkl", "wb") as f:
                pickle.dump(artifact.value, f)

    def load(
        self, system_spec: SystemSpec, task_spec: SystemAnalysisTaskSpec, task: Any
    ) -> SystemTaskArtifact:
        """Load a cached system-analysis task artifact.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the analyzed system.
        task_spec : SystemAnalysisTaskSpec
            Specification of the analysis task.
        task : Any
            Task implementation used for custom result loading when available.

        Returns
        -------
        SystemTaskArtifact
            Stored task artifact.
        """
        d = self._dir(system_spec, task_spec)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta.get("storage_mode") == "custom":
            value = task.load_result(task_dir=d)
        else:
            with open(d / "value.pkl", "rb") as f:
                value = pickle.load(f)

        return SystemTaskArtifact(
            task_key=meta["task_key"],
            task_kind=meta["task_kind"],
            value=value,
            meta=meta.get("meta", {}),
        )

    def get_or_create(
        self,
        *,
        system_spec: SystemSpec,
        system: PHSystem | LTISystem,
        task_spec: SystemAnalysisTaskSpec,
    ):
        """Load a task artifact or compute and cache it.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the analyzed system.
        system : PHSystem or LTISystem
            System instance passed to the task.
        task_spec : SystemAnalysisTaskSpec
            Specification that builds the task.

        Returns
        -------
        tuple[str, SystemTaskArtifact]
            Task identifier and artifact.
        """
        task_id = self.compute_id(system_spec=system_spec, task_spec=task_spec)
        task = task_spec.build()

        if self.exists(system_spec, task_spec) and not self.force_recompute:
            logger.info(
                f"System analysis task {task_id} - {task_spec.kind} already exists. Loading from disk..."
            )
            return task_id, self.load(system_spec, task_spec, task=task)

        logger.info(f"Computing system analysis task {task_id} - {task_spec.kind}...")
        value = task.compute(system=system)

        artifact = SystemTaskArtifact(
            task_key=task_spec.result_key(),
            task_kind=task_spec.kind,
            value=value,
            meta={
                "system_spec": _spec_to_dict(system_spec),
                "task_spec": task_spec.to_dict(),
            },
        )
        self.save(system_spec, task_spec, task=task, artifact=artifact)
        return task_id, artifact
