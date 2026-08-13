from __future__ import annotations

from dataclasses import dataclass

from pgopinf.specs.system.base import SystemSpec
from pgopinf.systems.analysis.artifact import (
    SystemAnalysisBundle,
)
from pgopinf.core.task_store import (
    SystemAnalysisTaskStore,
)


@dataclass
class SystemAnalysisManager:
    """Coordinate cached system-analysis task execution."""

    task_store: SystemAnalysisTaskStore

    def get_or_create(
        self, *, system_spec: SystemSpec, system, analysis_spec
    ) -> SystemAnalysisBundle:
        """Load or compute all tasks in a system-analysis specification.

        Parameters
        ----------
        system_spec : SystemSpec
            Specification of the analyzed system.
        system
            System instance passed to each task.
        analysis_spec
            Specification containing analysis task specs.

        Returns
        -------
        SystemAnalysisBundle
            Bundle of task IDs, values, and metadata.

        Raises
        ------
        ValueError
            If two tasks produce the same result key.
        """
        task_ids: dict[str, str] = {}
        values: dict[str, object] = {}

        for task_spec in analysis_spec.tasks:
            task_id, art = self.task_store.get_or_create(
                system_spec=system_spec,
                system=system,
                task_spec=task_spec,
            )

            key = task_spec.result_key()
            if key in values:
                raise ValueError(
                    f"Duplicate system-analysis result key {key!r}. "
                    "Use unique result_key() values."
                )

            task_ids[key] = task_id
            values[key] = art.value

        return SystemAnalysisBundle(
            task_ids=task_ids,
            values=values,
            meta={
                "system_spec": system_spec.to_dict(),
                "analysis_spec": analysis_spec.to_dict(),
            },
        )
