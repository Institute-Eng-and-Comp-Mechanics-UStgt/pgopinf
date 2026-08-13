from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.identification.base import (
    IdentificationSpec,
    identification_default_by_kind,
    identification_from_dict,
)
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)
from pgopinf.specs.overrides import validate_override_paths
from pgopinf.specs.system_analysis.base import (
    system_analysis_task_default_by_kind,
)
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)
from pgopinf.specs.system_analysis.tasks import EigenvaluesSpec


from .base import (
    apply_overrides_dict,
    dataclass_to_dict,
    stable_id,
    sanitize_name,
)
from .data.base import DataSpec
from .reduction.base import ReductionSpec
from .reduction.full_basis import full_basis_default_by_kind
from .reduction.mor import mor_default_by_kind
from .reduction.test_basis import test_basis_default_by_kind
from .system.base import (
    SystemSpec,
    system_default_by_kind,
    system_from_dict,
)


def _set_dotted(d: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def _reset_switched_kinds(
    *,
    spec_cls,
    d: dict[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """
    If a '.kind' override changes a polymorphic subtree, replace that subtree by
    the defaults of the new kind before applying the remaining overrides.

    Example:
        identification.kind = "opinf"

    should reset the whole 'identification' subtree to OperatorInferenceSpec()
    before applying e.g.
        identification.regularization = 1e-8
    """
    d = copy.deepcopy(d)

    if not hasattr(spec_cls, "polymorphic_subtrees"):
        return d

    subtree_factories = spec_cls.polymorphic_subtrees()

    for subtree_path, default_by_kind in subtree_factories.items():
        kind_override_key = f"{subtree_path}.kind"
        if kind_override_key in overrides:
            new_kind = str(overrides[kind_override_key])
            fresh_spec = default_by_kind(new_kind)
            fresh_spec_dict = (
                fresh_spec.to_dict()
                if hasattr(fresh_spec, "to_dict")
                else dataclass_to_dict(fresh_spec)
            )
            _set_dotted(d, subtree_path, fresh_spec_dict)

    return d


@dataclass(frozen=True)
class ExperimentSpec:
    """
    Top-level spec tying everything together.
    """

    name: str
    system: SystemSpec
    data: DataSpec
    reduction: ReductionSpec
    identification: IdentificationSpec
    evaluation: EvaluationSpec | None = None

    system_analysis: SystemAnalysisSpec | None = None

    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment specification."""
        d = dataclass_to_dict(self)
        d["system"] = dataclass_to_dict(self.system)
        d["data"] = self.data.to_dict()
        d["reduction"] = self.reduction.to_dict()
        d["identification"] = self.identification.to_dict()
        if self.evaluation is not None:
            d["evaluation"] = self.evaluation.to_dict()
        else:
            d.pop("evaluation", None)
        if self.system_analysis is not None:
            d["system_analysis"] = self.system_analysis.to_dict()
        else:
            d.pop("system_analysis", None)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExperimentSpec":
        """Deserialize an experiment specification."""
        for key in ("name", "seed"):
            if key not in d:
                raise KeyError(f"ExperimentSpec.from_dict: missing key '{key}'")
        for key in ("system", "data", "reduction", "identification"):
            if key not in d:
                raise KeyError(f"ExperimentSpec.from_dict: missing key '{key}'")

        system = system_from_dict(d["system"])
        data = DataSpec.from_dict(d["data"])
        reduction = ReductionSpec.from_dict(d["reduction"])
        identification = identification_from_dict(d["identification"])
        evaluation = (
            EvaluationSpec.from_dict(d["evaluation"]) if "evaluation" in d else None
        )
        system_analysis = (
            SystemAnalysisSpec.from_dict(d["system_analysis"])
            if "system_analysis" in d
            else None
        )
        name = str(d.get("name", "unnamed"))
        seed = int(d.get("seed", 0))

        return cls(
            name=name,
            system=system,
            system_analysis=system_analysis,
            data=data,
            reduction=reduction,
            identification=identification,
            evaluation=evaluation,
            seed=seed,
        )

    @classmethod
    def polymorphic_subtrees(cls) -> dict[str, Any]:
        """
        Polymorphic subtrees whose `.kind` can switch families.
        The override engine uses these factories to reset subtrees when kind changes.
        """
        return {
            "system": system_default_by_kind,
            "identification": identification_default_by_kind,
            "reduction.full_basis": full_basis_default_by_kind,
            "reduction.test_basis": test_basis_default_by_kind,
            "reduction.mor": mor_default_by_kind,
            "system_analysis": system_analysis_task_default_by_kind,
        }

    @classmethod
    def default(
        cls,
        system: str,
        name: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "ExperimentSpec":
        """Create a default experiment for a registered system kind."""
        sys_spec = system_default_by_kind(system)
        data_spec = _default_data_spec()
        reduction_spec = _default_reduction_spec()
        identification_spec = OperatorInferenceSpec()
        evaluation_spec = _default_evaluation_spec()
        system_analysis_spec = _default_system_analysis_spec()
        exp = cls(
            name=name or f"{system}_default",
            system=sys_spec,
            data=data_spec,
            reduction=reduction_spec,
            identification=identification_spec,
            evaluation=evaluation_spec,
            system_analysis=system_analysis_spec,
            seed=0,
        )
        if not overrides:
            return exp

        dd = cls.perform_overrides(exp, overrides)
        return cls.from_dict(dd)

    @classmethod
    def preset(
        cls, name: str, *, overrides: Mapping[str, Any] | None = None, **kwargs
    ) -> "ExperimentSpec":
        """Create an experiment from a named preset."""
        from pgopinf.specs.presets.experiment_presets import (
            experiment_preset,
        )

        exp = experiment_preset(name, **kwargs)

        if not overrides:
            return exp

        dd = cls.perform_overrides(exp, overrides)
        return cls.from_dict(dd)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "ExperimentSpec":
        """Return a copy with dotted-path overrides applied."""
        dd = self.perform_overrides(self, overrides)
        return ExperimentSpec.from_dict(dd)

    @staticmethod
    def perform_overrides(
        exp: ExperimentSpec,
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Return a dict representation of a new ExperimentSpec with dotted-path
        overrides applied.

        Handles kind switches by resetting polymorphic subtrees to the defaults
        of the new kind before applying all overrides.
        """
        validate_override_paths(exp, overrides)

        d = exp.to_dict()

        d = _reset_switched_kinds(
            spec_cls=ExperimentSpec,
            d=d,
            overrides=overrides,
        )

        dd = apply_overrides_dict(d, overrides)
        return dd

    @property
    def spec_id(self) -> str:
        """Stable content identifier for this experiment spec."""
        return stable_id(self.to_dict())

    @property
    def safe_name(self) -> str:
        """Filesystem-safe experiment name."""
        return sanitize_name(self.name)

    def save(self, paths: Path, spec_ids: dict[str, str] | None = None):
        """Write the experiment spec and optional artifact IDs to disk."""
        d = paths
        d.mkdir(parents=True, exist_ok=True)

        (d / "spec.json").write_text(
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if spec_ids is not None:
            (d / "spec_ids.json").write_text(
                json.dumps(
                    spec_ids,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )


def _default_data_spec() -> DataSpec:
    return DataSpec()


def _default_reduction_spec() -> ReductionSpec:
    return ReductionSpec()


def _default_evaluation_spec() -> EvaluationSpec:
    import pgopinf.specs.evaluation  # noqa: F401

    return EvaluationSpec.from_dict(
        {
            "metrics": [
                {
                    "kind": "state_values_over_time",
                    "split": "TEST",
                    "target": "identified",
                    "relative": True,
                },
                {
                    "kind": "state_values_over_time",
                    "split": "TEST",
                    "target": "intrusive",
                    "relative": True,
                },
            ]
        }
    )


def _default_system_analysis_spec() -> SystemAnalysisSpec:
    return SystemAnalysisSpec()
