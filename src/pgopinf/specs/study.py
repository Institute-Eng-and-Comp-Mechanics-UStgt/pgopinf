from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation_study.base import StudyEvaluationSpec


@dataclass(frozen=True)
class StudyVariant(SpecBase):
    """Named set of overrides for a study run family."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyAxis(SpecBase):
    """Parameter axis varied across a study."""

    parameter: str
    values: tuple[Any, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not self.parameter:
            raise ValueError("StudyAxis.parameter must not be empty")
        if len(self.values) == 0:
            raise ValueError("StudyAxis.values must not be empty")


@dataclass(frozen=True)
class StudySpec(SpecBase):
    """Specification for a multi-run study."""

    name: str = "study"
    variants: tuple[StudyVariant, ...] = field(default_factory=tuple)
    axes: tuple[StudyAxis, ...] = field(default_factory=tuple)
    evaluations: tuple[StudyEvaluationSpec, ...] = field(default_factory=tuple)
    include_base_spec: bool = True

    def __post_init__(self):
        if len(self.variants) == 0:
            raise ValueError("StudySpec.variants must not be empty")

    @classmethod
    def preset(cls, name: str, **kwargs) -> StudySpec:
        """Create a study spec from a named preset."""
        from pgopinf.specs.presets.study_presets import study_preset

        study_spec = study_preset(name, **kwargs)

        if study_spec is None:
            raise ValueError(f"Unknown study preset: {name}")
        else:
            return study_spec
