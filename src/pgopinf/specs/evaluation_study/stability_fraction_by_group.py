from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from pgopinf.specs.base import SpecBase
from pgopinf.specs.evaluation_study.base import (
    register_study_evaluation,
)
from pgopinf.specs.utils.utils import _as_tuple


@register_study_evaluation("stability_fraction_by_group")
@dataclass(frozen=True)
class StabilityFractionByGroupSpec(SpecBase):
    """
    Aggregate stable/unstable fractions over selected runs, grouped by e.g. variant.

    Example:
        group_by=("variant",)

    Then for each variant, aggregate over all selected runs:
        stable_fraction
        unstable_fraction
    """

    kind: Literal["stability_fraction_by_group"] = "stability_fraction_by_group"

    group_by: tuple[str, ...] = ("variant",)
    filters: dict[str, Any] = field(default_factory=dict)

    include_stable: bool = True
    include_unstable: bool = True

    title: str | None = None
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "group_by", _as_tuple(self.group_by))
        if not self.name:
            object.__setattr__(
                self,
                "name",
                self.kind if self.title is None else self.title,
            )
        object.__setattr__(self, "metric", "spectral_abscissa_cmp")

    def build(self):
        """Build the stability-fraction study evaluator."""
        from pgopinf.study.stability_fraction_by_group_evaluator import (
            StabilityFractionByGroupEvaluator,
        )

        return StabilityFractionByGroupEvaluator(
            metric=self.metric,
            group_by=self.group_by,
            filters=self.filters,
            include_stable=self.include_stable,
            include_unstable=self.include_unstable,
            title=self.title,
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StabilityFractionByGroupSpec":
        """Deserialize a stability-fraction study spec."""
        return cls(
            group_by=_as_tuple(d.get("group_by", ("variant",))),
            filters=dict(d.get("filters", {})),
            include_stable=bool(d.get("include_stable", True)),
            include_unstable=bool(d.get("include_unstable", True)),
            title=d.get("title"),
            name=d.get("name"),
        )
