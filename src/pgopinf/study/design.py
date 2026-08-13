from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.study import StudyVariant


@dataclass
class StudyDesigner:
    """Expand a study specification into concrete experiment specifications."""

    base_variant_name: str = "base"

    def build_experiments(
        self,
        *,
        base_spec: ExperimentSpec,
        study_spec,
    ) -> list[tuple[str, ExperimentSpec]]:
        """Build all experiment specs implied by a study.

        Parameters
        ----------
        base_spec : ExperimentSpec
            Base experiment specification to override.
        study_spec
            Study specification containing variants and axes.

        Returns
        -------
        list of tuple[str, ExperimentSpec]
            Variant name and concrete experiment specification for each job.
        """
        axis_names = [axis.parameter for axis in study_spec.axes]
        axis_values = [axis.values for axis in study_spec.axes]

        if len(axis_values) == 0:
            axis_combos = [()]
        else:
            axis_combos = list(product(*axis_values))

        variants = list(study_spec.variants)
        if study_spec.include_base_spec:
            variants = [
                StudyVariant(name=self.base_variant_name, overrides={})
            ] + variants

        out: list[tuple[str, ExperimentSpec]] = []

        for variant in variants:
            for combo in axis_combos:
                axis_overrides = {p: v for p, v in zip(axis_names, combo)}
                all_overrides = {**variant.overrides, **axis_overrides}

                spec_i = base_spec.with_overrides(all_overrides)

                suffix_parts = [f"variant={variant.name}"]
                suffix_parts += [
                    f"{p.split('.')[-1]}={v}" for p, v in axis_overrides.items()
                ]
                suffix = "__".join(suffix_parts)

                spec_i = spec_i.with_overrides({"name": f"{base_spec.name}__{suffix}"})

                out.append((variant.name, spec_i))

        return out

    def _make_base_variant(self):
        from pgopinf.specs.study import StudyVariant

        return StudyVariant(name=self.base_variant_name, overrides={})
