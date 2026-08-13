from __future__ import annotations

from dataclasses import dataclass, field

from pgopinf.specs.study import StudyAxis, StudySpec, StudyVariant
from pgopinf.study.design import StudyDesigner


@dataclass(frozen=True)
class FakeExperimentSpec:
    name: str
    values: dict = field(default_factory=dict)
    override_history: tuple[dict, ...] = ()

    def with_overrides(self, overrides):
        values = {**self.values, **overrides}
        return FakeExperimentSpec(
            name=values.get("name", self.name),
            values=values,
            override_history=self.override_history + (dict(overrides),),
        )


def test_build_experiments_includes_base_variant_and_axis_grid() -> None:
    base_spec = FakeExperimentSpec(name="base-exp")
    study_spec = StudySpec(
        name="study",
        variants=(StudyVariant(name="petrov", overrides={"reduction.basis": "pg"}),),
        axes=(
            StudyAxis(parameter="reduction.r", values=(2, 4)),
            StudyAxis(parameter="identification.lam", values=(0.1,)),
        ),
        include_base_spec=True,
    )
    designer = StudyDesigner()

    jobs = designer.build_experiments(base_spec=base_spec, study_spec=study_spec)

    assert [variant_name for variant_name, _ in jobs] == ["base", "base", "petrov", "petrov"]
    assert [spec.name for _, spec in jobs] == [
        "base-exp__variant=base__r=2__lam=0.1",
        "base-exp__variant=base__r=4__lam=0.1",
        "base-exp__variant=petrov__r=2__lam=0.1",
        "base-exp__variant=petrov__r=4__lam=0.1",
    ]
    assert jobs[0][1].override_history[0] == {
        "reduction.r": 2,
        "identification.lam": 0.1,
    }
    assert jobs[2][1].override_history[0] == {
        "reduction.basis": "pg",
        "reduction.r": 2,
        "identification.lam": 0.1,
    }


def test_axis_overrides_take_precedence_over_variant_overrides() -> None:
    base_spec = FakeExperimentSpec(name="base-exp")
    study_spec = StudySpec(
        variants=(
            StudyVariant(name="variant", overrides={"reduction.r": 99, "mode": "variant"}),
        ),
        axes=(StudyAxis(parameter="reduction.r", values=(3,)),),
        include_base_spec=False,
    )

    jobs = StudyDesigner().build_experiments(base_spec=base_spec, study_spec=study_spec)

    assert len(jobs) == 1
    assert jobs[0][1].values["reduction.r"] == 3
    assert jobs[0][1].values["mode"] == "variant"


def test_build_experiments_without_axes_builds_one_job_per_variant() -> None:
    base_spec = FakeExperimentSpec(name="base-exp")
    study_spec = StudySpec(
        variants=(
            StudyVariant(name="a", overrides={"mode": "a"}),
            StudyVariant(name="b", overrides={"mode": "b"}),
        ),
        axes=(),
        include_base_spec=False,
    )

    jobs = StudyDesigner().build_experiments(base_spec=base_spec, study_spec=study_spec)

    assert [(variant_name, spec.name) for variant_name, spec in jobs] == [
        ("a", "base-exp__variant=a"),
        ("b", "base-exp__variant=b"),
    ]
    assert [spec.values["mode"] for _, spec in jobs] == ["a", "b"]


def test_custom_base_variant_name_is_used_when_including_base_spec() -> None:
    base_spec = FakeExperimentSpec(name="base-exp")
    study_spec = StudySpec(
        variants=(StudyVariant(name="variant", overrides={}),),
        include_base_spec=True,
    )

    jobs = StudyDesigner(base_variant_name="reference").build_experiments(
        base_spec=base_spec,
        study_spec=study_spec,
    )

    assert [variant_name for variant_name, _ in jobs] == ["reference", "variant"]
    assert jobs[0][1].name == "base-exp__variant=reference"
