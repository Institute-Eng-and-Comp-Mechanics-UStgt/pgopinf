from __future__ import annotations

from dataclasses import dataclass

from pgopinf.specs.study import StudySpec, StudyVariant
from pgopinf.study.result import StudyResult
from pgopinf.study.runner import StudyRunner


@dataclass(frozen=True)
class FakeExperimentSpec:
    name: str

    def to_dict(self):
        return {"name": self.name}


class FakeDesigner:
    def __init__(self, jobs):
        self.jobs = jobs
        self.calls = []

    def build_experiments(self, *, base_spec, study_spec):
        self.calls.append({"base_spec": base_spec, "study_spec": study_spec})
        return self.jobs


class FakeExperimentRunner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def run(self, spec):
        self.calls.append(spec)
        return self.outputs.pop(0)


def make_study_spec(name: str = "study") -> StudySpec:
    return StudySpec(
        name=name,
        variants=(StudyVariant(name="variant", overrides={}),),
    )


def test_study_runner_builds_jobs_and_runs_each_experiment_in_order() -> None:
    base_spec = FakeExperimentSpec(name="base")
    study_spec = make_study_spec(name="sweep")
    spec_a = FakeExperimentSpec(name="variant-a")
    spec_b = FakeExperimentSpec(name="variant-b")
    run_paths_a = object()
    designer = FakeDesigner(jobs=[("base", spec_a), ("petrov", spec_b)])
    experiment_runner = FakeExperimentRunner(
        outputs=[
            {"run_id": "id-a", "run_paths": run_paths_a},
            {"run_id": "id-b"},
        ]
    )
    runner = StudyRunner(
        experiment_runner=experiment_runner,
        designer=designer,
    )

    result = runner.run(base_spec=base_spec, study_spec=study_spec)

    assert isinstance(result, StudyResult)
    assert result.study_name == "sweep"
    assert designer.calls == [{"base_spec": base_spec, "study_spec": study_spec}]
    assert experiment_runner.calls == [spec_a, spec_b]

    assert [run_ref.ident_id for run_ref in result.run_refs] == ["id-a", "id-b"]
    assert [run_ref.variant_name for run_ref in result.run_refs] == ["base", "petrov"]
    assert [run_ref.spec for run_ref in result.run_refs] == [spec_a, spec_b]
    assert result.run_refs[0].run_paths is run_paths_a
    assert result.run_refs[1].run_paths is None


def test_study_runner_returns_empty_result_when_designer_has_no_jobs() -> None:
    base_spec = FakeExperimentSpec(name="base")
    study_spec = make_study_spec(name="empty")
    designer = FakeDesigner(jobs=[])
    experiment_runner = FakeExperimentRunner(outputs=[])
    runner = StudyRunner(
        experiment_runner=experiment_runner,
        designer=designer,
    )

    result = runner.run(base_spec=base_spec, study_spec=study_spec)

    assert result == StudyResult(study_name="empty")
    assert designer.calls == [{"base_spec": base_spec, "study_spec": study_spec}]
    assert experiment_runner.calls == []
