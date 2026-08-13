from __future__ import annotations

import json
from dataclasses import dataclass

from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.study import StudySpec, StudyVariant
from pgopinf.study.manager import StudyManager
from pgopinf.study.result import StudyResult


class FakeStudyRunner:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, *, base_spec, study_spec):
        self.calls.append({"base_spec": base_spec, "study_spec": study_spec})
        return self.result


class FakeReportWriter:
    def __init__(self):
        self.scalar_calls = []
        self.array_calls = []
        self.grouped_calls = []

    def write_scalar_plot(self, **kwargs):
        self.scalar_calls.append(kwargs)

    def write_array_plot(self, **kwargs):
        self.array_calls.append(kwargs)

    def write_grouped_scalar_table(self, **kwargs):
        self.grouped_calls.append(kwargs)


@dataclass(frozen=True)
class FakeEvalSpec:
    kind: str
    name: str
    evaluator: object

    def build(self):
        return self.evaluator


def make_study_spec(*evaluations) -> StudySpec:
    return StudySpec(
        name="study",
        variants=(StudyVariant(name="variant", overrides={}),),
        evaluations=tuple(evaluations),
    )


def test_run_and_evaluate_runs_study_saves_result_and_dispatches_evaluations(tmp_path) -> None:
    base_spec = object()
    study_result = StudyResult(study_name="study")
    study_spec = make_study_spec(
        FakeEvalSpec(kind="scalar_plot", name="scalar", evaluator=object()),
        FakeEvalSpec(kind="array_plot", name="array", evaluator=object()),
        FakeEvalSpec(
            kind="stability_fraction_by_group",
            name="stability",
            evaluator=object(),
        ),
    )
    study_paths = ResultsPath(tmp_path).for_study(study_name="study")
    runner = FakeStudyRunner(result=study_result)
    report_writer = FakeReportWriter()
    manager = StudyManager(
        study_runner=runner,
        report_writer=report_writer,
        study_paths=study_paths,
    )

    result = manager.run_and_evaluate(base_spec=base_spec, study_spec=study_spec)

    assert result is study_result
    assert runner.calls == [{"base_spec": base_spec, "study_spec": study_spec}]
    assert json.loads((study_paths.study_dir / "spec.json").read_text()) == {
        "study_name": "study",
        "run_refs": [],
    }

    assert len(report_writer.scalar_calls) == 1
    assert report_writer.scalar_calls[0] == {
        "evaluator": study_spec.evaluations[0].evaluator,
        "study_result": study_result,
        "study_paths": study_paths,
        "evaluation_name": "scalar",
    }
    assert report_writer.array_calls[0]["evaluator"] is study_spec.evaluations[1].evaluator
    assert report_writer.array_calls[0]["evaluation_name"] == "array"
    assert report_writer.grouped_calls[0]["evaluator"] is study_spec.evaluations[2].evaluator
    assert report_writer.grouped_calls[0]["evaluation_name"] == "stability"


def test_run_and_evaluate_ignores_unknown_evaluation_kind_after_building_it(tmp_path) -> None:
    study_result = StudyResult(study_name="study")
    unknown_eval = FakeEvalSpec(kind="unknown", name="ignored", evaluator=object())
    study_spec = make_study_spec(unknown_eval)
    report_writer = FakeReportWriter()
    manager = StudyManager(
        study_runner=FakeStudyRunner(result=study_result),
        report_writer=report_writer,
        study_paths=ResultsPath(tmp_path).for_study(study_name="study"),
    )

    result = manager.run_and_evaluate(base_spec=object(), study_spec=study_spec)

    assert result is study_result
    assert report_writer.scalar_calls == []
    assert report_writer.array_calls == []
    assert report_writer.grouped_calls == []


def test_save_study_result_spec_writes_sorted_indented_json(tmp_path) -> None:
    study_result = StudyResult(study_name="study")
    study_paths = ResultsPath(tmp_path).for_study(study_name="study")
    manager = StudyManager(
        study_runner=FakeStudyRunner(result=study_result),
        report_writer=FakeReportWriter(),
        study_paths=study_paths,
    )

    manager.save_study_result_spec(study_result)

    text = (study_paths.study_dir / "spec.json").read_text(encoding="utf-8")
    assert text == '{\n  "run_refs": [],\n  "study_name": "study"\n}'
