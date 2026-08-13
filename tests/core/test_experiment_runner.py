from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from pgopinf.core.experiment_runner import ExperimentRunner, RunnerPaths
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.data.base import DataSpec
from pgopinf.specs.evaluation.evaluation import EvaluationSpec
from pgopinf.specs.experiment import ExperimentSpec
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.specs.system_analysis.system_analysis import (
    SystemAnalysisSpec,
)
from pgopinf.specs.system_analysis.tasks import EigenvaluesSpec


@dataclass(frozen=True)
class FakeSystemSpec:
    kind: str = "fake_system"
    name: str = "fake-system"

    def build(self):
        return SimpleNamespace(kind="runtime-system")


def make_spec(*, system_analysis=None) -> ExperimentSpec:
    return ExperimentSpec(
        name="runner test",
        system=FakeSystemSpec(),
        data=DataSpec(),
        reduction=ReductionSpec(r=2),
        identification=OperatorInferenceSpec(lambda_reg=1e-3),
        evaluation=EvaluationSpec(),
        system_analysis=system_analysis,
        seed=0,
    )


def make_dataset_artifact(prefix: str):
    return SimpleNamespace(
        train=SimpleNamespace(id=f"{prefix}-train"),
        test=SimpleNamespace(id=f"{prefix}-test"),
        train_id=f"{prefix}-train",
        test_id=f"{prefix}-test",
        data=SimpleNamespace(TRAIN=f"{prefix}-TRAIN", TEST=f"{prefix}-TEST"),
    )


class FakeDataStore:
    def __init__(self, label: str):
        self.label = label
        self.calls = []
        self.artifact = make_dataset_artifact(label)

    def get_or_create(self, **kwargs):
        self.calls.append(kwargs)
        return self.artifact


class FakeReductionStore:
    def __init__(self):
        self.get_calls = []
        self.compute_calls = []
        self.result = SimpleNamespace(
            basis="basis",
            mor="mor",
            reduced_dataset_projected="projected-dataset",
            reduced_system="reduced-system",
        )

    def get_or_create(self, **kwargs):
        self.get_calls.append(kwargs)
        return self.result

    def compute_id(self, **kwargs):
        self.compute_calls.append(kwargs)
        return "reduction-id"


class FakeIdentificationStore:
    def __init__(self):
        self.calls = []
        self.result = SimpleNamespace(
            system_identified="identified-system",
            diagnostics={"ok": True},
            meta={"kind": "fake"},
        )

    def get_or_create(self, **kwargs):
        self.calls.append(kwargs)
        return "ident-id", self.result


class FakeSystemAnalysisManager:
    def __init__(self):
        self.calls = []
        self.bundle = SimpleNamespace(values={"eigenvalues": [1.0]})

    def get_or_create(self, **kwargs):
        self.calls.append(kwargs)
        return self.bundle


class FakeMetricEvaluator:
    def __init__(self):
        self.calls = []
        self.outputs = ["metric-output"]

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs


class FakeReportWriter:
    def __init__(self):
        self.calls = []

    def write(self, **kwargs):
        self.calls.append(kwargs)


def make_runner(tmp_path) -> ExperimentRunner:
    runner = ExperimentRunner(
        paths=ResultsPath(tmp_path / "results"),
        force_recompute=True,
    )
    runner.data_store = FakeDataStore("full")
    runner.intrusive_data_store = FakeDataStore("intrusive")
    runner.identified_data_store = FakeDataStore("identified")
    runner.reduction_store = FakeReductionStore()
    runner.identification_store = FakeIdentificationStore()
    runner.system_analysis_manager = FakeSystemAnalysisManager()
    runner.metric_evaluator = FakeMetricEvaluator()
    runner.report_writer = FakeReportWriter()
    return runner


def test_runner_paths_properties(tmp_path) -> None:
    paths = RunnerPaths(results_root=tmp_path / "results")

    assert paths.datasets == tmp_path / "results" / "datasets"
    assert paths.reductions == tmp_path / "results" / "reductions"
    assert paths.runs == tmp_path / "results" / "runs"


def test_experiment_runner_initializes_stores_with_shared_paths_and_force_flag(
    tmp_path,
) -> None:
    paths = ResultsPath(tmp_path / "results")
    runner = ExperimentRunner(paths=paths, force_recompute=True)

    assert runner.paths is paths
    assert runner.force_recompute is True
    assert runner.data_store.paths is paths
    assert runner.data_store.force_recompute is True
    assert runner.intrusive_data_store.paths is paths
    assert runner.intrusive_data_store.force_recompute is True
    assert runner.identification_store.paths is paths
    assert runner.identification_store.force_recompute is True
    assert runner.identified_data_store.paths is paths
    assert runner.identified_data_store.force_recompute is True


def test_experiment_runner_run_orchestrates_pipeline_and_saves_run_metadata(
    tmp_path,
) -> None:
    runner = make_runner(tmp_path)
    spec = make_spec(system_analysis=SystemAnalysisSpec(tasks=(EigenvaluesSpec(),)))

    result = runner.run(spec)

    assert result["run_id"] == "ident-id"
    assert result["spec"] is spec
    assert result["metric_outputs"] == ["metric-output"]
    assert result["run_paths"].ident_id == "ident-id"

    original_system = runner.data_store.calls[0]["system"]
    assert original_system.kind == "runtime-system"
    assert runner.system_analysis_manager.calls == [
        {
            "system_spec": spec.system,
            "system": original_system,
            "analysis_spec": spec.system_analysis,
        }
    ]
    assert runner.reduction_store.get_calls == [
        {
            "dataset_artifact": runner.data_store.artifact,
            "reduction_spec": spec.reduction,
            "system": original_system,
        }
    ]
    assert runner.intrusive_data_store.calls[0]["reduction_result"] is (
        runner.reduction_store.result
    )
    assert runner.reduction_store.compute_calls == [
        {
            "data_train_id": "full-train",
            "reduction_spec": spec.reduction,
        }
    ]
    assert runner.identification_store.calls == [
        {
            "reduction_id": "reduction-id",
            "reduction_result": runner.reduction_store.result,
            "identification_spec": spec.identification,
            "original_system": original_system,
        }
    ]
    assert runner.identified_data_store.calls[0]["identification_result"] is (
        runner.identification_store.result
    )

    metric_call = runner.metric_evaluator.calls[0]
    eval_ctx = metric_call["eval_ctx"]
    assert metric_call["evaluation_spec"] is spec.evaluation
    assert eval_ctx.experiment_name == spec.name
    assert eval_ctx.run_id == "ident-id"
    assert eval_ctx.original_system is original_system
    assert eval_ctx.system_analysis is runner.system_analysis_manager.bundle
    assert eval_ctx.dataset_artifact is runner.data_store.artifact
    assert eval_ctx.reduction_result is runner.reduction_store.result
    assert eval_ctx.dataset_artifact_intrusive is runner.intrusive_data_store.artifact
    assert eval_ctx.identification_result is runner.identification_store.result
    assert eval_ctx.dataset_artifact_identified is runner.identified_data_store.artifact

    assert runner.report_writer.calls == [
        {
            "run_paths": result["run_paths"],
            "metric_outputs": ["metric-output"],
        }
    ]

    spec_ids = json.loads((result["run_paths"].run_dir / "spec_ids.json").read_text())
    assert spec_ids == {
        "dataset_train_id": "full-train",
        "dataset_test_id": "full-test",
        "dataset_intrusive_train_id": "intrusive-train",
        "dataset_intrusive_test_id": "intrusive-test",
        "dataset_identified_train_id": "identified-train",
        "dataset_identified_test_id": "identified-test",
        "reduction_id": "reduction-id",
        "identification_id": "ident-id",
    }
    saved_spec = json.loads((result["run_paths"].run_dir / "spec.json").read_text())
    assert saved_spec["name"] == "runner test"
    assert saved_spec["system"]["kind"] == "fake_system"


def test_experiment_runner_skips_system_analysis_when_not_requested(tmp_path) -> None:
    runner = make_runner(tmp_path)
    spec = make_spec(system_analysis=None)

    result = runner.run(spec)

    assert runner.system_analysis_manager.calls == []
    assert runner.metric_evaluator.calls[0]["eval_ctx"].system_analysis is None
    assert result["run_id"] == "ident-id"


def test_experiment_runner_skips_empty_system_analysis_spec(tmp_path) -> None:
    runner = make_runner(tmp_path)
    spec = make_spec(system_analysis=SystemAnalysisSpec())

    runner.run(spec)

    assert runner.system_analysis_manager.calls == []
    assert runner.metric_evaluator.calls[0]["eval_ctx"].system_analysis is None
