from __future__ import annotations

from dataclasses import dataclass

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.evaluator import MetricEvaluator
from pgopinf.evaluation.results import MetricOutput
from pgopinf.specs.evaluation.evaluation import EvaluationSpec


class FakeMetric:
    def __init__(self, name: str, calls: list[tuple[str, EvaluationContext]]):
        self.name = name
        self.calls = calls

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        self.calls.append((self.name, ctx))
        return MetricOutput(
            summaries=[
                {"metric": self.name, "value": 1.0},
                {"metric": f"{self.name}_detail", "run_id": "metric-run"},
            ],
            artifacts=[{"artifact": self.name}],
        )


@dataclass(frozen=True)
class FakeMetricSpec:
    name: str
    calls: list[tuple[str, EvaluationContext]]
    built: list[str]
    kind: str = "fake"

    def build(self) -> FakeMetric:
        self.built.append(self.name)
        return FakeMetric(self.name, self.calls)

    def to_dict(self):
        return {"kind": self.kind, "name": self.name}


def make_context() -> EvaluationContext:
    return EvaluationContext(
        experiment_name="experiment-a",
        run_id="run-a",
        original_system=object(),
        dataset_artifact=object(),
        reduction_result=object(),
        dataset_artifact_intrusive=None,
        identification_result=object(),
        dataset_artifact_identified=None,
    )


def test_metric_evaluator_builds_metrics_and_enriches_summary_rows() -> None:
    calls: list[tuple[str, EvaluationContext]] = []
    built: list[str] = []
    ctx = make_context()
    spec = EvaluationSpec(
        metrics=(
            FakeMetricSpec(name="m1", calls=calls, built=built),
            FakeMetricSpec(name="m2", calls=calls, built=built),
        )
    )

    outputs = MetricEvaluator().evaluate(eval_ctx=ctx, evaluation_spec=spec)

    assert built == ["m1", "m2"]
    assert calls == [("m1", ctx), ("m2", ctx)]
    assert len(outputs) == 2
    assert outputs[0].summaries == [
        {
            "metric": "m1",
            "value": 1.0,
            "experiment_name": "experiment-a",
            "run_id": "run-a",
        },
        {
            "metric": "m1_detail",
            "experiment_name": "experiment-a",
            "run_id": "run-a",
        },
    ]
    assert outputs[1].summaries[0]["metric"] == "m2"
    assert outputs[0].artifacts == [{"artifact": "m1"}]
    assert outputs[1].artifacts == [{"artifact": "m2"}]


def test_metric_evaluator_returns_empty_outputs_when_no_metrics() -> None:
    outputs = MetricEvaluator().evaluate(
        eval_ctx=make_context(),
        evaluation_spec=EvaluationSpec(),
    )

    assert outputs == []
