from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from pgopinf.io.results_path import ResultsPath
from pgopinf.study.result import StudyResult, StudyRunRef
from pgopinf.study.stability_fraction_by_group_evaluator import (
    StabilityFractionByGroupEvaluator,
    _expand_metadata_json,
    _get_group_value,
)


@dataclass(frozen=True)
class FakeSpec:
    name: str

    def to_dict(self):
        return {"name": self.name}


def write_scalar_metrics(paths: ResultsPath, spec: FakeSpec, ident_id: str, rows) -> None:
    csv_path = paths.for_run(
        experiment_name=spec.name,
        ident_id=ident_id,
    ).scalar_metrics_csv
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def test_expand_metadata_json_adds_missing_columns_and_ignores_bad_json() -> None:
    df = pd.DataFrame(
        {
            "metric": ["stability", "stability", "stability"],
            "split": ["existing", "existing", "existing"],
            "metadata_json": [
                '{"split": "TRAIN", "source": "identified"}',
                '{"source": "intrusive"}',
                "bad json",
            ],
        }
    )

    expanded = _expand_metadata_json(df.copy())

    assert expanded["split"].tolist() == ["existing", "existing", "existing"]
    assert expanded["source"].iloc[:2].tolist() == ["identified", "intrusive"]
    assert pd.isna(expanded["source"].iloc[2])


def test_get_group_value_supports_variant_and_row_columns() -> None:
    run_ref = StudyRunRef(
        ident_id="run-id",
        variant_name="base",
        spec=FakeSpec(name="exp"),
    )
    row = pd.Series({"split": "TRAIN"})

    assert _get_group_value(run_ref, row, "variant") == "base"
    assert _get_group_value(run_ref, row, "split") == "TRAIN"

    with pytest.raises(ValueError, match="group_by key 'missing' not available"):
        _get_group_value(run_ref, row, "missing")


def test_evaluate_groups_stability_fractions_by_variant_and_metadata(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec_a = FakeSpec(name="exp-a")
    spec_b = FakeSpec(name="exp-b")
    write_scalar_metrics(
        paths,
        spec_a,
        "id-a",
        [
            {
                "metric": "spectral_abscissa",
                "value": -0.1,
                "metadata_json": '{"split": "TRAIN", "source": "identified"}',
            },
            {
                "metric": "spectral_abscissa",
                "value": 0.2,
                "metadata_json": '{"split": "TRAIN", "source": "identified"}',
            },
            {
                "metric": "spectral_abscissa",
                "value": -0.3,
                "metadata_json": '{"split": "TEST", "source": "identified"}',
            },
            {
                "metric": "other",
                "value": -9.0,
                "metadata_json": '{"split": "TRAIN", "source": "identified"}',
            },
        ],
    )
    write_scalar_metrics(
        paths,
        spec_b,
        "id-b",
        [
            {
                "metric": "spectral_abscissa",
                "value": -0.4,
                "metadata_json": '{"split": "TRAIN", "source": "intrusive"}',
            },
            {
                "metric": "spectral_abscissa",
                "value": -0.5,
                "metadata_json": '{"split": "TRAIN", "source": "intrusive"}',
            },
        ],
    )
    study_result = StudyResult(
        study_name="study",
        run_refs=[
            StudyRunRef(ident_id="id-a", variant_name="base", spec=spec_a),
            StudyRunRef(ident_id="id-b", variant_name="petrov", spec=spec_b),
        ],
    )
    evaluator = StabilityFractionByGroupEvaluator(
        metric="spectral_abscissa",
        group_by=("variant", "source"),
        filters={"split": "TRAIN"},
    )

    df = evaluator.evaluate(study_result=study_result, paths=paths)

    assert df.to_dict("records") == [
        {
            "variant": "base",
            "source": "identified",
            "fraction_type": "stable",
            "count": 1,
            "total": 2,
            "fraction": 0.5,
            "percentage": 50.0,
        },
        {
            "variant": "base",
            "source": "identified",
            "fraction_type": "unstable",
            "count": 1,
            "total": 2,
            "fraction": 0.5,
            "percentage": 50.0,
        },
        {
            "variant": "petrov",
            "source": "intrusive",
            "fraction_type": "stable",
            "count": 2,
            "total": 2,
            "fraction": 1.0,
            "percentage": 100.0,
        },
        {
            "variant": "petrov",
            "source": "intrusive",
            "fraction_type": "unstable",
            "count": 0,
            "total": 2,
            "fraction": 0.0,
            "percentage": 0.0,
        },
    ]


def test_evaluate_can_include_only_stable_or_unstable_fractions(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp")
    write_scalar_metrics(
        paths,
        spec,
        "run-id",
        [
            {"metric": "stability", "value": -1.0, "split": "TRAIN"},
            {"metric": "stability", "value": 1.0, "split": "TRAIN"},
        ],
    )
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="run-id", variant_name="base", spec=spec)],
    )

    stable_only = StabilityFractionByGroupEvaluator(
        metric="stability",
        group_by=("variant",),
        filters={"split": "TRAIN"},
        include_stable=True,
        include_unstable=False,
    ).evaluate(study_result=study_result, paths=paths)
    unstable_only = StabilityFractionByGroupEvaluator(
        metric="stability",
        group_by=("variant",),
        filters={"split": "TRAIN"},
        include_stable=False,
        include_unstable=True,
    ).evaluate(study_result=study_result, paths=paths)

    assert stable_only["fraction_type"].tolist() == ["stable"]
    assert unstable_only["fraction_type"].tolist() == ["unstable"]


def test_evaluate_returns_empty_table_for_missing_or_empty_inputs(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    empty_spec = FakeSpec(name="empty")
    empty_csv = paths.for_run(experiment_name=empty_spec.name, ident_id="empty-id").scalar_metrics_csv
    pd.DataFrame(columns=["metric", "value"]).to_csv(empty_csv, index=False)
    study_result = StudyResult(
        study_name="study",
        run_refs=[
            StudyRunRef(ident_id="missing-id", variant_name="base", spec=FakeSpec(name="missing")),
            StudyRunRef(ident_id="empty-id", variant_name="base", spec=empty_spec),
        ],
    )
    evaluator = StabilityFractionByGroupEvaluator(
        metric="stability",
        group_by=("variant", "split"),
        filters={},
    )

    df = evaluator.evaluate(study_result=study_result, paths=paths)

    assert df.empty
    assert list(df.columns) == [
        "variant",
        "split",
        "fraction_type",
        "count",
        "total",
        "fraction",
        "percentage",
    ]


def test_evaluate_raises_for_missing_filter_column(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp")
    write_scalar_metrics(paths, spec, "run-id", [{"metric": "stability", "value": -1.0}])
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="run-id", variant_name="base", spec=spec)],
    )
    evaluator = StabilityFractionByGroupEvaluator(
        metric="stability",
        group_by=("variant",),
        filters={"split": "TRAIN"},
    )

    with pytest.raises(ValueError, match="Filter key 'split' not found"):
        evaluator.evaluate(study_result=study_result, paths=paths)


def test_evaluate_raises_for_missing_group_by_column(tmp_path) -> None:
    paths = ResultsPath(tmp_path)
    spec = FakeSpec(name="exp")
    write_scalar_metrics(paths, spec, "run-id", [{"metric": "stability", "value": -1.0}])
    study_result = StudyResult(
        study_name="study",
        run_refs=[StudyRunRef(ident_id="run-id", variant_name="base", spec=spec)],
    )
    evaluator = StabilityFractionByGroupEvaluator(
        metric="stability",
        group_by=("source",),
        filters={},
    )

    with pytest.raises(ValueError, match="group_by key 'source' not available"):
        evaluator.evaluate(study_result=study_result, paths=paths)
