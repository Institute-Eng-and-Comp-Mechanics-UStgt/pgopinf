from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pgopinf.study.result import StudyResult, StudyRunRef


class FakeSpec:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return dict(self.payload)


def test_study_run_ref_is_frozen_runtime_reference() -> None:
    spec = FakeSpec({"name": "experiment"})
    run_paths = object()

    run_ref = StudyRunRef(
        ident_id="run-1",
        spec=spec,
        variant_name="baseline",
        run_paths=run_paths,
    )

    assert run_ref.ident_id == "run-1"
    assert run_ref.spec is spec
    assert run_ref.variant_name == "baseline"
    assert run_ref.run_paths is run_paths

    with pytest.raises(FrozenInstanceError):
        run_ref.ident_id = "other"  # type: ignore[misc]


def test_study_result_defaults_to_empty_run_refs() -> None:
    result = StudyResult(study_name="empty-study")

    assert result.run_refs == []
    assert result.to_dict() == {"study_name": "empty-study", "run_refs": []}


def test_study_result_to_dict_serializes_run_refs_with_specs() -> None:
    result = StudyResult(
        study_name="sweep",
        run_refs=[
            StudyRunRef(
                ident_id="id-a",
                variant_name="base",
                spec=FakeSpec({"name": "exp-a", "reduction": {"r": 2}}),
                run_paths=object(),
            ),
            StudyRunRef(
                ident_id="id-b",
                variant_name="petrov",
                spec=FakeSpec({"name": "exp-b", "reduction": {"r": 4}}),
            ),
        ],
    )

    assert result.to_dict() == {
        "study_name": "sweep",
        "run_refs": [
            {
                "ident_id": "id-a",
                "variant_name": "base",
                "spec": {"name": "exp-a", "reduction": {"r": 2}},
            },
            {
                "ident_id": "id-b",
                "variant_name": "petrov",
                "spec": {"name": "exp-b", "reduction": {"r": 4}},
            },
        ],
    }
