from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.core.identification_store import IdentificationStore
from pgopinf.identification.identifier import IdentificationResult
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.identification.operator_inference import (
    OperatorInferenceSpec,
)
from pgopinf.systems.lti_system import LTISystem


def make_paths(tmp_path) -> ResultsPath:
    return ResultsPath(results_root=tmp_path / "results")


def make_system() -> LTISystem:
    return LTISystem(
        A=np.array([[-1.0, 0.2], [0.0, -2.0]]),
        B=np.array([[1.0], [0.5]]),
        C=np.array([[1.0, 0.0]]),
        D=np.array([[0.0]]),
        E=np.eye(2),
    )


def make_result(*, meta=None) -> IdentificationResult:
    return IdentificationResult(
        system_identified=make_system(),
        diagnostics={"loss": 0.25, "status": "ok"},
        meta=meta or {"kind": "opinf", "r": 2},
    )


def assert_system_equal(left: LTISystem, right: LTISystem) -> None:
    for key in ("A", "B", "C", "D", "E"):
        np.testing.assert_allclose(getattr(left, key), getattr(right, key))


@dataclass
class FakeIdentifier:
    calls: list
    result: IdentificationResult

    def fit(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


@dataclass
class FakeIdentificationSpec:
    calls: list
    result: IdentificationResult
    value: int = 1
    kind: str = "fake_identification"

    def to_dict(self):
        return {"kind": self.kind, "value": self.value}

    def build(self):
        return FakeIdentifier(calls=self.calls, result=self.result)


def test_compute_id_depends_on_reduction_and_identification_spec(tmp_path) -> None:
    store = IdentificationStore(make_paths(tmp_path))

    base_id = store.compute_id(
        reduction_id="red-a",
        identification_spec=OperatorInferenceSpec(lambda_reg=0.0),
    )

    assert base_id == store.compute_id(
        reduction_id="red-a",
        identification_spec=OperatorInferenceSpec(lambda_reg=0.0),
    )
    assert base_id != store.compute_id(
        reduction_id="red-b",
        identification_spec=OperatorInferenceSpec(lambda_reg=0.0),
    )
    assert base_id != store.compute_id(
        reduction_id="red-a",
        identification_spec=OperatorInferenceSpec(lambda_reg=1e-3),
    )


def test_save_and_load_round_trip_preserves_result_meta_and_system(tmp_path) -> None:
    paths = make_paths(tmp_path)
    store = IdentificationStore(paths)
    result = make_result(meta={"kind": "opinf", "r": 2, "note": "keep me"})

    store.save(
        "ident-id",
        reduction_id="red-id",
        identification_spec=OperatorInferenceSpec(lambda_reg=1e-3),
        result=result,
    )

    assert store.exists("ident-id")
    loaded = store.load("ident-id")
    assert loaded.diagnostics == result.diagnostics
    assert loaded.meta == result.meta
    assert_system_equal(loaded.system_identified, result.system_identified)

    d = paths.identification_dir("ident-id")
    assert json.loads((d / "spec.json").read_text()) == {
        "identification": {
            "kind": "opinf",
            "lambda_reg": 0.001,
            "use_E": True,
            "convert_to_ph": True,
            "seperate_output_inf": False,
        },
        "identification_id": "ident-id",
        "reduction_id": "red-id",
    }
    assert json.loads((d / "meta.json").read_text()) == result.meta
    assert json.loads((d / "system_meta.json").read_text()) == {"kind": "lti"}


def test_exists_requires_all_persisted_files(tmp_path) -> None:
    store = IdentificationStore(make_paths(tmp_path))
    d = store.paths.identification_dir("partial")
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text("{}", encoding="utf-8")
    (d / "diagnostics.json").write_text("{}", encoding="utf-8")
    (d / "meta.json").write_text("{}", encoding="utf-8")
    (d / "system_meta.json").write_text('{"kind": "lti"}', encoding="utf-8")

    assert store.exists("partial") is False


def test_load_rejects_unknown_system_kind(tmp_path) -> None:
    store = IdentificationStore(make_paths(tmp_path))
    d = store.paths.identification_dir("unknown")
    d.mkdir(parents=True, exist_ok=True)
    (d / "diagnostics.json").write_text("{}", encoding="utf-8")
    (d / "meta.json").write_text("{}", encoding="utf-8")
    (d / "system_meta.json").write_text('{"kind": "mystery"}', encoding="utf-8")
    np.savez_compressed(d / "matrices.npz", A=np.eye(1))

    with pytest.raises(ValueError, match="Unknown identified system kind"):
        store.load("unknown")


def test_get_or_create_runs_identifier_saves_and_reuses_cached_result(tmp_path) -> None:
    store = IdentificationStore(make_paths(tmp_path))
    calls = []
    result = make_result()
    spec = FakeIdentificationSpec(calls=calls, result=result, value=7)
    reduction_result = SimpleNamespace(
        reduced_dataset_projected="projected-data",
        reduced_system="reduced-system",
    )

    ident_id, first = store.get_or_create(
        reduction_id="red-id",
        reduction_result=reduction_result,
        identification_spec=spec,
        original_system="original-system",
    )
    second_id, second = store.get_or_create(
        reduction_id="red-id",
        reduction_result=reduction_result,
        identification_spec=spec,
        original_system="original-system",
    )

    assert ident_id == second_id
    assert first is result
    assert calls == [
        {
            "reduced_dataset": "projected-data",
            "reduced_system": "reduced-system",
            "original_system": "original-system",
        }
    ]
    assert second.diagnostics == result.diagnostics
    assert second.meta == result.meta
    assert_system_equal(second.system_identified, result.system_identified)


def test_get_or_create_honors_force_recompute(tmp_path) -> None:
    store = IdentificationStore(make_paths(tmp_path), force_recompute=True)
    calls = []
    result = make_result()
    spec = FakeIdentificationSpec(calls=calls, result=result)
    reduction_result = SimpleNamespace(
        reduced_dataset_projected="projected-data",
        reduced_system="reduced-system",
    )

    store.get_or_create(
        reduction_id="red-id",
        reduction_result=reduction_result,
        identification_spec=spec,
        original_system=None,
    )
    store.get_or_create(
        reduction_id="red-id",
        reduction_result=reduction_result,
        identification_spec=spec,
        original_system=None,
    )

    assert len(calls) == 2
