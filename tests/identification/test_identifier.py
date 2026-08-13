from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.identification import identifier as identifier_module
from pgopinf.identification.identifier import (
    ConvexPortHamiltonianIdentifier,
    IdentificationResult,
    OperatorInferenceIdentifier,
)


class FakeIdentifiedSystem:
    def __init__(self, n: int):
        self.n = n


def make_reduced_dataset():
    train = SimpleNamespace(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        y=np.array([[5.0, 6.0]]),
        u=np.array([[7.0, 8.0]]),
        dxdt=np.array([[9.0, 10.0], [11.0, 12.0]]),
    )
    return SimpleNamespace(TRAIN=train)


def test_identification_result_is_frozen_value_object() -> None:
    system = FakeIdentifiedSystem(n=2)
    result = IdentificationResult(
        system_identified=system,
        diagnostics={"a": 1},
        meta={"kind": "test"},
    )

    assert result.system_identified is system
    assert result.diagnostics == {"a": 1}
    assert result.meta == {"kind": "test"}

    with pytest.raises(FrozenInstanceError):
        result.meta = {}  # type: ignore[misc]


def test_operator_inference_identifier_uses_reduced_system_e_by_default(monkeypatch) -> None:
    calls = []
    identified = FakeIdentifiedSystem(n=2)

    def fake_operator_inference(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return identified

    monkeypatch.setattr(identifier_module, "operator_inference", fake_operator_inference)
    reduced_dataset = make_reduced_dataset()
    reduced_system = SimpleNamespace(E=np.diag([2.0, 3.0]), n=2)
    identifier = OperatorInferenceIdentifier(
        use_E=True,
        convert_to_ph=False,
        seperate_output_inf=True,
        lambda_reg=0.25,
    )

    result = identifier.fit(
        reduced_dataset=reduced_dataset,
        reduced_system=reduced_system,
        original_system=object(),
    )

    call = calls[0]
    assert call["args"] == (
        reduced_dataset.TRAIN.x,
        reduced_dataset.TRAIN.y,
        reduced_dataset.TRAIN.u,
    )
    assert call["kwargs"] == {
        "E": reduced_system.E,
        "dxdt": reduced_dataset.TRAIN.dxdt,
        "seperate_output_inf": True,
        "lambda_reg": 0.25,
        "convert_to_ph": False,
    }
    assert result.system_identified is identified
    assert result.diagnostics == {
        "use_E": True,
        "convert_to_ph": False,
        "seperate_output_inf": True,
        "lambda_reg": 0.25,
    }
    assert result.meta == {"kind": "opinf", "r": 2}


def test_operator_inference_identifier_can_use_identity_e(monkeypatch) -> None:
    calls = []

    def fake_operator_inference(*args, **kwargs):
        calls.append(kwargs)
        return FakeIdentifiedSystem(n=3)

    monkeypatch.setattr(identifier_module, "operator_inference", fake_operator_inference)

    OperatorInferenceIdentifier(use_E=False).fit(
        reduced_dataset=make_reduced_dataset(),
        reduced_system=SimpleNamespace(E=np.diag([2.0, 3.0, 4.0]), n=3),
    )

    np.testing.assert_allclose(calls[0]["E"], np.eye(3))


def test_convex_ph_identifier_forwards_options_and_known_j(monkeypatch) -> None:
    calls = []
    identified = FakeIdentifiedSystem(n=4)
    solver_stats = SimpleNamespace(
        solve_time=1.25,
        vector=np.array([1.0, 2.0]),
        nested={"array": np.array([[3.0, 4.0]]), "status": "ok"},
    )

    def fake_convex_ph_inference(**kwargs):
        calls.append(kwargs)
        return identified, solver_stats

    monkeypatch.setattr(identifier_module, "convex_ph_inference", fake_convex_ph_inference)
    reduced_dataset = make_reduced_dataset()
    reduced_system = SimpleNamespace(J=np.array([[0.0, 1.0], [-1.0, 0.0]]))
    identifier = ConvexPortHamiltonianIdentifier(
        J_is_known=True,
        add_regularization=True,
        add_dissipation_inequality_cost=True,
        lambdas=0.5,
        no_feedthrough=True,
        solver="clarabel",
        project_psd=True,
        return_system_type="lti",
        accept_unknown_mosek=True,
    )

    result = identifier.fit(
        reduced_dataset=reduced_dataset,
        reduced_system=reduced_system,
        original_system=object(),
    )

    assert calls[0] == {
        "x": reduced_dataset.TRAIN.x,
        "y": reduced_dataset.TRAIN.y,
        "u": reduced_dataset.TRAIN.u,
        "dxdt": reduced_dataset.TRAIN.dxdt,
        "J_known": reduced_system.J,
        "add_regularization": True,
        "lambdas": 0.5,
        "add_dissipation_inequality_cost": True,
        "no_feedthrough": True,
        "solver": "clarabel",
        "project_psd": True,
        "return_system_type": "lti",
        "return_solver_stats": True,
        "accept_unknown_mosek": True,
    }
    assert result.system_identified is identified
    assert result.meta == {"kind": "convex_ph_inference", "r": 4}
    assert result.diagnostics == {
        "J_is_known": True,
        "add_regularization": True,
        "add_dissipation_inequality_cost": True,
        "lambdas": 0.5,
        "no_feedthrough": True,
        "solver": "clarabel",
        "project_psd": True,
        "return_system_type": "lti",
        "solver_stats": {
            "solve_time": 1.25,
            "vector": [1.0, 2.0],
            "nested": {"array": [[3.0, 4.0]], "status": "ok"},
        },
        "accept_unknown_mosek": True,
    }


def test_convex_ph_identifier_omits_j_when_not_known(monkeypatch) -> None:
    calls = []

    def fake_convex_ph_inference(**kwargs):
        calls.append(kwargs)
        return FakeIdentifiedSystem(n=2), SimpleNamespace()

    monkeypatch.setattr(identifier_module, "convex_ph_inference", fake_convex_ph_inference)

    ConvexPortHamiltonianIdentifier(J_is_known=False).fit(
        reduced_dataset=make_reduced_dataset(),
        reduced_system=SimpleNamespace(J=np.array([[0.0, 1.0], [-1.0, 0.0]])),
    )

    assert calls[0]["J_known"] is None


def test_convex_ph_identifier_preserves_non_array_solver_stats(monkeypatch) -> None:
    def fake_convex_ph_inference(**kwargs):
        return FakeIdentifiedSystem(n=1), SimpleNamespace(
            status="optimal",
            nested={"iterations": 5},
        )

    monkeypatch.setattr(identifier_module, "convex_ph_inference", fake_convex_ph_inference)

    result = ConvexPortHamiltonianIdentifier().fit(
        reduced_dataset=make_reduced_dataset(),
        reduced_system=SimpleNamespace(J=np.zeros((2, 2))),
    )

    assert result.diagnostics["solver_stats"] == {
        "status": "optimal",
        "nested": {"iterations": 5},
    }
