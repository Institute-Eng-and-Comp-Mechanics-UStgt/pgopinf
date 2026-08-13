from __future__ import annotations

import numpy as np

from pgopinf.data.initial_condition.initial_condition import InitialCondition
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.reduction import mor as mor_module
from pgopinf.reduction.mor import LTIMORProjector, PHMORProjector


class FakeLTISystem:
    def __init__(self):
        self.reduce_calls = []

    def reduce(self, V, W):
        self.reduce_calls.append({"V": V, "W": W})
        return "reduced-lti"


class FakePHSystem:
    def __init__(self):
        self.reduce_calls = []

    def reduce(self, V, ph_reduction_type="Berlin"):
        self.reduce_calls.append(
            {"V": V, "ph_reduction_type": ph_reduction_type}
        )
        return "reduced-ph"


def test_lti_mor_projector_delegates_data_and_initial_condition_projection() -> None:
    projector = LTIMORProjector()
    basis = BasisArtifact(V=np.eye(2, 1), W=np.eye(2, 1), meta={})
    ic = InitialCondition(np.array([[1.0], [2.0]]))

    np.testing.assert_allclose(
        projector.project_initial_condition(ic, basis),
        np.array([[1.0]]),
    )


def test_lti_mor_projector_reduces_lti_like_system(monkeypatch) -> None:
    monkeypatch.setattr(mor_module, "LTISystem", FakeLTISystem)
    monkeypatch.setattr(mor_module, "PHSystem", FakePHSystem)
    projector = LTIMORProjector()
    system = FakeLTISystem()
    basis = BasisArtifact(V=np.eye(2, 1), W=np.ones((2, 1)), meta={})

    result = projector.reduce_system(system=system, basis=basis)

    assert result == "reduced-lti"
    assert system.reduce_calls == [{"V": basis.V, "W": basis.W}]


def test_ph_mor_projector_reduces_ph_system_with_configured_type(monkeypatch) -> None:
    monkeypatch.setattr(mor_module, "PHSystem", FakePHSystem)
    projector = PHMORProjector(ph_reduction_type="Gugercin")
    system = FakePHSystem()
    basis = BasisArtifact(V=np.eye(2, 1), W=None, meta={})

    result = projector.reduce_system(system=system, basis=basis)

    assert result == "reduced-ph"
    assert system.reduce_calls == [
        {"V": basis.V, "ph_reduction_type": "Gugercin"}
    ]
