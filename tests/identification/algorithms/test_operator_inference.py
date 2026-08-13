from __future__ import annotations

import numpy as np
import pytest

from pgopinf.identification.algorithms import (
    operator_inference as operator_inference_module,
)
from pgopinf.identification.algorithms.operator_inference import (
    operator_inference,
)


class FakeLTISystem:
    def __init__(self, *, A, B, C, D, E):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.n = A.shape[0]


def make_data():
    x = np.array([[1.0, 3.0, 7.0], [2.0, 6.0, 12.0]])
    y = np.array([[10.0, 20.0, 40.0]])
    u = np.array([[5.0, 7.0, 11.0]])
    dxdt = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return x, y, u, dxdt


def install_algorithm_fakes(monkeypatch):
    calls = {"iodmd": [], "unstack": [], "lti": []}
    Acal = np.arange(6.0).reshape(3, 2)

    def fake_iodmd(**kwargs):
        calls["iodmd"].append(kwargs)
        return Acal, 0.123

    def fake_unstack(arg, n):
        calls["unstack"].append({"Acal": arg, "n": n})
        return (
            np.eye(n),
            np.ones((n, 1)),
            np.ones((1, n)) * 2.0,
            np.ones((1, 1)) * 3.0,
        )

    class RecordingLTISystem(FakeLTISystem):
        def __init__(self, **kwargs):
            calls["lti"].append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(operator_inference_module, "iodmd", fake_iodmd)
    monkeypatch.setattr(operator_inference_module, "unstack", fake_unstack)
    monkeypatch.setattr(operator_inference_module, "LTISystem", RecordingLTISystem)
    return calls, Acal


def test_operator_inference_uses_provided_derivatives_and_builds_lti_system(monkeypatch) -> None:
    calls, Acal = install_algorithm_fakes(monkeypatch)
    x, y, u, dxdt = make_data()
    E = np.diag([2.0, 3.0])

    system = operator_inference(
        x,
        y,
        u,
        E=E,
        dxdt=dxdt,
        seperate_output_inf=True,
        lambda_reg=0.5,
        convert_to_ph=False,
    )

    assert isinstance(system, FakeLTISystem)
    assert calls["iodmd"] == [
        {
            "X": x,
            "Y": y,
            "U": u,
            "X1": dxdt,
            "E": E,
            "seperate_output_inf": True,
            "lambda_reg": 0.5,
        }
    ]
    assert calls["unstack"][0]["Acal"] is Acal
    assert calls["unstack"][0]["n"] == 2
    assert calls["lti"][0]["E"] is E
    np.testing.assert_allclose(system.A, np.eye(2))
    np.testing.assert_allclose(system.B, np.ones((2, 1)))
    np.testing.assert_allclose(system.C, np.ones((1, 2)) * 2.0)
    np.testing.assert_allclose(system.D, np.ones((1, 1)) * 3.0)


def test_operator_inference_derives_midpoint_data_when_dxdt_is_missing(monkeypatch) -> None:
    calls, _ = install_algorithm_fakes(monkeypatch)
    x, y, u, _ = make_data()

    operator_inference(
        x,
        y,
        u,
        E=None,
        delta_t=2.0,
        dxdt=None,
        convert_to_ph=False,
    )

    call = calls["iodmd"][0]
    np.testing.assert_allclose(call["X1"], np.array([[1.0, 2.0], [2.0, 3.0]]))
    np.testing.assert_allclose(call["X"], 0.5 * (x[:, 1:] + x[:, :-1]))
    np.testing.assert_allclose(call["U"], 0.5 * (u[:, 1:] + u[:, :-1]))
    np.testing.assert_allclose(call["Y"], 0.5 * (y[:, 1:] + y[:, :-1]))


def test_operator_inference_requires_delta_t_when_derivative_is_missing(monkeypatch) -> None:
    install_algorithm_fakes(monkeypatch)
    x, y, u, _ = make_data()

    with pytest.raises(AssertionError):
        operator_inference(x, y, u, dxdt=None, delta_t=None)


def test_operator_inference_can_convert_identified_lti_system_to_ph(monkeypatch) -> None:
    calls, _ = install_algorithm_fakes(monkeypatch)
    ph_system = object()
    riccati_calls = []

    def fake_get_Riccati_transform(system):
        riccati_calls.append(system)
        return "T", "X", ph_system

    monkeypatch.setattr(
        operator_inference_module,
        "get_Riccati_transform",
        fake_get_Riccati_transform,
    )
    x, y, u, dxdt = make_data()

    result = operator_inference(x, y, u, dxdt=dxdt, convert_to_ph=True)

    assert result is ph_system
    assert len(riccati_calls) == 1
    assert isinstance(riccati_calls[0], FakeLTISystem)
    assert calls["lti"][0]["A"].shape == (2, 2)


def test_operator_inference_does_not_mutate_input_arrays_when_midpointing(monkeypatch) -> None:
    install_algorithm_fakes(monkeypatch)
    x, y, u, _ = make_data()
    x_original = x.copy()
    y_original = y.copy()
    u_original = u.copy()

    operator_inference(x, y, u, delta_t=1.0, dxdt=None)

    np.testing.assert_allclose(x, x_original)
    np.testing.assert_allclose(y, y_original)
    np.testing.assert_allclose(u, u_original)
