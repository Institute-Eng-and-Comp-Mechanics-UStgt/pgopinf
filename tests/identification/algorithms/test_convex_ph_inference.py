from __future__ import annotations

import numpy as np
import pytest
import cvxpy as cp

from pgopinf.identification.algorithms import (
    convex_ph_inference as convex_module,
)
from pgopinf.identification.algorithms.convex_ph_inference import (
    convex_ph_inference,
    process_regularization,
)


class FakeLTISystem:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = kwargs["A"].shape[0]


class FakePHSystem:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.n = kwargs["J"].shape[0]


def make_data():
    x = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    y = np.array([[0.1, 0.2, 0.3]])
    u = np.array([[1.0, 0.0, -1.0]])
    dxdt = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    return x, y, u, dxdt


def install_fake_solver(monkeypatch, *, make_indefinite: bool = False):
    variables = []
    solve_calls = []
    real_variable = convex_module.cp.Variable
    real_solve = convex_module.cp.Problem.solve

    def recording_variable(*args, **kwargs):
        var = real_variable(*args, **kwargs)
        variables.append(var)
        return var

    def fake_solve(self, *args, **kwargs):
        solve_calls.append({"args": args, "kwargs": kwargs})
        two_by_two_seen = 0
        two_by_one_seen = 0
        one_by_one_seen = 0
        for var in variables:
            shape = var.shape
            if shape == (2, 2):
                two_by_two_seen += 1
                if two_by_two_seen == 1:
                    value = np.eye(2) if not make_indefinite else np.diag([1.0, -1.0])
                elif two_by_two_seen == 2:
                    value = (
                        np.eye(2) * 0.5 if not make_indefinite else np.diag([1.0, -2.0])
                    )
                else:
                    value = np.array([[0.0, 1.0], [-1.0, 0.0]])
            elif shape == (2, 1):
                two_by_one_seen += 1
                value = (
                    np.array([[2.0], [3.0]])
                    if two_by_one_seen == 1
                    else np.zeros((2, 1))
                )
            elif shape == (1, 1):
                one_by_one_seen += 1
                value = (
                    np.array([[0.0]]) if one_by_one_seen == 1 else np.array([[0.25]])
                )
            else:
                value = np.ones(shape)
            var.value = value
        return 0.0

    monkeypatch.setattr(convex_module.cp, "Variable", recording_variable)
    monkeypatch.setattr(convex_module.cp.Problem, "solve", fake_solve)
    return variables, solve_calls, real_solve


def test_process_regularization_accepts_scalar_list_and_ndarray_weights() -> None:
    matrices = [cp.Variable((1, 1)) for _ in range(7)]

    scalar_term = process_regularization(0.5, *matrices)
    list_term = process_regularization([1, 2, 3, 4, 5, 6, 7], *matrices)
    array_term = process_regularization(np.arange(1.0, 8.0), *matrices)

    assert scalar_term.is_convex()
    assert list_term.is_convex()
    assert array_term.is_convex()


def test_process_regularization_rejects_wrong_length_list_or_array() -> None:
    matrices = [cp.Variable((1, 1)) for _ in range(7)]

    with pytest.raises(AssertionError):
        process_regularization([1, 2], *matrices)

    with pytest.raises(AssertionError):
        process_regularization(np.ones(2), *matrices)


def test_process_regularization_skips_non_variable_matrices() -> None:
    H = cp.Variable((1, 1))

    term = process_regularization(
        1.0,
        H,
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
    )

    H.value = np.array([[3.0]])
    assert term.value == pytest.approx(3.0)


def test_convex_ph_inference_builds_lti_system_and_forwards_scs_solver(
    monkeypatch,
) -> None:
    install_fake_solver(monkeypatch)
    monkeypatch.setattr(convex_module, "LTISystem", FakeLTISystem, raising=False)
    monkeypatch.setattr(
        "pgopinf.systems.lti_system.LTISystem",
        FakeLTISystem,
    )
    x, y, u, dxdt = make_data()

    system = convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        solver="scs",
        return_system_type="lti",
    )

    assert isinstance(system, FakeLTISystem)
    np.testing.assert_allclose(system.E, np.eye(2))
    np.testing.assert_allclose(system.A, np.array([[-0.5, 1.0], [-1.0, -0.5]]))
    np.testing.assert_allclose(system.B, np.array([[2.0], [3.0]]))
    np.testing.assert_allclose(system.C, np.array([[2.0, 3.0]]))
    np.testing.assert_allclose(system.D, np.array([[0.25]]))


def test_convex_ph_inference_builds_ph_system_and_returns_solver_stats(
    monkeypatch,
) -> None:
    _, solve_calls, _ = install_fake_solver(monkeypatch)
    monkeypatch.setattr(
        "pgopinf.systems.ph_system.PHSystem",
        FakePHSystem,
    )
    x, y, u, dxdt = make_data()

    system, solver_stats = convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        solver="clarabel",
        return_system_type="ph",
        return_solver_stats=True,
    )

    assert isinstance(system, FakePHSystem)
    assert solver_stats is None
    assert solve_calls[0]["kwargs"]["solver"] == cp.CLARABEL
    assert solve_calls[0]["kwargs"]["max_iter"] == 50
    np.testing.assert_allclose(system.Q, np.eye(2))
    np.testing.assert_allclose(system.E, np.eye(2))
    np.testing.assert_allclose(system.P, np.zeros((2, 1)))
    np.testing.assert_allclose(system.S, np.array([[0.25]]))
    np.testing.assert_allclose(system.N, np.array([[0.0]]))


def test_convex_ph_inference_midpoints_data_when_dxdt_missing(monkeypatch) -> None:

    monkeypatch.setattr(
        convex_module, "get_supplied_power", lambda y, u: np.zeros((y.shape[1], 1))
    )
    install_fake_solver(monkeypatch)
    x, y, u, _ = make_data()

    convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=None,
        delta_t=2.0,
        solver="scs",
        return_system_type="lti",
    )

    # The fake solver sets values; reaching this point verifies midpoint-sized
    # expressions were accepted by CVXPY. The explicit missing-delta assertion is below.


def test_convex_ph_inference_requires_delta_t_when_dxdt_missing() -> None:
    x, y, u, _ = make_data()

    with pytest.raises(AssertionError):
        convex_ph_inference(x=x, y=y, u=u, dxdt=None, delta_t=None)


def test_convex_ph_inference_accepts_known_skew_j_and_rejects_non_skew(
    monkeypatch,
) -> None:
    install_fake_solver(monkeypatch)
    x, y, u, dxdt = make_data()
    J_known = np.array([[0.0, 1.0], [-1.0, 0.0]])

    convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        J_known=J_known,
        solver="scs",
        return_system_type="lti",
    )

    with pytest.raises(AssertionError):
        convex_ph_inference(
            x=x,
            y=y,
            u=u,
            dxdt=dxdt,
            J_known=np.ones((2, 2)),
            solver="scs",
            return_system_type="lti",
        )


def test_convex_ph_inference_no_feedthrough_builds_zero_feedthrough_blocks(
    monkeypatch,
) -> None:
    install_fake_solver(monkeypatch)
    monkeypatch.setattr(
        "pgopinf.systems.lti_system.LTISystem",
        FakeLTISystem,
    )
    x, y, u, dxdt = make_data()

    system = convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        no_feedthrough=True,
        solver="scs",
        return_system_type="lti",
    )

    np.testing.assert_allclose(system.D, np.zeros((1, 1)))


def test_convex_ph_inference_projects_indefinite_operators_when_requested(
    monkeypatch,
) -> None:
    install_fake_solver(monkeypatch, make_indefinite=True)
    monkeypatch.setattr(
        "pgopinf.systems.lti_system.LTISystem",
        FakeLTISystem,
    )
    x, y, u, dxdt = make_data()

    system = convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        project_psd=True,
        solver="scs",
        return_system_type="lti",
    )

    assert np.all(np.linalg.eigvalsh(system.E) > 0.0)


def test_convex_ph_inference_forwards_mosek_accept_unknown(monkeypatch) -> None:
    _, solve_calls, _ = install_fake_solver(monkeypatch)
    x, y, u, dxdt = make_data()

    convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        solver="mosek",
        accept_unknown_mosek=True,
        return_system_type="lti",
    )

    assert solve_calls[0]["kwargs"]["solver"] == cp.MOSEK
    assert solve_calls[0]["kwargs"]["accept_unknown"] is True


def test_convex_ph_inference_adds_regularization_and_dissipation_terms(
    monkeypatch,
) -> None:
    calls = {"regularization": [], "kron_x_dx": 0, "kron_x_x": 0, "supplied": 0}

    def fake_regularization(lambdas, *matrices):
        calls["regularization"].append(
            {"lambdas": lambdas, "n_matrices": len(matrices)}
        )
        return 0

    def fake_kron_x_dx(*args, **kwargs):
        calls["kron_x_dx"] += 1
        return np.ones((3, 3))

    def fake_kron_x_x(*args, **kwargs):
        calls["kron_x_x"] += 1
        return np.ones((3, 6))

    def fake_supplied(y, u):
        calls["supplied"] += 1
        return np.zeros((y.shape[1], 1))

    monkeypatch.setattr(convex_module, "process_regularization", fake_regularization)
    monkeypatch.setattr(convex_module, "get_kron_x_dx_data", fake_kron_x_dx)
    monkeypatch.setattr(convex_module, "get_kron_x_x_data", fake_kron_x_x)
    monkeypatch.setattr(convex_module, "get_supplied_power", fake_supplied)
    install_fake_solver(monkeypatch)
    x, y, u, dxdt = make_data()

    convex_ph_inference(
        x=x,
        y=y,
        u=u,
        dxdt=dxdt,
        add_regularization=True,
        lambdas=[1, 2, 3, 4, 5, 6, 7],
        add_dissipation_inequality_cost=True,
        solver="scs",
        return_system_type="lti",
    )

    assert calls["regularization"] == [
        {"lambdas": [1, 2, 3, 4, 5, 6, 7], "n_matrices": 7}
    ]
    assert calls["kron_x_dx"] == 1
    assert calls["kron_x_x"] == 1
    assert calls["supplied"] == 1
