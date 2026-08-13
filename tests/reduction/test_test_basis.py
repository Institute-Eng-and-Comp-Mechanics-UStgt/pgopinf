from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pgopinf.reduction import test_basis as test_basis_module
from pgopinf.reduction.test_basis import (
    GalerkinTestBasis,
    QVTestBasis,
    _as2d,
    _check_shapes,
    biorthonormalize,
)


def test_as2d_accepts_matrices_and_rejects_other_shapes() -> None:
    A = np.eye(2)

    assert _as2d(A) is A
    with pytest.raises(ValueError, match="Expected 2D array"):
        _as2d(np.ones(2))


def test_check_shapes_requires_matching_basis_shapes() -> None:
    _check_shapes(np.zeros((3, 2)), np.ones((3, 2)))

    with pytest.raises(ValueError, match="V and W must have same shape"):
        _check_shapes(np.zeros((3, 2)), np.ones((3, 1)))


def test_biorthonormalize_adjusts_w_so_wtv_is_identity() -> None:
    V = np.array([[2.0], [0.0]])
    W = np.array([[1.0], [0.0]])

    W_new = biorthonormalize(W, V)

    np.testing.assert_allclose(W_new.T @ V, np.eye(1))


def test_galerkin_test_basis_returns_v_as_array() -> None:
    V = [[1.0], [2.0]]

    W = GalerkinTestBasis().make_W(V=V)

    np.testing.assert_allclose(W, V)


def test_qv_test_basis_cache_options_include_q_affecting_options_only() -> None:
    basis = QVTestBasis(
        source="Hamiltonian",
        enforce_WtV_I=True,
        project_hamiltonian_Q_spsd=False,
    )

    assert basis.q_cache_options() == {
        "source": "Hamiltonian",
        "project_hamiltonian_Q_spsd": False,
    }


@pytest.mark.parametrize(
    ("source", "system_attr"),
    [("system_Q", "Q"), ("system_E", "E")],
)
def test_qv_test_basis_compute_q_from_system_matrix(
    source: str, system_attr: str
) -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    system = SimpleNamespace(**{system_attr: matrix})

    Q = QVTestBasis(source=source).compute_Q(system=system)

    assert Q is matrix


def test_qv_test_basis_compute_q_from_cached_matrix_is_used_by_make_w() -> None:
    V = np.array([[1.0], [2.0]])
    Q = np.array([[2.0, 0.0], [0.0, 3.0]])

    W = QVTestBasis(source="system_Q").make_W(V=V, Q=Q)

    np.testing.assert_allclose(W, np.array([[2.0], [6.0]]))


def test_qv_test_basis_can_enforce_biorthonormality() -> None:
    V = np.array([[2.0], [0.0]])
    Q = np.eye(2)

    W = QVTestBasis(source="system_Q", enforce_WtV_I=True).make_W(V=V, Q=Q)

    np.testing.assert_allclose(W.T @ V, np.eye(1))


def test_qv_test_basis_hamiltonian_source_computes_values_and_calls_identifier(
    monkeypatch,
) -> None:
    calls = []

    def fake_hamiltonian_identification(x, Ham, project):
        calls.append({"x": x, "Ham": Ham, "project": project})
        return np.eye(2) * 4.0

    monkeypatch.setattr(
        test_basis_module,
        "hamiltonian_identification",
        fake_hamiltonian_identification,
    )
    x = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    data_train = SimpleNamespace(x=x)
    system = SimpleNamespace(E=np.eye(2), Q=np.diag([2.0, 3.0]))

    Q = QVTestBasis(source="Hamiltonian", project_hamiltonian_Q_spsd=False).compute_Q(
        system=system,
        data_train=data_train,
    )

    np.testing.assert_allclose(Q, np.eye(2) * 4.0)
    np.testing.assert_allclose(calls[0]["x"], x)
    np.testing.assert_allclose(calls[0]["Ham"], np.array([1.0, 6.0, 2.5]))
    assert calls[0]["project"] is False


def test_qv_test_basis_hamiltonian_source_uses_lsmr_for_large_data(monkeypatch) -> None:
    calls = []

    def fake_lsmr(**kwargs):
        calls.append(kwargs)
        return np.eye(2) * 5.0, {"info": True}

    monkeypatch.setattr(test_basis_module, "hamiltonian_identification_lsmr", fake_lsmr)
    x = np.ones((2, 10001))
    data_train = SimpleNamespace(x=x)
    system = SimpleNamespace(E=np.eye(2), Q=np.eye(2))

    Q = QVTestBasis(source="Hamiltonian").compute_Q(
        system=system, data_train=data_train
    )

    np.testing.assert_allclose(Q, np.eye(2) * 5.0)
    assert calls[0]["x"] is x
    assert calls[0]["project"] is True


def test_qv_test_basis_hamiltonian_red_requires_v_and_passes_it(monkeypatch) -> None:
    calls = []

    def fake_hamiltonian_identification(x, Ham, project, V=None):
        calls.append({"x": x, "Ham": Ham, "project": project, "V": V})
        return np.eye(2)

    monkeypatch.setattr(
        test_basis_module,
        "hamiltonian_identification",
        fake_hamiltonian_identification,
    )
    x = np.eye(2)
    data_train = SimpleNamespace(x=x)
    system = SimpleNamespace(E=np.eye(2), Q=np.eye(2))
    V = np.eye(2)

    Q = QVTestBasis(source="Hamiltonian_red").compute_Q(
        system=system,
        data_train=data_train,
        V=V,
    )

    np.testing.assert_allclose(Q, np.eye(2))
    assert calls[0]["V"] is V

    with pytest.raises(ValueError, match="requires V"):
        QVTestBasis(source="Hamiltonian_red").compute_Q(
            system=system,
            data_train=data_train,
        )


def test_qv_test_basis_kyp_source_calls_riccati_solver(monkeypatch) -> None:
    system = object()
    monkeypatch.setattr(
        test_basis_module,
        "solve_Riccati",
        lambda arg: np.eye(2) if arg is system else None,
    )

    Q = QVTestBasis(source="kyp").compute_Q(system=system)

    np.testing.assert_allclose(Q, np.eye(2))


def test_qv_test_basis_validates_required_inputs_and_q_shape() -> None:
    with pytest.raises(ValueError, match="requires system.Q"):
        QVTestBasis(source="system_Q").compute_Q(system=SimpleNamespace())

    with pytest.raises(ValueError, match="requires system.E"):
        QVTestBasis(source="system_E").compute_Q(system=SimpleNamespace())

    with pytest.raises(ValueError, match="requires data_train"):
        QVTestBasis(source="Hamiltonian").compute_Q(
            system=SimpleNamespace(E=np.eye(2), Q=np.eye(2))
        )

    with pytest.raises(ValueError, match="Invalid source"):
        QVTestBasis(source="missing").compute_Q(system=SimpleNamespace())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Q must be square"):
        QVTestBasis(source="system_Q").compute_Q(system=SimpleNamespace(Q=np.ones(2)))


def test_qv_test_basis_make_w_rejects_incompatible_q_shape() -> None:
    with pytest.raises(ValueError, match="V and W must have same shape"):
        QVTestBasis().make_W(V=np.ones((2, 1)), Q=np.ones((3, 2)))
