from __future__ import annotations

import logging

import numpy as np
import pytest

from pgopinf.data.hamiltonian_identification import (
    _make_hamiltonian_operator,
    _unpack_symmetric_upper,
    duplication_matrix,
    hamiltonian_identification,
    hamiltonian_identification_lsmr,
)


def hamiltonian_values(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum(X * (Q @ X), axis=0)


def test_duplication_matrix_maps_upper_triangular_entries_to_symmetric_vector() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    Q = (duplication_matrix(3) @ q).reshape(3, 3)

    np.testing.assert_allclose(
        Q,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 5.0],
                [3.0, 5.0, 6.0],
            ]
        ),
    )


def test_unpack_symmetric_upper_reconstructs_full_symmetric_matrix() -> None:
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    Q = _unpack_symmetric_upper(q, 3)

    np.testing.assert_allclose(Q, (duplication_matrix(3) @ q).reshape(3, 3))


def test_hamiltonian_operator_matvec_matches_quadratic_hamiltonian() -> None:
    X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    q = np.array([2.0, 0.5, 1.0])
    Q = _unpack_symmetric_upper(q, 2)
    operator = _make_hamiltonian_operator(X)

    np.testing.assert_allclose(operator @ q, hamiltonian_values(Q, X))


def test_hamiltonian_operator_rmatvec_matches_explicit_adjoint() -> None:
    X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    operator = _make_hamiltonian_operator(X)
    y = np.array([3.0, -1.0, 2.0])
    eye = np.eye(operator.shape[1])
    explicit = np.column_stack([operator @ eye[:, i] for i in range(operator.shape[1])])

    np.testing.assert_allclose(operator.rmatvec(y), explicit.T @ y)


def test_hamiltonian_identification_recovers_exact_symmetric_matrix() -> None:
    Q_true = np.array([[2.0, 0.5], [0.5, 1.0]])
    X = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, -1.0]])
    Ham = hamiltonian_values(Q_true, X)

    Q_id = hamiltonian_identification(X, Ham)

    np.testing.assert_allclose(Q_id, Q_true)


def test_hamiltonian_identification_warns_for_underdetermined_data(caplog) -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Ham = np.array([1.0, 2.0])

    with caplog.at_level(logging.WARNING):
        hamiltonian_identification(X, Ham)

    assert "not enough data points to identify Q" in caplog.text


def test_hamiltonian_identification_projects_and_lifts_reduced_basis() -> None:
    V = np.array([[1.0], [0.0]])
    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    Ham = 0.5 * 4.0 * X[0, :] ** 2

    Q_id = hamiltonian_identification(X, Ham, V=V)

    np.testing.assert_allclose(Q_id, np.array([[4.0, 0.0], [0.0, 0.0]]))


def test_hamiltonian_identification_lsmr_recovers_exact_symmetric_matrix() -> None:
    Q_true = np.array([[2.0, 0.5], [0.5, 1.0]])
    X = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, -1.0]])
    Ham = hamiltonian_values(Q_true, X)

    Q_id, info = hamiltonian_identification_lsmr(
        X,
        Ham,
        atol=1e-12,
        btol=1e-12,
    )

    np.testing.assert_allclose(Q_id, Q_true, atol=1e-12)
    assert set(info) == {"istop", "itn", "normr", "normar", "norma", "conda", "normx"}
    assert info["normr"] < 1e-10


def test_hamiltonian_identification_lsmr_rejects_mismatched_hamiltonian_length() -> None:
    X = np.ones((2, 3))
    Ham = np.ones(2)

    with pytest.raises(ValueError, match="Shape mismatch"):
        hamiltonian_identification_lsmr(X, Ham)


def test_hamiltonian_identification_lsmr_warns_for_underdetermined_data(caplog) -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    Ham = np.array([1.0, 2.0])

    with caplog.at_level(logging.WARNING):
        hamiltonian_identification_lsmr(X, Ham)

    assert "not enough data points to uniquely identify" in caplog.text


def test_hamiltonian_identification_lsmr_can_project_identified_matrix_to_psd() -> None:
    Q_indefinite = np.array([[1.0, 0.0], [0.0, -1.0]])
    X = np.array([[1.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, -1.0]])
    Ham = hamiltonian_values(Q_indefinite, X)

    Q_id, _ = hamiltonian_identification_lsmr(
        X,
        Ham,
        project=True,
        atol=1e-12,
        btol=1e-12,
    )

    assert np.all(np.linalg.eigvalsh(Q_id) >= -1e-12)
