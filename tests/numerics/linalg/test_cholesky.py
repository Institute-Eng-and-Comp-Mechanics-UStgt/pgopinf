from __future__ import annotations

import numpy as np

from pgopinf.numerics.linalg.cholesky import lrcholesky, truncation


def test_truncation_sorts_modes_by_absolute_eigenvalue_and_drops_small_modes() -> None:
    d = np.array([1.0, 4.0, 1e-14])
    L = np.eye(3)

    dr, Lr = truncation(d, L, trunc_tol=1e-12)

    np.testing.assert_allclose(dr, np.array([4.0, 1.0]))
    np.testing.assert_allclose(np.abs(Lr), np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]))


def test_truncation_handles_complex_low_rank_factorization() -> None:
    d = np.array([4.0, 1.0])
    L = np.array([[1.0 + 0.0j, 0.0], [0.0, 1.0j]])

    dr, Lr = truncation(d, L)

    assert dr.shape == (2,)
    assert Lr.shape == (2, 2)
    np.testing.assert_allclose(Lr.conj().T @ Lr, np.eye(2), atol=1e-12)


def test_lrcholesky_reconstructs_positive_semidefinite_matrix_up_to_truncation() -> None:
    X = np.diag([4.0, 1.0, 0.0])

    L = lrcholesky(X, trunc_tol=1e-12)

    assert L.shape == (3, 2)
    np.testing.assert_allclose(L @ L.conj().T, X)


def test_lrcholesky_uses_hermitian_part_and_discards_negative_modes() -> None:
    X = np.array([[4.0, 1.0], [3.0, -1.0]])

    L = lrcholesky(X, trunc_tol=1e-12)
    reconstructed = L @ L.conj().T

    assert L.shape[0] == 2
    assert np.all(np.linalg.eigvalsh(reconstructed) >= -1e-12)
    np.testing.assert_allclose(reconstructed, reconstructed.conj().T)
