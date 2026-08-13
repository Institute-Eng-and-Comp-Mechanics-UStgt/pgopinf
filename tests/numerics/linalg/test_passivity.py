from __future__ import annotations

import numpy as np
import cvxpy as cp

from pgopinf.numerics.linalg.passivity import kyp_lmi


def test_kyp_lmi_builds_numeric_matrix_and_checks_spsd() -> None:
    A = np.array([[-1.0]])
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[1.0]])
    X = np.array([[1.0]])

    W, is_spsd = kyp_lmi(A, B, C, D, X)

    np.testing.assert_allclose(W, np.array([[2.0, 0.0], [0.0, 2.0]]))
    assert is_spsd is True


def test_kyp_lmi_uses_descriptor_matrix_when_provided() -> None:
    A = np.array([[-1.0]])
    B = np.array([[2.0]])
    C = np.array([[3.0]])
    D = np.array([[0.5]])
    X = np.array([[4.0]])
    E = np.array([[2.0]])

    W, _ = kyp_lmi(A, B, C, D, X, E=E)

    np.testing.assert_allclose(W, np.array([[16.0, -13.0], [-13.0, 1.0]]))


def test_kyp_lmi_reports_non_spsd_numeric_matrix() -> None:
    A = np.array([[1.0]])
    B = np.array([[0.0]])
    C = np.array([[0.0]])
    D = np.array([[-1.0]])
    X = np.array([[1.0]])

    W, is_spsd = kyp_lmi(A, B, C, D, X)

    np.testing.assert_allclose(W, np.array([[-2.0, 0.0], [0.0, -2.0]]))
    assert is_spsd is False


def test_kyp_lmi_builds_cvxpy_expression_for_variable_system_matrix() -> None:
    A = cp.Variable((1, 1))
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[2.0]])
    X = np.array([[3.0]])

    W = kyp_lmi(A, B, C, D, X)

    assert W.shape == (2, 2)
    assert W.is_affine()
    A.value = np.array([[-1.0]])
    np.testing.assert_allclose(W.value, np.array([[6.0, 0.0], [0.0, 4.0]]))


def test_kyp_lmi_relaxed_variable_branch_adds_scalar_delta_variable() -> None:
    A = cp.Variable((1, 1))
    B = np.array([[1.0]])
    C = np.array([[1.0]])
    D = np.array([[2.0]])
    X = np.array([[3.0]])

    W = kyp_lmi(A, B, C, D, X, relaxed=True)

    assert W.shape == (2, 2)
    assert W.is_affine()
    assert len(W.variables()) == 2
