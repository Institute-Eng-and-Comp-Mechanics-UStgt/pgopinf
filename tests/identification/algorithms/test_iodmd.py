from __future__ import annotations

import logging

import numpy as np
import pytest

from pgopinf.identification.algorithms.iodmd import (
    iodmd,
    row_scale,
    unstack,
)


def make_exact_data():
    A = np.array([[1.0, 0.5], [0.0, 2.0]])
    B = np.array([[1.0], [-1.0]])
    C = np.array([[2.0, -1.0]])
    D = np.array([[0.25]])
    X = np.array(
        [
            [1.0, 0.0, 2.0, -1.0],
            [0.0, 1.0, -1.0, 2.0],
        ]
    )
    U = np.array([[1.0, -1.0, 2.0, 0.5]])
    X1 = A @ X + B @ U
    Y = C @ X + D @ U
    return A, B, C, D, X, U, X1, Y


def test_iodmd_recovers_exact_operator_with_explicit_x1() -> None:
    A, B, C, D, X, U, X1, Y = make_exact_data()

    Acal, error = iodmd(X=X, Y=Y, U=U, X1=X1)
    A_id, B_id, C_id, D_id = unstack(Acal, n=2)

    np.testing.assert_allclose(A_id, A, atol=1e-12)
    np.testing.assert_allclose(B_id, B, atol=1e-12)
    np.testing.assert_allclose(C_id, C, atol=1e-12)
    np.testing.assert_allclose(D_id, D, atol=1e-12)
    assert error == pytest.approx(0.0, abs=1e-12)


def test_iodmd_applies_descriptor_matrix_to_x1_target() -> None:
    A, B, C, D, X, U, X1, Y = make_exact_data()
    E = np.diag([2.0, 3.0])

    Acal, _ = iodmd(X=X, Y=Y, U=U, X1=X1, E=E)
    A_id, B_id, C_id, D_id = unstack(Acal, n=2)

    np.testing.assert_allclose(A_id, E @ A, atol=1e-12)
    np.testing.assert_allclose(B_id, E @ B, atol=1e-12)
    np.testing.assert_allclose(C_id, C, atol=1e-12)
    np.testing.assert_allclose(D_id, D, atol=1e-12)


def test_iodmd_uses_shifted_snapshots_when_x1_is_missing() -> None:
    A = np.array([[2.0]])
    B = np.array([[3.0]])
    C = np.array([[4.0]])
    D = np.array([[5.0]])
    U0 = np.array([[0.5, -1.0, 2.0]])
    X = np.zeros((1, 4))
    X[:, 0] = [1.0]
    for i in range(U0.shape[1]):
        X[:, i + 1] = (A @ X[:, i : i + 1] + B @ U0[:, i : i + 1]).ravel()
    U = np.concatenate([U0, np.array([[99.0]])], axis=1)
    Y = np.concatenate([C @ X[:, :-1] + D @ U0, np.array([[123.0]])], axis=1)

    Acal, error = iodmd(X=X, Y=Y, U=U, X1=None)
    A_id, B_id, C_id, D_id = unstack(Acal, n=1)

    np.testing.assert_allclose(A_id, A, atol=1e-12)
    np.testing.assert_allclose(B_id, B, atol=1e-12)
    np.testing.assert_allclose(C_id, C, atol=1e-12)
    np.testing.assert_allclose(D_id, D, atol=1e-12)
    assert error == pytest.approx(0.0, abs=1e-12)


def test_iodmd_separate_output_inference_matches_joint_solution_for_exact_data() -> (
    None
):
    _, _, _, _, X, U, X1, Y = make_exact_data()

    joint, _ = iodmd(X=X, Y=Y, U=U, X1=X1, seperate_output_inf=False)
    separate, _ = iodmd(X=X, Y=Y, U=U, X1=X1, seperate_output_inf=True)

    np.testing.assert_allclose(separate, joint)


def test_iodmd_ridge_regularization_augments_problem_and_keeps_shape() -> None:
    _, _, _, _, X, U, X1, Y = make_exact_data()

    Acal, error = iodmd(X=X, Y=Y, U=U, X1=X1, lambda_reg=0.1)

    assert Acal.shape == (3, 3)
    assert np.isfinite(error)


def test_iodmd_rejects_negative_regularization() -> None:
    _, _, _, _, X, U, X1, Y = make_exact_data()

    with pytest.raises(AssertionError):
        iodmd(X=X, Y=Y, U=U, X1=X1, lambda_reg=-1.0)


def test_iodmd_logs_warning_for_ill_conditioned_data(caplog) -> None:
    X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0 + 1e-14]])
    U = np.array([[0.0, 0.0, 0.0]])
    X1 = X.copy()
    Y = np.zeros((1, 3))

    with caplog.at_level(logging.WARNING):
        iodmd(X=X, Y=Y, U=U, X1=X1)

    assert "ill-conditioned" in caplog.text


def test_row_scale_normalizes_rows_and_returns_scale_factors() -> None:
    A = np.array([[3.0, 4.0], [0.0, 0.0]])

    scaled, scale = row_scale(A, eps=0.5)

    np.testing.assert_allclose(scale, np.array([5.0, 0.5]))
    np.testing.assert_allclose(scaled, np.array([[0.6, 0.8], [0.0, 0.0]]))


def test_unstack_splits_block_matrix_and_can_zero_feedthrough() -> None:
    AA = np.arange(12.0).reshape(3, 4)

    A, B, C, D = unstack(AA, n=2)
    A_no, B_no, C_no, D_no = unstack(AA, n=2, no_feedtrough=True)

    np.testing.assert_allclose(A, AA[:2, :2])
    np.testing.assert_allclose(B, AA[:2, 2:])
    np.testing.assert_allclose(C, AA[2:, :2])
    np.testing.assert_allclose(D, AA[2:, 2:])
    np.testing.assert_allclose(A_no, A)
    np.testing.assert_allclose(B_no, B)
    np.testing.assert_allclose(C_no, C)
    np.testing.assert_allclose(D_no, np.zeros_like(D))
