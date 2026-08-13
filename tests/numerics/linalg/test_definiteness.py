from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.sparse

from pgopinf.numerics.linalg.definiteness import (
    check_spsd,
    project_definite_cone,
    project_snd,
    project_snsd,
    project_spd,
    project_spsd,
)


def test_check_spsd_accepts_positive_definite_and_semidefinite_matrices() -> None:
    assert check_spsd(np.array([[2.0, 0.0], [0.0, 1.0]]))
    assert check_spsd(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert check_spsd(scipy.sparse.csr_matrix([[1.0, 0.0], [0.0, 2.0]]))


def test_check_spsd_rejects_indefinite_and_nonsymmetric_matrices(caplog) -> None:
    with caplog.at_level(logging.INFO):
        assert not check_spsd(np.array([[1.0, 0.0], [0.0, -1.0]]))
        assert not check_spsd(np.array([[1.0, 2.0], [0.0, 1.0]]))

    assert "Minimal eigenvalue" in caplog.text
    assert "not symmetric" in caplog.text


def test_project_definite_cone_clips_eigenvalues_to_positive_cone() -> None:
    A = np.array([[1.0, 0.0], [0.0, -2.0]])

    projected = project_definite_cone(A, max_min_eig_val=0.25, definite_type="pos")

    np.testing.assert_allclose(projected, np.array([[1.0, 0.0], [0.0, 0.25]]))
    assert np.all(np.linalg.eigvalsh(projected) >= 0.25 - 1e-12)


def test_project_definite_cone_clips_eigenvalues_to_negative_cone() -> None:
    A = np.array([[1.0, 0.0], [0.0, -2.0]])

    projected = project_definite_cone(A, max_min_eig_val=-0.25, definite_type="neg")

    np.testing.assert_allclose(projected, np.array([[-0.25, 0.0], [0.0, -2.0]]))
    assert np.all(np.linalg.eigvalsh(projected) <= -0.25 + 1e-12)


def test_projection_wrappers_enforce_expected_definiteness() -> None:
    A = np.array([[1.0, 2.0], [2.0, -3.0]])

    assert np.all(np.linalg.eigvalsh(project_spsd(A)) >= -1e-12)
    assert np.all(np.linalg.eigvalsh(project_spd(A, min_eig_val=0.1)) >= 0.1 - 1e-12)
    assert np.all(np.linalg.eigvalsh(project_snsd(A)) <= 1e-12)
    assert np.all(np.linalg.eigvalsh(project_snd(A, max_eig_val=-0.1)) <= -0.1 + 1e-12)


def test_project_definite_cone_warns_for_threshold_that_breaks_requested_cone(caplog) -> None:
    A = np.eye(2)

    with caplog.at_level(logging.WARNING):
        project_definite_cone(A, max_min_eig_val=-1.0, definite_type="pos")
        project_definite_cone(A, max_min_eig_val=1.0, definite_type="neg")

    assert "Negative minimal eigenvalue" in caplog.text
    assert "Positive minimal eigenvalue" in caplog.text


def test_project_definite_cone_raises_for_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown definite_type"):
        project_definite_cone(np.eye(2), definite_type="missing")
