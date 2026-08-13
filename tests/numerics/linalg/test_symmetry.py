from __future__ import annotations

import numpy as np
import scipy.sparse

from pgopinf.numerics.linalg.symmetry import (
    check_sparsity_and_convert_to_dense,
    hermitian_part,
    is_skewsym,
    is_sym,
    skew_hermitian,
    skewsym,
    sym,
)


def test_is_sym_accepts_dense_and_sparse_symmetric_matrices() -> None:
    A = np.array([[1.0, 2.0], [2.0, 3.0]])

    assert is_sym(A)
    assert is_sym(scipy.sparse.csr_matrix(A))
    assert not is_sym(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_is_skewsym_accepts_dense_and_sparse_skew_symmetric_matrices() -> None:
    A = np.array([[0.0, 2.0], [-2.0, 0.0]])

    assert is_skewsym(A)
    assert is_skewsym(scipy.sparse.coo_matrix(A))
    assert not is_skewsym(np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_sym_and_skewsym_decompose_matrix() -> None:
    A = np.array([[1.0, 4.0], [2.0, 3.0]])

    np.testing.assert_allclose(sym(A), np.array([[1.0, 3.0], [3.0, 3.0]]))
    np.testing.assert_allclose(skewsym(A), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    np.testing.assert_allclose(sym(A) + skewsym(A), A)


def test_check_sparsity_and_convert_to_dense_only_converts_sparse_inputs() -> None:
    dense = np.eye(2)
    sparse = scipy.sparse.csr_matrix(dense)

    assert check_sparsity_and_convert_to_dense(dense) is dense
    np.testing.assert_allclose(check_sparsity_and_convert_to_dense(sparse), dense)


def test_hermitian_and_skew_hermitian_parts_decompose_complex_matrix() -> None:
    X = np.array([[1.0 + 0.0j, 2.0 + 3.0j], [4.0 - 1.0j, 5.0 + 2.0j]])

    H = hermitian_part(X)
    K = skew_hermitian(X)

    np.testing.assert_allclose(H, H.conj().T)
    np.testing.assert_allclose(K, -K.conj().T)
    np.testing.assert_allclose(H + K, X)
