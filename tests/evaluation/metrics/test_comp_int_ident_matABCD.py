from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.comp_int_ident_matABCD import (
    CompIntIdentMatABCDMetric,
)


def make_system(*, A, B, C, D, issparse: bool = False):
    return SimpleNamespace(A=A, B=B, C=C, D=D, issparse=issparse)


def make_context(*, intrusive_system, identified_system) -> EvaluationContext:
    return EvaluationContext(
        experiment_name="experiment",
        run_id="run",
        original_system=object(),
        dataset_artifact=object(),
        reduction_result=SimpleNamespace(reduced_system=intrusive_system),
        dataset_artifact_intrusive=None,
        identification_result=SimpleNamespace(system_identified=identified_system),
        dataset_artifact_identified=None,
    )


def artifact_values(output):
    return {artifact.name: artifact.value for artifact in output.artifacts}


def test_comp_int_ident_mat_abcd_computes_relative_dense_frobenius_errors() -> None:
    intrusive = make_system(
        A=np.array([[3.0, 4.0], [0.0, 0.0]]),
        B=np.array([[1.0], [2.0]]),
        C=np.array([[2.0, 0.0]]),
        D=np.array([[4.0]]),
    )
    identified = make_system(
        A=np.array([[0.0, 0.0], [0.0, 0.0]]),
        B=np.array([[0.0], [0.0]]),
        C=np.array([[1.0, 0.0]]),
        D=np.array([[2.0]]),
    )

    output = CompIntIdentMatABCDMetric().compute(
        make_context(intrusive_system=intrusive, identified_system=identified)
    )

    assert output.summaries == []
    values = artifact_values(output)
    assert values["comp_int_ident_matA"] == 1.0
    assert values["comp_int_ident_matB"] == 1.0
    assert values["comp_int_ident_matC"] == 0.5
    assert values["comp_int_ident_matD"] == 0.5
    assert [artifact.name for artifact in output.artifacts] == [
        "comp_int_ident_matA",
        "comp_int_ident_matB",
        "comp_int_ident_matC",
        "comp_int_ident_matD",
    ]
    assert all(
        artifact.meta == {"relative": True, "rel_threshold": 1e-8}
        for artifact in output.artifacts
    )


def test_comp_int_ident_mat_abcd_can_compute_absolute_errors() -> None:
    intrusive = make_system(
        A=np.eye(2),
        B=np.array([[1.0], [0.0]]),
        C=np.array([[0.0, 2.0]]),
        D=np.array([[3.0]]),
    )
    identified = make_system(
        A=np.zeros((2, 2)),
        B=np.zeros((2, 1)),
        C=np.zeros((1, 2)),
        D=np.zeros((1, 1)),
    )

    output = CompIntIdentMatABCDMetric(relative=False).compute(
        make_context(intrusive_system=intrusive, identified_system=identified)
    )

    values = artifact_values(output)
    assert values["comp_int_ident_matA"] == np.sqrt(2.0)
    assert values["comp_int_ident_matB"] == 1.0
    assert values["comp_int_ident_matC"] == 2.0
    assert values["comp_int_ident_matD"] == 3.0
    assert all(
        artifact.meta == {"relative": False, "rel_threshold": 1e-8}
        for artifact in output.artifacts
    )


def test_comp_int_ident_mat_abcd_uses_relative_threshold_for_zero_reference() -> None:
    intrusive = make_system(
        A=np.zeros((1, 1)),
        B=np.zeros((1, 1)),
        C=np.zeros((1, 1)),
        D=np.zeros((1, 1)),
    )
    identified = make_system(
        A=np.array([[2.0]]),
        B=np.array([[4.0]]),
        C=np.array([[6.0]]),
        D=np.array([[8.0]]),
    )

    output = CompIntIdentMatABCDMetric(rel_threshold=2.0).compute(
        make_context(intrusive_system=intrusive, identified_system=identified)
    )

    assert artifact_values(output) == {
        "comp_int_ident_matA": 1.0,
        "comp_int_ident_matB": 2.0,
        "comp_int_ident_matC": 3.0,
        "comp_int_ident_matD": 4.0,
    }


def test_comp_int_ident_mat_abcd_uses_sparse_frobenius_norms() -> None:
    intrusive = make_system(
        A=scipy.sparse.csr_matrix([[3.0, 4.0]]),
        B=scipy.sparse.csr_matrix([[2.0]]),
        C=scipy.sparse.csr_matrix([[0.0, 6.0]]),
        D=scipy.sparse.csr_matrix([[5.0]]),
        issparse=True,
    )
    identified = make_system(
        A=scipy.sparse.csr_matrix([[0.0, 0.0]]),
        B=scipy.sparse.csr_matrix([[1.0]]),
        C=scipy.sparse.csr_matrix([[0.0, 3.0]]),
        D=scipy.sparse.csr_matrix([[0.0]]),
        issparse=True,
    )

    output = CompIntIdentMatABCDMetric().compute(
        make_context(intrusive_system=intrusive, identified_system=identified)
    )

    values = artifact_values(output)
    assert values["comp_int_ident_matA"] == 1.0
    assert values["comp_int_ident_matB"] == 0.5
    assert values["comp_int_ident_matC"] == 0.5
    assert values["comp_int_ident_matD"] == 1.0
