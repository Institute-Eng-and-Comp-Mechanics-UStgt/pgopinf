from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.opinf_theory import OpInfTheoryMetric
from pgopinf.reduction.interfaces import BasisArtifact


class FakeSystem:
    def __init__(self, *, A, B, C, D, n_u: int = 1, issparse: bool = False):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n_u = n_u
        self.issparse = issparse
        self.convert_to_dense_calls = 0

    def convert_to_dense(self):
        self.convert_to_dense_calls += 1


def make_context(
    *,
    V: np.ndarray | None = None,
    W: np.ndarray | None = None,
    full_basis=None,
    lambda_reg=None,
    identified_system=None,
) -> EvaluationContext:
    V = np.array([[1.0], [0.0]]) if V is None else V
    W = V.copy() if W is None else W
    original_system = FakeSystem(
        A=np.array([[1.0, 0.0], [0.0, 2.0]]),
        B=np.array([[0.0], [0.0]]),
        C=np.array([[0.0, 3.0]]),
        D=np.array([[0.0]]),
    )
    intrusive_system = FakeSystem(
        A=np.array([[2.0]]),
        B=np.array([[1.0]]),
        C=np.array([[4.0]]),
        D=np.array([[3.0]]),
    )
    identified_system = identified_system or FakeSystem(
        A=np.array([[2.5]]),
        B=np.array([[0.5]]),
        C=np.array([[5.0]]),
        D=np.array([[1.0]]),
    )
    train_data = SimpleNamespace(
        x=np.array([[1.0, 2.0], [3.0, 4.0]]),
        u=np.array([[5.0, 6.0]]),
        dxdt=np.array([[7.0, 8.0], [9.0, 10.0]]),
        y=np.array([[11.0, 12.0]]),
    )
    diagnostics = {} if lambda_reg is None else {"lambda_reg": lambda_reg}
    if full_basis is None:
        full_basis = SimpleNamespace(
            Vh=np.eye(2),
            S=np.array([4.0, 2.0]),
        )

    return EvaluationContext(
        experiment_name="experiment",
        run_id="run",
        original_system=original_system,
        dataset_artifact=SimpleNamespace(data=SimpleNamespace(TRAIN=train_data)),
        reduction_result=SimpleNamespace(
            reduced_system=intrusive_system,
            basis=BasisArtifact(V=V, W=W, meta={}),
            full_basis=full_basis,
        ),
        dataset_artifact_intrusive=None,
        identification_result=SimpleNamespace(
            system_identified=identified_system,
            diagnostics=diagnostics,
        ),
        dataset_artifact_identified=None,
    )


def artifact_values(output):
    return {artifact.name: artifact.value for artifact in output.artifacts}


def test_opinf_theory_computes_operator_data_bias_and_bound_metrics() -> None:
    ctx = make_context()

    output = OpInfTheoryMetric().compute(ctx)

    assert output.summaries == []
    assert ctx.original_system.convert_to_dense_calls == 1
    assert ctx.reduction_result.reduced_system.convert_to_dense_calls == 1
    assert ctx.identification_result.system_identified.convert_to_dense_calls == 1

    values = artifact_values(output)
    assert values["A_B_conc_int_frob_norm"] == np.sqrt(5.0)
    assert values["C_D_conc_int_frob_norm"] == 5.0
    assert values["A_B_conc_id_frob_norm"] == np.sqrt(6.5)
    assert values["C_D_conc_id_frob_norm"] == np.sqrt(26.0)
    assert values["bias_term_states_frob_norm"] == 0.0
    assert values["bias_term_output_frob_norm"] == 15.0
    assert values["rank_train_data"] == 2
    assert values["rank_rhs_data_reduced"] == 2

    rhs = np.array([[1.0, 2.0], [5.0, 6.0]])
    bias_states = np.array([[0.0, 0.0]])
    bias_output = np.array([[9.0, 12.0]])
    bias_states_pinv = bias_states @ np.linalg.pinv(rhs, rcond=1e-12)
    bias_output_pinv = bias_output @ np.linalg.pinv(rhs, rcond=1e-12)
    np.testing.assert_allclose(
        values["abs_operator_bias_states"],
        np.linalg.norm(bias_states_pinv, "fro"),
    )
    np.testing.assert_allclose(
        values["abs_operator_bias_output"],
        np.linalg.norm(bias_output_pinv, "fro"),
    )
    np.testing.assert_allclose(values["abs_error_A_B"], np.linalg.norm([[0.5, -0.5]]))
    np.testing.assert_allclose(values["abs_error_C_D"], np.linalg.norm([[1.0, -2.0]]))
    assert "X_perp_pinv_T_r_frobnorm" in values
    assert "bound_Xperp_Tr" in values
    assert values["singular_value_r"] == 4.0


def test_opinf_theory_regularization_branch_uses_intrusive_system_matrices() -> None:
    ctx = make_context(lambda_reg=0.25)

    output = OpInfTheoryMetric().compute(ctx)

    values = artifact_values(output)
    assert "abs_operator_bias_states" in values
    assert np.isfinite(values["abs_operator_bias_states"])
    assert np.isfinite(values["abs_operator_bias_output"])
    assert values["rank_rhs_data_reduced"] == 2


def test_opinf_theory_skips_bound_metrics_when_petrov_galerkin_basis_is_used(
    caplog,
) -> None:
    V = np.array([[1.0], [0.0]])
    W = np.array([[1.0], [1.0]])
    ctx = make_context(V=V, W=W)

    output = OpInfTheoryMetric().compute(ctx)

    values = artifact_values(output)
    assert "rel_error_operators_combined" in values
    assert "bound_Xperp_Tr" not in values
    assert "V and W need to be equal" in caplog.text


def test_opinf_theory_skips_bound_metrics_without_square_full_vh() -> None:
    ctx = make_context(full_basis=SimpleNamespace(Vh=np.ones((1, 2)), S=np.array([1.0, 0.5])))

    output = OpInfTheoryMetric().compute(ctx)

    values = artifact_values(output)
    assert "rel_error_operators_combined" in values
    assert "bound_Xperp_Tr" not in values
