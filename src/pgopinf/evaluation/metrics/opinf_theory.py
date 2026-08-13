from dataclasses import dataclass
import numpy as np
import scipy
import logging

from pgopinf.evaluation.context import EvaluationContext
from pgopinf.evaluation.metrics.subroutines.hinf import hinf_error
from pgopinf.evaluation.results import (
    MetricOutput,
    ScalarMetricResult,
)
from pgopinf.reduction.project_data import define_reduction_projector

logger = logging.getLogger(__name__)


@dataclass
class OpInfTheoryMetric:
    """Compute diagnostic quantities from the operator-inference error theory."""

    name: str = "opinf_theory"
    rcond: float = 1e-12

    def compute(self, ctx: EvaluationContext) -> MetricOutput:
        """Compute operator-inference diagnostic scalar artifacts.

        Parameters
        ----------
        ctx : EvaluationContext
            Evaluation context containing original, intrusive, and identified
            systems and training data.

        Returns
        -------
        MetricOutput
            Scalar artifacts for matrix norms, bias terms, data conditioning,
            and related bounds.
        """

        intrusive_system = ctx.reduction_result.reduced_system
        identified_system = ctx.identification_result.system_identified

        # convert system matrices to dense
        ctx.original_system.convert_to_dense()
        intrusive_system.convert_to_dense()
        identified_system.convert_to_dense()

        if identified_system.issparse or intrusive_system.issparse:
            frobenius_norm_function = lambda x: scipy.sparse.linalg.norm(x, "fro")
        else:
            frobenius_norm_function = lambda x: np.linalg.norm(x, "fro")

        artifacts = []

        # -------------------------------------------------------------------------------------------
        # Differences between intrusive and identified model matrices. [A,B] and [C,D] with bias term
        # --------------------------------------------------------------------------------------------
        A_B_conc_int = np.concatenate(
            [intrusive_system.A, intrusive_system.B],
            axis=1,
        )
        A_B_conc_id = np.concatenate([identified_system.A, identified_system.B], axis=1)
        C_D_conc_int = np.concatenate(
            [intrusive_system.C, intrusive_system.D],
            axis=1,
        )
        C_D_conc_id = np.concatenate([identified_system.C, identified_system.D], axis=1)

        """
        Frobenius norms of concatenated matrices
        A_B_conc_int_frob_norm: ||A B]_intrusive||_F
        C_D_conc_int_frob_norm: ||[C D]_intrusive||_F
        A_B_conc_id_frob_norm:  ||[A B]_identified||_F
        C_D_conc_id_frob_norm:  ||[C D]_identified||_F
        """
        A_B_conc_int_frob_norm = frobenius_norm_function(A_B_conc_int)
        C_D_conc_int_frob_norm = frobenius_norm_function(C_D_conc_int)
        A_B_conc_id_frob_norm = frobenius_norm_function(A_B_conc_id)
        C_D_conc_id_frob_norm = frobenius_norm_function(C_D_conc_id)

        artifacts.append(
            ScalarMetricResult(
                name=f"A_B_conc_int_frob_norm",
                value=A_B_conc_int_frob_norm,
                unit="-",
                meta={"target": "intrusive"},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"C_D_conc_int_frob_norm",
                value=C_D_conc_int_frob_norm,
                unit="-",
                meta={"target": "intrusive"},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"A_B_conc_id_frob_norm",
                value=A_B_conc_id_frob_norm,
                unit="-",
                meta={"target": "identified"},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"C_D_conc_id_frob_norm",
                value=C_D_conc_id_frob_norm,
                unit="-",
                meta={"target": "identified"},
            )
        )

        # reduction values
        V = ctx.reduction_result.basis.V
        W = ctx.reduction_result.basis.W
        reduction_projector = define_reduction_projector(V, W)
        # original data values
        original_data_x = ctx.dataset_artifact.data.TRAIN.x
        original_data_u = ctx.dataset_artifact.data.TRAIN.u

        residual_data = (np.eye(V.shape[0]) - V @ reduction_projector) @ original_data_x

        # if self.mor.reduce_sample_n:
        #     residual_data = residual_data[:, :: self.mor.reduce_sample_n]
        bias_term_states = W.T @ ctx.original_system.A @ residual_data
        bias_term_output = ctx.original_system.C @ residual_data

        if "lambda_reg" in ctx.identification_result.diagnostics:
            lambda_reg = ctx.identification_result.diagnostics["lambda_reg"]
            if lambda_reg > 0.0:
                # add regularization term to bias term calculation
                bias_term_states = np.hstack(
                    [
                        bias_term_states,
                        -lambda_reg * intrusive_system.A,
                        -lambda_reg * intrusive_system.B,
                    ]
                )
                bias_term_output = np.hstack(
                    [
                        bias_term_output,
                        -lambda_reg * intrusive_system.C,
                        -lambda_reg * intrusive_system.D,
                    ]
                )
        else:
            lambda_reg = None

        """
        Frobenius norms of bias terms
        bias_term_states_frob_norm: |W.T@A@Xperp||_F = ||E_x||_F
        bias_term_output_frob_norm: ||C@Xperp||_F = ||E_y||_F
        """
        bias_term_states_frob_norm = frobenius_norm_function(bias_term_states)
        bias_term_output_frob_norm = frobenius_norm_function(bias_term_output)

        artifacts.append(
            ScalarMetricResult(
                name=f"bias_term_states_frob_norm",
                value=bias_term_states_frob_norm,
                unit="-",
                meta={},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"bias_term_output_frob_norm",
                value=bias_term_output_frob_norm,
                unit="-",
                meta={},
            )
        )

        """
        Data properties
        rank_rhs_data_reduced: rank(T_r)
        cond_rhs_data_reduced: cond(T_r)
        """
        # reduced states and inputs of inference data
        rhs_data_reduced = np.concatenate(
            [
                reduction_projector @ original_data_x,
                original_data_u,
            ],
            axis=0,
        )

        if lambda_reg and lambda_reg > 0.0:
            # regularization term
            n_cols_rhs = rhs_data_reduced.shape[0]
            reg_term = lambda_reg * np.eye(n_cols_rhs)
            rhs_data_reduced = np.hstack(
                [rhs_data_reduced, reg_term],
            )
            residual_data_for_pinv = np.hstack(
                [residual_data, np.zeros((residual_data.shape[0], n_cols_rhs))]
            )
        else:
            residual_data_for_pinv = residual_data

        # check rank condition
        rank_train_data = np.linalg.matrix_rank(original_data_x)
        rank_rhs_data_reduced = np.linalg.matrix_rank(rhs_data_reduced)
        if rank_rhs_data_reduced < min(
            rhs_data_reduced.shape[0], rhs_data_reduced.shape[1]
        ):
            logger.warning(
                f"Warning: rhs_data_reduced is rank deficient with rank {np.linalg.matrix_rank(rhs_data_reduced)} < {min(rhs_data_reduced.shape[0], rhs_data_reduced.shape[1])}.\n This may lead to incorrect bias term calculation."
            )

        #  condition of the data
        cond_rhs_data_reduced = np.linalg.cond(rhs_data_reduced)

        artifacts.append(
            ScalarMetricResult(
                name=f"rank_train_data",
                value=rank_train_data,
                unit="-",
                meta={},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rank_rhs_data_reduced",
                value=rank_rhs_data_reduced,
                unit="-",
                meta={},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"cond_rhs_data_reduced",
                value=cond_rhs_data_reduced,
                unit="-",
                meta={},
            )
        )

        """
        Frobenius norms of bias terms with pseudoinverse of data
        abs_operator_bias_states: ||E_x@pinv(T_r)||_F
        abs_operator_bias_output: ||E_y@pinv(T_r)||_F
        rel_operator_bias_states: ||E_x@pinv(T_r)||_F / ||[A B]_intrusive||_F
        rel_operator_bias_output: ||E_y@pinv(T_r)||_F / ||[C D]_intrusive||_F
        """
        rhs_data_reduced_pinv = np.linalg.pinv(rhs_data_reduced, rcond=self.rcond)
        bias_term_states_data_pinv = bias_term_states @ rhs_data_reduced_pinv
        bias_term_output_data_pinv = bias_term_output @ rhs_data_reduced_pinv

        abs_operator_bias_states = frobenius_norm_function(bias_term_states_data_pinv)
        abs_operator_bias_output = frobenius_norm_function(bias_term_output_data_pinv)
        rel_operator_bias_states = abs_operator_bias_states / frobenius_norm_function(
            A_B_conc_int
        )
        rel_operator_bias_output = abs_operator_bias_output / frobenius_norm_function(
            C_D_conc_int
        )

        artifacts.append(
            ScalarMetricResult(
                name=f"abs_operator_bias_states",
                value=abs_operator_bias_states,
                unit="-",
                meta={"relative": False},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"abs_operator_bias_output",
                value=abs_operator_bias_output,
                unit="-",
                meta={"relative": False},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_operator_bias_states",
                value=rel_operator_bias_states,
                unit="-",
                meta={"relative": True},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_operator_bias_output",
                value=rel_operator_bias_output,
                unit="-",
                meta={"relative": True},
            )
        )

        """
        Compare intrusive (A,B) and (C,D) with identified
        abs_error_A_B: ||[A B]_identified-[A B]_intrusive||_F
        abs_error_C_D: ||[C D]_identified-[C D]_intrusive||_F
        rel_error_A_B: ||[A B]_identified-[A B]_intrusive||_F / ||[A B]_intrusive||_F
        rel_error_C_D: ||[C D]_identified-[C D]_intrusive||_F / ||[C D]_intrusive||_F
        """
        abs_error_A_B = frobenius_norm_function(A_B_conc_id - A_B_conc_int)
        abs_error_C_D = frobenius_norm_function(C_D_conc_id - C_D_conc_int)
        rel_error_A_B = abs_error_A_B / frobenius_norm_function(A_B_conc_int)
        rel_error_C_D = abs_error_C_D / frobenius_norm_function(C_D_conc_int)

        artifacts.append(
            ScalarMetricResult(
                name=f"abs_error_A_B",
                value=abs_error_A_B,
                unit="-",
                meta={"relative": False},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"abs_error_C_D",
                value=abs_error_C_D,
                unit="-",
                meta={"relative": False},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_A_B",
                value=rel_error_A_B,
                unit="-",
                meta={"relative": True},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_C_D",
                value=rel_error_C_D,
                unit="-",
                meta={"relative": True},
            )
        )

        """
        Compare intrusive (A,B) and (C,D) with identified with bias term
        rel_error_A_B_incl_bias: ||[A B]_identified - ([A B]_intrusive + E_x@pinv(T_r))||_F/||[A B]_intrusive||_F
        rel_error_C_D_incl_bias: ||[C D]_identified - ([C D]_intrusive + E_y@pinv(T_r))||_F/||[C D]_intrusive||_F
        """
        rel_error_A_B_incl_bias = frobenius_norm_function(
            A_B_conc_id - (A_B_conc_int + bias_term_states_data_pinv)
        ) / frobenius_norm_function(A_B_conc_int)
        rel_error_C_D_incl_bias = frobenius_norm_function(
            C_D_conc_id - (C_D_conc_int + bias_term_output_data_pinv)
        ) / frobenius_norm_function(C_D_conc_int)

        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_A_B_incl_bias",
                value=rel_error_A_B_incl_bias,
                unit="-",
                meta={"relative": True},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_C_D_incl_bias",
                value=rel_error_C_D_incl_bias,
                unit="-",
                meta={"relative": True},
            )
        )

        """
        relative error of combined system without bias in data space (this should decrease over r)
        rel_error_ABCD_in_data_space:  ||([A B;C D]_identified - [A B;C D]_intrusive) @ T_r||_F/||[dxdt;y]||_F
        bias_term_combined_data_frob_norm:||E_x@pinv(T_r);E_y@pinv(T_r)||_F
        rel_error_bias_term_combined_data_frob_norm (eps_corr in paper): ||E_x@pinv(T_r);E_y@pinv(T_r)||_F / ||[A B;C D]_intrusive||_F
        rel_error_ABCD_in_data_space_with_bias: ||([A B;C D]_identified - ([A B;C D]_intrusive + E@pinv(T_r))) @ T_r||_F/||[dxdt;y]||_F
        """
        original_data_dxdt = ctx.dataset_artifact.data.TRAIN.dxdt
        original_data_y = ctx.dataset_artifact.data.TRAIN.y
        lhs_data_reduced = np.concatenate(
            [
                reduction_projector @ original_data_dxdt,
                original_data_y,
            ],
            axis=0,
        )

        A_B_C_D_conc_id = np.concatenate([A_B_conc_id, C_D_conc_id], axis=0)
        A_B_C_D_conc_int = np.concatenate([A_B_conc_int, C_D_conc_int], axis=0)
        rel_error_ABCD_in_data_space = frobenius_norm_function(
            (A_B_C_D_conc_id - A_B_C_D_conc_int) @ rhs_data_reduced
        ) / frobenius_norm_function(lhs_data_reduced)

        # relative error of combined system with bias in data space (this should be machine precision close to zero)
        bias_term_combined_data = np.concatenate(
            [bias_term_states_data_pinv, bias_term_output_data_pinv], axis=0
        )
        bias_term_combined_data_frob_norm = frobenius_norm_function(
            bias_term_combined_data
        )
        rel_error_bias_term_combined_data_frob_norm = (
            bias_term_combined_data_frob_norm
            / frobenius_norm_function(A_B_C_D_conc_int)
        )

        rel_error_ABCD_in_data_space_with_bias = frobenius_norm_function(
            (A_B_C_D_conc_id - (A_B_C_D_conc_int + bias_term_combined_data))
            @ rhs_data_reduced
        ) / frobenius_norm_function(lhs_data_reduced)

        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_ABCD_in_data_space",
                value=rel_error_ABCD_in_data_space,
                unit="-",
                meta={"relative": True},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"bias_term_combined_data_frob_norm",
                value=bias_term_combined_data_frob_norm,
                unit="-",
                meta={},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_bias_term_combined_data_frob_norm",
                value=rel_error_bias_term_combined_data_frob_norm,
                unit="-",
                meta={"relative": True},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_ABCD_in_data_space_with_bias",
                value=rel_error_ABCD_in_data_space_with_bias,
                unit="-",
                meta={"relative": True},
            )
        )

        """
        Relative error of combined system of operators
        rel_error_operators_combined (eps_ref in paper): ||[A B;C D]_identified - [A B;C D]_intrusive||_F/||[A B;C D]_intrusive||_F
        rel_error_operators_combined_with_bias: ||[A B;C D]_identified - ([A B;C D]_intrusive + E@pinv(T_r))||_F/||[A B;C D]_intrusive||_F
        """
        # without bias term (this should decrease over r)
        rel_error_operators_combined = frobenius_norm_function(
            A_B_C_D_conc_id - A_B_C_D_conc_int
        ) / frobenius_norm_function(A_B_C_D_conc_int)
        # with bias term (this should be small)
        rel_error_operators_combined_with_bias = frobenius_norm_function(
            A_B_C_D_conc_id - (A_B_C_D_conc_int + bias_term_combined_data)
        ) / frobenius_norm_function(A_B_C_D_conc_int)

        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_operators_combined",
                value=rel_error_operators_combined,
                unit="-",
                meta={"relative": True},
            )
        )
        artifacts.append(
            ScalarMetricResult(
                name=f"rel_error_operators_combined_with_bias",
                value=rel_error_operators_combined_with_bias,
                unit="-",
                meta={"relative": True},
            )
        )

        """
        Bound on ||X_perp @ pinv(T_r)||_F. Bound proven for n_u = 1 but also calculated for n_u > 1. It is assumed to hold also for higher n_u.
        X_perp_pinv_T_r_frobnorm: ||X_perp @ pinv(T_r)||_F
        bound_Xperp_Tr: sigma_r+1/sigma_r * ||U_perp||_F* sqrt(||U_parallel||_F^2 + n_u*sigma_r^2)
        rel_bound_Xperp_Tr (eps_est in paper): bound_Xperp_Tr / ||[A B;C D]_intrusive||_F
        bound_operators_combined: bound_Xperp_Tr * ||[W.T@A;C]_intrusive||_F
        rel_error_bound_operators_combined (eps_bound in paper): bound_operators_combined / ||[A B;C D]_intrusive||_F
        """

        # check assumptions of proposition
        if np.allclose(V, W):
            prop_assumptions_satisfied = True
        else:
            prop_assumptions_satisfied = False
            logger.warning(f"V and W need to be equal. Bound will not be calculated.")

        if not ctx.original_system.n_u == 1:
            logger.warning(
                f"Warning: Assumptions of proposition not satisfied. n_u={ctx.original_system.n_u} (needs to be 1). The bound may not hold in this case. It will still be calculated."
            )

        # check if Vh is attribute of full basis artifact (should be the case for POD full basis)
        if (
            hasattr(ctx.reduction_result.full_basis, "Vh")
            and prop_assumptions_satisfied
            and ctx.reduction_result.full_basis.Vh.shape[0]
            == ctx.reduction_result.full_basis.Vh.shape[
                1
            ]  # check if full_matrices was used
        ):
            # ||X_perp @ pinv(T_r)||_2
            X_perp_pinv_T_r = residual_data_for_pinv @ rhs_data_reduced_pinv
            X_perp_pinv_T_r_2norm = np.linalg.norm(X_perp_pinv_T_r, 2)
            X_perp_pinv_T_r_frobnorm = frobenius_norm_function(X_perp_pinv_T_r)

            r = ctx.reduction_result.basis.V.shape[1]
            sigma_rplusone = ctx.reduction_result.full_basis.S[r]
            sigma_r = ctx.reduction_result.full_basis.S[r - 1]
            U1 = original_data_u @ ctx.reduction_result.full_basis.Vh[:r, :].T
            U2 = original_data_u @ ctx.reduction_result.full_basis.Vh[r:, :].T
            U2_pinv = np.linalg.pinv(U2, rcond=self.rcond)

            assert np.allclose(
                original_data_u,
                U1 @ ctx.reduction_result.full_basis.Vh[:r, :]
                + U2 @ ctx.reduction_result.full_basis.Vh[r:, :],
            )

            bound_Xperp_Tr = (
                sigma_rplusone
                / sigma_r
                * frobenius_norm_function(U2_pinv)
                * np.sqrt(
                    frobenius_norm_function(U1) ** 2
                    + ctx.original_system.n_u * sigma_r**2
                )
            )

            rel_bound_Xperp_Tr = bound_Xperp_Tr / frobenius_norm_function(
                A_B_C_D_conc_int
            )

            bound_operators_combined = bound_Xperp_Tr * frobenius_norm_function(
                np.vstack((W.T @ ctx.original_system.A, ctx.original_system.C))
            )
            rel_error_bound_operators_combined = (
                bound_operators_combined / frobenius_norm_function(A_B_C_D_conc_int)
            )

            artifacts.append(
                ScalarMetricResult(
                    name=f"X_perp_pinv_T_r_frobnorm",
                    value=X_perp_pinv_T_r_frobnorm,
                    unit="-",
                    meta={"relative": False},
                )
            )
            artifacts.append(
                ScalarMetricResult(
                    name=f"bound_Xperp_Tr",
                    value=bound_Xperp_Tr,
                    unit="-",
                    meta={"relative": False},
                )
            )
            artifacts.append(
                ScalarMetricResult(
                    name=f"rel_bound_Xperp_Tr",
                    value=rel_bound_Xperp_Tr,
                    unit="-",
                    meta={"relative": True},
                )
            )
            artifacts.append(
                ScalarMetricResult(
                    name=f"bound_operators_combined",
                    value=bound_operators_combined,
                    unit="-",
                    meta={"relative": False},
                )
            )
            artifacts.append(
                ScalarMetricResult(
                    name=f"rel_error_bound_operators_combined",
                    value=rel_error_bound_operators_combined,
                    unit="-",
                    meta={"relative": True},
                )
            )
            artifacts.append(
                ScalarMetricResult(
                    name=f"singular_value_r",
                    value=sigma_r,
                    unit="-",
                    meta={},
                )
            )
            # print(f"||X_perp @ pinv(T_r)||_2 = {X_perp_pinv_T_r_2norm}")
            # print(
            #     f"sigma_r+1/sigma_r * ||U_perp||_2* sqrt(||U_parallel||_2^2 + sigma_r^2) = {bound_Xperp_Tr}"
            # )
            # print(
            #     f"X_perp_pinv_T_r_2norm <= bound_Xperp: {X_perp_pinv_T_r_frobnorm <= bound_Xperp_Tr}; Overestimation factor: {bound_Xperp_Tr/X_perp_pinv_T_r_2norm}"
            # )

        summaries = []

        return MetricOutput(
            summaries=summaries,
            artifacts=artifacts,
        )
