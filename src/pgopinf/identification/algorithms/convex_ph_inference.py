import logging
import numpy as np

import cvxpy as cp
from typing import Literal

from pgopinf.data.utils.data_manipulating import (
    get_kron_x_dx_data,
    get_kron_x_x_data,
    get_supplied_power,
)
from pgopinf.systems.utils.unstack import unstack
from pgopinf.numerics.linalg.definiteness import check_spsd
from pgopinf.numerics.linalg.definiteness import project_spd
from pgopinf.numerics.linalg.symmetry import is_skewsym

logger = logging.getLogger(__name__)


def convex_ph_inference(
    x,
    y,
    u,
    dxdt=None,
    delta_t=None,
    J_known=None,
    add_regularization: bool = False,
    lambdas: float | int | np.ndarray | list = 1e-6,
    add_dissipation_inequality_cost=False,
    no_feedthrough=False,
    solver="mosek",
    project_psd=False,
    return_system_type: Literal["ph", "lti"] = "ph",
    return_solver_stats: bool = False,
    accept_unknown_mosek=False,
):
    """Identify a port-Hamiltonian system by convex optimization.

    The method fits pH operators to state, output, input, and derivative data.
    It solves a convex least-squares problem with structural constraints on the
    interconnection and dissipation operators. If derivatives are not supplied,
    finite differences are formed from adjacent state snapshots and the data are
    shifted to midpoint samples.

    Parameters
    ----------
    x : ndarray, shape (n, n_samples)
        State snapshots in sample format.
    y : ndarray, shape (n_y, n_samples)
        Output snapshots in sample format.
    u : ndarray, shape (n_u, n_samples)
        Input snapshots in sample format.
    dxdt : ndarray, optional, shape (n, n_samples)
        State derivative snapshots. If omitted, derivatives are computed from
        ``x`` and ``delta_t``.
    delta_t : float, optional
        Time step used to approximate derivatives when ``dxdt`` is ``None``.
    J_known : ndarray, optional, shape (n, n)
        Known skew-symmetric interconnection matrix. If supplied, it is fixed
        in the optimization.
    add_regularization : bool, optional
        If ``True``, add L1 regularization to selected inferred matrices.
    lambdas : float, int, ndarray, or list, optional
        Regularization weights. Scalars are applied to all seven matrix groups;
        arrays and lists must contain weights for ``H``, ``J``, ``R``, ``G``,
        ``P``, ``S``, and ``N``.
    add_dissipation_inequality_cost : bool, optional
        If ``True``, add a dissipation-inequality residual to the objective.
    no_feedthrough : bool, optional
        If ``True``, enforce ``P = 0``, ``S = 0``, and ``N = 0``.
    solver : str, optional
        CVXPY solver name. Recognized strings include ``"mosek"``, ``"scs"``,
        and ``"clarabel"``; other values are passed through to CVXPY.
    project_psd : bool, optional
        If ``True``, project inferred ``R`` and ``H`` matrices onto the
        positive semidefinite cone when the post-solve checks fail.
    return_system_type : {"ph", "lti"}, optional
        Type of system to return. ``"ph"`` returns a ``PHSystem`` and ``"lti"``
        returns an equivalent ``LTISystem``.
    return_solver_stats : bool, optional
        If ``True`` and ``return_system_type`` is ``"ph"``, also return CVXPY
        solver statistics.
    accept_unknown_mosek : bool, optional
        If ``True`` and MOSEK is used, pass ``accept_unknown=True`` to CVXPY.

    Returns
    -------
    PHSystem
        Identified pH system when ``return_system_type="ph"`` and
        ``return_solver_stats=False``.
    tuple[PHSystem, SolverStats]
        Identified pH system and CVXPY solver statistics when
        ``return_system_type="ph"`` and ``return_solver_stats=True``.
    LTISystem
        Identified LTI system when ``return_system_type="lti"``. In this code
        path, solver statistics are not returned.

    Raises
    ------
    AssertionError
        If ``dxdt`` is ``None`` and ``delta_t`` is not provided, or if
        ``J_known`` is supplied but not skew-symmetric.
    """

    # data at mid timepoints + get dXdt
    if dxdt is None:
        assert delta_t is not None
        dxdt = (x[:, 1:] - x[:, :-1]) / delta_t
        x = 1 / 2 * (x[:, 1:] + x[:, :-1])
        u = 1 / 2 * (u[:, 1:] + u[:, :-1])
        y = 1 / 2 * (y[:, 1:] + y[:, :-1])

    use_midpoints = False
    # For MSD example midpoints lead to worse results
    if use_midpoints:
        dxdt = 1 / 2 * (dxdt[:, 1:] + dxdt[:, :-1])
        x = 1 / 2 * (x[:, 1:] + x[:, :-1])
        u = 1 / 2 * (u[:, 1:] + u[:, :-1])
        y = 1 / 2 * (y[:, 1:] + y[:, :-1])

    n = x.shape[0]
    n_u = u.shape[0]

    # state equation level
    H_inf = cp.Variable((n, n), symmetric=True)
    R_inf = cp.Variable((n, n), symmetric=True)
    J_inf = cp.Variable((n, n))
    G_inf = cp.Variable((n, n_u))
    epsilon = 1e-10
    constraints = [H_inf - epsilon * np.eye(n) >> 0]
    if no_feedthrough:
        P_inf = np.zeros((n, n_u))
        N_inf = np.zeros((n_u, n_u))
        S_inf = np.zeros((n_u, n_u))

        constraints += [J_inf == -J_inf.T]
        constraints += [R_inf - epsilon * np.eye(n) >> 0]
        constraints = [H_inf - epsilon * np.eye(n) >> 0]

        optimization_cost = cp.norm(
            H_inf @ dxdt - (J_inf - R_inf) @ x - G_inf @ u, "fro"
        ) + cp.norm(y - G_inf.T @ x, "fro")

    else:
        P_inf = cp.Variable((n, n_u))
        N_inf = cp.Variable((n_u, n_u))
        S_inf = cp.Variable((n_u, n_u), symmetric=True)

        # input-ouput equation level
        Z_data = cp.bmat([[H_inf @ dxdt], [-y]])
        T_data = cp.bmat([[x], [u]])
        R_op = cp.bmat([[R_inf, P_inf], [P_inf.T, S_inf]])

        if J_known is not None:
            # J known
            assert np.allclose(J_known, -J_known.T)
            J_op = cp.bmat([[J_known, G_inf], [-G_inf.T, N_inf]])
        else:
            J_op = cp.bmat([[J_inf, G_inf], [-G_inf.T, N_inf]])

        constraints += [J_op == -J_op.T]
        constraints += [R_op - epsilon * np.eye(n + n_u) >> 0]

        # always part of optimization: The ODE cost
        optimization_cost = cp.norm(Z_data - (J_op - R_op) @ T_data, "fro")

    if add_dissipation_inequality_cost:
        # Berlin system (e.g. reduced); Q=eye;
        # add_dissipation_as = "cost"  # 'constraint' | 'cost'
        dissipation_inequality_cost = 0
        kron_x_dx = get_kron_x_dx_data(x, dxdt, symmetric=True)
        power_in_storage = H_inf[np.triu_indices(n)] @ kron_x_dx.T
        supplied_power = get_supplied_power(y, u)
        kron_xu_xu = get_kron_x_x_data(np.vstack((x, u)), symmetric=True)
        dissipated_power = R_op[np.triu_indices(n + n_u)] @ kron_xu_xu.T
        dissipation_inequality_cost += cp.norm(
            power_in_storage + dissipated_power - np.squeeze(supplied_power, axis=1)
        )
        optimization_cost += dissipation_inequality_cost

    if add_regularization:
        regularization_term = process_regularization(
            lambdas, H_inf, J_inf, R_inf, G_inf, P_inf, S_inf, N_inf
        )
        optimization_cost += regularization_term
        # minimize_IO_level = cp.Minimize(
        #     cp.norm(Z_data - (J_op - R_op) @ T_data, "fro") + regularization_term
        # )
    else:
        pass
        # minimize_IO_level = cp.Minimize(cp.norm(Z_data - (J_op - R_op) @ T_data, "fro"))
    minimize_IO_level = cp.Minimize(optimization_cost)
    prob = cp.Problem(minimize_IO_level, constraints)

    # solve optimization problem
    solver_kwargs = {}
    if solver.lower() == "mosek":
        solver = cp.MOSEK
        if accept_unknown_mosek:
            solver_kwargs = {"accept_unknown": True}
    elif solver.lower() == "scs":
        solver = cp.SCS
        solver_kwargs = {"max_iters": 10000}
    elif solver.lower() == "clarabel":
        solver = cp.CLARABEL
        solver_kwargs = {"max_iter": 50}
    else:
        # try to use input
        solver = solver
    logger.info(f"Using solver {solver} to solve optimization problem.")
    prob.solve(solver=solver, verbose=True, **solver_kwargs)

    if no_feedthrough:
        J_check_value = J_inf.value
        R_check_value = R_inf.value
    else:
        J_check_value = J_op.value
        R_check_value = R_op.value
    H_inf_value = H_inf.value
    # check properties
    logger.info(f"J operator is skew-symmetric {is_skewsym(J_check_value)}")
    R_is_psd = check_spsd(R_check_value)
    logger.info(f"R operator is symmetric and positive definite: {R_is_psd}")
    if project_psd and not R_is_psd:
        logger.info(f"Projecting R operator onto pos. semi-def. cone.")
        R_check_value = project_spd(R_check_value)
    H_is_psd = check_spsd(H_inf_value)
    logger.info(f"H operator is symmetric and positive definite: {H_is_psd}")
    if project_psd and not H_is_psd:
        logger.info(f"Projecting H onto pos. semi-def. cone.")
        H_inf_value = project_spd(H_inf_value)

    if no_feedthrough:
        R_inf_value = R_check_value
        P_inf_value = np.zeros((n, n_u))
        S_inf_value = np.zeros((n_u, n_u))
        N_inf_value = np.zeros((n_u, n_u))
        logger.info(
            f"Norm of residual for identified operators {np.linalg.norm(H_inf_value @ dxdt - (J_inf.value - R_inf_value) @ x - G_inf.value @ u, 'fro') + np.linalg.norm(y - G_inf.value.T @ x, 'fro')}"
        )
    else:
        # get system matrices from combined operator
        # J, G, _, N = unstack(JJ, n, no_feedtrough)
        R_inf_value, P_inf_value, _, S_inf_value = unstack(
            R_check_value, n, no_feedthrough
        )
        P_inf_value = P_inf_value
        S_inf_value = S_inf_value
        N_inf_value = N_inf.value
        logger.info(
            f"Norm of residual for identified operators {np.linalg.norm(Z_data.value -(J_check_value - R_check_value)@T_data.value,'fro')}"
        )
    if return_system_type.lower() == "lti":
        from pgopinf.systems.lti_system import LTISystem

        lti_system = LTISystem(
            A=(J_inf.value - R_inf_value),
            B=G_inf.value - P_inf_value,
            C=(G_inf.value + P_inf_value).T,
            D=S_inf_value - N_inf_value,
            E=H_inf_value,
        )
        return lti_system
    elif return_system_type.lower() == "ph":
        from pgopinf.systems.ph_system import PHSystem

        ph_system = PHSystem(
            J=J_inf.value,
            R=R_inf_value,
            G=G_inf.value,
            Q=np.eye(n),
            E=H_inf_value,
            P=P_inf_value,
            S=S_inf_value,
            N=N_inf_value,
        )

    if return_solver_stats:
        return ph_system, prob.solver_stats
    else:
        return ph_system


def process_regularization(lambdas, H, J, R, G, P, S, N):
    """Create an L1 regularization term for the convex objective.

    Parameters
    ----------
    lambdas : float, int, ndarray, or list
        Regularization weights for ``H``, ``J``, ``R``, ``G``, ``P``, ``S``,
        and ``N``. Scalars are broadcast to all matrix groups.
    H, J, R, G, P, S, N
        CVXPY variables or constant arrays representing the inferred matrix
        groups. Only CVXPY variables are regularized.

    Returns
    -------
    cvxpy.Expression
        Sum of weighted L1 norms for variable matrix groups.

    Raises
    ------
    AssertionError
        If a list or ndarray of weights does not contain seven entries.
    """

    expected_size_lambdas = 7
    if isinstance(lambdas, float) or isinstance(lambdas, int):
        lambdas = lambdas * np.ones((expected_size_lambdas,))
    elif isinstance(lambdas, list):
        assert len(lambdas) == expected_size_lambdas
        lambdas = np.array(lambdas)
    elif isinstance(lambdas, np.ndarray):
        # expected shape (expected_size_lambdas,)
        assert lambdas.shape[0] == expected_size_lambdas

    matrix_names = ["H", "J", "R", "G", "P", "S", "N"]
    regularization_term = 0
    for i_matrix, matrix in enumerate((H, J, R, G, P, S, N)):
        if isinstance(matrix, cp.Variable):
            logger.info(
                f"Using regularization {lambdas[i_matrix]} for {matrix_names[i_matrix]}."
            )
            regularization_term += lambdas[i_matrix] * cp.norm(matrix, 1)

    return regularization_term
