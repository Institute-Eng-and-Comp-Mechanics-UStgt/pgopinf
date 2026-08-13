import numpy as np
import scipy
import logging
from timeit import default_timer as timer
from tqdm import tqdm

import scipy.sparse


def implicit_midpoint(
    E, A, t, x_init, B=None, u_mid=None, decomp_option="lu", verbose=False
):
    """
    Calculate time integration of a linear ODE system using the implicit midpoint rule.

    This function solves the ODE system defined by:
        E * dz_dt = A * z + B * u

    Parameters:
    -----------
    E : numpy.ndarray
        Descriptor matrix of the ODE system.
    A : numpy.ndarray
        System matrix of the ODE system.
    t : numpy.ndarray
        Time vector with equidistant time steps.
    z_init : numpy.ndarray
        Initial state vector.
    B : numpy.ndarray, optional
        Input matrix. Defaults to None, which sets B to zero.
    u_mid : numpy.ndarray or callable (dependent on t, return np.ndarray), optional
        Input function at time midpoints. Defaults to None, which sets u to zero.
    decomp_option : str, optional
        Option for matrix decomposition or solving. Choices are 'lu' for LU decomposition or
        'linalg_solve' for solving without decomposition. Defaults to 'lu'.

    Returns:
    --------
    z : numpy.ndarray
        Array of state vectors at each time step.
    t : numpy.ndarray
        Array of time points.

    Theory:
    --------
    we got a pH-system E*Dx = (J-R)*Q*x + B*u
    we define A:=(J-R)*Q and the RHS as f(t,x)
    use the differential slope equation at midpoint
    (x(t+h)-x(t))/h=Dx(t+h/2)=E^-1 * f(t+h/2,x(t+h/2))
    since x(t+h/2) is unknown we use the approximation
    x(t+h/2) = 1/2*(x(t)+x(t+h))
    insert the linear system into the differential equation leads to
    x(t+h) = x(t) + h * E^-1 *(1/2*A*(x(t)+x(t+h))+ B*u(t+h/2))
    reformulate the equation to
    (E-h/2*A)x(t+h) = (E+h/2*A)*x(t) + h*B*u(t+h/2)
    solve the linear equation system, e.g. via LU-decomposition

    The linear system is solved for x(t + h) using the specified decomposition method.

    Examples:
    ---------
    >>> E = np.array([[1, 0], [0, 1]])
    >>> A = np.array([[0, -1], [1, 0]])
    >>> t = np.linspace(0, 10, 100)
    >>> z_init = np.array([1, 0])
    >>> z, t = implicit_midpoint(E, A, t, z_init)
    """

    # number of time samples
    n_t = len(t)
    step_size = t[1] - t[0]
    t_mid = np.zeros(n_t)
    # initialize state array
    n_f = len(x_init)
    x = np.zeros((n_f, n_t))
    x[:, 0] = x_init

    if n_f != A.shape[0]:
        raise ValueError(
            f"Initial condition and number of states must be the same. Currently, they are n_IC={n_f} and n_states={A.shape[0]}."
        )

    # default input values
    if B is None:
        n_u = 1
        B = np.zeros((n_f, n_u))
        u_mid = np.zeros((n_u, n_t))

    if verbose:
        # time different decomposition methods
        start_time_solver = timer()

    if decomp_option == "lu":
        # LU decomposition of lhs
        M = E - (step_size / 2) * A
        lhs_is_sparse = False
        if scipy.sparse.issparse(M):
            lhs_is_sparse = True
            # convert sparse to dense
            if M.getformat() == "csr":
                # convert to csc to prevent splu giving a warning
                M = scipy.sparse.csc_matrix(M)
            invM_sparse = scipy.sparse.linalg.splu(M)
        else:
            lu_dense, piv_dense = scipy.linalg.lu_factor(M)
    elif decomp_option == "linalg_solve":
        M = E - (step_size / 2) * A
    else:
        raise ValueError(f"Decomposition option {decomp_option} is unknown.")

    # loop over time samples
    A_imr = E + (step_size / 2) * A
    try:
        for i_t in tqdm(range(n_t - 1)):
            t_mid[i_t] = t[0] + ((i_t + 1) - 0.5) * step_size
            if decomp_option == "lu":
                if lhs_is_sparse:
                    x[:, i_t + 1] = invM_sparse.solve(
                        A_imr @ x[:, i_t] + step_size * B @ input(u_mid, t_mid, i_t),
                    )
                else:
                    x[:, i_t + 1] = scipy.linalg.lu_solve(
                        (lu_dense, piv_dense),
                        A_imr @ x[:, i_t] + step_size * B @ input(u_mid, t_mid, i_t),
                    )
            elif decomp_option == "linalg_solve":
                x[:, i_t + 1] = np.linalg.solve(
                    M,
                    A_imr @ x[:, i_t] + step_size * B @ input(u_mid, t_mid, i_t),
                )
            else:
                raise ValueError(f"Decomposition option {decomp_option} is unknown.")
    except ValueError as value_error:
        if np.isnan(x[:, i_t]).any():
            # check if nans are contained in the solution
            logging.info(
                f"NaN values occured in the solution at time step {i_t}. This might be due to an unstable system."
            )
        else:
            logging.error(
                f"ValueError {value_error} occured while performing the implicit midpoint rule."
            )

    if verbose:
        end_time_solver = timer()
        solver_time = end_time_solver - start_time_solver
        logging.info(f"Calculation time of ODE solver has been {solver_time} s.\n")

    return x, t[:, np.newaxis]


def input(u_mid, t_mid, i_t):
    """
    u_mid: array or callable
    t: time_array
    i_t: time iteration
    """
    if callable(u_mid):
        return u_mid(t_mid[i_t])
    elif isinstance(u_mid, np.ndarray):
        return u_mid[:, i_t]
    else:
        raise ValueError(f"Unknown type for {u_mid}.")
