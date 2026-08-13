import numpy as np
from scipy import linalg
import logging

from pgopinf.systems.ph_system import PHSystem
from pgopinf.systems.lti_system import LTISystem

# from pgopinf.models.model import Model


# %% define functions
def mass_spring_damper_system(
    n_mass,
    mass_vals=1,
    damp_vals=1,
    stiff_vals=1,
    input_vals=None,
    system_type="ph",
    use_Berlin=True,
    J_Poisson=False,
) -> PHSystem | LTISystem:
    """
    Creates a mass-spring-damper system
    :param n_mass: number of masses, i.e. second order system size (integer value)
    :param mass_vals: mass values either (n_mass,) array or scalar value (same value applied to all masses)
    :param damp_vals: damping values either (n_mass,) array or scalar value (same value applied to all dampers)
    :param stiff_vals: stiffness values either (n_mass,) array or scalar value (same value applied to all springs)
    :param input_vals: array of size (number of inputs,) creates an array of size (n_mass,len(input_vals)) with ones at indices from input_vals (index of excited mass)
    :param system: system output matrices, string with either {'ph'}, '2nd', or 'lti' for port-Hamiltonian, second order or state-space matrices
    :param J_Poisson: Transform J to Poisson form (identities on subdiagonals)
    :return: return the matrices that were requested from 'system'
    """

    # number of states
    n = 2 * n_mass
    # create mass matrix
    if isinstance(mass_vals, (list, np.ndarray)):
        M = np.diag(mass_vals)
    else:  # scalar value
        M = np.eye(n_mass) * mass_vals

    # create stiffness matrix
    if not isinstance(stiff_vals, (list, np.ndarray)):
        # scalar value
        stiff_vals = np.ones(n_mass) * stiff_vals
    K = np.zeros((n_mass, n_mass))
    K[:, :] = np.diag(stiff_vals[:])
    K[1:, 1:] += np.diag(stiff_vals[:-1])
    K += -np.diag(stiff_vals[:-1], -1)
    K += -np.diag(stiff_vals[:-1], 1)

    # create damping matrix
    if isinstance(damp_vals, (list, np.ndarray)):
        D = np.diag(damp_vals)
    elif isinstance(damp_vals, (tuple)) and len(damp_vals) == 2:
        # proportional damping
        logging.info("Using proportional Rayleigh damping.")
        alpha, beta = damp_vals
        D = alpha * M + beta * K
    else:  # scalar value
        D = np.eye(n_mass) * damp_vals

    # create input vector
    if input_vals is not None:
        if isinstance(input_vals, int):
            # single input
            assert input_vals < n_mass
            B_2nd = np.zeros((n_mass, 1))
            B_2nd[input_vals, 0] = 1
        elif isinstance(input_vals, np.ndarray):
            # array is B_2nd
            B_2nd = input_vals
        else:
            assert max(list(input_vals)) < n_mass
            B_2nd = np.zeros((n_mass, len(input_vals)))
            B_2nd[input_vals, np.arange(len(input_vals))] = 1
    else:
        B_2nd = np.zeros((n_mass))

    # number of inputs
    n_u = B_2nd.shape[-1]

    # create state-space system
    M_inv = np.linalg.inv(M)
    # A = np.block(
    #     [[np.zeros((n_mass, n_mass)), np.eye(n_mass)], [-M_inv @ K, -M_inv @ D]]
    # )

    # # scale input with M_inv
    # B = np.concatenate((np.zeros(B_2nd.shape), M_inv @ B_2nd), axis=0)

    # %%
    modeling_from = "benchmark"  # "benchmark" | "rettberg"
    if modeling_from == "rettberg":
        # convert to port-Hamiltonian system
        J = np.diag(np.ones(n_mass), n_mass)
        J += -np.diag(np.ones(n_mass), -n_mass)

        R = linalg.block_diag(np.zeros((n_mass, n_mass)), D)

        Q = linalg.block_diag(K, M_inv)

        # B_ph cancels out M with M_inv due to momentum description
        B_ph = np.concatenate((np.zeros(B_2nd.shape), B_2nd), axis=0)
    elif modeling_from == "benchmark":
        # convert mass_vals, stiff_vals, damp_vals to arrays if they are given as scalars
        if isinstance(mass_vals, (float, int)):
            mass_vals_array = np.ones(n_mass) * mass_vals
        else:
            mass_vals_array = mass_vals
        if isinstance(stiff_vals, (float, int)):
            stiff_vals_array = np.ones(n_mass) * stiff_vals
        else:
            stiff_vals_array = stiff_vals
        if isinstance(damp_vals, (float, int)):
            damp_vals_array = np.ones(n_mass) * damp_vals
        else:
            damp_vals_array = damp_vals
        # add n_mass block-diagonal blocks
        J = np.zeros((n, n))
        J_subblocks = np.array([[0, 1], [-1, 0]])
        R = np.zeros((n, n))
        Q = np.zeros((n, n))
        B_ph = np.zeros((n, n_u))
        for i in range(n_mass):
            R_subblocks = np.array([[0, 0], [0, damp_vals_array[i]]])
            if i == n_mass - 1:
                Q_subblocks = np.array(
                    [[stiff_vals_array[i], 0], [0, 1 / mass_vals_array[i]]]
                )
            else:
                Q_subblocks = np.array(
                    [
                        [stiff_vals_array[i], 0, -stiff_vals_array[i]],
                        [0, 1 / mass_vals_array[i], 0],
                        [-stiff_vals_array[i], 0, stiff_vals_array[i]],
                    ]
                )
            J[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = J_subblocks
            R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = R_subblocks
            Q[2 * i : 2 * i + 3, 2 * i : 2 * i + 3] = (
                Q[2 * i : 2 * i + 3, 2 * i : 2 * i + 3] + Q_subblocks
            )
        if input_vals is not None:
            if isinstance(input_vals, int):
                # single input
                input_vals = [input_vals]
            for j in range(n_u):
                B_ph[2 * input_vals[j] + 1, j] = 1

    H = np.eye(n)
    G = B_ph
    P = np.zeros_like(G)
    S = np.zeros((n_u, n_u))
    N = np.zeros_like(S)

    if use_Berlin:
        # bring Q to left-hand side
        H = Q.T @ H
        J = Q.T @ J @ Q
        R = Q.T @ R @ Q
        G = Q.T @ G
        P = Q.T @ P
        Q = np.eye(J.shape[0])

    if J_Poisson:
        if use_Berlin:
            # assume skew-symmetric matrix J of form [[0, E.T],[-E 0]]
            J12 = J[:n_mass, n_mass : 2 * n_mass].T
            # transformation matrix
            T = np.block(
                [
                    [np.zeros((n_mass, n_mass)), J12.T],
                    [-np.eye(n_mass), np.zeros((n_mass, n_mass))],
                ]
            )

            T_inv = np.linalg.solve(T, np.identity(T.shape[0]))

            J = T_inv.T @ J @ T_inv
            # check transformation (should be in Poisson form)
            if np.allclose(
                J,
                np.block(
                    [
                        [np.zeros((n_mass, n_mass)), np.eye(n_mass)],
                        [-np.eye(n_mass), np.zeros((n_mass, n_mass))],
                    ]
                ),
            ):
                print(f"J successfully transformed to Poisson form.")
            else:
                print(f"J could not be transformed to Poisson form.")

            H = T_inv.T @ H @ T_inv
            R = T_inv.T @ R @ T_inv
            G = T_inv.T @ G
            P = T_inv.T @ P
        else:
            # J already in desired format for Q energy
            pass

    # MSD options
    options = {
        "n_mass": n_mass,
        "mass_vals": mass_vals,
        "damp_vals": damp_vals,
        "stiff_vals": stiff_vals,
        "input_vals": input_vals,
        "system_type": system_type,
        "use_Berlin": use_Berlin,
        "J_Poisson": J_Poisson,
    }
    if system_type == "ph":
        # rename H to E
        E = H
        system = PHSystem(J=J, R=R, G=G, Q=Q, E=E, P=P, S=S, N=N)
        return system

    elif system_type == "2nd":
        return M, D, K, B_2nd
    elif system_type == "lti":
        E = H
        system = LTISystem(A=(J - R) @ Q, B=G - P, C=(G + P).T @ Q, D=S - N, E=E)
        return system
    else:
        raise Exception("system input not known. Choose either 'ph','2nd' or 'lti' ")


if __name__ == "__main__":
    msd_model = mass_spring_damper_system(
        n_mass=3,
        mass_vals=4,
        damp_vals=1,
        stiff_vals=4,
        input_vals=0,
        system_type="ph",
        use_Berlin=True,
        J_Poisson=True,
    )
