from __future__ import annotations  # deprecated for Python 3.14+

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import control as ct

from pymor.models.iosys import LTIModel as PymorLTIModel
from pgopinf.numerics.discretization.integrators import (
    implicit_midpoint,
)
from pgopinf.systems.utils.stable_decomposition import (
    stable_part_lti,
)

# from phidentification.systems.ph_system import PHSystem
# from phidentification.systems.lti_system import LTISystem


class LTISystem:
    """Continuous-time descriptor LTI system.

    The system is represented as ``E xdot = A x + B u`` and
    ``y = C x + D u``.
    """

    def __init__(self, A: np.ndarray, B=None, C=None, D=None, E=None) -> None:
        """Initialize an LTI system.

        Parameters
        ----------
        A : ndarray or sparse matrix, shape (n, n)
            State matrix.
        B : ndarray or sparse matrix, optional, shape (n, n_u)
            Input matrix. Defaults to a zero single-input matrix.
        C : ndarray or sparse matrix, optional, shape (n_y, n)
            Output matrix. Defaults to a zero single-output matrix.
        D : ndarray or sparse matrix, optional, shape (n_y, n_u)
            Feedthrough matrix. Defaults to zero.
        E : ndarray or sparse matrix, optional, shape (n, n)
            Descriptor matrix. Defaults to the identity.
        """

        # system matrix
        assert A.shape[0] == A.shape[1]  # quadratic
        self.A = A
        self.n = A.shape[0]

        # input
        if B is None:
            self.B = np.zeros((self.n, 1))
        else:
            self.B = B
        self.n_u = self.B.shape[1]

        # output
        if C is None:
            self.C = np.zeros((1, self.n))
        else:
            self.C = C
        self.n_y = self.C.shape[0]

        # feedthrough
        if D is None:
            self.D = np.zeros((self.n_y, self.n_u))
        else:
            self.D = D

        # descriptor part
        if E is None:
            self.E = np.identity(self.n)
        else:
            self.E = E

        if scipy.sparse.issparse(self.A) or scipy.sparse.issparse(self.E):
            self.issparse = True
            # convert all matrices to sparse
            self.convert_to_sparse()
        else:
            self.issparse = False
            # convert all matrices to dense
            self.convert_to_dense()

    @property
    def abcde(self):
        """Return the state-space matrices.

        Returns
        -------
        tuple
            ``(A, B, C, D, E)``.
        """
        return self.A, self.B, self.C, self.D, self.E

    @property
    def matrices(self) -> dict[str, np.ndarray]:
        """State-space matrices keyed by name.

        Returns
        -------
        dict of str to ndarray
            Dictionary containing ``A``, ``B``, ``C``, ``D``, and ``E``.
        """
        return {"A": self.A, "B": self.B, "C": self.C, "D": self.D, "E": self.E}

    @property
    def kind(self) -> str:
        """System kind label.

        Returns
        -------
        str
            The string ``"lti"``.
        """
        return "lti"

    @property
    def EinvA(self):
        """Return ``E^{-1} A``.

        Returns
        -------
        ndarray or sparse matrix
            Matrix product obtained by solving with ``E`` when needed.
        """

        if scipy.sparse.issparse(self.E):
            if np.allclose(self.E.toarray(), np.eye(self.E.shape[0])):
                return self.A
            E_inv_A = scipy.sparse.linalg.spsolve(self.E, self.A)
        else:
            if np.allclose(self.E, np.eye(self.E.shape[0])):
                return self.A
            E_inv_A = np.linalg.solve(self.E, self.A)
        return E_inv_A

    @property
    def EinvB(self):
        """Return ``E^{-1} B``.

        Returns
        -------
        ndarray or sparse matrix
            Matrix product obtained by solving with ``E`` when needed.
        """
        if scipy.sparse.issparse(self.E):
            if np.allclose(self.E.toarray(), np.eye(self.E.shape[0])):
                return self.B
            E_inv_B = scipy.sparse.linalg.spsolve(self.E, self.B)
        else:
            if np.allclose(self.E, np.eye(self.E.shape[0])):
                return self.B
            E_inv_B = np.linalg.solve(self.E, self.B)
        return E_inv_B

    def convert_to_dense(self):
        """Convert all system matrices to dense arrays in place."""
        for var in ["A", "B", "C", "D", "E"]:
            matrix = getattr(self, var)
            if scipy.sparse.issparse(matrix):
                setattr(self, var, matrix.toarray())

    def convert_to_sparse(self):
        """Convert all system matrices to CSR sparse matrices in place."""
        for var in ["A", "B", "C", "D", "E"]:
            matrix = getattr(self, var)
            if not scipy.sparse.issparse(matrix):
                setattr(self, var, scipy.sparse.csr_matrix(matrix))

    def reduce(self, V, W):
        """
        Docstring for reduce
        """
        A_red = W.T @ self.A @ V
        B_red = W.T @ self.B
        C_red = self.C @ V
        D_red = self.D
        E_red = W.T @ self.E @ V
        if np.allclose(E_red, np.eye(E_red.shape[0])):
            E_red = np.eye(E_red.shape[0])

        return LTISystem(A=A_red, B=B_red, C=C_red, D=D_red, E=E_red)

    def solve(self, t, U, x0, integrator="imr", decomp_option="lu", return_dXdt=False):
        """Simulate the LTI system on a time grid.

        Parameters
        ----------
        t : ndarray, shape (n_t,)
            Time grid.
        U : ndarray or callable
            Input values with shape ``(n_u, n_t)`` or callable input.
        x0 : ndarray, shape (n,)
            Initial state.
        integrator : {"imr"}, optional
            Time integration method.
        decomp_option : str, optional
            Linear solver decomposition option passed to the integrator.
        return_dXdt : bool, optional
            If ``True``, also return state derivatives.

        Returns
        -------
        tuple
            ``(x, y)`` or ``(x, y, dxdt)`` depending on ``return_dXdt``.
        """
        if integrator.lower() == "imr":
            if callable(U):
                U_mid = U
            else:
                U_mid = (U[:, 1:] + U[:, :-1]) / 2
            # implicit midpoint rule
            x, _ = implicit_midpoint(
                self.E,
                self.A,
                t.ravel(),
                x0,
                B=self.B,
                u_mid=U_mid,
                decomp_option=decomp_option,
                verbose=True,
            )

        else:
            raise ValueError(f"Unknown integrator option {integrator}.")

        y = self.C @ x + self.D @ U

        if return_dXdt:
            solve_lse = True
            if solve_lse:
                # solve linear system of equations
                if scipy.sparse.issparse(self.E):
                    dxdt = scipy.sparse.linalg.spsolve(self.E, self.A @ x + self.B @ U)
                else:
                    dxdt = np.linalg.solve(self.E, self.A @ x + self.B @ U)
            else:
                # numerically calculate gradient (for testing purposes)
                dt = t[1] - t[0]
                dxdt = np.gradient(x, dt, axis=1)
            return x, y, dxdt
        else:
            return x, y

    @classmethod
    def from_matrices(cls, A, B=None, C=None, D=None, E=None):
        """Create an LTI system from state-space matrices.

        Parameters
        ----------
        A, B, C, D, E : ndarray or sparse matrix
            State-space matrices.

        Returns
        -------
        LTISystem
            Constructed LTI system.
        """
        return cls(A=A, B=B, C=C, D=D, E=E)

    @classmethod
    def from_pymor(cls, phlti_pymor_model):
        """Create an LTI system from a pyMOR LTI model.

        Parameters
        ----------
        phlti_pymor_model : pymor.models.iosys.LTIModel
            pyMOR model providing ``to_abcde_matrices``.

        Returns
        -------
        LTISystem
            Constructed LTI system.
        """
        assert isinstance(phlti_pymor_model, PymorLTIModel)
        A, B, C, D, E = phlti_pymor_model.to_abcde_matrices()
        return cls(A, B, C, D, E)

    def eigvals(self):
        """Compute the eigenvalues of the LTI system."""
        if scipy.sparse.issparse(self.A) or scipy.sparse.issparse(self.E):
            # calculate only the k largest real part of all eigenvalues
            k = 3
            sigma = 0  # shift for shift-invert mode
            if np.allclose(self.E.toarray(), np.eye(self.E.shape[0])):
                eigvals = scipy.sparse.linalg.eigs(
                    self.A, k=k, which="LR", sigma=sigma, return_eigenvectors=False
                )
            else:
                eigvals = scipy.sparse.linalg.eigs(
                    self.A,
                    M=self.E,
                    k=k,
                    which="LR",
                    sigma=sigma,
                    return_eigenvectors=False,
                )
        else:
            eigvals = scipy.linalg.eigvals(a=self.A, b=self.E)
        return eigvals

    def isstable(self, tol=1e-10):
        """Check if the LTI system is stable."""
        eigvals = self.eigvals()
        if np.all(np.real(eigvals) < -tol):
            return True
        else:
            return False

    def hinf_norm(
        self, mu=None, return_fpeak=False, ab13dd_equilibrate=False, tol=1e-10
    ):
        """Compute the H-infinity norm of the LTI system."""
        pymor_model = PymorLTIModel.from_matrices(
            A=self.A, B=self.B, C=self.C, D=self.D, E=self.E
        )
        if not self.isstable(tol=tol):
            return float("inf")
        return pymor_model.hinf_norm(
            mu=mu,
            return_fpeak=return_fpeak,
            ab13dd_equilibrate=ab13dd_equilibrate,
            tol=tol,
        )

    def hinf_error(
        self,
        second_system: LTISystem | PHSystem,
        mu=None,
        return_fpeak=False,
        ab13dd_equilibrate=False,
        tol=1e-10,
    ):
        """Compute the H-infinity norm of the LTI system."""
        pymor_model = PymorLTIModel.from_matrices(
            A=self.A, B=self.B, C=self.C, D=self.D, E=self.E
        )
        pymor_model_2 = PymorLTIModel.from_matrices(
            A=second_system.A,
            B=second_system.B,
            C=second_system.C,
            D=second_system.D,
            E=second_system.E,
        )
        pymor_model_diff = pymor_model - pymor_model_2
        if not self.isstable(tol=tol) or not second_system.isstable(tol=tol):
            return float("inf")
        return pymor_model_diff.hinf_norm(
            mu=mu,
            return_fpeak=return_fpeak,
            ab13dd_equilibrate=ab13dd_equilibrate,
            tol=tol,
        )

    def h2_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False, tol=1e-10):
        """Compute the H2 norm of the LTI system."""
        pymor_model = PymorLTIModel.from_matrices(
            A=self.A, B=self.B, C=self.C, D=self.D, E=self.E
        )
        return pymor_model.h2_norm(
            mu=mu,
            return_fpeak=return_fpeak,
            ab13dd_equilibrate=ab13dd_equilibrate,
            tol=tol,
        )

    def stable_decomposition(self, discrete=False, tol=1e-9):
        """Return the stable subsystem.

        Parameters
        ----------
        discrete : bool, optional
            If ``True``, use discrete-time stability.
        tol : float, optional
            Stability tolerance.

        Returns
        -------
        LTISystem
            LTI realization containing only stable modes.
        """
        A_s, B_s, C_s, D_s, stable_eigs, unstable_eigs = stable_part_lti(
            self.EinvA, self.EinvB, self.C, self.D, discrete=discrete, tol=tol
        )

        return LTISystem(A=A_s, B=B_s, C=C_s, D=D_s)

    def singular_value_plot(self, name="sigma_plot.png", return_only_figure=False):
        """Create a singular-value plot of the transfer function.

        Parameters
        ----------
        name : str, optional
            Output filename used when ``return_only_figure`` is ``False``.
        return_only_figure : bool, optional
            If ``True``, return the matplotlib figure instead of saving it.

        Returns
        -------
        matplotlib.figure.Figure or None
            Figure when requested; otherwise ``None`` after saving.
        """
        #
        import matplotlib as mpl

        sys = ct.ss(
            (
                self.EinvA
                if not scipy.sparse.issparse(self.EinvA)
                else self.EinvA.toarray()
            ),
            (
                self.EinvB
                if not scipy.sparse.issparse(self.EinvB)
                else self.EinvB.toarray()
            ),
            (self.C if not scipy.sparse.issparse(self.C) else self.C.toarray()),
            (self.D if not scipy.sparse.issparse(self.D) else self.D.toarray()),
        )
        control_plot = ct.singular_values_plot(
            [sys], Hz=True, figure=mpl.pyplot.figure(figsize=(8, 6))
        )
        if return_only_figure:
            return control_plot.figure
        else:
            control_plot.figure.savefig(name)

    def minimal_realization(self):
        """Compute a minimal realization.

        Raises
        ------
        NotImplementedError
            Always raised for ``LTISystem``.
        """
        raise NotImplementedError(
            "Minimal realization not implemented yet for LTISystem."
        )
