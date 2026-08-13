import numpy as np
import scipy
import logging

import scipy.sparse
import control as ct

from pymor.models.iosys import PHLTIModel

from pgopinf.numerics.linalg.cholesky import (
    lrcholesky,
)
from pgopinf.systems.lti_system import LTISystem
from pgopinf.numerics.linalg.definiteness import check_spsd
from pgopinf.numerics.linalg.symmetry import is_skewsym

logger = logging.getLogger(__name__)


class PHSystem(LTISystem):
    """Port-Hamiltonian system in descriptor form.

    The pH matrices define the associated LTI realization through
    ``A = (J - R) Q``, ``B = G - P``, ``C = (G + P).T Q``, and
    ``D = S - N``.
    """

    def __init__(self, J, R, G, Q=None, E=None, P=None, S=None, N=None) -> None:
        """Initialize a port-Hamiltonian system.

        Parameters
        ----------
        J : ndarray or sparse matrix, shape (n, n)
            Skew-symmetric interconnection matrix.
        R : ndarray or sparse matrix, shape (n, n)
            Dissipation matrix.
        G : ndarray or sparse matrix, shape (n, n_u)
            Input matrix.
        Q : ndarray or sparse matrix, optional, shape (n, n)
            Energy matrix. Defaults to the identity.
        E : ndarray or sparse matrix, optional, shape (n, n)
            Descriptor matrix. Defaults to the identity.
        P : ndarray or sparse matrix, optional, shape (n, n_u)
            Port coupling matrix. Defaults to zero.
        S : ndarray or sparse matrix, optional, shape (n_u, n_u)
            Symmetric feedthrough dissipation matrix. Defaults to zero.
        N : ndarray or sparse matrix, optional, shape (n_u, n_u)
            Skew-symmetric feedthrough matrix. Defaults to zero.
        """
        self.J = J
        self.n = J.shape[0]
        self.R = R
        self.G = G
        assert self.G.shape[0] == self.n
        self.n_u = G.shape[1]
        self.n_y = self.n_u  # same number of inputs and outputs
        if Q is None:
            self.Q = np.identity(self.n)
        else:
            self.Q = Q
        if E is None:
            self.E = np.identity(self.n)
        else:
            self.E = E
        if P is None:
            self.P = np.zeros((self.n, self.n_u))
        else:
            self.P = P
        if S is None:
            self.S = np.zeros((self.n_y, self.n_u))
        else:
            self.S = S
        if N is None:
            self.N = np.zeros((self.n_y, self.n_u))
        else:
            self.N = N

        if not self.check_pH_properties():
            logger.error(
                f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n THE SYSTEM IS NOT PH!!!!!!!!\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
            )
            self.is_ph = False
        else:
            self.is_ph = True
        # assert self.check_pH_properties()

        if (
            scipy.sparse.issparse(self.J)
            or scipy.sparse.issparse(self.R)
            or scipy.sparse.issparse(self.Q)
        ):
            # sparse matrices
            self.issparse = True
            # convert all matrices to sparse
            self.convert_to_sparse()
        else:
            self.issparse = False
            # convert all matrices to dense
            self.convert_to_dense()

        self.categorize_ph_system()

        # state-space form
        A = (self.J - self.R) @ self.Q
        B = self.G - self.P
        C = (self.G + self.P).T @ self.Q
        D = self.S - self.N
        super().__init__(A, B, C, D, E)

    def categorize_ph_system(self):
        """Set boolean flags describing the pH representation.

        The method updates flags such as ``is_standard``, ``is_berlin``,
        ``is_descriptor``, ``is_normalized``, and ``is_poisson`` in place.
        """
        # categorize pH-system
        self.is_berlin = False  # Q=I and E!=I
        self.is_standard = False  # E=I and Q anything
        self.is_normalized = False  # Q=I and E=I
        self.is_descriptor = False  # Q!=I and E!=I
        self.is_poisson = False  # J = skew-identity [[0, I],[-I,0]]
        if np.allclose(
            np.eye(self.n),
            self.E.toarray() if scipy.sparse.issparse(self.E) else self.E,
        ):
            self.is_standard = True
            if np.allclose(
                np.eye(self.n),
                self.Q.toarray() if scipy.sparse.issparse(self.Q) else self.Q,
            ):
                self.is_normalized = True
                self.is_berlin = True
        else:
            if np.allclose(
                np.eye(self.n),
                self.Q.toarray() if scipy.sparse.issparse(self.Q) else self.Q,
            ):
                self.is_berlin = True
            else:
                self.is_descriptor = True

        # check Poisson form
        if self.n % 2 == 0:  # only even numbers
            n_half = int(self.n / 2)
            if np.allclose(
                np.block(
                    [
                        [np.zeros((n_half, n_half)), np.eye(n_half)],
                        [-np.eye(n_half), np.zeros((n_half, n_half))],
                    ]
                ),
                self.J.toarray() if scipy.sparse.issparse(self.J) else self.J,
            ):
                self.is_poisson = True

    @property
    def matrices(self) -> dict[str, np.ndarray]:
        """Port-Hamiltonian matrices keyed by name.

        Returns
        -------
        dict of str to ndarray
            Dictionary containing ``J``, ``R``, ``G``, ``Q``, ``E``, ``P``,
            ``S``, and ``N``.
        """
        return {
            "J": self.J,
            "R": self.R,
            "G": self.G,
            "Q": self.Q,
            "E": self.E,
            "P": self.P,
            "S": self.S,
            "N": self.N,
        }

    @classmethod
    def from_matrices(cls, J, R, G, Q=None, E=None, P=None, S=None, N=None):
        """Create a pH system from pH matrices.

        Parameters
        ----------
        J, R, G, Q, E, P, S, N : ndarray or sparse matrix
            Port-Hamiltonian matrices.

        Returns
        -------
        PHSystem
            Constructed pH system.
        """
        return cls(J=J, R=R, G=G, Q=Q, E=E, P=P, S=S, N=N)

    @property
    def kind(self) -> str:
        """System kind label.

        Returns
        -------
        str
            The string ``"ph"``.
        """
        return "ph"

    def create_block_pH_matrices(self):
        """Create block matrices used in pH property checks.

        Returns
        -------
        tuple
            ``(Rcal, Jcal)`` where ``Rcal = [[R, P], [P.T, S]]`` and
            ``Jcal = [[J, G], [-G.T, N]]``.
        """
        if any(
            [
                scipy.sparse.issparse(self.J),
                scipy.sparse.issparse(self.R),
                scipy.sparse.issparse(self.P),
                scipy.sparse.issparse(self.S),
                scipy.sparse.issparse(self.G),
                scipy.sparse.issparse(self.N),
            ]
        ):
            # sparse matrices
            self.Rcal = scipy.sparse.block_array(
                ([[self.R, self.P], [self.P.T, self.S]])
            )
            self.Jcal = scipy.sparse.block_array(
                ([[self.J, self.G], [-self.G.T, self.N]])
            )
        else:
            # dense matrices
            self.Rcal = np.block(([[self.R, self.P], [self.P.T, self.S]]))
            self.Jcal = np.block(([[self.J, self.G], [-self.G.T, self.N]]))
        return self.Rcal, self.Jcal

    def check_pH_properties(self):
        """Check dissipativity, skew symmetry, and Hamiltonian positivity.

        Returns
        -------
        bool
            ``True`` if the pH block matrices satisfy the required properties.
        """
        Rcal, Jcal = self.create_block_pH_matrices()
        # check dissipativity part
        if check_spsd(Rcal):
            # logger.info(f"Rcal is spd.")
            Rcal_is_spd = True
        else:
            logger.warning(f"Rcal is NOT spd.")
            Rcal_is_spd = False

        # check interconnection part
        if is_skewsym(Jcal):
            # logger.info(f"Jcal is skew-symmetric.")
            Jcal_is_skewsym = True
        else:
            logger.warning(f"Jcal is NOT skew-symmetric.")
            Jcal_is_skewsym = False

        # check Hamiltonian spsd
        H = self.E.T @ self.Q
        if check_spsd(H):
            # logger.info(f"Rcal is spd.")
            H_is_spsd = True
        else:
            logger.warning(f"Rcal is NOT spd.")
            H_is_spsd = False

        # all pH properties fulfilled?
        if all([Rcal_is_spd, Jcal_is_skewsym, H_is_spsd]):
            logger.info("System fulfills pH properties")
            return True
        else:
            logger.warning("System does NOT fulfill pH properties.")
            return False

    def reduce(self, V, ph_reduction_type: str = "Berlin"):
        """
        Structure-preserving reduction by choosing W=Q@V and approximate x=V@xred
        reduction_type (str): Gugercin | {Berlin} |
        """
        if ph_reduction_type.lower() == "berlin":
            # projection matrix (from left)
            W = self.Q @ V

            # reduce matrices
            Jr = W.T @ self.J @ self.Q @ V
            Rr = W.T @ self.R @ self.Q @ V
            Er = W.T @ self.E @ V
            Qr = np.identity(V.shape[1])
            Gr = W.T @ self.G
            Pr = W.T @ self.P
            Sr = self.S
            Nr = self.N
        elif ph_reduction_type.lower() == "gugercin":
            # left projection matrix
            W = self.Q @ V @ np.linalg.solve(V.T @ self.Q @ V, np.eye(V.shape[1]))

            # reduce ph matrices
            Jr = W.T @ self.J @ W
            Rr = W.T @ self.R @ W
            Er = W.T @ self.E @ V  # check if this is structure-preserving
            Qr = V.T @ self.Q @ V
            Gr = W.T @ self.G
            Pr = W.T @ self.P
            Sr = self.S
            Nr = self.N
        else:
            raise ValueError(
                f"Unknown reduction type {ph_reduction_type}. Choose Berlin or Gugercin."
            )

        return PHSystem(J=Jr, R=Rr, G=Gr, Q=Qr, E=Er, P=Pr, S=Sr, N=Nr)

    def transform_to_poisson(self):
        """
        Transform pH system into Poisson form, i.e. with J as Poisson matrix (skew-identity)
        Based on [BennerEtAl00] Cholesky-like factorizations of skew-symmetric matrices
        """
        if self.is_poisson:
            # already in Poisson form
            return self

        elif self.is_berlin:

            n_half = int(self.n / 2)
            J12 = self.J[:n_half, n_half : 2 * n_half].T
            # assume skew-symmetric matrix J of form [[0, E.T],[-E 0]]
            if not (
                np.allclose(
                    self.J,
                    np.block(
                        [
                            [np.zeros((n_half, n_half)), J12.T],
                            [-J12, np.zeros((n_half, n_half))],
                        ]
                    ),
                )
            ):
                raise ValueError(f"J needs to be of the form [[0, J12.T],[-J12 0]].")

            # transformation matrix
            T = np.block(
                [
                    [np.zeros((n_half, n_half)), J12.T],
                    [-np.eye(n_half), np.zeros((n_half, n_half))],
                ]
            )

            T_inv = np.linalg.solve(T, np.eye(self.n))

            J = T_inv.T @ self.J @ T_inv

            E = T_inv.T @ self.E @ T_inv
            R = T_inv.T @ self.R @ T_inv
            G = T_inv.T @ self.G
            P = T_inv.T @ self.P
        else:
            raise NotImplementedError(
                f"The Cholesky-like transformation of J into canonical form is currently only available for pH systems in Berlin form."
            )

        ph_system = PHSystem(J=J, R=R, G=G, Q=self.Q, E=E, P=P, S=self.S, N=self.N)
        assert ph_system.is_poisson

        return ph_system

    def transform_to_standard(self):
        """ """
        # check if E has full rank
        assert np.linalg.matrix_rank(self.E) == self.E.shape[0]
        E_inv = np.linalg.solve(self.E, np.identity(self.E.shape[0]))
        Q = self.Q @ E_inv
        assert np.allclose(np.identity(self.E.shape[0]), self.E @ E_inv)
        E = np.identity(self.E.shape[0])

        ph_system = PHSystem(
            J=self.J, R=self.R, G=self.G, Q=Q, E=E, P=self.P, S=self.S, N=self.N
        )
        assert ph_system.is_standard

        return ph_system

    def transform_to_berlin(self):
        """ """
        # same transformation as reduce() with V as identity
        ph_system = self.reduce(V=np.identity(self.n))
        assert ph_system.is_berlin

        return ph_system

    @classmethod
    def from_pymor(cls, phlti_pymor_model):
        """Create a pH system from a pyMOR PHLTI model.

        Parameters
        ----------
        phlti_pymor_model : pymor.models.iosys.PHLTIModel
            pyMOR model providing pH matrices.

        Returns
        -------
        PHSystem
            Constructed pH system.
        """
        assert isinstance(phlti_pymor_model, PHLTIModel)
        J, R, G, P, S, N, E, Q = phlti_pymor_model.to_matrices()
        return cls(J, R, G, Q=Q, E=E, P=P, S=S, N=N)

    def convert_to_sparse(self):
        """Convert pH matrices to CSR sparse matrices in place."""
        for var in ["J", "R", "Q", "E", "P", "S", "N"]:
            mat = getattr(self, var)
            if not scipy.sparse.issparse(mat):
                setattr(self, var, scipy.sparse.csr_matrix(mat))

    def convert_to_dense(self):
        """Convert pH matrices to dense arrays in place."""
        for var in ["J", "R", "Q", "E", "P", "S", "N"]:
            mat = getattr(self, var)
            if scipy.sparse.issparse(mat):
                setattr(self, var, mat.toarray())

    def minimal_realization(self, trunc_tol=1e-12):
        """
        Computes a structure preserving minimal realization of a pH system.
        Assumes 'sys' has attributes J, R, Q, G, P, S, N.
        """
        # transform to standard form
        standard_ph_system = self.transform_to_standard()
        # compute observability Gramian
        sys = ct.ss(
            standard_ph_system.A,
            standard_ph_system.B,
            standard_ph_system.C,
            standard_ph_system.D,
        )
        Y = ct.gram(sys, type="o")

        # X = Q^{-1}
        X = np.linalg.solve(standard_ph_system.Q, np.eye(standard_ph_system.Q.shape[0]))

        Lx = lrcholesky(X, trunc_tol=trunc_tol)
        Ly = lrcholesky(Y, trunc_tol=trunc_tol)

        # SVD of Lx.H * Ly
        _, sigma, Vh = scipy.linalg.svd(Lx.conj().T @ Ly, full_matrices=False)
        V = Vh.conj().T  # SciPy returns Vh (V-Hermitian)

        # Rank determination
        r = np.sum((sigma / sigma[0]) > trunc_tol)
        V_r = V[:, :r]
        sigma_r = sigma[:r]

        # Transformation matrix Wr
        Wr = Ly @ V_r @ np.diag(sigma_r ** (-0.5))

        # Reconstruct system matrices
        from pgopinf.numerics.linalg.symmetry import (
            skew_hermitian,
            hermitian_part,
        )

        J_red = skew_hermitian(Wr.conj().T @ standard_ph_system.J @ Wr)
        R_red = hermitian_part(Wr.conj().T @ standard_ph_system.R @ Wr)
        Q_red = np.diag(1.0 / sigma_r)
        G_red = Wr.conj().T @ standard_ph_system.G
        P_red = Wr.conj().T @ standard_ph_system.P

        # Returns the new system (Assuming a constructor PHSystem exists)
        return PHSystem(
            J=J_red,
            R=R_red,
            Q=Q_red,
            G=G_red,
            P=P_red,
            S=standard_ph_system.S,
            N=standard_ph_system.N,
        )
