import logging

from pgopinf.identification.algorithms.iodmd import iodmd, unstack
from pgopinf.identification.subroutines.transformations import (
    get_Riccati_transform,
)
from pgopinf.systems.lti_system import LTISystem

logger = logging.getLogger(__name__)


def operator_inference(
    x,
    y,
    u,
    E=None,
    delta_t=None,
    dxdt=None,
    seperate_output_inf: bool = False,
    lambda_reg: float = 0.0,
    convert_to_ph: bool = False,
):
    """Identify a reduced system by linear operator inference.

    The method infers the block operator ``[[A, B], [C, D]]`` from state,
    output, input, and derivative data using ioDMD. If derivatives are not
    supplied, finite differences are formed from adjacent state snapshots and
    the data are shifted to midpoint samples.

    Parameters
    ----------
    x : ndarray, shape (n, n_samples)
        State snapshots in sample format.
    y : ndarray, shape (n_y, n_samples)
        Output snapshots in sample format.
    u : ndarray, shape (n_u, n_samples)
        Input snapshots in sample format.
    E : ndarray, optional, shape (n, n)
        Descriptor matrix used in the inferred dynamics. If omitted, ioDMD
        treats the descriptor as the identity.
    delta_t : float, optional
        Time step used to approximate derivatives when ``dxdt`` is ``None``.
    dxdt : ndarray, optional, shape (n, n_samples)
        State derivative snapshots. If omitted, derivatives are computed from
        ``x`` and ``delta_t``.
    seperate_output_inf : bool, optional
        If ``True``, infer the state equation and output equation in separate
        least-squares problems.
    lambda_reg : float, optional
        Ridge regularization weight passed to ioDMD. A value of zero disables
        regularization.
    convert_to_ph : bool, optional
        If ``True``, attempt to convert the inferred LTI system to a
        port-Hamiltonian system via a Riccati transformation.

    Returns
    -------
    LTISystem or PHSystem
        Identified LTI system, or a pH system when ``convert_to_ph`` is
        ``True``.

    Raises
    ------
    AssertionError
        If ``dxdt`` is ``None`` and ``delta_t`` is not provided.
    """

    if dxdt is None:
        assert delta_t is not None
        dxdt = (x[:, 1:] - x[:, :-1]) / delta_t
        x = 1 / 2 * (x[:, 1:] + x[:, :-1])
        u = 1 / 2 * (u[:, 1:] + u[:, :-1])
        y = 1 / 2 * (y[:, 1:] + y[:, :-1])

    Acal, e = iodmd(
        X=x,
        Y=y,
        U=u,
        X1=dxdt,
        E=E,
        seperate_output_inf=seperate_output_inf,
        lambda_reg=lambda_reg,
    )
    A, B, C, D = unstack(Acal, x.shape[0])
    lti_system = LTISystem(A=A, B=B, C=C, D=D, E=E)
    # %% create pH system
    if convert_to_ph:
        logger.info(f"Create pH system from state-space system...")
        _, _, ph_system = get_Riccati_transform(lti_system)
        return ph_system
    else:
        return lti_system
