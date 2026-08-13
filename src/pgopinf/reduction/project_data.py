from __future__ import annotations
from typing import Any
import numpy as np

from pgopinf.data.initial_condition.initial_condition import (
    InitialCondition,
)
from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.data.reduced_data import ReducedData
from pgopinf.data.data import Data


def define_reduction_projector(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    returns the data projector according to data_projection_type
    From full to reduced dimension
    """
    r = V.shape[1]
    if np.allclose(W.T @ V, np.eye(r)):
        reduction_projector = W.T
    else:
        # projector: (W.T*V)^-1*W.T
        reduction_projector = np.linalg.solve(W.T @ V, W.T)
    return reduction_projector


def project_data_linear(
    *,
    data: Data,
    basis: BasisArtifact,
) -> ReducedData:
    """
    Projects data to reduced space using the provided basis.
    """
    V = basis.V
    W = basis.W
    if W is None:
        raise ValueError("W must be provided for linear projection.")

    reduction_projector = define_reduction_projector(V, W)

    xr = reduction_projector @ data.x
    dxdtr = (
        reduction_projector @ data.dxdt
    )  # dxdt is reduced; in accordance with [MorandinNicodemusUnger22]

    Xr, U, Y, dXdtr = Data.convert_xuy_to_time_step_format(
        x=xr,
        u=data.u,
        y=data.y,
        dxdt=dxdtr,
        n=basis.r,
        n_u=data.n_u,
        n_y=data.n_y,
        n_t=data.n_t,
        n_sim=data.n_sim,
    )

    return ReducedData(
        time=data.time,
        X=Xr,
        U=U,
        Y=Y,
        dXdt=dXdtr,
        deriv_method=data.deriv_method,
        V=basis.V,
        meta="projected",
    )


def project_ic_linear(ic: InitialCondition, basis: BasisArtifact) -> np.ndarray:
    """Project initial conditions with a Petrov-Galerkin projector.

    Parameters
    ----------
    ic : InitialCondition
        Full-order initial conditions.
    basis : BasisArtifact
        Trial and test basis.

    Returns
    -------
    ndarray, shape (r, n_sim)
        Reduced initial-condition columns.
    """
    V = basis.V
    W = basis.W
    if W is None:
        raise ValueError("W must be provided for linear projection.")

    reduction_projector = define_reduction_projector(V, W)

    return ic.reduce(projector=reduction_projector).x0
