from __future__ import annotations
import logging
import numpy as np

from pgopinf.data.dataset import DataSet
from pgopinf.data.data import Data
from pgopinf.data.time.time import Time
from pgopinf.specs.data.base import SplitDataSpec
from pgopinf.systems.lti_system import LTISystem
from pgopinf.reduction.interfaces import BasisArtifact, MORProjector
from pgopinf.data.reduced_data import ReducedData
from pgopinf.systems.ph_system import PHSystem

logger = logging.getLogger(__name__)


def generate_split_data(
    *,
    system: LTISystem | PHSystem,
    split_spec: SplitDataSpec,
    seed: int,
    split_name: str,
) -> Data:
    """Generate one full-order data split.

    Parameters
    ----------
    system : LTISystem or PHSystem
        System to simulate.
    split_spec : SplitDataSpec
        Data-generation specification for the split.
    seed : int
        Random seed for initial conditions and inputs.
    split_name : str
        Split label stored in metadata.

    Returns
    -------
    Data
        Generated full-order data.
    """
    return _generate_split(
        system=system,
        split_spec=split_spec,
        seed=seed,
        split_name=split_name,
    )


def generate_reduced_split_data(
    *,
    reduced_system: LTISystem | PHSystem,
    split_spec: SplitDataSpec,
    seed: int,
    split_name: str,
    mor: MORProjector,
    basis: BasisArtifact,
):
    """Generate one reduced data split by simulating a reduced system.

    Parameters
    ----------
    reduced_system : LTISystem or PHSystem
        Reduced system to simulate.
    split_spec : SplitDataSpec
        Data-generation specification for the split.
    seed : int
        Random seed for initial conditions and inputs.
    split_name : str
        Split label stored in metadata.
    mor : MORProjector
        Projector used to map initial conditions to reduced coordinates.
    basis : BasisArtifact
        Basis artifact associated with the reduced system.

    Returns
    -------
    ReducedData
        Generated reduced data.
    """
    return _generate_split(
        system=reduced_system,
        split_spec=split_spec,
        seed=seed,
        split_name=split_name,
        mor=mor,
        basis=basis,
    )


def generate_dataset(*, system_spec, data_spec) -> DataSet:
    """Generate train and test data from system and data specifications.

    Parameters
    ----------
    system_spec
        Specification that builds the full-order system.
    data_spec
        Specification containing train and test data settings.

    Returns
    -------
    DataSet
        Generated train/test dataset.
    """
    system = system_spec.build()
    ds = DataSet(
        train=generate_split_data(
            system=system,
            split_spec=data_spec.train,
            seed=int(data_spec.seed),
            split_name="train",
        ),
        test=generate_split_data(
            system=system,
            split_spec=data_spec.test,
            seed=int(data_spec.seed) + 1,
            split_name="test",
        ),
    )
    return ds


def generate_reduced_dataset(*, reduced_system, data_spec, mor, basis) -> DataSet:
    """Generate train and test data from a reduced system.

    Parameters
    ----------
    reduced_system
        Reduced system to simulate.
    data_spec
        Specification containing train and test data settings.
    mor : MORProjector
        Projector used to map initial conditions to reduced coordinates.
    basis : BasisArtifact
        Basis artifact associated with the reduced system.

    Returns
    -------
    DataSet
        Generated train/test reduced dataset.
    """
    ds = DataSet(
        train=generate_reduced_split_data(
            reduced_system=reduced_system,
            split_spec=data_spec.train,
            seed=int(data_spec.seed),
            split_name="train",
            mor=mor,
            basis=basis,
        ),
        test=generate_reduced_split_data(
            reduced_system=reduced_system,
            split_spec=data_spec.test,
            seed=int(data_spec.seed) + 1,
            split_name="test",
            mor=mor,
            basis=basis,
        ),
    )
    return ds


def _generate_split(
    *,
    system: LTISystem,
    split_spec: SplitDataSpec,
    seed: int,
    split_name: str,
    mor: MORProjector | None = None,
    basis: BasisArtifact | None = None,
) -> Data:
    time = split_spec.time.build()  # -> Time runtime (your class)
    n_sim = int(split_spec.n_sim)

    # initial condition
    if mor is None:
        ic = split_spec.initial_condition.build(n=system.n, n_sim=n_sim, seed=seed)
        x0 = ic.x0
    else:
        if basis is None:
            raise ValueError("Basis must be provided if MOR projector is provided.")
        n_full = basis.V.shape[0]
        ic = split_spec.initial_condition.build(n=n_full, n_sim=n_sim, seed=seed)
        x0 = mor.project_initial_condition(ic=ic, basis=basis)

    # input
    inp = split_spec.input.build(time=time, seed=seed, n_sim=n_sim)
    U = inp.evaluate(time.t)  # (n_u, n_t, n_sim)

    assert U.shape[2] == x0.shape[1]  # same number of simulations

    # simulate trajectories
    X, Y, dXdt, deriv_method = simulate_trajectories(
        system=system,
        time=time,
        U=U,
        x0=x0,
        split_spec=split_spec,
    )

    if mor is None:
        return Data(
            time=time,
            X=X,
            U=U,
            Y=Y,
            dXdt=dXdt,
            deriv_method=deriv_method,
            meta={
                "split": split_name,
                "seed": seed,
                "dXdt_derivation": split_spec.dXdt_derivation,
            },
        )
    else:
        return ReducedData(
            time=time,
            X=X,
            U=U,
            Y=Y,
            dXdt=dXdt,
            deriv_method=deriv_method,
            V=basis.V,
            meta="integrated",
        )


def simulate_trajectories(system, time, U, x0, split_spec):
    """Simulate all trajectories for a data split.

    Parameters
    ----------
    system
        System with a ``solve`` method.
    time : Time
        Time grid for simulation.
    U : ndarray, shape (n_u, n_t, n_sim)
        Input trajectories.
    x0 : ndarray, shape (n, n_sim)
        Initial state for each simulation.
    split_spec : SplitDataSpec
        Split specification, including derivative handling.

    Returns
    -------
    tuple
        ``(X, Y, dXdt, deriv_method)`` with trajectory arrays and derivative
        metadata.
    """
    n = system.n
    n_y = system.n_y
    n_t = time.n_t
    n_sim = split_spec.n_sim
    X = np.empty((n, n_t, n_sim))
    Y = np.empty((n_y, n_t, n_sim))

    if split_spec.dXdt_derivation == "return":
        return_dXdt = True
        deriv_method = None
    else:
        return_dXdt = False
        deriv_method = split_spec.dXdt_derivation

    if return_dXdt:
        dXdt = np.empty((n, n_t, n_sim))
    for i_sim in range(n_sim):
        # get system matrices
        integration_outputs = system.solve(
            time.t,
            U[:, :, i_sim],
            x0[:, i_sim],
            integrator="imr",
            decomp_option="lu",
            return_dXdt=return_dXdt,
        )
        if return_dXdt:
            X[:, :, i_sim], Y[:, :, i_sim], dXdt[:, :, i_sim] = integration_outputs
        else:
            X[:, :, i_sim], Y[:, :, i_sim] = integration_outputs
            dXdt = None

    return X, Y, dXdt, deriv_method


# def reduce_samples(
#     X,
#     U,
#     Y,
#     dXdt,
#     time,
#     n: int = 2,
#     mode: str = "every_nth",
# ):
#     """
#     mode:
#         every_nth:  use only every nth sample, requires n
#     n: value for every nth
#     """
#     # if self.samples_reduced == True:
#     #     logging.info(f"Samples have already been reduced once. Skipping...")
#     #     return
#     # else:
#     if isinstance(n, float):
#         n = int(n)

#     if mode == "every_nth":
#         logging.info(f"Reduce samples by using only every {n}-th entry of the data.")
#         X = X[:, ::n, :]
#         U = U[:, ::n, :]
#         Y = Y[:, ::n, :]
#         if dXdt is not None:
#             dXdt = dXdt[:, ::n, :]
#         t = time.t[::n]
#         time = Time(t)

#         # self.convert_to_sample_format()
#     else:
#         raise ValueError(f"Unknown mode {mode} to reduce samples.")

#     return X, U, Y, dXdt, time
