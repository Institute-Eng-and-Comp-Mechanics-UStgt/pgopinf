import logging

import numpy as np
from pymor.models.iosys import LTIModel as PymorLTIModel

from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


def hinf_norm(
    first_system: LTISystem | PHSystem,
    mu=None,
    return_fpeak=False,
    ab13dd_equilibrate=False,
    tol=1e-10,
):
    """Compute the H-infinity norm of the LTI system."""
    pymor_model = create_pymor_model(first_system)
    if not first_system.isstable(tol=tol):
        return float("inf")
    return pymor_model.hinf_norm(
        mu=mu,
        return_fpeak=return_fpeak,
        ab13dd_equilibrate=ab13dd_equilibrate,
        tol=tol,
    )


def hinf_error(
    first_system: LTISystem | PHSystem,
    second_system: LTISystem | PHSystem,
    mu=None,
    return_fpeak=False,
    ab13dd_equilibrate=False,
    tol=1e-10,
    return_spectral_abscissas=False,
    first_system_eigvals=None,
):
    """Compute the H-infinity norm of the LTI system."""
    pymor_model = create_pymor_model(first_system)
    pymor_model_2 = create_pymor_model(second_system)
    pymor_model_diff: PymorLTIModel = pymor_model - pymor_model_2

    if first_system_eigvals is None:
        first_system_eigvals = first_system.eigvals()
    second_system_eigvals = second_system.eigvals()
    if return_spectral_abscissas:
        first_system_spectral_abscissa = np.max(np.real(first_system_eigvals))
        if second_system_eigvals.size == 0:
            # happens if stable-unstable decomposition results in empty stable part
            second_system_spectral_abscissa = np.nan
        else:
            second_system_spectral_abscissa = np.max(np.real(second_system_eigvals))
    tol = 1e-10
    if not np.all(np.real(first_system_eigvals) < -tol) or not np.all(
        np.real(second_system_eigvals) < -tol
    ):
        # at least one system is unstable - Pymor.hinf_norm expects stable systems
        hinf_error = float("inf")
    else:
        # The difference of two pymor models leads to BlockDiagonal Operators
        # These operators will be converted and raise a warning `|WARNING|LTIModel: Converting operator A to a NumPy array.` in PyMOR.
        # We suppress this warning here, because it is not relevant for the H-infinity norm computation and can be misleading.
        hinf_error = suppress_warnings(
            pymor_model_diff.hinf_norm,
            mu=mu,
            return_fpeak=return_fpeak,
            ab13dd_equilibrate=ab13dd_equilibrate,
            tol=tol,
        )

    if return_spectral_abscissas:
        return (
            hinf_error,
            first_system_spectral_abscissa,
            second_system_spectral_abscissa,
        )
    else:
        return hinf_error


def create_pymor_model(
    system: LTISystem | PHSystem,
) -> PymorLTIModel:
    """Create a PyMOR LTI model from a given system."""
    pymor_model = PymorLTIModel.from_matrices(
        A=system.A,
        B=system.B,
        C=system.C,
        D=system.D,
        E=system.E,
    )
    return pymor_model


def suppress_warnings(func, *args, **kwargs):
    previous_level = logging.root.manager.disable

    try:
        logging.disable(logging.WARNING)
        return func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError("H-infinity norm computation failed.") from e
    finally:
        logging.disable(previous_level)
