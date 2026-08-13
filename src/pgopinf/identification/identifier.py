from typing import Protocol, Any


from dataclasses import dataclass
from typing import Any

import numpy as np

from pgopinf.data.data import Data
from pgopinf.identification.algorithms.convex_ph_inference import (
    convex_ph_inference,
)
from pgopinf.identification.algorithms.operator_inference import (
    operator_inference,
)
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass(frozen=True)
class IdentificationResult:
    """Result produced by an identification method.

    Attributes
    ----------
    system_identified : LTISystem or PHSystem
        Identified reduced-order system.
    diagnostics : dict of str to Any
        Method-specific diagnostics and solver information.
    meta : dict of str to Any
        Lightweight metadata describing the result.
    """

    system_identified: LTISystem | PHSystem
    diagnostics: dict[str, Any]
    meta: dict[str, Any]


class Identifier(Protocol):
    """Protocol for reduced-system identification methods."""

    def fit(
        self,
        *,
        reduced_dataset: Data,
        reduced_system=None,
        original_system=None,
    ) -> IdentificationResult:
        """Fit an identified system from reduced data."""
        ...


@dataclass
class OperatorInferenceIdentifier:
    """Identifier based on operator inference."""

    use_E: bool = True
    convert_to_ph: bool = True
    seperate_output_inf: bool = False
    lambda_reg: float = 0.0

    def fit(
        self, *, reduced_dataset, reduced_system=None, original_system=None
    ) -> IdentificationResult:
        """Identify a reduced system with operator inference.

        Parameters
        ----------
        reduced_dataset
            Dataset containing projected training data.
        reduced_system
            Intrusive reduced system used to provide the descriptor matrix when
            ``use_E`` is ``True``.
        original_system
            Unused; accepted for the identifier protocol.

        Returns
        -------
        IdentificationResult
            Identified system and diagnostics.
        """
        data = reduced_dataset.TRAIN

        if self.use_E:
            E = reduced_system.E
        else:
            E = np.eye(reduced_system.n)

        system_identified = operator_inference(
            data.x,
            data.y,
            data.u,
            E=E,
            dxdt=data.dxdt,
            seperate_output_inf=self.seperate_output_inf,
            lambda_reg=self.lambda_reg,
            convert_to_ph=self.convert_to_ph,
        )
        diagnostics = {
            "use_E": self.use_E,
            "convert_to_ph": self.convert_to_ph,
            "seperate_output_inf": self.seperate_output_inf,
            "lambda_reg": self.lambda_reg,
        }
        return IdentificationResult(
            system_identified=system_identified,
            diagnostics=diagnostics,
            meta={"kind": "opinf", "r": system_identified.n},
        )


@dataclass
class ConvexPortHamiltonianIdentifier:
    """Identifier based on convex port-Hamiltonian inference."""

    J_is_known: bool = False
    add_regularization: bool = False
    add_dissipation_inequality_cost: bool = False
    lambdas: float = 1e-6
    no_feedthrough: bool = False
    solver: str = "mosek"
    project_psd: bool = False
    return_system_type: str = "ph"
    accept_unknown_mosek: bool = False

    def fit(
        self, *, reduced_dataset, reduced_system=None, original_system=None
    ) -> IdentificationResult:
        """Identify a reduced system with convex pH inference.

        Parameters
        ----------
        reduced_dataset
            Dataset containing projected training data.
        reduced_system
            Intrusive reduced system. Its ``J`` matrix is used when
            ``J_is_known`` is ``True``.
        original_system
            Unused; accepted for the identifier protocol.

        Returns
        -------
        IdentificationResult
            Identified system, serialized solver diagnostics, and metadata.
        """
        data = reduced_dataset.TRAIN

        if self.J_is_known:
            J_known = reduced_system.J
        else:
            J_known = None

        system_identified, solver_stats = convex_ph_inference(
            x=data.x,
            y=data.y,
            u=data.u,
            dxdt=data.dxdt,
            J_known=J_known,
            add_regularization=self.add_regularization,
            lambdas=self.lambdas,
            add_dissipation_inequality_cost=self.add_dissipation_inequality_cost,
            no_feedthrough=self.no_feedthrough,
            solver=self.solver,
            project_psd=self.project_psd,
            return_system_type=self.return_system_type,
            return_solver_stats=True,
            accept_unknown_mosek=self.accept_unknown_mosek,
        )

        # remove np.ndarrays from diagnostics nested dict for json serialization
        # go into nested dicts
        solver_stats_serializable = {}
        for key, value in solver_stats.__dict__.items():
            if isinstance(value, np.ndarray):
                solver_stats_serializable[key] = value.tolist()  # convert to list
            elif isinstance(value, dict):
                solver_stats_serializable[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                solver_stats_serializable[key] = value

        diagnostics = {
            "J_is_known": self.J_is_known,
            "add_regularization": self.add_regularization,
            "add_dissipation_inequality_cost": self.add_dissipation_inequality_cost,
            "lambdas": self.lambdas,
            "no_feedthrough": self.no_feedthrough,
            "solver": self.solver,
            "project_psd": self.project_psd,
            "return_system_type": self.return_system_type,
            "solver_stats": solver_stats_serializable,
            "accept_unknown_mosek": self.accept_unknown_mosek,
        }

        return IdentificationResult(
            system_identified=system_identified,
            diagnostics=diagnostics,
            meta={"kind": "convex_ph_inference", "r": system_identified.n},
        )
