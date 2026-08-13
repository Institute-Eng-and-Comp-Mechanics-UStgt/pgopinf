from dataclasses import dataclass
from typing import Literal, Protocol, Mapping, Any

from pgopinf.reduction.interfaces import BasisArtifact
from pgopinf.reduction.project_data import (
    project_data_linear,
    project_ic_linear,
)
from pgopinf.specs import system
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass
class LTIMORProjector:
    """Projection backend for LTI reduced models."""

    def project_data(self, *, data, basis):
        """Project trajectory data with a linear reduction projector.

        Parameters
        ----------
        data
            Full-order data to project.
        basis : BasisArtifact
            Trial and test basis.

        Returns
        -------
        ReducedData
            Projected data.
        """
        return project_data_linear(data=data, basis=basis)

    def project_initial_condition(self, ic, basis):
        """Project initial conditions with a linear reduction projector.

        Parameters
        ----------
        ic
            Initial-condition object.
        basis : BasisArtifact
            Trial and test basis.

        Returns
        -------
        ndarray
            Reduced initial-condition array.
        """
        return project_ic_linear(ic=ic, basis=basis)

    def reduce_system(
        self, *, system: LTISystem | PHSystem, basis: BasisArtifact
    ) -> LTISystem:
        """Reduce an LTI or pH system as a generic LTI system.

        Parameters
        ----------
        system : LTISystem or PHSystem
            System to reduce.
        basis : BasisArtifact
            Trial and test basis.

        Returns
        -------
        LTISystem
            Reduced LTI system.
        """

        if isinstance(system, PHSystem):
            # use LTISystem reduce method
            return LTISystem.reduce(system, basis.V, basis.W)
        elif isinstance(system, LTISystem):
            return system.reduce(basis.V, basis.W)


@dataclass
class PHMORProjector:
    """Structure-preserving projection backend for pH systems."""

    ph_reduction_type: Literal["Berlin", "Gugercin"] = "Berlin"

    def project_data(self, *, data, basis):
        """Project trajectory data with a linear reduction projector.

        Parameters
        ----------
        data
            Full-order data to project.
        basis : BasisArtifact
            Trial and test basis.

        Returns
        -------
        ReducedData
            Projected data.
        """
        return project_data_linear(data=data, basis=basis)

    def project_initial_condition(self, ic, basis):
        """Project initial conditions with a linear reduction projector.

        Parameters
        ----------
        ic
            Initial-condition object.
        basis : BasisArtifact
            Trial and test basis.

        Returns
        -------
        ndarray
            Reduced initial-condition array.
        """
        return project_ic_linear(ic=ic, basis=basis)

    def reduce_system(self, *, system: PHSystem, basis: BasisArtifact) -> PHSystem:
        """Reduce a pH system while preserving pH structure.

        Parameters
        ----------
        system : PHSystem
            Port-Hamiltonian system to reduce.
        basis : BasisArtifact
            Reduction basis.

        Returns
        -------
        PHSystem
            Reduced pH system.
        """
        return system.reduce(basis.V, ph_reduction_type=self.ph_reduction_type)
