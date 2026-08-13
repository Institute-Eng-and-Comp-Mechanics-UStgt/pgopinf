from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any

from pgopinf.data.dataset import DataSet
from pgopinf.reduction.interfaces import (
    BasisArtifact,
    FullBasisArtifact,
    MORProjector,
)
from pgopinf.specs.reduction.base import ReductionSpec
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem


@dataclass
class ReductionResult:
    """Result of a reduction pipeline step.

    Attributes
    ----------
    basis : BasisArtifact
        Trial and test basis used for reduction.
    mor : MORProjector
        Projection backend used for data and system reduction.
    reduced_dataset_projected : DataSet or None
        Projected train/test data, if requested.
    reduced_system : LTISystem or PHSystem or None
        Reduced system, if requested.
    full_basis : FullBasisArtifact or None
        Full basis artifact from which ``basis`` was truncated.
    """

    basis: BasisArtifact
    mor: MORProjector
    reduced_dataset_projected: DataSet | None
    reduced_system: LTISystem | PHSystem | None
    full_basis: FullBasisArtifact | None = None


class Reducer:
    """Apply a reduction specification to data and a system."""

    def __init__(self, spec: ReductionSpec):
        """Initialize the reducer.

        Parameters
        ----------
        spec : ReductionSpec
            Reduction specification defining basis, MOR method, and outputs.
        """
        self.spec = spec
        self.test_basis = spec.test_basis.build()
        self.mor = spec.mor.build()

    def reduce(
        self,
        dataset: DataSet,
        system: LTISystem | PHSystem,
        full_basis: FullBasisArtifact,
        *,
        test_basis_kwargs: dict[str, Any] | None = None,
    ) -> ReductionResult:
        """Run basis truncation, optional data projection, and system reduction.

        Parameters
        ----------
        dataset : DataSet
            Full-order train/test data.
        system : LTISystem or PHSystem
            Full-order system.
        full_basis : FullBasisArtifact
            Full basis artifact to truncate.
        test_basis_kwargs : dict of str to Any, optional
            Extra arguments passed to test-basis construction.

        Returns
        -------
        ReductionResult
            Basis, projector, and requested reduced artifacts.
        """

        basis = full_basis.truncate(self.spec.r)

        if basis.W is None:
            # use replace because BasisArtifact is frozen - keep frozen for safety
            W = self.test_basis.make_W(
                V=basis.V,
                data_train=dataset.TRAIN,
                system=system,
                **(test_basis_kwargs or {}),
            )
            basis = replace(basis, W=W)

        reduced_dataset_projected = None
        if self.spec.project_data:
            train_red = self.mor.project_data(data=dataset.TRAIN, basis=basis)
            test_red = self.mor.project_data(data=dataset.TEST, basis=basis)
            reduced_dataset_projected = DataSet(train=train_red, test=test_red)

        reduced_system = None
        if self.spec.reduce_system:
            reduced_system = self.mor.reduce_system(system=system, basis=basis)

        return ReductionResult(
            basis=basis,
            mor=self.mor,
            reduced_dataset_projected=reduced_dataset_projected,
            reduced_system=reduced_system,
            full_basis=full_basis,
        )
