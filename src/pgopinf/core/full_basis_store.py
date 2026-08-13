from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import logging

import numpy as np

from pgopinf.data.dataset import DatasetArtifact
from pgopinf.io.results_path import ResultsPath
from pgopinf.specs.base import stable_id

from pgopinf.reduction.full_basis.pod import PODFullBasisArtifact
from pgopinf.reduction.full_basis.modal import ModalFullBasisArtifact
from pgopinf.specs.reduction.full_basis import FullBasisSpec

logger = logging.getLogger(__name__)


class FullBasisStore:
    """Store and reuse full-basis artifacts.

    Full bases are cached independently of the reduced dimension ``r`` so that
    different reductions can truncate the same artifact without recomputing it.
    """

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the full-basis store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager used to locate the full-basis cache.
        force_recompute : bool, optional
            If ``True``, recompute artifacts even when cached files exist.
        """
        self.paths = paths
        self.force_recompute = force_recompute

    def compute_id(self, *, data_train_id: str | None, full_basis_spec) -> str:
        """Compute the stable identifier for a full-basis artifact.

        Parameters
        ----------
        data_train_id : str or None
            Identifier of the training data used to fit the basis, or ``None``
            for bases that are independent of data.
        full_basis_spec
            Specification that builds the full-basis method.

        Returns
        -------
        str
            Stable full-basis identifier.
        """
        key = {
            "dataset_id": data_train_id,
            "full_basis": full_basis_spec.to_dict(),
        }
        return stable_id(key)

    def get_or_create(
        self,
        *,
        dataset_artifact: DatasetArtifact | None,
        system=None,
        full_basis_spec: FullBasisSpec,
    ):
        """Load an existing full basis or compute and cache a new one.

        Parameters
        ----------
        dataset_artifact : DatasetArtifact or None
            Dataset used to fit data-driven bases. Use ``None`` for system-only
            bases.
        system
            System passed to the full-basis builder.
        full_basis_spec : FullBasisSpec
            Specification of the full-basis construction.

        Returns
        -------
        tuple[str, object]
            Full-basis identifier and fitted full-basis artifact.
        """
        data_train_id = (
            dataset_artifact.train_id if dataset_artifact is not None else None
        )
        fb_id = self.compute_id(
            data_train_id=data_train_id, full_basis_spec=full_basis_spec
        )
        if self.exists(fb_id) and not self.force_recompute:
            logger.info(f"Full basis already exists (id={fb_id}), loading...")
            return fb_id, self.load(fb_id)

        full_basis_builder = full_basis_spec.build()
        if dataset_artifact is not None:
            dataset = dataset_artifact.data
        else:
            dataset = None
        logger.info(f"Computing full basis (id={fb_id})...")
        full_basis = full_basis_builder.fit(dataset=dataset, system=system)
        self.save(
            fb_id,
            data_train_id=data_train_id,
            full_basis_spec=full_basis_spec,
            full_basis=full_basis,
        )
        return fb_id, full_basis

    def exists(self, fb_id: str) -> bool:
        """Return whether a full-basis artifact is present on disk.

        Parameters
        ----------
        fb_id : str
            Full-basis identifier.

        Returns
        -------
        bool
            ``True`` if both metadata and array data exist.
        """
        d = self.paths.full_bases_dir(fb_id)
        return (d / "full_basis.npz").exists() and (d / "spec.json").exists()

    def save(
        self, fb_id: str, *, data_train_id: str | None, full_basis_spec, full_basis
    ):
        """Persist a full-basis artifact and its specification.

        Parameters
        ----------
        fb_id : str
            Full-basis identifier.
        data_train_id : str or None
            Identifier of the training data used to fit the basis.
        full_basis_spec
            Specification used to build the artifact.
        full_basis
            Full-basis artifact implementing ``to_npz``.
        """
        d = self.paths.full_bases_dir(fb_id)

        (d / "spec.json").write_text(
            json.dumps(
                {
                    "full_basis_id": fb_id,
                    "data_train_id": data_train_id,
                    "full_basis": full_basis_spec.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        np.savez_compressed(d / "full_basis.npz", **full_basis.to_npz())

    def load(self, fb_id: str):
        """Load a cached full-basis artifact.

        Parameters
        ----------
        fb_id : str
            Full-basis identifier.

        Returns
        -------
        PODFullBasisArtifact or ModalFullBasisArtifact
            Reconstructed full-basis artifact.

        Raises
        ------
        ValueError
            If the stored basis kind is unknown.
        """
        d = self.paths.full_bases_dir(fb_id)
        z = dict(np.load(d / "full_basis.npz", allow_pickle=True))
        kind = str(z.get("kind", "pod"))

        logger.info(f"Loading full basis (id={fb_id})...")
        if kind == "pod":
            return PODFullBasisArtifact.from_npz(z)
        if kind == "modal":
            return ModalFullBasisArtifact.from_npz(z)
        raise ValueError(f"Unknown full_basis artifact kind: {kind!r}")
