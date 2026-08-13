from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import logging

from pgopinf.identification.identifier import IdentificationResult
from pgopinf.identification.identifier import IdentificationResult
from pgopinf.io.results_path import ResultsPath
from pgopinf.reduction.reducer import ReductionResult
from pgopinf.specs.base import stable_id
from pgopinf.specs.identification.base import IdentificationSpec
from pgopinf.systems.lti_system import LTISystem
from pgopinf.systems.ph_system import PHSystem

logger = logging.getLogger(__name__)


class IdentificationStore:
    """Store identified reduced systems and identification metadata."""

    def __init__(self, paths: ResultsPath, force_recompute: bool = False):
        """Initialize the identification store.

        Parameters
        ----------
        paths : ResultsPath
            Result-directory manager.
        force_recompute : bool, optional
            If ``True``, rerun identification even when cached files exist.
        """
        self.paths = paths
        self.force_recompute = force_recompute

    def compute_id(
        self,
        *,
        reduction_id: str,
        identification_spec: IdentificationSpec,
    ) -> str:
        """Compute the stable identifier for an identification result.

        Parameters
        ----------
        reduction_id : str
            Identifier of the reduction used for identification.
        identification_spec : IdentificationSpec
            Identification method specification.

        Returns
        -------
        str
            Stable identification identifier.
        """
        key = {
            "reduction_id": reduction_id,
            "identification": identification_spec.to_dict(),
        }
        return stable_id(key)

    def get_or_create(
        self,
        *,
        reduction_id: str,
        reduction_result: ReductionResult,
        identification_spec: IdentificationSpec,
        original_system: LTISystem | PHSystem | None = None,
    ) -> tuple[str, IdentificationResult]:
        """Load an identification result or compute and cache a new one.

        Parameters
        ----------
        reduction_id : str
            Identifier of the reduction used for identification.
        reduction_result : ReductionResult
            Reduced data and reduced system used by the identifier.
        identification_spec : IdentificationSpec
            Specification that builds the identifier.
        original_system : LTISystem or PHSystem, optional
            Original system passed to identifiers that require it.

        Returns
        -------
        tuple[str, IdentificationResult]
            Identification identifier and result.
        """
        ident_id = self.compute_id(
            reduction_id=reduction_id,
            identification_spec=identification_spec,
        )

        if self.exists(ident_id) and not self.force_recompute:
            logger.info(
                f"Identification {ident_id} already exists. Loading from disk..."
            )
            return ident_id, self.load(ident_id)

        identifier = identification_spec.build()
        logger.info(
            f"Running identification (id={ident_id}) with method {identification_spec.kind}..."
        )
        identification_result = identifier.fit(
            reduced_dataset=reduction_result.reduced_dataset_projected,
            reduced_system=reduction_result.reduced_system,
            original_system=original_system,
        )

        self.save(
            ident_id,
            reduction_id=reduction_id,
            identification_spec=identification_spec,
            result=identification_result,
        )
        return ident_id, identification_result

    def exists(self, ident_id: str) -> bool:
        """Return whether an identification result exists on disk.

        Parameters
        ----------
        ident_id : str
            Identification identifier.

        Returns
        -------
        bool
            ``True`` if metadata, diagnostics, and system matrices exist.
        """
        d = self._dir(ident_id)
        return (
            (d / "spec.json").exists()
            and (d / "diagnostics.json").exists()
            and (d / "meta.json").exists()
            and (d / "system_meta.json").exists()
            and (d / "matrices.npz").exists()
        )

    def _dir(self, ident_id: str) -> Path:
        return self.paths.identification_dir(ident_id)

    def save(
        self,
        ident_id: str,
        *,
        reduction_id: str,
        identification_spec: IdentificationSpec,
        result: IdentificationResult,
    ):
        """Persist an identification result.

        Parameters
        ----------
        ident_id : str
            Identification identifier.
        reduction_id : str
            Identifier of the reduction used for identification.
        identification_spec : IdentificationSpec
            Identification specification.
        result : IdentificationResult
            Identified system, diagnostics, and metadata.
        """
        d = self._dir(ident_id)
        d.mkdir(parents=True, exist_ok=True)

        (d / "spec.json").write_text(
            json.dumps(
                {
                    "identification_id": ident_id,
                    "reduction_id": reduction_id,
                    "identification": identification_spec.to_dict(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # diagnostics/meta
        (d / "diagnostics.json").write_text(
            json.dumps(result.diagnostics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (d / "meta.json").write_text(
            json.dumps(result.meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # identified system serialization
        sys_dir = d
        sys_dir.mkdir(exist_ok=True)
        self._save_system(sys_dir, result.system_identified)

    def load(self, ident_id: str):
        """Load a cached identification result.

        Parameters
        ----------
        ident_id : str
            Identification identifier.

        Returns
        -------
        IdentificationResult
            Reconstructed identified system with diagnostics and metadata.
        """
        d = self._dir(ident_id)
        diagnostics = json.loads((d / "diagnostics.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        identified_system = self._load_system(d)

        from pgopinf.identification.identifier import (
            IdentificationResult,
        )

        return IdentificationResult(
            system_identified=identified_system,
            diagnostics=diagnostics,
            meta=meta,
        )

    def _save_system(self, folder: Path, system_obj):
        # same serialization approach as for reduced intrusive systems
        meta = {}
        meta["kind"] = system_obj.kind
        matrices = system_obj.matrices
        (folder / "system_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        np.savez_compressed(folder / "matrices.npz", **matrices)

    def _load_system(self, folder: Path):
        meta_path = folder / "system_meta.json"
        if not meta_path.exists():
            meta_path = folder / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        matrices = dict(np.load(folder / "matrices.npz"))
        kind = meta["kind"]

        from pgopinf.systems.lti_system import LTISystem
        from pgopinf.systems.ph_system import PHSystem

        if kind == "lti":
            return LTISystem.from_matrices(**matrices)
        if kind == "ph":
            return PHSystem.from_matrices(**matrices)
        raise ValueError(f"Unknown identified system kind {kind!r}")
