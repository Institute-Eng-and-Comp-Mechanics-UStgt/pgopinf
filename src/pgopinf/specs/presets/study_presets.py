from pgopinf.specs.presets.studies.opinf_theory_over_r import (
    opinf_theory_over_r,
)
from pgopinf.specs.presets.studies.petrov_galerkin_and_ph import (
    petrov_galerkin_and_ph,
)
from pgopinf.specs.presets.studies.petrov_vs_galerkin import (
    petrov_vs_galerkin,
)
from pgopinf.specs.presets.studies.g_pg_hamcvx_ph import (
    g_pg_hamcvx_ph,
)
from pgopinf.specs.presets.studies.test_convex_ph_and_g_opinf import (
    test_convex_ph_and_g_opinf,
)
from pgopinf.specs.study import StudySpec


def study_preset(name: str, **kwargs) -> StudySpec:
    """Return a named study preset."""
    name = name.lower()

    if name == "opinf_theory_over_r":
        study_spec = opinf_theory_over_r(**kwargs)
    elif name == "petrov_vs_galerkin":
        study_spec = petrov_vs_galerkin(**kwargs)
    elif name == "petrov_vs_galerkin_fine_r":
        kwargs["name"] = "petrov_vs_galerkin_fine_r"
        study_spec = petrov_vs_galerkin(**kwargs)
    elif name == "petrov_galerkin_and_ph":
        study_spec = petrov_galerkin_and_ph(**kwargs)
    elif name == "g_pg_hamcvx_ph":
        study_spec = g_pg_hamcvx_ph(**kwargs)
    elif name == "test_convex_ph_and_g_opinf":
        study_spec = test_convex_ph_and_g_opinf(**kwargs)
    else:
        raise ValueError(f"Unknown study preset name: {name}")
    return study_spec
