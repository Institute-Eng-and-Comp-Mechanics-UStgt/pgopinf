from __future__ import annotations

from typing import Any


LABEL_MAP: dict[str, str] = {
    # metrics
    # "hinf_error": r"$\mathcal{E}_{\mathcal{H}_\infty}$",
    "rel_error_operators_combined": r"eps_ref",  # r"$\varepsilon_{\mathrm{ref}}$",
    "rel_error_bias_term_combined_data_frob_norm": r"eps_corr",  # r"$\varepsilon_{\mathrm{corr}}$",
    "rel_bound_Xperp_Tr": r"eps_est",  # r"$\varepsilon_{\mathrm{est}}$",
    "rel_error_bound_operators_combined": r"eps_bound",  # r"$\varepsilon_{\mathrm{bound}}$",
    "singular_value_r": r"sigma",  # r"$\sigma_r$",
    "cond_rhs_data_reduced": r"kappa_T_r",  # r"$\kappa(\mathbf{T}_r)$",
    "u_1": r"$u_1$",
    # common spec parameters
    "reduction.r": r"r",
    "seed": r"seed",
}


def pretty_label(name: Any) -> str:
    """
    Convert an internal key/value into a display label.
    Falls back to str(name) if no mapping exists.
    """
    s = str(name)
    return LABEL_MAP.get(s, s)


def pretty_join(parts: list[Any], sep: str = " | ") -> str:
    """Join values after applying display-label mappings.

    Parameters
    ----------
    parts : list of Any
        Values to format.
    sep : str, optional
        Separator inserted between formatted values.

    Returns
    -------
    str
        Joined display string.
    """
    return sep.join(pretty_label(p) for p in parts)


def pretty_kv(key: Any, value: Any) -> str:
    """
    Format 'key=value' using pretty labels.
    """
    return f"{pretty_label(key)}={pretty_label(value)}"
