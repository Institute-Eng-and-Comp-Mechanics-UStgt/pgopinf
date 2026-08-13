from __future__ import annotations

import copy
from dataclasses import is_dataclass, fields
from typing import Any, get_origin, get_args

from pgopinf.specs.base import dataclass_to_dict


class UnknownOverrideKey(ValueError):
    """Raised when a dotted override path does not match a spec field."""

    pass


def _unwrap_type(tp):
    # handles Optional[T] / Union[T, None]
    origin = get_origin(tp)
    if origin is None:
        return tp
    if origin is list or origin is dict or origin is tuple:
        return tp
    args = [a for a in get_args(tp) if a is not type(None)]
    return args[0] if len(args) == 1 else tp


# def _get_dotted(obj: Any, path: str) -> Any:
#     cur = obj
#     for part in path.split("."):
#         cur = getattr(cur, part)
#     return cur


# def _set_dotted_obj(obj: Any, path: str, value: Any) -> None:
#     """
#     Set a nested attribute on an object graph.
#     This assumes intermediate objects already exist.
#     """
#     parts = path.split(".")
#     cur = obj
#     for part in parts[:-1]:
#         cur = getattr(cur, part)
#     setattr(cur, parts[-1], value)


def _apply_kind_switches_for_validation(
    spec_obj: Any, overrides: dict[str, Any]
) -> Any:
    """
    Build a temporary object for validation where polymorphic subtrees whose
    '.kind' changes are replaced by defaults of the new kind.

    Requires the root spec class to define:
        polymorphic_subtrees() -> dict[str, callable]
    mapping subtree path -> default_by_kind factory.
    """
    if not hasattr(type(spec_obj), "polymorphic_subtrees"):
        return spec_obj

    subtree_factories = type(spec_obj).polymorphic_subtrees()
    if not subtree_factories:
        return spec_obj

    # easiest robust route: work through dict -> from_dict
    d = spec_obj.to_dict()

    for subtree_path, default_by_kind in subtree_factories.items():
        kind_key = f"{subtree_path}.kind"
        if kind_key in overrides:
            new_kind = str(overrides[kind_key])
            fresh_spec = default_by_kind(new_kind)

            # set subtree dict
            parts = subtree_path.split(".")
            cur = d
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = (
                fresh_spec.to_dict()
                if hasattr(fresh_spec, "to_dict")
                else dataclass_to_dict(fresh_spec)
            )

    # reconstruct same top-level type
    return type(spec_obj).from_dict(d)


def validate_override_paths(spec_obj: Any, overrides: dict[str, Any]) -> None:
    """
    Validate override paths against the effective spec shape.

    Important:
    If overrides switch a polymorphic subtree kind, e.g.
        identification.kind = "convex_ph_inference"
    then validation of nested keys under 'identification.*' is performed against
    the switched spec, not the old one.
    """
    effective_obj = _apply_kind_switches_for_validation(spec_obj, overrides)

    for path in overrides.keys():
        _validate_single_path(effective_obj, path)


def _validate_single_path(obj: Any, path: str) -> None:
    parts = path.split(".")
    cur = obj

    for i, part in enumerate(parts):
        if not is_dataclass(cur):
            raise UnknownOverrideKey(
                f"Override path '{path}': '{'.'.join(parts[:i])}' is not a dataclass, "
                f"cannot set '{part}'."
            )

        fdict = {f.name: f for f in fields(cur)}
        if part not in fdict:
            valid = ", ".join(sorted(fdict.keys()))
            raise UnknownOverrideKey(
                f"Unknown override key '{path}': '{part}' is not a field of "
                f"{type(cur).__name__}. Valid fields: {valid}"
            )

        f = fdict[part]

        if i == len(parts) - 1:
            return

        cur = getattr(cur, part)

        if cur is None:
            tp = _unwrap_type(f.type)
            raise UnknownOverrideKey(
                f"Override path '{path}': '{'.'.join(parts[:i+1])}' is None; "
                "cannot validate deeper path. Provide a non-None default for that spec."
            )
