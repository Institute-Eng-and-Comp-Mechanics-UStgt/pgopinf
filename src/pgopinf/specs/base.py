from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import is_dataclass, asdict
from typing import Any, Mapping, MutableMapping
import numpy as np

Jsonable = dict[str, Any] | list[Any] | str | int | float | bool | None


# ---------------------------
# Canonicalization + hashing
# ---------------------------


def canonicalize(obj: Any) -> Jsonable:
    """
    Convert nested dataclasses (and basic python structures) into JSON-safe,
    deterministic structures for stable hashing and saving.

    Rules:
    - dataclass -> dict of fields (including nested dataclasses)
    - tuple -> list
    - dict keys are coerced to str
    - sets/frozensets -> sorted list
    - floats are kept as floats (no rounding here; do it outside if needed)
    """
    if obj is None:
        return None
    if is_dataclass(obj):
        d = dataclasses.asdict(
            obj
        )  # recursive, converts tuples->lists inside dataclasses too
        return canonicalize(d)
    if isinstance(obj, Mapping):
        # sort keys deterministically later in dumps; here just canonicalize values
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = canonicalize(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(
            [canonicalize(x) for x in obj], key=lambda x: json.dumps(x, sort_keys=True)
        )
    # numpy scalars handling (optional): avoid importing numpy
    # If obj has item() method like numpy scalars, convert
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return canonicalize(obj.item())
        except Exception:
            pass
    if isinstance(obj, (str, int, float, bool)):
        return obj
    # last resort: string representation (keeps hashing stable, but be cautious)
    return str(obj)


def _jsonify(x: Any) -> Any:
    """Convert to JSON-stable primitives for hashing/serialization."""
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, np.generic):  # numpy scalar (np.float64, ...)
        return x.item()
    if is_dataclass(x):
        # use spec's to_dict if it has one, else asdict
        if hasattr(x, "to_dict"):
            return x.to_dict()
        return _jsonify(asdict(x))
    if isinstance(x, set):
        return sorted(_jsonify(v) for v in x)
    # Don’t silently accept ndarray (too big/ambiguous); require explicit handling
    if isinstance(x, np.ndarray):
        raise TypeError(
            "np.ndarray is not JSON-stable for specs; store a list or define explicit encoding."
        )
    # fallback: try string
    raise TypeError(f"Unsupported spec value type: {type(x)}")


def stable_json_dumps(data: Any) -> str:
    """
    Stable JSON string (sorted keys, compact separators) for hashing.
    """
    canon = canonicalize(data)
    return json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_id(data: Any, algo: str = "sha256") -> str:
    """
    Stable content hash of any JSON-serializable structure after canonicalization.
    """
    s = stable_json_dumps(data).encode("utf-8")
    h = hashlib.new(algo)
    h.update(s)
    return h.hexdigest()[:12]


def short_id(hex_id: str, n: int = 8) -> str:
    """Return the first ``n`` characters of a hexadecimal identifier."""
    return hex_id[:n]


# ---------------------------
# Dotted-path overrides
# ---------------------------


def apply_overrides_dict(
    base: MutableMapping[str, Any], overrides: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    """
    Apply dotted-path overrides into a nested dict structure.

    Example:
      base = {"a":{"b":1}}
      overrides = {"a.b": 3, "a.c": 5}
      -> {"a":{"b":3,"c":5}}

    Note: This works on dicts. The recommended workflow is:
      spec_obj -> to_dict() -> apply overrides -> from_dict()
    """
    for path, value in overrides.items():
        set_in_dict(base, path, value)
    return base


def set_in_dict(d: MutableMapping[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value in a nested dictionary using a dotted path."""
    parts = dotted_path.split(".")
    cur: MutableMapping[str, Any] = d
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


# ---------------------------
# Dataclass (de)serialization helpers
# ---------------------------


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """
    Convert dataclass -> plain dict without losing 'kind' tags etc.
    Uses asdict (recursive) then canonicalizes for JSON safety.
    """
    if not is_dataclass(obj):
        raise TypeError("dataclass_to_dict expects a dataclass instance")
    return canonicalize(dataclasses.asdict(obj))


def require_keys(d: Mapping[str, Any], keys: list[str], where: str = "") -> None:
    """Raise an error if required keys are missing from a mapping."""
    missing = [k for k in keys if k not in d]
    if missing:
        prefix = f"{where}: " if where else ""
        raise KeyError(f"{prefix}missing required keys: {missing}")


# ---------------------------
# Naming helpers
# ---------------------------


def sanitize_name(name: str, max_len: int = 60) -> str:
    """
    Filesystem-friendly name: lowercase, alnum/_/- only.
    """
    import re

    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


class SpecBase:
    """
    Mixin for spec dataclasses: provides to_dict() for caching/hashing.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize a spec dataclass to a JSON-safe dictionary."""
        if not is_dataclass(self):
            raise TypeError("SpecBase must be used with dataclasses")
        d = asdict(self)
        return _jsonify(d)
