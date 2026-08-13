from __future__ import annotations

from dataclasses import fields
from typing import (
    Any,
    Callable,
    Mapping,
    Generic,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from .base import require_keys

T = TypeVar("T")


class KindRegistry(Generic[T]):
    """Registry for polymorphic specs keyed by a ``kind`` field."""

    def __init__(self, label: str):
        """Initialize an empty kind registry."""
        self.label = label
        self._ctors: dict[str, Callable[[Mapping[str, Any]], T]] = {}
        self._default_ctors: dict[str, Callable[[], T]] = {}

    def register(self, kind: str):
        """
        Register a spec dataclass class.

        Assumes:
        - cls(**d) works
        - cls() works for default construction
        """

        def deco(cls):
            """Register a class and return it unchanged."""
            def ctor(d: Mapping[str, Any]) -> T:
                """Construct a registered spec from a dictionary."""
                from_dict = getattr(cls, "from_dict", None)
                if callable(from_dict):
                    return from_dict(d)
                return cls(**restore_dataclass_kwargs(cls, d))

            def default_ctor() -> T:
                """Construct the default registered spec."""
                return cls()

            self._ctors[kind] = ctor
            self._default_ctors[kind] = default_ctor
            return cls

        return deco

    def from_dict(self, d: Mapping[str, Any]) -> T:
        """Construct a registered spec from a dictionary."""
        require_keys(d, ["kind"], where=f"{self.label}.from_dict")
        kind = str(d["kind"])
        if kind not in self._ctors:
            known = ", ".join(sorted(self._ctors.keys()))
            raise ValueError(f"{self.label}: unknown kind {kind!r}. Known: {known}")
        return self._ctors[kind](d)

    def default_by_kind(self, kind: str) -> T:
        """Return the default spec for a registered kind."""
        if kind not in self._default_ctors:
            known = ", ".join(sorted(self._default_ctors.keys()))
            raise ValueError(f"{self.label}: unknown kind {kind!r}. Known: {known}")
        return self._default_ctors[kind]()

    def known_kinds(self) -> list[str]:
        """Return all registered kind names."""
        return sorted(self._ctors.keys())


def restore_dataclass_kwargs(cls, d: Mapping[str, Any]) -> dict[str, Any]:
    """Restore JSON-loaded values to dataclass constructor-compatible types."""
    type_hints = get_type_hints(cls)
    restored: dict[str, Any] = dict(d)

    for field in fields(cls):
        value = restored.get(field.name)
        if value is None:
            continue

        field_type = type_hints.get(field.name, field.type)
        origin = get_origin(field_type)

        if origin is tuple and isinstance(value, list):
            restored[field.name] = tuple(value)

    return restored
