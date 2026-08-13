from __future__ import annotations

from typing import Any, Mapping, Protocol

from pgopinf.specs.registry import KindRegistry


class MetricSpec(Protocol):
    """
    Configuration object for a metric.

    Concrete metric specs should be frozen dataclasses with a `kind` field
    and a `.build()` method that returns the runtime metric object.
    """

    kind: str

    def build(self):
        """
        Construct the runtime metric object.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize spec to a dictionary (used for hashing and persistence).
        """
        ...


# registry: metric kind -> spec constructor
METRIC_REGISTRY: KindRegistry[MetricSpec] = KindRegistry("metric")


def register_metric(kind: str):
    """
    Decorator used on metric spec dataclasses.

    Example:

        @register_metric("state_error_over_time")
        @dataclass(frozen=True)
        class StateErrorOverTimeSpec(...)
    """
    return METRIC_REGISTRY.register(kind)


def metric_from_dict(d: Mapping[str, Any]) -> MetricSpec:
    """
    Parse a metric spec from a dictionary.

    The dictionary must contain a 'kind' key.
    """
    return METRIC_REGISTRY.from_dict(d)


def metric_to_dict(spec: MetricSpec) -> dict[str, Any]:
    """
    Serialize a metric spec.

    This assumes concrete metric specs are dataclasses
    implementing `.to_dict()` (via SpecBase).
    """
    return spec.to_dict()
