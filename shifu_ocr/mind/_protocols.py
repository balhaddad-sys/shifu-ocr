"""
Structural typing contracts for the Shifu cognitive architecture.

These are Protocol classes — a module satisfies the contract by having
the right method signatures, not by inheriting from a base class.
This matches the existing duck-typing convention in ShifuEnsemble.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable


@runtime_checkable
class Feedable(Protocol):
    """Can absorb information from input."""
    def feed(self, data: Any) -> int: ...


@runtime_checkable
class Queryable(Protocol):
    """Can answer questions about its knowledge."""
    def activate(self, key: str) -> Dict[str, float]: ...


@runtime_checkable
class Temporal(Protocol):
    """Tracks its own temporal history."""
    @property
    def birth_epoch(self) -> int: ...
    @property
    def last_active(self) -> int: ...
    @property
    def myelinated(self) -> bool: ...
    def decay(self, factor: float) -> float: ...


@runtime_checkable
class Serializable(Protocol):
    """Can persist and restore its state."""
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> Any: ...
