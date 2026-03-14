"""Persistent-memory method registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .baseline import Baseline
from .m1_prefix import M1Prefix
from .m2_xattn import M2XAttn
from .m3_kv_ext import M3KVExtension
from .m4_hebbian import M4Hebbian
from .m5_gated import M5Gated
from .m6_slot import M6Slot

if TYPE_CHECKING:
    from .base import BaseMemoryMethod

METHOD_REGISTRY: dict[str, type[BaseMemoryMethod]] = {
    "baseline": Baseline,
    "m1_prefix": M1Prefix,
    "m2_xattn": M2XAttn,
    "m3_kv_ext": M3KVExtension,
    "m4_hebbian": M4Hebbian,
    "m5_gated": M5Gated,
    "m6_slot": M6Slot,
}


def build_method(name: str, **kwargs) -> BaseMemoryMethod:
    """Instantiate a method by config name."""
    cls = METHOD_REGISTRY[name]
    return cls(**kwargs)
