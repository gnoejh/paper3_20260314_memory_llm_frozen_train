"""Experiment-wide configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


MethodName = Literal[
    "baseline",
    "m1_prefix",
    "m2_xattn",
    "m3_kv_ext",
    "m4_hebbian",
    "m5_gated",
    "m6_slot",
]

ALL_METHODS: list[MethodName] = [
    "baseline",
    "m1_prefix",
    "m2_xattn",
    "m3_kv_ext",
    "m4_hebbian",
    "m5_gated",
    "m6_slot",
]

TRAINED_METHODS: list[MethodName] = [
    "m1_prefix", "m2_xattn", "m3_kv_ext",
    "m4_hebbian", "m5_gated", "m6_slot",
]

DatasetName = Literal["locomo", "msc"]


# Checkpoint objective tag used to prevent evaluating legacy adapters trained
# with the old next-turn objective against the QA benchmark.
CHECKPOINT_OBJECTIVE = "locomo_qa_v1"


@dataclass
class BackboneConfig:
    name: str = "google/flan-t5-base"
    max_src_len: int = 384
    max_tgt_len: int = 64
    dtype: str = "bfloat16"
    device: str = "auto"


@dataclass
class MemoryConfig:
    """Shared hyper-parameters for the persistent memory bank."""
    n_p: int = 64          # number of memory slots / prefix tokens
    gamma: float = 0.95    # decay for write rule
    d_h: int = 256         # Hebbian associative dimension (M.4)
    n_slots: int = 64      # slot count for M.6
    write_top_k: int = 8   # top-k for M.6 sparse write



@dataclass
class TrainConfig:
    batch_size: int = 4
    grad_accum: int = 4
    num_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    tbptt_k: int = 8       # truncated BPTT window
    seed: int = 42
    save_dir: str = "checkpoints"
    eval_every_epoch: int = 1
    # Dataset-specific overrides handled in DatasetConfig


@dataclass
class DatasetConfig:
    locomo_max_src: int = 384
    locomo_max_tgt: int = 64
    msc_max_src: int = 256
    msc_max_tgt: int = 64
    max_sessions: int | None = None
    max_turns_per_session: int | None = None


@dataclass
class ExperimentConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)
    methods: list[MethodName] = field(default_factory=lambda: list(ALL_METHODS))
    datasets: list[DatasetName] = field(default_factory=lambda: ["locomo"])
