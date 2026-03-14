"""Baseline: stateless frozen T5 (no persistent memory)."""

from __future__ import annotations

import torch

from .base import BaseMemoryMethod, FrozenT5Backbone


class Baseline(BaseMemoryMethod):
    """Stateless baseline — Eq. 1 in the paper.

    Z_t = E_frozen(x_t),  ŷ_t = D_frozen(Z_t)
    No persistent state whatsoever.
    """

    requires_training = False

    def __init__(self, backbone: FrozenT5Backbone, **kwargs):
        super().__init__(backbone)

    def write(self, Z_t: torch.Tensor) -> None:
        pass  # no memory

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
