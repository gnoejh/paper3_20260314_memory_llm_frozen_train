"""M.1: Memory as Encoder-Input Prefix (simplified).

Paper formulation: prepend S_t = P W_P to encoder input embeddings, so
encoder self-attention mixes memory with current tokens.

Implementation: project P into soft prefix tokens passed as
extra_encoder_hidden_states, so the decoder cross-attends to [Z_t; S_t].
This keeps the training loop uniform across all methods.

Write: attention-coupled update  P_t = γ P_{t-1} + A^T V
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseMemoryMethod, FrozenT5Backbone


class M1Prefix(BaseMemoryMethod):
    requires_training = True

    def __init__(self, backbone: FrozenT5Backbone, *,
                 n_p: int = 64, gamma: float = 0.95, **kwargs):
        super().__init__(backbone)
        d = self.d_model
        self.n_p = n_p
        self.gamma = gamma

        # Persistent memory bank (small random init so read-side gradients flow)
        self.P = nn.Parameter(torch.randn(1, n_p, d, device=self.device, dtype=self.dtype) * 0.02,
                              requires_grad=False)

        # Read projection: soft prefix in encoder-output space
        # (zero-init so memory starts silent — safe startup)
        self.W_P = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        nn.init.zeros_(self.W_P.weight)

        # Write projections (attention-coupled; frozen — write runs under no_grad)
        self.W_Q = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.W_Q.requires_grad_(False)
        self.W_K = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.W_K.requires_grad_(False)
        self.W_V = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.W_V.requires_grad_(False)

    def reset_memory(self) -> None:
        super().reset_memory()
        nn.init.normal_(self.P, std=0.02)

    def write(self, Z_t: torch.Tensor) -> None:
        """Attention-coupled write: P_t = γ P + A^T V.

        Q and K address which memory slots are relevant to the current turn.
        V comes from Z_t so that *new content* actually enters memory.
        A^T aggregates the current-turn values into the n_p memory rows.
        """
        Q = self.W_Q(Z_t)           # (B, n, d)
        K = self.W_K(self.P)        # (B, n_p, d)
        V = self.W_V(Z_t)           # (B, n, d)  — content from current turn
        A = torch.softmax(Q @ K.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)  # (B, n, n_p)
        update = A.transpose(-2, -1) @ V   # (B, n_p, d)
        self.P = nn.Parameter(
            (self.gamma * self.P.data + update).detach(),
            requires_grad=False,
        )

    def _get_prefix(self, batch_size: int) -> torch.Tensor:
        """Project P into soft prefix tokens."""
        return self.W_P(self.P.expand(batch_size, -1, -1))  # (B, n_p, d)

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        extra = self._get_prefix(Z_t.size(0))
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        extra = self._get_prefix(Z_t.size(0))
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            max_new_tokens=max_new_tokens,
        )
