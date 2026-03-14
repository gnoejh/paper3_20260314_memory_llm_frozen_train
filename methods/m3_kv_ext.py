"""M.3: Decoder KV Extension (Trained).

Read:  concatenate learned memory KV to frozen encoder KV at every decoder
       cross-attention layer.
Write: attention-coupled update.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseMemoryMethod, FrozenT5Backbone


class M3KVExtension(BaseMemoryMethod):
    requires_training = True

    def __init__(self, backbone: FrozenT5Backbone, *,
                 n_p: int = 64, gamma: float = 0.95, **kwargs):
        super().__init__(backbone)
        d = self.d_model
        self.n_p = n_p
        self.gamma = gamma

        self.P = nn.Parameter(torch.randn(1, n_p, d, device=self.device, dtype=self.dtype) * 0.02,
                              requires_grad=False)

        # Project P into pseudo-encoder hidden states; the frozen decoder
        # applies its own W_K / W_V during cross-attention (zero-init → safe startup).
        self.W_mem = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        nn.init.zeros_(self.W_mem.weight)

        # Write projections (frozen — write runs under no_grad)
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
        """Attention-coupled write: V from Z_t injects new content into P."""
        Q = self.W_Q(Z_t)           # (B, n, d)
        K = self.W_K(self.P)        # (B, n_p, d)
        V = self.W_V(Z_t)           # (B, n, d)  — content from current turn
        A = torch.softmax(Q @ K.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)  # (B, n, n_p)
        update = A.transpose(-2, -1) @ V   # (B, n_p, d)
        self.P = nn.Parameter(
            (self.gamma * self.P.data + update).detach(),
            requires_grad=False,
        )

    def _get_memory_kv(self, batch_size: int) -> torch.Tensor:
        """Project P into extra encoder-like hidden states for KV extension."""
        P_exp = self.P.expand(batch_size, -1, -1)
        return self.W_mem(P_exp)

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        extra = self._get_memory_kv(Z_t.size(0))
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        extra = self._get_memory_kv(Z_t.size(0))
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            max_new_tokens=max_new_tokens,
        )
