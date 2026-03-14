"""M.5: Context-Gated Decoder Memory Branch.

Read:  c_mem = XAttn_mem(s, P);  g = σ(W_g [s; c_mem] + b_g);  s' += g ⊙ c_mem
Write: attention-coupled update + gated write
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseMemoryMethod, FrozenT5Backbone


class M5Gated(BaseMemoryMethod):
    requires_training = True

    def __init__(self, backbone: FrozenT5Backbone, *,
                 n_p: int = 64, gamma: float = 0.95, **kwargs):
        super().__init__(backbone)
        d = self.d_model
        self.n_p = n_p
        self.gamma = gamma

        self.P = nn.Parameter(torch.randn(1, n_p, d, device=self.device, dtype=self.dtype) * 0.02,
                              requires_grad=False)

        # Memory cross-attention
        self.mem_q = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.mem_k = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.mem_v = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)

        # Context gate: σ(W_g [s; c_mem] + b_g)
        self.W_g = nn.Linear(2 * d, d, bias=True, device=self.device, dtype=self.dtype)
        # Init bias negative so gate starts near 0 → baseline behavior
        nn.init.constant_(self.W_g.bias, -2.0)

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

    def _gated_memory_read(self, Z_t: torch.Tensor) -> torch.Tensor:
        """Compute gated memory contribution."""
        P_exp = self.P.expand(Z_t.size(0), -1, -1)
        Q = self.mem_q(Z_t)
        K = self.mem_k(P_exp)
        V = self.mem_v(P_exp)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        c_mem = attn @ V  # (B, n, d)
        # Gate
        g = torch.sigmoid(self.W_g(torch.cat([Z_t, c_mem], dim=-1)))  # (B, n, d)
        return g * c_mem

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        extra = self._gated_memory_read(Z_t)
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        extra = self._gated_memory_read(Z_t)
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            max_new_tokens=max_new_tokens,
        )
