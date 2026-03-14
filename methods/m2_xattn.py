"""M.2: Parallel Decoder Cross-Attention.

Read:  c_mem = XAttn_mem(s, P);  s' = s + XAttn_frozen(s, Z) + β · c_mem
Write: attention-coupled update
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseMemoryMethod, FrozenT5Backbone


class M2XAttn(BaseMemoryMethod):
    requires_training = True

    def __init__(self, backbone: FrozenT5Backbone, *,
                 n_p: int = 64, gamma: float = 0.95, **kwargs):
        super().__init__(backbone)
        d = self.d_model
        self.n_p = n_p
        self.gamma = gamma

        self.P = nn.Parameter(torch.randn(1, n_p, d, device=self.device, dtype=self.dtype) * 0.02,
                              requires_grad=False)

        # Per-layer memory cross-attention (shared across layers for param efficiency)
        self.mem_q = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.mem_k = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.mem_v = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.mem_o = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)

        # Mixing coefficient β per layer, init 0 → starts as baseline
        self.beta = nn.Parameter(torch.zeros(backbone.n_layers, device=self.device, dtype=self.dtype))

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

    def _memory_xattn(self, s: torch.Tensor) -> torch.Tensor:
        """Compute memory cross-attention: softmax(s W_Q · P W_K / √d) · P W_V."""
        Q = self.mem_q(s)
        K = self.mem_k(self.P.expand(s.size(0), -1, -1))
        V = self.mem_v(self.P.expand(s.size(0), -1, -1))
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (self.d_model ** 0.5), dim=-1)
        return self.mem_o(attn @ V)

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        # We cannot easily inject into each decoder layer, so we approximate
        # by adding memory as extra KV (equivalent effect for a single read head)
        c_mem = self._memory_xattn(Z_t)  # (B, n, d) — use Z as proxy for decoder states
        extra = c_mem * self.beta.mean()  # simplified: single scalar blend
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        c_mem = self._memory_xattn(Z_t)
        extra = c_mem * self.beta.mean()
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            max_new_tokens=max_new_tokens,
        )
