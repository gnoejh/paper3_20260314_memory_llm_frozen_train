"""M.4: Hebbian / Associative Memory.

Write: M_t = γ M_{t-1} + (Z W_K_H)^T (Z W_V_H)   (outer-product accumulation)
Read:  R_t = (Z W_Q_H) M_{t-1}  → project into decoder KV extension
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseMemoryMethod, FrozenT5Backbone


class M4Hebbian(BaseMemoryMethod):
    requires_training = True

    def __init__(self, backbone: FrozenT5Backbone, *,
                 d_h: int = 256, gamma: float = 0.95, **kwargs):
        super().__init__(backbone)
        d = self.d_model
        self.d_h = d_h
        self.gamma = gamma

        # Associative matrix M ∈ R^{d_h × d_h} (small random init for gradient flow)
        self.M = nn.Parameter(torch.randn(1, d_h, d_h, device=self.device, dtype=self.dtype) * 0.02,
                              requires_grad=False)

        # Hebbian projections
        # W_Q_H: small random init so R_t is non-zero → W_mem can bootstrap.
        # W_mem stays zero-init (safe startup: extra output = 0 at init).
        self.W_Q_H = nn.Linear(d, d_h, bias=False, device=self.device, dtype=self.dtype)
        nn.init.normal_(self.W_Q_H.weight, std=0.02)
        self.W_K_H = nn.Linear(d, d_h, bias=False, device=self.device, dtype=self.dtype)
        self.W_K_H.requires_grad_(False)
        self.W_V_H = nn.Linear(d, d_h, bias=False, device=self.device, dtype=self.dtype)
        self.W_V_H.requires_grad_(False)

        # Read: project retrieval back to model dim for decoder cross-attention
        # (zero-init so memory read starts silent; W_Q_H random provides gradient signal)
        self.W_mem = nn.Linear(d_h, d, bias=False, device=self.device, dtype=self.dtype)
        nn.init.zeros_(self.W_mem.weight)

    def reset_memory(self) -> None:
        super().reset_memory()
        nn.init.normal_(self.M, std=0.02)

    def write(self, Z_t: torch.Tensor) -> None:
        """Hebbian outer-product: M_t = γ M + mean(K_H^T V_H), Frobenius-normalised."""
        K_H = self.W_K_H(Z_t)  # (B, n, d_h)
        V_H = self.W_V_H(Z_t)  # (B, n, d_h)
        # Mean outer product over sequence positions (not sum → bounded update)
        n = K_H.size(1)
        update = K_H.transpose(-2, -1) @ V_H / n  # (B, d_h, d_h)
        raw = self.gamma * self.M.data + update
        # Frobenius normalisation: keep ||M||_F ≤ 1 so the read path stays bounded
        fnorm = raw.norm(p="fro", dim=(-2, -1), keepdim=True).clamp(min=1.0)
        self.M = nn.Parameter((raw / fnorm).detach(), requires_grad=False)

    def _read(self, Z_t: torch.Tensor) -> torch.Tensor:
        """Query associative memory → R_t ∈ R^{n × d_h}."""
        Q_H = self.W_Q_H(Z_t)   # (B, n, d_h)
        R_t = Q_H @ self.M.expand(Z_t.size(0), -1, -1)   # (B, n, d_h)
        return self.W_mem(R_t)  # (B, n, d)

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        extra = self._read(Z_t)
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        extra = self._read(Z_t)
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            max_new_tokens=max_new_tokens,
        )
