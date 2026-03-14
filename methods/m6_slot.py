"""M.6: Slot-Based Memory with Sparse Write.

Fixed S slots, top-k sparse write, decoder KV extension for read.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseMemoryMethod, FrozenT5Backbone


class M6Slot(BaseMemoryMethod):
    requires_training = True

    def __init__(self, backbone: FrozenT5Backbone, *,
                 n_slots: int = 64, write_top_k: int = 8,
                 gamma: float = 0.95, **kwargs):
        super().__init__(backbone)
        d = self.d_model
        self.n_slots = n_slots
        self.top_k = write_top_k
        self.gamma = gamma

        # Slot bank P ∈ R^{S × d} (small random init for gradient flow)
        self.P = nn.Parameter(torch.randn(1, n_slots, d, device=self.device, dtype=self.dtype) * 0.02,
                              requires_grad=False)

        # Address network (frozen — write runs under no_grad)
        self.W_a = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.W_a.requires_grad_(False)
        # Candidate projection (frozen — write runs under no_grad)
        self.W_u = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        self.W_u.requires_grad_(False)

        # Read: project slots into pseudo-encoder hidden states (zero-init → safe startup)
        self.W_mem = nn.Linear(d, d, bias=False, device=self.device, dtype=self.dtype)
        nn.init.zeros_(self.W_mem.weight)

    def reset_memory(self) -> None:
        super().reset_memory()
        nn.init.normal_(self.P, std=0.02)

    def write(self, Z_t: torch.Tensor) -> None:
        """Sparse top-k slot update."""
        # Mean-pool current latent
        z_bar = Z_t.mean(dim=1)  # (B, d)

        # Address: softmax(z_bar W_a · P^T / √d)
        addr_query = self.W_a(z_bar)  # (B, d)
        scores = torch.softmax(
            addr_query.unsqueeze(1) @ self.P.data.transpose(-2, -1) / (self.d_model ** 0.5),
            dim=-1,
        ).squeeze(1)  # (B, S)

        # Top-k mask
        _, topk_idx = scores.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1.0)  # (B, S)

        # Candidate
        u_t = self.W_u(z_bar)  # (B, d)

        # Update selected slots
        P_new = self.P.data.clone()
        mask_3d = mask.unsqueeze(-1)  # (B, S, 1)
        u_3d = u_t.unsqueeze(1).expand_as(P_new)  # (B, S, d)
        P_new = (1 - mask_3d) * P_new + mask_3d * (self.gamma * P_new + (1 - self.gamma) * u_3d)
        self.P = nn.Parameter(P_new.detach(), requires_grad=False)

    def _read(self, batch_size: int) -> torch.Tensor:
        P_exp = self.P.expand(batch_size, -1, -1)
        return self.W_mem(P_exp)

    def forward_with_memory(self, Z_t, attention_mask, decoder_input_ids, labels=None):
        extra = self._read(Z_t.size(0))
        return self.backbone.decode_with_kv(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    def generate_with_memory(self, Z_t, attention_mask=None, max_new_tokens=64):
        extra = self._read(Z_t.size(0))
        return self.backbone.generate(
            encoder_hidden_states=Z_t,
            encoder_attention_mask=attention_mask,
            extra_encoder_hidden_states=extra,
            max_new_tokens=max_new_tokens,
        )
