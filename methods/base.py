"""Frozen T5 backbone and base class for all memory methods.

Every method shares:
  - A frozen encoder E and frozen decoder D (from Flan-T5)
  - A tokenizer
  - A persistent memory state P_t
  - write(P_{t-1}, Z_t) -> P_t
  - forward(x_t, P_{t-1}) -> logits, P_t
"""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    T5EncoderModel,
    T5ForConditionalGeneration,
)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype]


class FrozenT5Backbone(nn.Module):
    """Holds the frozen Flan-T5 encoder, decoder, lm_head, and tokenizer."""

    def __init__(self, name: str = "google/flan-t5-base",
                 device: str = "auto", dtype: str = "bfloat16"):
        super().__init__()
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
            name, torch_dtype=self.dtype,
        )
        self.encoder = t5.encoder
        self.decoder = t5.decoder
        self.lm_head = t5.lm_head
        self.shared = t5.shared
        self.config = t5.config
        self.d_model: int = t5.config.d_model
        self.n_layers: int = t5.config.num_decoder_layers
        self.n_heads: int = t5.config.num_heads
        self.d_kv: int = t5.config.d_kv

        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)

        self.to(self.device)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run the frozen encoder -> Z_t ∈ R^{n × d}."""
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def decode_with_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        extra_encoder_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the frozen decoder with optional KV extension.

        If extra_encoder_hidden_states is provided, it is concatenated to
        encoder_hidden_states along the sequence dimension before cross-attention.

        Returns (logits, loss_or_None).
        """
        if extra_encoder_hidden_states is not None:
            enc_hs = torch.cat([encoder_hidden_states, extra_encoder_hidden_states], dim=1)
            if encoder_attention_mask is not None:
                extra_mask = torch.ones(
                    extra_encoder_hidden_states.shape[:2],
                    device=encoder_attention_mask.device,
                    dtype=encoder_attention_mask.dtype,
                )
                enc_mask = torch.cat([encoder_attention_mask, extra_mask], dim=1)
            else:
                enc_mask = None
        else:
            enc_hs = encoder_hidden_states
            enc_mask = encoder_attention_mask

        # NOTE: no torch.no_grad() here — gradients must flow through
        # trainable memory projections injected via extra_encoder_hidden_states.
        # Backbone weights are already frozen (requires_grad=False).
        out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc_hs,
            encoder_attention_mask=enc_mask,
        )
        hidden = out.last_hidden_state
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100,
            )
        return logits, loss

    def tokenize_src(self, text: str, max_length: int = 384) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def tokenize_tgt(self, text: str, max_length: int = 64) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text, max_length=max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        extra_encoder_hidden_states: torch.Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Greedy decode from encoder hidden states with KV caching."""
        if extra_encoder_hidden_states is not None:
            enc_hs = torch.cat([encoder_hidden_states, extra_encoder_hidden_states], dim=1)
            if encoder_attention_mask is not None:
                extra_mask = torch.ones(
                    extra_encoder_hidden_states.shape[:2],
                    device=encoder_attention_mask.device,
                    dtype=encoder_attention_mask.dtype,
                )
                enc_mask = torch.cat([encoder_attention_mask, extra_mask], dim=1)
            else:
                enc_mask = None
        else:
            enc_hs = encoder_hidden_states
            enc_mask = encoder_attention_mask

        # Start with decoder_start_token_id
        dec_ids = torch.full(
            (enc_hs.size(0), 1),
            self.config.decoder_start_token_id,
            dtype=torch.long, device=self.device,
        )
        eos_id = self.config.eos_token_id
        past_key_values = None

        for _ in range(max_new_tokens):
            out = self.decoder(
                input_ids=dec_ids if past_key_values is None else dec_ids[:, -1:],
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            logits = self.lm_head(out.last_hidden_state[:, -1:, :])
            next_id = logits.argmax(dim=-1)
            dec_ids = torch.cat([dec_ids, next_id], dim=1)
            if (next_id == eos_id).all():
                break

        return self.tokenizer.decode(dec_ids[0, 1:], skip_special_tokens=True)


class BaseMemoryMethod(nn.Module, abc.ABC):
    """Abstract base for all 11 configurations.

    Subclasses must implement:
      - write(Z_t) -> None   (update self.P in-place or return new P)
      - forward_with_memory(Z_t, attention_mask, decoder_input_ids, labels)
      - generate_with_memory(Z_t, attention_mask, max_new_tokens) -> str

    The base class manages the backbone and provides helpers.
    """

    requires_training: bool = False  # override in trained methods

    def __init__(self, backbone: FrozenT5Backbone, **kwargs: Any):
        super().__init__()
        self.backbone = backbone
        self.d_model = backbone.d_model
        self.device = backbone.device
        self.dtype = backbone.dtype
        self._turn_count = 0

    def reset_memory(self) -> None:
        """Reset persistent state for a new conversation."""
        self._turn_count = 0

    @abc.abstractmethod
    def write(self, Z_t: torch.Tensor) -> None:
        """Update persistent memory P_t given new encoder output Z_t."""
        ...

    @abc.abstractmethod
    def forward_with_memory(
        self,
        Z_t: torch.Tensor,
        attention_mask: torch.Tensor | None,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass using memory. Returns (logits, loss)."""
        ...

    @abc.abstractmethod
    def generate_with_memory(
        self,
        Z_t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate text using memory."""
        ...

    def process_turn(
        self,
        input_text: str,
        max_src_len: int = 384,
    ) -> torch.Tensor:
        """Encode a turn and update memory. Returns Z_t."""
        tok = self.backbone.tokenize_src(input_text, max_length=max_src_len)
        Z_t = self.backbone.encode(tok["input_ids"], tok["attention_mask"])
        self.write(Z_t)
        self._turn_count += 1
        return Z_t

    def answer(
        self,
        question_text: str,
        context_text: str = "",
        max_src_len: int = 384,
        max_new_tokens: int = 64,
    ) -> str:
        """Answer a question using current memory state."""
        # Question first so it survives right-truncation when context is long.
        src = f"question: {question_text} context: {context_text}" if context_text else f"question: {question_text}"
        tok = self.backbone.tokenize_src(src, max_length=max_src_len)
        Z_t = self.backbone.encode(tok["input_ids"], tok["attention_mask"])
        return self.generate_with_memory(Z_t, tok["attention_mask"], max_new_tokens)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())
