"""Training loop for trained methods (M.1–M.6).

Only θ_mem is optimised; the backbone stays frozen.
Uses truncated BPTT with gradient accumulation, AMP (bfloat16),
and cosine-annealing LR.  Saves adapter-only checkpoints.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import CHECKPOINT_OBJECTIVE, ExperimentConfig, TRAINED_METHODS
from data.locomo import load_locomo, locomo_turns_as_text
from methods.base import BaseMemoryMethod, FrozenT5Backbone

log = logging.getLogger(__name__)


def _make_teacher_forcing_ids(
    backbone: FrozenT5Backbone,
    answer_text: str,
    max_tgt_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenise answer into decoder_input_ids and labels."""
    enc = backbone.tokenize_tgt(answer_text, max_length=max_tgt_len)
    input_ids = enc["input_ids"]  # (1, L)
    # Shift right for teacher forcing
    decoder_input_ids = input_ids.new_zeros(input_ids.shape)
    decoder_input_ids[:, 1:] = input_ids[:, :-1]
    decoder_input_ids[:, 0] = backbone.config.decoder_start_token_id
    labels = input_ids.clone()
    labels[labels == backbone.tokenizer.pad_token_id] = -100
    return decoder_input_ids, labels


def train_method(
    method: BaseMemoryMethod,
    cfg: ExperimentConfig,
) -> dict:
    """Train a single method on LoCoMo QA supervision.

    Returns a summary dict with loss curves and timing.
    """
    assert method.requires_training, f"{type(method).__name__} does not require training"

    backbone = method.backbone
    device = backbone.device
    max_src = cfg.data.locomo_max_src
    max_tgt = cfg.data.locomo_max_tgt

    # Load training data
    conversations = load_locomo(split="train")
    if cfg.data.max_sessions is not None:
        conversations = conversations[:cfg.data.max_sessions]

    # Optimiser setup — only adapter params
    trainable = method.trainable_parameters()
    log.info("Trainable params: %d (%.3fM)",
             method.trainable_param_count(),
             method.trainable_param_count() / 1e6)

    optimizer = AdamW(trainable, lr=cfg.training.lr,
                      weight_decay=cfg.training.weight_decay)
    total_steps = cfg.training.num_epochs * sum(
        len(c.qa_pairs) for c in conversations
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16

    save_dir = pathlib.Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    global_step = 0
    epoch_grad_norms: list[float] = []
    t0 = time.time()

    for epoch in range(cfg.training.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        total_qas = 0
        for c in conversations:
            turns = locomo_turns_as_text(c)
            if cfg.data.max_turns_per_session:
                turns = turns[:cfg.data.max_turns_per_session]
            max_turn_idx = len(turns)
            total_qas += sum(
                1
                for qa in c.qa_pairs
                if not qa.evidence_turn_ids or min(qa.evidence_turn_ids) < max_turn_idx
            )
        pbar = tqdm(total=int(total_qas),
                    desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}",
                    unit="qa", leave=True)

        for conv in conversations:
            method.reset_memory()
            turns = locomo_turns_as_text(conv)
            if cfg.data.max_turns_per_session:
                turns = turns[:cfg.data.max_turns_per_session]

            # Build persistent memory from the conversation turns.  No gradients
            # flow through writes, matching the evaluation protocol.
            for turn_text in turns:
                tok = backbone.tokenize_src(
                    f"context: {turn_text}", max_length=max_src,
                )
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    Z_t = backbone.encode(tok["input_ids"], tok["attention_mask"])
                method.write(Z_t.detach())
                method._turn_count += 1

            qa_pairs = conv.qa_pairs
            if cfg.data.max_turns_per_session:
                max_turn_idx = len(turns)
                qa_pairs = [
                    qa for qa in qa_pairs
                    if not qa.evidence_turn_ids or min(qa.evidence_turn_ids) < max_turn_idx
                ]

            chunk_loss = torch.tensor(0.0, device=device)
            chunk_n = 0
            accum_count = 0

            for qa_idx, qa in enumerate(qa_pairs):
                src = f"question: {qa.question}"
                tok = backbone.tokenize_src(src, max_length=max_src)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    Z_q = backbone.encode(tok["input_ids"], tok["attention_mask"])

                dec_ids, labels = _make_teacher_forcing_ids(backbone, qa.answer, max_tgt)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    _, loss = method.forward_with_memory(
                        Z_q, tok["attention_mask"], dec_ids, labels,
                    )

                if loss is not None:
                    chunk_loss = chunk_loss + loss
                    chunk_n += 1
                    epoch_loss += loss.item()
                    epoch_steps += 1

                if chunk_n >= cfg.training.tbptt_k or qa_idx == len(qa_pairs) - 1:
                    if chunk_n > 0:
                        scaled = (chunk_loss / chunk_n) / cfg.training.grad_accum
                        scaled.backward()
                        accum_count += 1
                    chunk_loss = torch.tensor(0.0, device=device)
                    chunk_n = 0

                    if accum_count >= cfg.training.grad_accum or qa_idx == len(qa_pairs) - 1:
                        if accum_count > 0:
                            gn = nn.utils.clip_grad_norm_(trainable, cfg.training.max_grad_norm)
                            epoch_grad_norms.append(gn.item())
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                            accum_count = 0
                            global_step += 1

                pbar.update(1)
                pbar.set_postfix(loss=f"{epoch_loss / max(epoch_steps, 1):.4f}")

        pbar.close()
        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_gn = sum(epoch_grad_norms) / max(len(epoch_grad_norms), 1) if epoch_grad_norms else 0.0
        elapsed = time.time() - t0
        log.info("Epoch %d/%d  loss=%.4f  grad_norm=%.6f  step=%d  time=%.0fs",
                 epoch + 1, cfg.training.num_epochs, avg_loss, avg_gn, global_step, elapsed)
        history.append({"epoch": epoch + 1, "loss": avg_loss, "grad_norm": avg_gn, "global_step": global_step})
        epoch_grad_norms.clear()

        # Save checkpoint
        adapter_state = {
            k: v for k, v in method.state_dict().items()
            if not k.startswith("backbone.")
        }
        ckpt_path = save_dir / f"{type(method).__name__}_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_objective": CHECKPOINT_OBJECTIVE,
            "adapter_state": adapter_state,
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
        }, ckpt_path)
        log.info("Saved checkpoint → %s", ckpt_path)

    return {"history": history, "total_time_s": time.time() - t0}


def train_all(cfg: ExperimentConfig, backbone: FrozenT5Backbone) -> dict:
    """Train all trained methods sequentially."""
    from methods import build_method

    results = {}
    for method_name in cfg.methods:
        if method_name not in TRAINED_METHODS:
            continue
        log.info("=== Training %s ===", method_name)
        method = build_method(
            method_name,
            backbone=backbone,
            n_p=cfg.memory.n_p,
            gamma=cfg.memory.gamma,
            d_h=cfg.memory.d_h,
            n_slots=cfg.memory.n_slots,
            write_top_k=cfg.memory.write_top_k,
        )
        method.to(backbone.device)
        result = train_method(method, cfg)
        results[method_name] = result
        log.info("Finished %s — final loss %.4f", method_name, result["history"][-1]["loss"])

    return results
