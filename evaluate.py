"""Evaluation loop for absolute forgetting curves on LoCoMo.

For each QA pair, we compare the method's answer with accumulated persistent
memory against the same method with its persistent state forced to zero. The
absolute retained-memory score is the positive excess of the former over the
latter, so the stateless baseline is identically zero.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from tqdm import tqdm

from config import CHECKPOINT_OBJECTIVE, ExperimentConfig, ALL_METHODS
from data.locomo import load_locomo, locomo_turns_as_text
from data.msc import load_msc
from metrics import (
    ForgettingCurveTracker,
    MetricAccumulator,
    token_f1,
)
from methods.base import BaseMemoryMethod, FrozenT5Backbone

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval presets  (control wall-clock time)
# ---------------------------------------------------------------------------
EVAL_PRESETS: dict[str, dict] = {
    "quick": {        # ~1 hour
        "locomo_max_convs": 1,
        "msc_max_dialogues": 30,
        "locomo_max_new_tokens": 32,
    },
    "standard": {     # ~2 hours
        "locomo_max_convs": 2,
        "msc_max_dialogues": 50,
        "locomo_max_new_tokens": 32,
    },
    "full": {         # ~3 hours
        "locomo_max_convs": 3,
        "msc_max_dialogues": 100,
        "locomo_max_new_tokens": 32,
    },
}

# Active preset values (set by apply_eval_preset before evaluation)
LOCOMO_MAX_CONVS: int = 3
MSC_MAX_DIALOGUES: int = 100
LOCOMO_MAX_NEW_TOKENS: int = 32


def apply_eval_preset(name: str) -> None:
    """Activate an evaluation speed preset."""
    global LOCOMO_MAX_CONVS, MSC_MAX_DIALOGUES, LOCOMO_MAX_NEW_TOKENS
    preset = EVAL_PRESETS[name]
    LOCOMO_MAX_CONVS = preset["locomo_max_convs"]
    MSC_MAX_DIALOGUES = preset["msc_max_dialogues"]
    LOCOMO_MAX_NEW_TOKENS = preset["locomo_max_new_tokens"]
    log.info("Eval preset '%s': %d LoCoMo convs, %d MSC dlgs, max_new=%d",
             name, LOCOMO_MAX_CONVS, MSC_MAX_DIALOGUES, LOCOMO_MAX_NEW_TOKENS)


@dataclass
class EvalResult:
    method: str
    dataset: str
    metrics: dict
    latency_ms: float = 0.0


def _lag_from_evidence(total_turns: int, evidence_turn_ids: list[int]) -> int:
    """Return lag in turns using the oldest annotated supporting fact."""
    if evidence_turn_ids:
        return max((total_turns - 1) - min(evidence_turn_ids), 0)
    return total_turns


def _retained_memory_score(pred_with_memory: str, pred_without_memory: str, gold: str) -> float:
    """Absolute retained-memory score for one QA item."""
    with_score = token_f1(pred_with_memory, gold)
    without_score = token_f1(pred_without_memory, gold)
    return max(0.0, with_score - without_score)


def _zero_memory_state(method: BaseMemoryMethod) -> None:
    """Force a method into a memoryless control state without changing weights."""
    method.reset_memory()
    if hasattr(method, "P"):
        tensor = getattr(method, "P")
        if isinstance(tensor, nn.Parameter):
            setattr(method, "P", nn.Parameter(torch.zeros_like(tensor.data), requires_grad=False))
    if hasattr(method, "M"):
        tensor = getattr(method, "M")
        if isinstance(tensor, nn.Parameter):
            setattr(method, "M", nn.Parameter(torch.zeros_like(tensor.data), requires_grad=False))
    method._turn_count = 0


# ---------------------------------------------------------------------------
# LoCoMo evaluation
# ---------------------------------------------------------------------------

def evaluate_locomo(
    method_name: str,
    method: BaseMemoryMethod,
    cfg: ExperimentConfig,
    control_method: BaseMemoryMethod | None = None,
    split: str = "test",
) -> EvalResult:
    """Evaluate a method on LoCoMo with an absolute forgetting curve."""
    conversations = load_locomo(split=split)
    if cfg.data.max_sessions is not None:
        conversations = conversations[:cfg.data.max_sessions]
    # Apply preset conversation limit
    if len(conversations) > LOCOMO_MAX_CONVS:
        conversations = conversations[:LOCOMO_MAX_CONVS]

    backbone = method.backbone
    max_src = cfg.data.locomo_max_src
    fct = ForgettingCurveTracker()
    use_amp = backbone.device.type == "cuda"
    amp_dtype = torch.bfloat16

    t0 = time.time()

    for conv in tqdm(conversations, desc="LoCoMo", unit="conv"):
        turns = locomo_turns_as_text(conv)
        if cfg.data.max_turns_per_session:
            turns = turns[:cfg.data.max_turns_per_session]

        if method_name == "baseline":
            for qa in conv.qa_pairs:
                fct.record(_lag_from_evidence(len(turns), qa.evidence_turn_ids), 0.0)
            continue

        method.reset_memory()
        for turn_text in turns:
            tok = backbone.tokenize_src(f"context: {turn_text}", max_length=max_src)
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                Z_t = backbone.encode(tok["input_ids"], tok["attention_mask"])
            method.write(Z_t)
            method._turn_count += 1

        if control_method is not None:
            _zero_memory_state(control_method)

        for qa in conv.qa_pairs:
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred_with_memory = method.answer(
                    question_text=qa.question,
                    context_text="",
                    max_src_len=max_src,
                    max_new_tokens=LOCOMO_MAX_NEW_TOKENS,
                )
                pred_without_memory = control_method.answer(
                    question_text=qa.question,
                    context_text="",
                    max_src_len=max_src,
                    max_new_tokens=LOCOMO_MAX_NEW_TOKENS,
                ) if control_method is not None else pred_with_memory
            lag = _lag_from_evidence(len(turns), qa.evidence_turn_ids)
            retained_score = _retained_memory_score(
                pred_with_memory,
                pred_without_memory,
                qa.answer,
            )
            fct.record(lag, retained_score)

    elapsed = time.time() - t0
    result_metrics = {
        "curve_type": "absolute_retained_memory",
        "forgetting_curve": fct.summary(),
    }

    return EvalResult(
        method=method_name,
        dataset="locomo",
        metrics=result_metrics,
        latency_ms=elapsed * 1000 / max(len(conversations), 1),
    )


# ---------------------------------------------------------------------------
# MSC evaluation
# ---------------------------------------------------------------------------

def evaluate_msc(
    method: BaseMemoryMethod,
    cfg: ExperimentConfig,
    split: str = "test",
) -> EvalResult:
    """Evaluate a method on MSC (sessions 2–5 personalisation).

    For each dialogue: reset memory, process sessions 1–5, evaluate on
    turns from sessions 2–5 where the model must generate responses
    conditioned on conversation history and persistent memory.
    """
    dialogues = load_msc(split=split)
    if cfg.data.max_sessions is not None:
        dialogues = dialogues[:cfg.data.max_sessions]

    # Sub-sample for speed: 100 dialogues is statistically sufficient.
    if len(dialogues) > MSC_MAX_DIALOGUES:
        rng = random.Random(42)  # deterministic across methods
        dialogues = rng.sample(dialogues, MSC_MAX_DIALOGUES)
        log.info("Sampled %d / %d MSC dialogues", MSC_MAX_DIALOGUES, len(dialogues) + MSC_MAX_DIALOGUES)

    backbone = method.backbone
    max_src = cfg.data.msc_max_src
    acc = MetricAccumulator()
    use_amp = backbone.device.type == "cuda"
    amp_dtype = torch.bfloat16

    t0 = time.time()

    for dlg in tqdm(dialogues, desc="MSC", unit="dlg"):
        method.reset_memory()

        # Build full history and evaluate within each dialogue
        full_history: list[str] = []
        for sess_num in sorted(dlg.sessions.keys()):
            for i, turn in enumerate(dlg.sessions[sess_num]):
                turn_text = f"{turn.speaker}: {turn.text}"

                # Evaluate on sessions 2-5 (need at least one prior turn)
                if sess_num >= 2 and i > 0 and full_history:
                    # Only the last utterance (partner's turn) is the input.
                    # No conversation history as context — memory methods
                    # rely on P; baseline has nothing.
                    question = full_history[-1] if full_history else ""
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        pred = method.answer(
                            question_text=question,
                            context_text="",
                            max_src_len=max_src,
                            max_new_tokens=32,
                        )
                    acc.add(pred, turn.text, category=f"session_{sess_num}")

                # Write the current turn only after scoring it.
                # Otherwise the target response leaks into persistent memory.
                tok = backbone.tokenize_src(
                    f"context: {turn_text}", max_length=max_src,
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    Z_t = backbone.encode(tok["input_ids"], tok["attention_mask"])
                method.write(Z_t)
                method._turn_count += 1

                full_history.append(turn_text)

    elapsed = time.time() - t0

    result_metrics = {
        "personalisation_f1": acc.mean_f1("all") * 100,
        "per_session": acc.summary(),
    }

    return EvalResult(
        method=type(method).__name__,
        dataset="msc",
        metrics=result_metrics,
        latency_ms=elapsed * 1000 / max(len(dialogues), 1),
    )


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    cfg: ExperimentConfig,
    backbone: FrozenT5Backbone,
    checkpoint_dir: str | None = None,
) -> dict[str, dict[str, EvalResult]]:
    """Evaluate all methods on all datasets.

    Returns {method_name: {dataset_name: EvalResult}}.
    """
    from methods import build_method

    all_results: dict[str, dict[str, EvalResult]] = {}

    for method_name in tqdm(cfg.methods, desc="Methods", unit="method"):
        log.info("=== Evaluating %s ===", method_name)
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

        # Load checkpoint if trained method
        if method.requires_training and checkpoint_dir:
            ckpt_dir = pathlib.Path(checkpoint_dir)
            pattern = f"{type(method).__name__}_epoch*.pt"
            ckpts = sorted(ckpt_dir.glob(pattern),
                           key=lambda p: int(p.stem.split("epoch")[-1]))
            if ckpts:
                ckpt = torch.load(ckpts[-1], map_location=backbone.device, weights_only=True)
                objective = ckpt.get("train_objective")
                if objective != CHECKPOINT_OBJECTIVE:
                    raise RuntimeError(
                        f"Checkpoint {ckpts[-1].name} was trained with objective {objective!r}; "
                        f"expected {CHECKPOINT_OBJECTIVE!r}. Retrain adapters before evaluation."
                    )
                method.load_state_dict(ckpt["adapter_state"], strict=False)
                log.info("Loaded checkpoint %s (epoch %d)", ckpts[-1].name, ckpt["epoch"])
            else:
                log.warning("No checkpoint found for %s — evaluating untrained", method_name)

        method_results: dict[str, EvalResult] = {}
        control_method: BaseMemoryMethod | None = None
        if "locomo" in cfg.datasets and method_name != "baseline":
            control_method = build_method(
                method_name,
                backbone=backbone,
                n_p=cfg.memory.n_p,
                gamma=cfg.memory.gamma,
                d_h=cfg.memory.d_h,
                n_slots=cfg.memory.n_slots,
                write_top_k=cfg.memory.write_top_k,
            )
            control_method.to(backbone.device)
            control_method.load_state_dict(method.state_dict(), strict=False)
            _zero_memory_state(control_method)

        if "locomo" in cfg.datasets:
            log.info("  → LoCoMo")
            result = evaluate_locomo(method_name, method, cfg, control_method=control_method)
            method_results["locomo"] = result
            curve = result.metrics["forgetting_curve"]
            start = curve.get("0_31", {}).get("score", 0.0)
            end = curve.get("256_plus", {}).get("score", 0.0)
            log.info("    forgetting curve: start=%.1f  end=%.1f", start, end)

        if "msc" in cfg.datasets:
            log.warning(
                "  → MSC skipped: the rebuilt evaluation reports only absolute forgetting curves, "
                "which require explicit evidence-turn annotations that MSC does not provide."
            )

        all_results[method_name] = method_results

    return all_results


def save_results(
    results: dict[str, dict[str, EvalResult]],
    path: str = "results.json",
) -> None:
    """Flatten results to JSON-serializable dict and save."""
    flat: dict = {}
    for method_name, datasets in results.items():
        flat[method_name] = {}
        for ds_name, result in datasets.items():
            flat[method_name][ds_name] = {
                "metrics": result.metrics,
                "latency_ms": result.latency_ms,
            }

    out = pathlib.Path(path)
    out.write_text(json.dumps(flat, indent=2, default=str), encoding="utf-8")
    log.info("Results saved → %s", out)
