"""CLI entry point for the persistent-memory experiment.

Usage:
    python main.py train   [--methods m1_prefix,m2_xattn,...] [--backbone google/flan-t5-base]
    python main.py eval    [--methods baseline,m1_prefix,...] [--datasets locomo]
    python main.py smoke   (quick sanity check with tiny data)
    python main.py run_all (train → eval → save results)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

import torch
from dotenv import load_dotenv

# Ensure UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load .env from repo root (two levels up from experiment/)
_env_path = pathlib.Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)

# Map HUGGINGFACE_API_KEY → HF_TOKEN (what huggingface_hub expects)
_hf_key = os.environ.get("HUGGINGFACE_API_KEY", "")
if _hf_key and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = _hf_key

# Add experiment dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ALL_METHODS, TRAINED_METHODS,
    ExperimentConfig, BackboneConfig, MemoryConfig, TrainConfig, DatasetConfig,
)
from methods.base import FrozenT5Backbone

log = logging.getLogger("experiment")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent Memory Experiment")
    sub = p.add_subparsers(dest="command", required=True)

    # --- train ---
    tr = sub.add_parser("train", help="Train methods M.1–M.6")
    tr.add_argument("--methods", type=str, default=",".join(TRAINED_METHODS))
    tr.add_argument("--backbone", type=str, default="google/flan-t5-base")
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--grad-accum", type=int, default=4)
    tr.add_argument("--save-dir", type=str, default="checkpoints")
    tr.add_argument("--dtype", type=str, default="bfloat16")
    tr.add_argument("--max-sessions", type=int, default=None)
    tr.add_argument("--max-turns", type=int, default=None)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("-v", "--verbose", action="store_true")

    # --- eval ---
    ev = sub.add_parser("eval", help="Evaluate absolute forgetting curves")
    ev.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    ev.add_argument("--datasets", type=str, default="locomo")
    ev.add_argument("--backbone", type=str, default="google/flan-t5-base")
    ev.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ev.add_argument("--output", type=str, default="results.json")
    ev.add_argument("--dtype", type=str, default="bfloat16")
    ev.add_argument("--max-sessions", type=int, default=None)
    ev.add_argument("--max-turns", type=int, default=None)
    ev.add_argument("--preset", type=str, default="full",
                    choices=["quick", "standard", "full"],
                    help="Speed preset: quick ~1h, standard ~2h, full ~3h")
    ev.add_argument("-v", "--verbose", action="store_true")

    # --- smoke ---
    sm = sub.add_parser("smoke", help="Quick sanity check")
    sm.add_argument("--backbone", type=str, default="google/flan-t5-small")
    sm.add_argument("-v", "--verbose", action="store_true")

    # --- run_all ---
    ra = sub.add_parser("run_all", help="Train + evaluate absolute forgetting curves")
    ra.add_argument("--backbone", type=str, default="google/flan-t5-base")
    ra.add_argument("--epochs", type=int, default=10)
    ra.add_argument("--save-dir", type=str, default="checkpoints")
    ra.add_argument("--output", type=str, default="results.json")
    ra.add_argument("--dtype", type=str, default="bfloat16")
    ra.add_argument("--max-sessions", type=int, default=None)
    ra.add_argument("--max-turns", type=int, default=None)
    ra.add_argument("--preset", type=str, default="full",
                    choices=["quick", "standard", "full"],
                    help="Speed preset: quick ~1h, standard ~2h, full ~3h")
    ra.add_argument("-v", "--verbose", action="store_true")

    return p.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build ExperimentConfig from CLI args."""
    backbone_name = getattr(args, "backbone", "google/flan-t5-base")
    dtype = getattr(args, "dtype", "bfloat16")
    methods = getattr(args, "methods", ",".join(ALL_METHODS)).split(",")
    datasets = getattr(args, "datasets", "locomo").split(",")

    return ExperimentConfig(
        backbone=BackboneConfig(name=backbone_name, dtype=dtype),
        memory=MemoryConfig(),
        training=TrainConfig(
            num_epochs=getattr(args, "epochs", 10),
            lr=getattr(args, "lr", 1e-4),
            batch_size=getattr(args, "batch", 4),
            grad_accum=getattr(args, "grad_accum", 4),
            save_dir=getattr(args, "save_dir", "checkpoints"),
            seed=getattr(args, "seed", 42),
        ),
        data=DatasetConfig(
            max_sessions=getattr(args, "max_sessions", None),
            max_turns_per_session=getattr(args, "max_turns", None),
        ),
        methods=methods,
        datasets=datasets,
    )


def cmd_train(args: argparse.Namespace) -> None:
    from train import train_all

    cfg = make_config(args)
    log.info("Loading backbone %s ...", cfg.backbone.name)
    backbone = FrozenT5Backbone(cfg.backbone.name, dtype=cfg.backbone.dtype)
    log.info("Backbone: d_model=%d, n_layers=%d, device=%s",
             backbone.d_model, backbone.n_layers, backbone.device)

    torch.manual_seed(cfg.training.seed)
    results = train_all(cfg, backbone)
    log.info("Training complete. Results: %s", json.dumps(
        {k: v["history"][-1] for k, v in results.items()}, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    from evaluate import evaluate_all, save_results, apply_eval_preset

    preset = getattr(args, "preset", "full")
    apply_eval_preset(preset)

    cfg = make_config(args)
    log.info("Loading backbone %s ...", cfg.backbone.name)
    backbone = FrozenT5Backbone(cfg.backbone.name, dtype=cfg.backbone.dtype)

    results = evaluate_all(
        cfg, backbone,
        checkpoint_dir=getattr(args, "checkpoint_dir", "checkpoints"),
    )
    save_results(results, path=getattr(args, "output", "results.json"))


def cmd_smoke(args: argparse.Namespace) -> None:
    """Quick smoke test: load backbone, instantiate all methods, run 1 turn."""
    from methods import build_method

    log.info("=== Smoke Test ===")
    backbone_name = getattr(args, "backbone", "google/flan-t5-small")
    log.info("Loading %s ...", backbone_name)
    backbone = FrozenT5Backbone(backbone_name, dtype="float32")
    log.info("Backbone loaded: d=%d, layers=%d, device=%s",
             backbone.d_model, backbone.n_layers, backbone.device)

    mem_cfg = MemoryConfig(n_p=8, d_h=64, n_slots=8)

    for method_name in ALL_METHODS:
        log.info("Testing %s ...", method_name)
        method = build_method(
            method_name,
            backbone=backbone,
            n_p=mem_cfg.n_p,
            gamma=mem_cfg.gamma,
            d_h=mem_cfg.d_h,
            n_slots=mem_cfg.n_slots,
            write_top_k=mem_cfg.write_top_k,
        )
        method.to(backbone.device)

        # Process a turn
        method.reset_memory()
        tok = backbone.tokenize_src("context: Hello, my name is Alice.", max_length=64)
        with torch.no_grad():
            Z = backbone.encode(tok["input_ids"], tok["attention_mask"])
        method.write(Z)
        method._turn_count += 1

        # Generate an answer
        answer = method.answer(
            "What is my name?",
            context_text="Hello, my name is Alice.",
            max_src_len=64,
            max_new_tokens=16,
        )
        params = method.trainable_param_count()
        log.info("  %s → '%s'  (trainable params: %d)", method_name, answer, params)

    log.info("=== Smoke test passed ===")


def cmd_run_all(args: argparse.Namespace) -> None:
    from train import train_all
    from evaluate import evaluate_all, save_results, apply_eval_preset

    preset = getattr(args, "preset", "full")
    apply_eval_preset(preset)

    cfg = make_config(args)
    cfg.methods = list(ALL_METHODS)
    cfg.datasets = ["locomo"]

    log.info("Loading backbone %s ...", cfg.backbone.name)
    backbone = FrozenT5Backbone(cfg.backbone.name, dtype=cfg.backbone.dtype)

    torch.manual_seed(cfg.training.seed)

    # Phase 1: Train
    log.info("=== Phase 1: Training ===")
    train_results = train_all(cfg, backbone)

    # Phase 2: Evaluate all
    log.info("=== Phase 2: Evaluation ===")
    eval_results = evaluate_all(cfg, backbone, checkpoint_dir=cfg.training.save_dir)
    save_results(eval_results, path=getattr(args, "output", "results.json"))
    log.info("=== Done ===")


def main() -> None:
    args = parse_args()
    setup_logging(getattr(args, "verbose", False))

    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "smoke": cmd_smoke,
        "run_all": cmd_run_all,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
