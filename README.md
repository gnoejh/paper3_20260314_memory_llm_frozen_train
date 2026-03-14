# Persistent Memory for Frozen Encoder–Decoder LLMs

Official experiment code for the paper:

> **Persistent Memory for Frozen Encoder–Decoder LLMs: A Latent-Space Approach**
> John Doe
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

We train a **latent-space persistent-memory module** on top of a **frozen Flan-T5-base** backbone (248 M parameters, encoder–decoder, bfloat16).
Six architecturally distinct methods are compared against a stateless baseline using an absolute forgetting-curve protocol on the [LoCoMo](https://huggingface.co/datasets/KhangPTT373/locomo_preprocess) benchmark.

## Methods

| Key | Name | Description |
|---|---|---|
| `baseline` | Baseline | Stateless frozen backbone (no memory) |
| `m1_prefix` | M.1 Prefix | Learnable prefix tokens prepended to the encoder |
| `m2_xattn` | M.2 XAttn | Cross-attention memory injection |
| `m3_kv_ext` | M.3 KV Extension | Persistent key–value pairs concatenated to cross-attention |
| `m4_hebbian` | M.4 Hebbian | Hebbian associative memory with decay |
| `m5_gated` | M.5 Gated | Gated memory update with soft attention |
| `m6_slot` | M.6 Slot | Slot-based memory with sparse top-k write |

## Requirements

- Python ≥ 3.11
- CUDA-capable GPU (tested with CUDA 12.4)
- A HuggingFace account for downloading the LoCoMo dataset

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

> **Note:** PyTorch is installed from the cu124 index by default.
> If you use a different CUDA version, adjust the `[[tool.uv.index]]` in `pyproject.toml` or install PyTorch manually first.

## Usage

### Smoke test

Verifies that the backbone loads and all methods run correctly:

```bash
python main.py smoke
```

### Train all methods

Trains M.1–M.6 for 10 epochs (AdamW, lr=1e-4, cosine schedule) with the backbone frozen:

```bash
python main.py train
```

Checkpoints are saved to `checkpoints/` (one `.pt` file per method per epoch).

### Evaluate

Runs the absolute forgetting-curve evaluation on LoCoMo:

```bash
python main.py eval --preset full --output results.json
```

Presets control wall-clock time: `quick` (~1 h), `standard` (~2 h), `full` (~3 h).

### Train + Evaluate

```bash
python main.py run_all --preset full
```

### Generate LaTeX macros

Patches forgetting-curve coordinates in the paper's `main.tex`:

```bash
python gen_macros.py --results results.json
```

## Project structure

```
├── main.py          # CLI entry point (train / eval / smoke / run_all)
├── config.py        # Dataclass configs and hyperparameters
├── train.py         # Training loop (frozen backbone, adapter-only)
├── evaluate.py      # Absolute forgetting-curve evaluation
├── metrics.py       # Token F1, exact match, knowledge tracking
├── gen_macros.py    # Patch LaTeX figures from results.json
├── data/
│   ├── locomo.py    # LoCoMo dataset loader (HuggingFace)
│   └── msc.py       # MSC dataset loader
├── methods/
│   ├── base.py      # FrozenT5Backbone + BaseMemoryMethod ABC
│   ├── baseline.py  # Stateless baseline
│   ├── m1_prefix.py # M.1 Prefix
│   ├── m2_xattn.py  # M.2 Cross-Attention
│   ├── m3_kv_ext.py # M.3 KV Extension
│   ├── m4_hebbian.py# M.4 Hebbian
│   ├── m5_gated.py  # M.5 Gated
│   └── m6_slot.py   # M.6 Slot
└── pyproject.toml   # Project metadata and dependencies
```

## Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | Flan-T5-base (248 M, frozen) |
| Precision | bfloat16 |
| Optimiser | AdamW (lr=1e-4, wd=1e-2, warmup 200 steps) |
| Epochs | 10 |
| Effective batch size | 16 (4 × 4 grad accum) |
| TBPTT window | k = 8 |
| Memory prefix tokens | n_P = 64 |
| Hebbian dimension | d_h = 256 |
| Decay (γ) | 0.95 |
| Slot count (M.6) | S = 64, top-k = 8 |
| Seed | 42 |

## Citation

```bibtex
@article{doe2026persistent,
  title   = {Persistent Memory for Frozen Encoder--Decoder LLMs:
             A Latent-Space Approach},
  author  = {Doe, John},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE).
