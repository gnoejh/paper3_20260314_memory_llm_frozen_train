"""Patch forgetting-curve coordinates in main.tex from results.json.

Usage (from the project root, i.e. 20260313_memory_llm_frozen_train/):
    python experiment/gen_macros.py [--results experiment/results.json]

Reads results.json produced by `python experiment/main.py eval` and:
    1. Writes a minimal results_macros.tex stub for compatibility.
    2. Patches the \addplot coordinates in main.tex for the forgetting-curve figure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


# Method key in results.json → (macro prefix, legend label)
METHOD_MAP = [
    ("baseline",   "Baseline", "Baseline"),
    ("m1_prefix",  "Mi",       "M.1 Prefix"),
    ("m2_xattn",   "Mii",      "M.2 XAttn"),
    ("m3_kv_ext",  "Miii",     "M.3 KV Ext"),
    ("m4_hebbian", "Miv",      "M.4 Hebbian"),
    ("m5_gated",   "Mv",       "M.5 Gated"),
    ("m6_slot",    "Mvi",      "M.6 Slot"),
]

BUCKET_ORDER = ["0_31", "32_63", "64_127", "128_255", "256_plus"]


def _require_forgetting_curve(results: dict, method_key: str) -> dict:
    """Validate that results contain a complete absolute forgetting curve."""
    if method_key not in results:
        raise ValueError(f"Missing results for method '{method_key}'.")

    mdata = results[method_key]
    locomo = mdata.get("locomo") if isinstance(mdata, dict) else None
    if not isinstance(locomo, dict):
        raise ValueError(f"Method '{method_key}' is missing LoCoMo results.")

    metrics = locomo.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Method '{method_key}' is missing metrics.")

    curve_type = metrics.get("curve_type")
    if curve_type != "absolute_retained_memory":
        raise ValueError(
            f"Method '{method_key}' has curve_type={curve_type!r}; "
            "expected 'absolute_retained_memory'."
        )

    curve = metrics.get("forgetting_curve")
    if not isinstance(curve, dict):
        raise ValueError(f"Method '{method_key}' is missing forgetting_curve data.")

    missing_buckets = [bucket for bucket in BUCKET_ORDER if bucket not in curve]
    if missing_buckets:
        raise ValueError(
            f"Method '{method_key}' is missing forgetting-curve buckets: {', '.join(missing_buckets)}."
        )

    return curve


def gen_macros(results: dict) -> str:
    return (
        "% Auto-generated from results.json — do not edit by hand.\n"
        "% Run:  python experiment/gen_macros.py\n"
        "% Scalar result macros were retired; the forgetting-curve figure is patched directly into main.tex.\n"
    )


def gen_curve_coords(results: dict) -> dict[str, str]:
    """Return {legend_label: '(1,0.12) (2,0.34) ...'} for forgetting-curve pgfplots."""
    coords: dict[str, str] = {}
    for method_key, _prefix, label in METHOD_MAP:
        curve = _require_forgetting_curve(results, method_key)
        pts = " ".join(
            f"({idx},{curve.get(bucket, {}).get('score', 0.0):.4f})"
            for idx, bucket in enumerate(BUCKET_ORDER, start=1)
        )
        coords[label] = pts
    return coords


def patch_main_tex(tex_path: pathlib.Path, coords: dict[str, str]) -> None:
    """Replace placeholder \\addplot coordinates in main.tex."""
    text = tex_path.read_text(encoding="utf-8")

    for label, pts in coords.items():
        # Pattern: \addplot coordinates {(...)};  followed by \addlegendentry{label}
        pattern = (
            r"(\\addplot\s+coordinates\s*\{)[^}]*(};)\s*\n"
            r"(\s*\\addlegendentry\{" + re.escape(label) + r"\})"
        )
        replacement = rf"\g<1>{pts}\g<2>\n\g<3>"
        text, n = re.subn(pattern, replacement, text, count=1)
        if n == 0:
            print(f"  Warning: could not find addplot for '{label}'", file=sys.stderr)

    tex_path.write_text(text, encoding="utf-8")
    print(f"  Updated {tex_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="experiment/results.json",
                   help="Path to results.json")
    p.add_argument("--tex", default="main.tex",
                   help="Path to main.tex")
    args = p.parse_args()

    results_path = pathlib.Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run the experiment first.",
              file=sys.stderr)
        sys.exit(1)

    results = json.loads(results_path.read_text(encoding="utf-8"))

    # 1. Generate macros
    macros_path = pathlib.Path("results_macros.tex")
    macros_path.write_text(gen_macros(results), encoding="utf-8")
    print(f"  Wrote {macros_path}")

    # 2. Patch forgetting curve in main.tex
    tex_path = pathlib.Path(args.tex)
    if tex_path.exists():
        coords = gen_curve_coords(results)
        patch_main_tex(tex_path, coords)
    else:
        print(f"  Warning: {tex_path} not found, skipping curve patch",
              file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
