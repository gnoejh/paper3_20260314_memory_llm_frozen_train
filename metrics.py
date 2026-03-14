"""Evaluation helpers for absolute forgetting-curve analysis."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Text normalisation (SQuAD convention)
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Lower-case, strip articles, punctuation, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = normalise(prediction).split()
    gold_tokens = normalise(gold).split()
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------

def exact_match(prediction: str, gold: str) -> float:
    """1.0 if normalised prediction equals normalised gold, else 0.0."""
    return 1.0 if normalise(prediction) == normalise(gold) else 0.0


# ---------------------------------------------------------------------------
# Knowledge metric K_t and derived quantities
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeTracker:
    """Tracks conversational knowledge K_t across session boundaries.

    Call `record_probe(f1)` with the raw token-F1 for each probe question.
    After all probes at a session boundary, call `close_session(session_idx)`.
    K_t is the mean F1 at each session boundary (continuous, not binary).
    """

    # session_idx → K_t value
    session_scores: dict[int, float] = field(default_factory=dict)
    _current_sum: float = 0.0
    _current_total: int = 0

    def record_probe(self, f1: float) -> None:
        """Record a raw F1 value for the current session."""
        self._current_sum += f1
        self._current_total += 1

    def close_session(self, session_idx: int) -> float:
        """Finalise K_t for this session boundary."""
        if self._current_total == 0:
            k_t = 0.0
        else:
            k_t = self._current_sum / self._current_total
        self.session_scores[session_idx] = k_t
        self._current_sum = 0.0
        self._current_total = 0
        return k_t

    def delta_k(self) -> float:
        """ΔK = K_last - K_first.  Net knowledge gain."""
        if len(self.session_scores) < 2:
            return 0.0
        keys = sorted(self.session_scores)
        return self.session_scores[keys[-1]] - self.session_scores[keys[0]]

    def rho(self) -> float:
        """ρ = K_last / K_first.  Forgetting resistance (1 = no forgetting)."""
        if len(self.session_scores) < 2:
            return 0.0
        keys = sorted(self.session_scores)
        k_first = self.session_scores[keys[0]]
        if k_first == 0.0:
            return 0.0
        return self.session_scores[keys[-1]] / k_first

    def as_curve(self) -> list[tuple[int, float]]:
        """Return sorted (session_idx, K_t) pairs for plotting."""
        return sorted(self.session_scores.items())


# ---------------------------------------------------------------------------
# Retention-by-distance tracker
# ---------------------------------------------------------------------------

@dataclass
class RetentionTracker:
    """Bins (distance, F1) into retention-distance buckets.

    For each QA pair, *distance* = total_turns − min(evidence_turn_ids),
    i.e. how many turns ago the answer evidence appeared.

    Default bins:
        near   [0, 100)   — evidence in last 100 turns
        mid    [100, 300)  — medium-term recall
        far    [300, ∞)    — long-term recall
    """

    BINS: list[tuple[int, float, str]] = field(default_factory=lambda: [
        (0, 100, "near"),
        (100, 300, "mid"),
        (300, float("inf"), "far"),
    ])
    _records: list[tuple[int, float]] = field(default_factory=list)

    def record(self, distance: int, f1: float) -> None:
        self._records.append((distance, f1))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ``{bin_name: {"f1": …, "n": …}}``."""
        result: dict[str, dict[str, float]] = {}
        for lo, hi, name in self.BINS:
            vals = [f1 for d, f1 in self._records if lo <= d < hi]
            result[name] = {
                "f1": (sum(vals) / len(vals) * 100) if vals else 0.0,
                "n": len(vals),
            }
        return result


@dataclass
class ForgettingCurveTracker:
    """Track absolute retained-memory score as a function of turn lag.

    The recorded value is not raw QA F1. It is the method-internal retained
    memory score after ablating that same method's persistent state, which makes
    the stateless baseline identically zero.
    """

    BINS: list[tuple[int, float, str]] = field(default_factory=lambda: [
        (0, 32, "0_31"),
        (32, 64, "32_63"),
        (64, 128, "64_127"),
        (128, 256, "128_255"),
        (256, float("inf"), "256_plus"),
    ])
    _records: list[tuple[int, float]] = field(default_factory=list)

    def record(self, lag_turns: int, score: float) -> None:
        self._records.append((lag_turns, score))

    @staticmethod
    def _pava_non_decreasing(values: list[float], weights: list[float]) -> list[float]:
        """Weighted pool-adjacent-violators fit for a non-decreasing sequence."""
        if not values:
            return []

        blocks: list[dict[str, float | int]] = []
        for idx, (value, weight) in enumerate(zip(values, weights)):
            blocks.append({
                "start": idx,
                "end": idx,
                "sum_w": float(weight),
                "sum_y": float(weight) * float(value),
            })
            while len(blocks) >= 2:
                left = blocks[-2]
                right = blocks[-1]
                left_mean = float(left["sum_y"]) / max(float(left["sum_w"]), 1e-12)
                right_mean = float(right["sum_y"]) / max(float(right["sum_w"]), 1e-12)
                if left_mean <= right_mean:
                    break
                merged = {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "sum_w": float(left["sum_w"]) + float(right["sum_w"]),
                    "sum_y": float(left["sum_y"]) + float(right["sum_y"]),
                }
                blocks[-2:] = [merged]

        fitted = [0.0] * len(values)
        for block in blocks:
            mean = float(block["sum_y"]) / max(float(block["sum_w"]), 1e-12)
            for idx in range(int(block["start"]), int(block["end"]) + 1):
                fitted[idx] = mean
        return fitted

    @classmethod
    def _pava_non_increasing(cls, values: list[float], weights: list[float]) -> list[float]:
        """Weighted isotonic fit constrained to be non-increasing."""
        fitted = cls._pava_non_decreasing([-value for value in values], weights)
        return [-value for value in fitted]

    def summary(self) -> dict[str, dict[str, float | int | None]]:
        """Return bucketed absolute forgetting-curve stats.

        The reported ``score`` is a weighted non-increasing isotonic fit of the
        raw bucket means. ``raw_score`` is preserved for auditability.
        """
        bucket_names: list[str] = []
        raw_scores: list[float] = []
        counts: list[int] = []
        bounds: list[tuple[int, int | None]] = []

        for lo, hi, name in self.BINS:
            vals = [score for lag, score in self._records if lo <= lag < hi]
            bucket_names.append(name)
            raw_scores.append((sum(vals) / len(vals) * 100) if vals else 0.0)
            counts.append(len(vals))
            bounds.append((lo, None if hi == float("inf") else int(hi)))

        fitted_scores = self._pava_non_increasing(
            raw_scores,
            [max(count, 1) for count in counts],
        )

        result: dict[str, dict[str, float | int | None]] = {}
        for name, raw_score, fitted_score, count, (lag_lo, lag_hi) in zip(
            bucket_names,
            raw_scores,
            fitted_scores,
            counts,
            bounds,
        ):
            result[name] = {
                "score": fitted_score,
                "raw_score": raw_score,
                "n": count,
                "lag_lo": lag_lo,
                "lag_hi": lag_hi,
            }
        return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

@dataclass
class MetricAccumulator:
    """Accumulates F1 and EM scores, optionally by category."""

    scores: dict[str, list[float]] = field(default_factory=dict)
    em_scores: dict[str, list[float]] = field(default_factory=dict)

    def add(self, prediction: str, gold: str, category: str = "all") -> None:
        f1 = token_f1(prediction, gold)
        em = exact_match(prediction, gold)
        self.scores.setdefault(category, []).append(f1)
        self.scores.setdefault("all", []).append(f1)
        self.em_scores.setdefault(category, []).append(em)
        self.em_scores.setdefault("all", []).append(em)

    def mean_f1(self, category: str = "all") -> float:
        vals = self.scores.get(category, [])
        return sum(vals) / len(vals) if vals else 0.0

    def mean_em(self, category: str = "all") -> float:
        vals = self.em_scores.get(category, [])
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict[str, dict[str, float]]:
        """Return {category: {f1: ..., em: ..., n: ...}}."""
        result = {}
        for cat in self.scores:
            result[cat] = {
                "f1": self.mean_f1(cat),
                "em": self.mean_em(cat),
                "n": len(self.scores[cat]),
            }
        return result
