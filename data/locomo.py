"""LoCoMo dataset loader.

Uses ``KhangPTT373/locomo_preprocess`` on HuggingFace (only "test" split).
10 multi-session conversations, 1 986 QA pairs with integer categories.
We split 7 / 3 for train / test when the caller requests those splits.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

from datasets import load_dataset as hf_load

log = logging.getLogger(__name__)

LOCOMO_HF_PATH = "KhangPTT373/locomo_preprocess"

# The dataset stores integer category codes.
INT_CATEGORY_MAP: dict[int, str] = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "adversarial",
    5: "open_domain",
}

_TRAIN_CONV_IDS = range(0, 7)   # first 7 conversations for training
_TEST_CONV_IDS  = range(7, 10)  # last 3 for evaluation


@dataclass
class Turn:
    """A single conversational turn."""
    session_id: str
    turn_idx: int
    speaker: str
    text: str


@dataclass
class QAPair:
    """A question-answer pair with metadata."""
    question: str
    answer: str
    category: str          # single_hop | multi_hop | temporal | open_domain | adversarial
    evidence_turn_ids: list[int]      # global 0-based turn indices
    evidence_session_ids: list[int]   # 1-based session ids
    session_id: str


@dataclass
class LoCoMoConversation:
    """One full multi-session conversation."""
    conv_id: str
    turns: list[Turn]
    qa_pairs: list[QAPair]


# ------------------------------------------------------------------
# Turn parsing
# ------------------------------------------------------------------
_TURN_RE = re.compile(
    r"^(?P<ts>[^\n]+)\n\s*(?P<speaker>[^:]+):\s*(?P<text>.*)",
    re.DOTALL,
)
_SPEAKER_MARKER_RE = re.compile(r"(?m)^\s*([^:\n]+):\s*")


def _parse_turn(raw: str, idx: int, session_id: str = "0") -> Turn:
    """Parse a raw turn string ``timestamp\\n Speaker: text``."""
    raw = raw.strip()
    m = _TURN_RE.match(raw)
    if m:
        return Turn(
            session_id=session_id, turn_idx=idx,
            speaker=m.group("speaker").strip(),
            text=m.group("text").strip(),
        )
    # Fallback: treat entire string as text
    return Turn(session_id=session_id, turn_idx=idx, speaker="unknown", text=raw)


def _parse_session_turns(raw_session: str, session_idx: int, global_start_idx: int) -> list[Turn]:
    """Parse one LoCoMo session transcript into ordered turns.

    The raw dataset stores:
      - ``turns``: flattened list of turns across all sessions
      - ``sessions``: list of full session transcripts

    Evidence annotations reference ``[session_id, turn_id]`` pairs, so we must
    recover per-session turn numbering instead of flattening everything into a
    single anonymous stream.
    """
    body = raw_session.split("\n", 1)[1] if "\n" in raw_session else raw_session
    matches = list(_SPEAKER_MARKER_RE.finditer(body))
    turns: list[Turn] = []
    for local_idx, match in enumerate(matches):
        speaker = match.group(1).strip()
        text_start = match.end()
        text_end = matches[local_idx + 1].start() if local_idx + 1 < len(matches) else len(body)
        text = body[text_start:text_end].strip()
        turns.append(Turn(
            session_id=str(session_idx),
            turn_idx=global_start_idx + local_idx,
            speaker=speaker,
            text=text,
        ))
    return turns


def _flatten_evidence_pairs(ev: object) -> list[tuple[int, int]]:
    """Return evidence references as ``(session_id, turn_in_session)`` pairs.

    The HF row encodes evidence as nested lists, e.g. ``[[1, 2]]`` or
    ``[[1, 5], [2, 1], [6, 4]]``.
    """
    pairs: list[tuple[int, int]] = []
    if isinstance(ev, list):
        for item in ev:
            if (
                isinstance(item, list)
                and len(item) == 2
                and all(isinstance(x, int) for x in item)
            ):
                pairs.append((int(item[0]), int(item[1])))
            else:
                pairs.extend(_flatten_evidence_pairs(item))
    return pairs


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_locomo(split: str = "test") -> list[LoCoMoConversation]:
    """Load LoCoMo conversations with their QA pairs.

    Parameters
    ----------
    split : str
        ``"train"`` returns the first 7 conversations (for memory training),
        ``"test"`` returns the last 3 (for evaluation).
        ``"all"`` returns all 10.
    """
    log.info("Loading LoCoMo from %s (virtual split=%s)", LOCOMO_HF_PATH, split)
    ds = hf_load(LOCOMO_HF_PATH, split="test")  # only split available

    conversations: list[LoCoMoConversation] = []
    for row_idx, row in enumerate(ds):
        # --- turns and session mapping ---
        raw_turns: list[str] = row["turns"]
        raw_sessions: list[str] = row.get("sessions", [])

        turns: list[Turn] = []
        session_turn_to_global: dict[tuple[int, int], int] = {}
        global_idx = 0
        for session_idx, raw_session in enumerate(raw_sessions, start=1):
            session_turns = _parse_session_turns(raw_session, session_idx, global_idx)
            for local_idx, turn in enumerate(session_turns, start=1):
                session_turn_to_global[(session_idx, local_idx)] = turn.turn_idx
            turns.extend(session_turns)
            global_idx += len(session_turns)

        # Fallback to the flattened ``turns`` field if parsing fails.
        if len(turns) != len(raw_turns):
            log.warning(
                "Session parsing mismatch for row %d: parsed %d turns, raw has %d; falling back to flat turns",
                row_idx,
                len(turns),
                len(raw_turns),
            )
            turns = [_parse_turn(t, i) for i, t in enumerate(raw_turns)]
            session_turn_to_global = {}

        # --- QA pairs (parallel lists) ---
        questions: list[str] = row["questions"]
        answers: list[str] = row["answers"]
        categories: list[int] = row["category"]
        evidences: list = row["evidences"]

        qa_pairs: list[QAPair] = []
        for i in range(len(questions)):
            ev = evidences[i]
            evidence_pairs = _flatten_evidence_pairs(ev)
            evidence_session_ids = sorted({session_id for session_id, _ in evidence_pairs})
            global_evidence_turn_ids: list[int] = []
            for session_id, turn_in_session in evidence_pairs:
                global_turn = session_turn_to_global.get((session_id, turn_in_session))
                if global_turn is not None:
                    global_evidence_turn_ids.append(global_turn)
            cat_int = categories[i] if i < len(categories) else 0
            qa_pairs.append(QAPair(
                question=questions[i],
                answer=answers[i],
                category=INT_CATEGORY_MAP.get(cat_int, f"cat_{cat_int}"),
                evidence_turn_ids=global_evidence_turn_ids,
                evidence_session_ids=evidence_session_ids,
                session_id=str(max(evidence_session_ids) if evidence_session_ids else 0),
            ))

        conversations.append(LoCoMoConversation(
            conv_id=str(row_idx), turns=turns, qa_pairs=qa_pairs,
        ))

    # Virtual train / test split
    if split == "train":
        conversations = [conversations[i] for i in _TRAIN_CONV_IDS if i < len(conversations)]
    elif split == "test":
        conversations = [conversations[i] for i in _TEST_CONV_IDS if i < len(conversations)]
    # else "all" → keep everything

    log.info("Loaded %d LoCoMo conversations with %d total QA pairs",
             len(conversations), sum(len(c.qa_pairs) for c in conversations))
    return conversations


def locomo_turns_as_text(conv: LoCoMoConversation) -> list[str]:
    """Return conversation as a list of 'speaker: text' strings."""
    return [f"{t.speaker}: {t.text}" for t in conv.turns]


def build_locomo_qa_inputs(
    conv: LoCoMoConversation,
    max_context_turns: int = 20,
) -> list[dict]:
    """Build model-ready input dicts for each QA pair.

    Each dict has keys: context, question, answer, category, session_id.
    """
    all_text = locomo_turns_as_text(conv)
    results = []
    for qa in conv.qa_pairs:
        context = " ".join(all_text[-max_context_turns:])
        results.append({
            "context": context,
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
            "session_id": qa.session_id,
        })
    return results


def get_locomo_by_category(
    conversations: Sequence[LoCoMoConversation],
) -> dict[str, list[QAPair]]:
    """Group all QA pairs across conversations by category."""
    by_cat: dict[str, list[QAPair]] = {}
    for conv in conversations:
        for qa in conv.qa_pairs:
            by_cat.setdefault(qa.category, []).append(qa)
    return by_cat
