"""MSC (Multi-Session Chat) dataset loader.

Uses ``nayohan/multi_session_chat`` on HuggingFace.
501 dialogues × 5 sessions (session_id 0–4) in test split.
Each row contains parallel ``dialogue`` / ``speaker`` lists and persona facts.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from datasets import load_dataset as hf_load

log = logging.getLogger(__name__)

MSC_HF_PATH = "nayohan/multi_session_chat"


@dataclass
class MSCTurn:
    speaker: str
    text: str
    session: int      # 1-indexed (session_id + 1)
    turn_idx: int


@dataclass
class MSCDialogue:
    """One multi-session dialogue (sessions 1–5)."""
    dialogue_id: str
    sessions: dict[int, list[MSCTurn]]  # session_num (1-indexed) -> turns
    personas: list[str]  # persona facts (union of persona1 + persona2)


def load_msc(
    split: str = "test",
    sessions: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> list[MSCDialogue]:
    """Load MSC dialogues spanning the requested sessions.

    Parameters
    ----------
    split : str
        ``"train"``, ``"validation"``, or ``"test"``.
    sessions : tuple[int, ...]
        Which session numbers to include (1-indexed, matching paper convention).
    """
    log.info("Loading MSC split=%s from %s", split, MSC_HF_PATH)
    ds = hf_load(MSC_HF_PATH, split=split)

    # Group rows by dialogue id
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in ds:
        did = str(row["dialoug_id"])
        sess = int(row["session_id"])  # 0-indexed in dataset
        grouped[did][sess] = row

    dialogues: list[MSCDialogue] = []
    for did in sorted(grouped, key=int):
        sess_map: dict[int, list[MSCTurn]] = {}
        personas: list[str] = []
        for sess_idx in sorted(grouped[did]):
            sess_num = sess_idx + 1  # convert to 1-indexed
            if sess_num not in sessions:
                continue
            row = grouped[did][sess_idx]

            # Collect persona facts once
            if not personas:
                p1 = row.get("persona1") or []
                p2 = row.get("persona2") or []
                personas = [str(p) for p in p1] + [str(p) for p in p2]

            utts: list[str] = row["dialogue"]
            speakers: list[str] = row["speaker"]
            turns: list[MSCTurn] = []
            for i, (utt, spk) in enumerate(zip(utts, speakers)):
                turns.append(MSCTurn(
                    speaker=spk, text=utt,
                    session=sess_num, turn_idx=i,
                ))
            sess_map[sess_num] = turns

        if sess_map:
            dialogues.append(MSCDialogue(
                dialogue_id=did, sessions=sess_map, personas=personas,
            ))

    total_turns = sum(len(t) for d in dialogues for t in d.sessions.values())
    log.info("Loaded %d MSC dialogues (%d total turns)", len(dialogues), total_turns)
    return dialogues


def build_msc_eval_pairs(
    dialogues: list[MSCDialogue],
    eval_sessions: tuple[int, ...] = (2, 3, 4, 5),
    context_window: int = 20,
) -> list[dict]:
    """Build evaluation pairs for MSC.

    For each turn in ``eval_sessions``, the model must generate a response
    conditioned on the conversation history (including prior sessions).
    Even-indexed turns are treated as user queries; odd-indexed as gold
    responses, following the MSC convention.

    Returns list of dicts with keys: context, question, answer, session.
    """
    pairs = []
    for dlg in dialogues:
        # Build full history across sessions
        full_history: list[str] = []
        for sess_num in sorted(dlg.sessions):
            for turn in dlg.sessions[sess_num]:
                full_history.append(f"{turn.speaker}: {turn.text}")

        # Build eval pairs from requested sessions
        offset = 0
        for sess_num in sorted(dlg.sessions):
            sess_turns = dlg.sessions[sess_num]
            if sess_num in eval_sessions:
                for i, turn in enumerate(sess_turns):
                    if i == 0:
                        continue  # need at least one prior turn
                    ctx_start = max(0, offset + i - context_window)
                    ctx_end = offset + i
                    context = " ".join(full_history[ctx_start:ctx_end])
                    question = full_history[ctx_end - 1] if ctx_end > 0 else ""
                    pairs.append({
                        "context": context,
                        "question": question,
                        "answer": turn.text,
                        "session": sess_num,
                    })
            offset += len(sess_turns)

    return pairs
