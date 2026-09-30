"""
Baseline conflict detector: compare the model's closed-book answer with its
context-grounded answer. Offline on the v2 PopQA set this catches 39% (Llama) /
22% (Qwen) of adopted false answers at a 23% / 11% false-alarm rate on true
context; it is the baseline later detectors are measured against.
"""
import re
import string
from dataclasses import dataclass
from enum import Enum

REFUSAL_MARKERS = ("don't know", "dont know", "not mentioned", "no information")


class Verdict(str, Enum):
    CONSISTENT = "consistent"             # closed-book agrees with context answer
    NO_PRIOR = "no_prior"                 # model has no closed-book answer; trust context
    CONFLICT = "conflict"                 # closed-book and context answers disagree
    CONTEXT_REFUSED = "context_refused"   # model would not answer from the context


@dataclass(frozen=True)
class Detection:
    verdict: Verdict
    overlap: float  # token overlap between the two answers, 0..1


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[-_']", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def is_refusal(answer: str) -> bool:
    return not normalize(answer) or any(m in answer.lower() for m in REFUSAL_MARKERS)


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(normalize(a).split()), set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def detect(closed_book: str, context_answer: str, threshold: float = 0.5) -> Detection:
    if is_refusal(context_answer):
        return Detection(Verdict.CONTEXT_REFUSED, 0.0)
    if is_refusal(closed_book):
        return Detection(Verdict.NO_PRIOR, 0.0)
    overlap = token_overlap(closed_book, context_answer)
    verdict = Verdict.CONSISTENT if overlap >= threshold else Verdict.CONFLICT
    return Detection(verdict, overlap)
