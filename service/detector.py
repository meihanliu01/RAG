"""
Conflict detectors.

agreement (step-1 baseline): compare the closed-book answer with the
    context-grounded answer. Test split, llama3:8b: catches 34% of adopted false
    answers at a 21% false-alarm rate on true context.
verify (default): ask the model whether the context answer is correct and flag
    when log P(Yes) is below a dev-tuned threshold. Test split, llama3:8b at the
    10% budget: catches 44% at an 11% false-alarm rate. Recognising a wrong answer
    works even when the model cannot recall the right one, which is where the
    baseline has no signal (60% of its misses).
See scripts/detectors/eval_detectors.py.
"""
import math
import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import Optional

REFUSAL_MARKERS = ("don't know", "dont know", "not mentioned", "no information")


class Verdict(str, Enum):
    CONSISTENT = "consistent"             # closed-book agrees with context answer
    NO_PRIOR = "no_prior"                 # model has no closed-book answer; trust context
    CONFLICT = "conflict"                 # closed-book and context answers disagree
    CONTEXT_REFUSED = "context_refused"   # model would not answer from the context
    VERIFIED = "verified"                 # verifier accepts the context answer
    REJECTED = "rejected"                 # verifier rejects the context answer


@dataclass(frozen=True)
class Detection:
    verdict: Verdict
    overlap: float = 0.0                # agreement: token overlap between the two answers
    log_p_yes: Optional[float] = None   # verify: normalized log P(Yes)


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


def log_p_yes(top_logprobs: dict[str, float]) -> float:
    """log P(Yes) renormalized over Yes/No from the first-token candidates."""
    p_yes = sum(math.exp(lp) for tok, lp in top_logprobs.items() if tok.strip().lower().startswith("yes"))
    p_no = sum(math.exp(lp) for tok, lp in top_logprobs.items() if tok.strip().lower().startswith("no"))
    if p_yes + p_no == 0:
        return math.log(0.5)
    return math.log(max(p_yes / (p_yes + p_no), 1e-9))


def detect_verified(context_answer: str, log_p: Optional[float], threshold: float) -> Detection:
    if is_refusal(context_answer) or log_p is None:
        return Detection(Verdict.CONTEXT_REFUSED)
    verdict = Verdict.REJECTED if log_p <= threshold else Verdict.VERIFIED
    return Detection(verdict, log_p_yes=log_p)
