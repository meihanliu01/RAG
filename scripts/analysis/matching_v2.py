"""
matching_v2.py
==============
Fixed answer-matching logic for conflict classification.

Bug in v1 (behavioral_classifier.py): the gold answer was checked FIRST with a
lenient soft match (substring or >=50% token overlap). Because counterfactuals
are built from the gold answer (e.g. "alt_Drigum Tsenpo", "$3.1/W installed cost"),
an output that copies the fake answer also "matches" gold and was labeled
Persistence.

v2 compares only the tokens that DIFFER between gold and fake, so an output is
credited to whichever answer its distinguishing tokens belong to.
"""
import re
import string

REFUSAL_MARKERS = ["don't know", "dont know", "not mentioned", "no information"]


def normalize(s):
    """Lowercase, drop articles, split on hyphens/underscores/apostrophes, strip punctuation."""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r"[-_']", ' ', s)
    s = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', s)   # 131st == 131th
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def perturbation_type(fake):
    """How the counterfactual was built (see smart_perturb in generate_balanced_data.py)."""
    fake = str(fake)
    if fake.startswith('alt_'):
        return 'alt_prefix'
    if re.fullmatch(r'\d{4}', fake):
        return 'year'
    if re.search(r'\d', fake):
        return 'number'
    return 'synthetic_entity'   # Low tier: hand-written fakes like "Pseudo-Zfyve26"


def _tokens(s):
    return set(normalize(s).split())


def matches_answer(pred, target, other):
    """True if `pred` contains `target` in a way that distinguishes it from `other`.

    - If target has tokens that `other` lacks, all of them must appear in pred.
    - If target's tokens are a subset of other's (e.g. gold "X" vs fake "alt X"),
      pred must NOT contain other's extra tokens, and must contain target's phrase
      or at least half of its tokens.
    """
    p_norm = normalize(pred)
    t_norm = normalize(target)
    if not p_norm or not t_norm:
        return False
    p_tok, t_tok, o_tok = set(p_norm.split()), set(t_norm.split()), _tokens(other)

    unique = t_tok - o_tok
    if unique:
        return unique <= p_tok
    other_unique = o_tok - t_tok
    if other_unique and other_unique <= p_tok:
        return False
    # Allow partial answers ("Lord Irwin" for "Viceroy Lord Irwin")
    return t_norm in p_norm or len(p_tok & t_tok) / len(t_tok) >= 0.5


def classify(pred, golds, fake):
    """Return one of: Persistence, Adherence, Mixed, Uncertain/Other."""
    if isinstance(golds, str):
        golds = [golds]
    pred = str(pred or '')
    if not normalize(pred) or any(m in pred.lower() for m in REFUSAL_MARKERS):
        return "Uncertain/Other"

    gold_hit = any(matches_answer(pred, g, fake) for g in golds if g)
    fake_hit = matches_answer(pred, fake, golds[0]) if fake and golds else False

    if gold_hit and fake_hit:
        return "Mixed"          # mentions both, e.g. "the context says X, but it is Y"
    if gold_hit:
        return "Persistence"
    if fake_hit:
        return "Adherence"
    return "Uncertain/Other"
