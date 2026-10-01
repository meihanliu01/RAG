"""
eval_detectors.py
=================
Compare conflict detectors on the dev/test split from collect_signals.py.

A detector "flags" a (question, context) pair; flagged or abstained answers are
not served as-is. Metrics, matching service/bench/run_benchmark.py:
    catch rate   = flagged / false-context answers that adopted the fake
    false alarm  = flagged / true-context answers

Detectors:
    D0 baseline    closed-book and context answers disagree (service/detector.py)
    D1 +confidence D0, but only when the closed-book answer is confident
    D2 verify      model's P(Yes) that the context answer is correct is low
    D3 D1 or D2
    D4 logistic regression over all signals

Thresholds are chosen on dev for a false-alarm budget, then reported on test.

Usage:
    rag_env/bin/python scripts/detectors/eval_detectors.py --model llama3:8b
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from service.detector import Verdict, detect, is_refusal, token_overlap  # noqa: E402

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "popqa_conflicts_v2.json")
V2_DIR = os.path.join(PROJECT_ROOT, "results", "v2")
SIG_DIR = os.path.join(PROJECT_ROOT, "results", "detectors")
FA_BUDGETS = (0.05, 0.10, 0.20)


def build_rows(model):
    tag = model.replace(":", "_")
    with open(DATA_FILE, encoding="utf-8") as f:
        meta = {s["id"]: s for s in json.load(f)["details"]}
    with open(os.path.join(V2_DIR, f"{tag}.json"), encoding="utf-8") as f:
        v2 = json.load(f)["results"]
    with open(os.path.join(SIG_DIR, f"{tag}_signals.json"), encoding="utf-8") as f:
        sig = json.load(f)["signals"]

    rows = []
    for sid, s in sig.items():
        if "closed_book" not in s:
            continue
        cb = s["closed_book"]
        for cond in ("conflict", "support"):
            ctx = v2[sid][cond]
            if cond == "conflict" and ctx["label"] != "Adherence":
                continue   # catch rate is measured on answers that adopted the fake
            p_yes = s[f"verify_{cond}"]["p_yes_norm"]
            rows.append({
                "id": sid, "split": s["split"], "tier": meta[sid]["saliency"],
                "positive": cond == "conflict",
                "verdict": detect(cb["pred"], ctx["pred"]).verdict,
                "ctx_refused": is_refusal(ctx["pred"]),
                "cb_refused": is_refusal(cb["pred"]),
                "overlap": token_overlap(cb["pred"], ctx["pred"]),
                "cb_conf": cb["mean_logprob"],
                "cb_first_p": cb["first_token_p"],
                "log_p_yes": math.log(max(p_yes, 1e-9)),
            })
    return rows


def features(r):
    conflict = float(r["verdict"] is Verdict.CONFLICT)
    return [conflict, float(r["cb_refused"]), r["overlap"], r["cb_conf"], r["cb_first_p"],
            r["log_p_yes"], conflict * r["cb_conf"]]


def metrics(rows, flags):
    pos = [f for r, f in zip(rows, flags) if r["positive"]]
    neg = [f for r, f in zip(rows, flags) if not r["positive"]]
    return (sum(pos) / len(pos) if pos else float("nan"),
            sum(neg) / len(neg) if neg else float("nan"))


def pick_threshold(scores, rows, budget):
    """Lowest threshold (most flags) whose dev false-alarm rate stays within budget; flag if score >= t."""
    best = math.inf
    for t in sorted(set(scores), reverse=True):
        _, fa = metrics(rows, [s >= t for s in scores])
        if fa <= budget:
            best = t
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3:8b")
    args = ap.parse_args()

    rows = build_rows(args.model)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"{args.model}: dev {len(dev)} rows, test {len(test)} rows "
          f"(positives = adopted false answers, negatives = true-context answers)")

    # Scores: higher = more suspicious. Context refusals are always flagged (abstain).
    big = 1e9
    score_fns = {
        "D1 +confidence": lambda r: big if r["ctx_refused"] else (r["cb_conf"] if r["verdict"] is Verdict.CONFLICT else -big),
        "D2 verify": lambda r: big if r["ctx_refused"] else -r["log_p_yes"],
    }

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    trainable = [r for r in dev if not r["ctx_refused"]]
    clf.fit(np.array([features(r) for r in trainable]), np.array([r["positive"] for r in trainable]))
    score_fns["D4 logistic"] = lambda r: big if r["ctx_refused"] else clf.predict_proba([features(r)])[0][1]

    def baseline(r):
        return r["ctx_refused"] or r["verdict"] is Verdict.CONFLICT

    print(f"\n{'Detector':<16}{'FA budget':>10} | {'dev catch':>9}{'dev FA':>8} | {'test catch':>10}{'test FA':>8}")
    c, fa = metrics(dev, [baseline(r) for r in dev])
    tc, tfa = metrics(test, [baseline(r) for r in test])
    print(f"{'D0 baseline':<16}{'-':>10} | {100 * c:8.1f}%{100 * fa:7.1f}% | {100 * tc:9.1f}%{100 * tfa:7.1f}%")

    chosen = {}
    for name, fn in score_fns.items():
        dev_scores = [fn(r) for r in dev]
        test_scores = [fn(r) for r in test]
        for budget in FA_BUDGETS:
            t = pick_threshold(dev_scores, dev, budget)
            c, fa = metrics(dev, [s >= t for s in dev_scores])
            tc, tfa = metrics(test, [s >= t for s in test_scores])
            chosen[(name, budget)] = (t, [s >= t for s in test_scores])
            print(f"{name:<16}{int(100 * budget):>9}% | {100 * c:8.1f}%{100 * fa:7.1f}% | {100 * tc:9.1f}%{100 * tfa:7.1f}%")

    # D3: D1 OR D2, each at half the budget
    for budget in FA_BUDGETS:
        flags_dev, flags_test = [], []
        for part, out in ((dev, flags_dev), (test, flags_test)):
            s1 = [score_fns["D1 +confidence"](r) for r in part]
            s2 = [score_fns["D2 verify"](r) for r in part]
            t1 = pick_threshold([score_fns["D1 +confidence"](r) for r in dev], dev, budget / 2)
            t2 = pick_threshold([score_fns["D2 verify"](r) for r in dev], dev, budget / 2)
            out.extend(a >= t1 or b >= t2 for a, b in zip(s1, s2))
        c, fa = metrics(dev, flags_dev)
        tc, tfa = metrics(test, flags_test)
        print(f"{'D3 D1 or D2':<16}{int(100 * budget):>9}% | {100 * c:8.1f}%{100 * fa:7.1f}% | {100 * tc:9.1f}%{100 * tfa:7.1f}%")

    # Per-tier view of the logistic detector at the 10% budget
    _, flags = chosen[("D4 logistic", 0.10)]
    print("\nD4 logistic @10% budget, test set by tier:")
    for tier in ("High", "Medium", "Low"):
        sub = [(r, f) for r, f in zip(test, flags) if r["tier"] == tier]
        c, fa = metrics([r for r, _ in sub], [f for _, f in sub])
        print(f"  {tier:<7} catch {100 * c:5.1f}%   false alarm {100 * fa:5.1f}%")

    names = ["conflict", "cb_refused", "overlap", "cb_conf", "cb_first_p", "log_p_yes", "conflict*cb_conf"]
    print("\nD4 coefficients:", ", ".join(f"{n}={w:+.2f}" for n, w in zip(names, clf.coef_[0])))


if __name__ == "__main__":
    main()
