"""
build_popqa_conflicts.py
========================
v2 dataset: knowledge-conflict samples built from PopQA (Mallen et al., 2023).

Fixes over v1 (generate_balanced_data.py):
  - Saliency comes from real Wikipedia page views (s_pop), not question keywords,
    and tiers are cut by quantile *within each relation type*, so every tier has
    the same relation mix (no construction-method confound across tiers).
  - Counterfactuals are real entities of the same relation type (another real
    city for "place of birth"), never "alt_X" strings the model can trivially spot.
  - Every question also gets a SUPPORTING context (true answer), so we can tell
    "resists false context" apart from "ignores context altogether".

Input:  data/popqa/popQA.tsv
        (https://github.com/AlexTMallen/adaptive-retrieval/blob/main/data/popQA.tsv)
Output: data/popqa_conflicts_v2.json

Usage:
    rag_env/bin/python scripts/data_prep/build_popqa_conflicts.py
"""
import json
import os
import random
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INPUT_FILE = os.path.join(DATA_DIR, "popqa", "popQA.tsv")
OUTPUT_FILE = os.path.join(DATA_DIR, "popqa_conflicts_v2.json")

SEED = 657
PER_TIER = 600
# Within-relation popularity quantiles. The middle band is left out on purpose
# so High and Low are well separated.
TIER_QUANTILES = {'Low': (0.0, 0.25), 'Medium': (0.40, 0.60), 'High': (0.80, 1.0)}

# One declarative sentence per relation, used to write the evidence passage.
TEMPLATES = {
    'occupation':    "{subj} is best known for working as a {obj}.",
    'place of birth': "{subj} was born in {obj}.",
    'genre':         "{subj} is generally classified in the {obj} genre.",
    'father':        "The father of {subj} is {obj}.",
    'mother':        "The mother of {subj} is {obj}.",
    'country':       "{subj} is located in {obj}.",
    'producer':      "{subj} was produced by {obj}.",
    'director':      "{subj} was directed by {obj}.",
    'screenwriter':  "The screenplay for {subj} was written by {obj}.",
    'composer':      "The music for {subj} was composed by {obj}.",
    'author':        "{subj} was written by {obj}.",
    'capital of':    "{subj} is the capital of {obj}.",
    'capital':       "The capital of {subj} is {obj}.",
    'religion':      "{subj} is affiliated with {obj}.",
    'sport':         "{subj} is associated with the sport of {obj}.",
    'color':         "The color most associated with {subj} is {obj}.",
}
PASSAGE = "{fact} This is documented in reference sources that cover {subj}. {fact_again}"


def parse_list(cell):
    try:
        out = json.loads(cell)
        return [str(x) for x in out] if isinstance(out, list) else [str(out)]
    except (TypeError, ValueError):
        return [str(cell)] if isinstance(cell, str) and cell else []


def make_passage(prop, subj, obj):
    fact = TEMPLATES[prop].format(subj=subj, obj=obj)
    fact_again = f"Sources consistently give {obj} for this."
    return PASSAGE.format(fact=fact, subj=subj, fact_again=fact_again)


def pick_fake(row, pool, rng):
    """A real object of the same relation that is not any alias of the true answer."""
    banned = {a.lower() for a in row['answers']}
    candidates = [o for o in pool if o.lower() not in banned and o.lower() not in row['subj'].lower()]
    return rng.choice(candidates) if candidates else None


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(INPUT_FILE, sep='\t')
    df = df[df['prop'].isin(TEMPLATES)].dropna(subset=['subj', 'obj', 'question']).copy()
    df['subj'] = df['subj'].astype(str)
    df['obj'] = df['obj'].astype(str)
    df['answers'] = df['possible_answers'].apply(parse_list)
    df = df[df['answers'].map(len) > 0]

    # Relation-level answer pools for counterfactuals
    pools = {p: sorted(set(g['obj'].astype(str))) for p, g in df.groupby('prop')}

    # Assign tiers by within-relation popularity quantile
    df['pop_q'] = df.groupby('prop')['s_pop'].rank(pct=True)
    df['saliency'] = None
    for tier, (lo, hi) in TIER_QUANTILES.items():
        df.loc[(df['pop_q'] > lo) & (df['pop_q'] <= hi), 'saliency'] = tier
    df = df[df['saliency'].notna()]

    # Sample each tier with the same per-relation quota
    props = sorted(df['prop'].unique())
    samples = []
    for tier in TIER_QUANTILES:
        tier_df = df[df['saliency'] == tier]
        quota = {p: 0 for p in props}
        # Round-robin across relations until the tier is full or relations run out
        by_prop = {p: tier_df[tier_df['prop'] == p].sample(frac=1, random_state=SEED).to_dict('records')
                   for p in props}
        taken = 0
        while taken < PER_TIER and any(by_prop.values()):
            for p in props:
                if taken >= PER_TIER or not by_prop[p]:
                    continue
                row = by_prop[p].pop()
                fake = pick_fake(row, pools[p], rng)
                if fake is None:
                    continue
                samples.append({
                    'id': f"popqa-{row['id']}",
                    'q': row['question'],
                    'prop': p,
                    'subj': row['subj'],
                    'gt': row['answers'],
                    'fake': fake,
                    's_pop': int(row['s_pop']),
                    'saliency': tier,
                    'conflicting_context': make_passage(p, row['subj'], fake),
                    'supporting_context': make_passage(p, row['subj'], row['obj']),
                })
                quota[p] += 1
                taken += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"details": samples}, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(samples)} samples -> {OUTPUT_FILE}")
    tiers = Counter(s['saliency'] for s in samples)
    print(f"Tiers: {dict(tiers)}")
    for tier in TIER_QUANTILES:
        pops = sorted(s['s_pop'] for s in samples if s['saliency'] == tier)
        if pops:
            print(f"  {tier:<7} median page views = {pops[len(pops) // 2]:,}")
    print("Relations per tier:")
    table = Counter((s['saliency'], s['prop']) for s in samples)
    for p in props:
        print(f"  {p:<15}" + ''.join(f"{table[(t, p)]:>7}" for t in TIER_QUANTILES))


if __name__ == "__main__":
    main()
