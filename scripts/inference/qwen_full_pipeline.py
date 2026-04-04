"""
Qwen2.5-7B Full Pipeline
=========================
Complete pipeline for Qwen2.5-7B: Parametric Probe + RAG Inference + Classification
Mirrors the Llama pipeline for cross-model comparison.

Usage:
    python scripts/inference/qwen_full_pipeline.py
"""
import os
import sys
import json
import re
import string
import requests
from tqdm import tqdm

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b"
INPUT_FILE = os.path.join(DATA_DIR, "rag_balanced_2400.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "qwen_final_labeled.json")


# ============================================================
# Soft-Match Classification (identical to behavioral_classifier.py)
# ============================================================
def normalize(s):
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return ' '.join(s.split())

def is_soft_match(pred, target):
    if not pred or not target: return False
    p_norm = normalize(pred)
    t_norm = normalize(target)
    if t_norm in p_norm or p_norm in t_norm:
        return True
    p_tokens = set(p_norm.split())
    t_tokens = set(t_norm.split())
    if not t_tokens: return False
    return len(p_tokens & t_tokens) / len(t_tokens) >= 0.5


def ollama_query(prompt, max_tokens=20):
    """Send a query to Ollama and return the response text."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens
        }
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=30).json()
        return res.get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"


def main():
    # --- 1. Load data ---
    print(f">>> 1. Loading balanced data: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        print("   请先运行: python scripts/data_prep/generate_balanced_data.py")
        sys.exit(1)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('details', data) if isinstance(data, dict) else data
    print(f"   Total samples: {len(samples)}")
    
    # Check if we can resume from a partial run
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        existing_samples = existing.get('details', existing) if isinstance(existing, dict) else existing
        if len(existing_samples) == len(samples):
            done_probe = sum(1 for s in existing_samples if s.get('pred_base'))
            done_rag = sum(1 for s in existing_samples if s.get('pred'))
            print(f"   Found existing results: {done_probe} probed, {done_rag} RAG-inferred")
            samples = existing_samples
    
    # --- 2. Parametric Probe (Zero-Context) ---
    to_probe = [s for s in samples if not s.get('pred_base')]
    print(f"\n>>> 2. Parametric Probe: {len(to_probe)}/{len(samples)} samples need probing...")
    
    for item in tqdm(samples, desc="[Qwen] Probe"):
        if item.get('pred_base'):
            continue
        
        question = item['q']
        prompt = (
            f"Instruction: Answer the following question in 1-3 words based on your internal knowledge. "
            f"If the answer is unknown, strictly respond with 'I don't know'.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
        
        pred_base = ollama_query(prompt, max_tokens=10).lower().rstrip('.')
        item['pred_base'] = pred_base
        
        gold_answers = [str(ans).lower() for ans in item.get('gt', [])]
        item['is_correct_base'] = any(ans in pred_base for ans in gold_answers)
    
    # Intermediate save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"details": samples}, f, indent=4, ensure_ascii=False)
    print("   ✅ Probe complete. Intermediate save done.")
    
    # --- 3. RAG Inference (with conflicting context) ---
    to_rag = [s for s in samples if not s.get('pred') or s['pred'].startswith('Error')]
    print(f"\n>>> 3. RAG Inference: {len(to_rag)}/{len(samples)} samples need processing...")
    
    for item in tqdm(samples, desc="[Qwen] RAG"):
        existing_pred = item.get('pred', '')
        if existing_pred and not existing_pred.startswith('Error'):
            continue
        
        question = item['q']
        context = item.get('conflicting_context', item.get('context', 'No context provided'))
        
        prompt = f"""Use the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say you don't know. 
Keep the answer as short as possible.

Context: {context}

Question: {question}
Answer:"""
        
        item['pred'] = ollama_query(prompt, max_tokens=50)
    
    # Intermediate save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"details": samples}, f, indent=4, ensure_ascii=False)
    print("   ✅ RAG inference complete. Intermediate save done.")
    
    # --- 4. Behavioral Classification ---
    print(f"\n>>> 4. Classifying behavior for {len(samples)} samples...")
    
    for entry in samples:
        pred_rag = entry.get('pred', '')
        pred_base = entry.get('pred_base', '')
        
        gt_data = entry.get('gt', entry.get('gold', ''))
        gold = gt_data[0] if isinstance(gt_data, list) else gt_data
        fake = entry.get('fake', '')
        
        # is_known_by_model (soft match on probe answer)
        entry['is_known_by_model'] = is_soft_match(pred_base, gold)
        
        # Unified taxonomy (same as Llama classifier)
        pred_lower = pred_rag.lower()
        if any(msg in pred_lower for msg in ["don't know", "dont know", "not mentioned", "no information"]) or not normalize(pred_rag):
            entry['label'] = "Uncertain/Other"
        elif is_soft_match(pred_rag, gold):
            entry['label'] = "Persistence"
        elif is_soft_match(pred_rag, fake):
            entry['label'] = "Adherence"
        else:
            entry['label'] = "Uncertain/Other"
    
    # --- 5. Final save ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump({"details": samples}, f, indent=4, ensure_ascii=False)
    
    # --- 6. Summary ---
    from collections import Counter
    total = len(samples)
    label_counts = Counter(s.get('label') for s in samples)
    
    print(f"\n{'='*60}")
    print(f"🚀 Qwen2.5-7B Pipeline Complete!")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Total: N={total}")
    print(f"   Persistence: {label_counts.get('Persistence', 0)} ({label_counts.get('Persistence', 0)/total*100:.1f}%)")
    print(f"   Adherence:   {label_counts.get('Adherence', 0)} ({label_counts.get('Adherence', 0)/total*100:.1f}%)")
    print(f"   Uncertain:   {label_counts.get('Uncertain/Other', 0)} ({label_counts.get('Uncertain/Other', 0)/total*100:.1f}%)")
    
    # Per-tier breakdown
    for tier in ['High', 'Medium', 'Low']:
        subset = [s for s in samples if s.get('saliency') == tier]
        tc = Counter(s.get('label') for s in subset)
        n = len(subset)
        if n > 0:
            print(f"   {tier:>6}: P={tc.get('Persistence',0)/n*100:.1f}% A={tc.get('Adherence',0)/n*100:.1f}% U={tc.get('Uncertain/Other',0)/n*100:.1f}% (N={n})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
