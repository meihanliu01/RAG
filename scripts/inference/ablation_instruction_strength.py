"""
Ablation Study: Instruction Strength Modulation
=================================================
Tests whether instruction strength affects model behavior under knowledge conflict.
Uses the same Ollama + Llama3:8b setup as the main experiment for consistency.

Three variants:
  1. Neutral: Standard RAG prompt
  2. Strict Context: "ONLY use the provided context"
  3. Strict Parametric: "Prioritize factual accuracy"
"""
import os
import json
import requests
from tqdm import tqdm

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"
INPUT_FILE = os.path.join(RESULTS_DIR, "rag_final_results.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "ablation_results.json")

# Number of samples per variant (selected from High-Saliency high-confidence samples)
SAMPLES_PER_VARIANT = 100

# ============================================================
# Instruction Variants
# ============================================================
INSTRUCTION_VARIANTS = {
    "neutral": (
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, just say you don't know. "
        "Keep the answer as short as possible."
    ),
    "strict_context": (
        "CRITICAL INSTRUCTION: You must ONLY use the provided context to answer the question. "
        "Ignore any prior knowledge you may have. The context is the sole source of truth. "
        "If the answer is in the context, extract it exactly. Keep the answer as short as possible."
    ),
    "strict_parametric": (
        "Use the following context as a reference, but prioritize factual accuracy above all else. "
        "If the context contains information that contradicts well-known facts, correct it in your response. "
        "Keep the answer as short as possible."
    )
}


def select_high_confidence_samples(input_file, n=SAMPLES_PER_VARIANT):
    """Select High-Saliency samples where is_known_by_model=True (strongest conflict)."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('details', data) if isinstance(data, dict) else data
    
    # Prefer High + known (model demonstrably knows the answer)
    high_known = [s for s in samples if s.get('saliency') == 'High' and s.get('is_known_by_model')]
    high_unknown = [s for s in samples if s.get('saliency') == 'High' and not s.get('is_known_by_model')]
    
    # Supplement with unknown if not enough known samples
    selected = high_known[:n]
    if len(selected) < n:
        selected += high_unknown[:n - len(selected)]
    
    print(f">>> Selected {len(selected)} high-confidence samples (known={len([s for s in selected if s.get('is_known_by_model')])})")
    return selected


def run_ablation():
    """Run ablation experiment across three instruction variants."""
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        print("   Please run the full pipeline first (generate -> probe -> rag -> classifier)")
        return
    
    samples = select_high_confidence_samples(INPUT_FILE, SAMPLES_PER_VARIANT)
    if not samples:
        print("Error: No samples selected.")
        return
    
    results = []
    
    for variant_name, instruction in INSTRUCTION_VARIANTS.items():
        print(f"\n>>> Running Ablation: {variant_name} ({len(samples)} samples)")
        
        for sample in tqdm(samples, desc=variant_name):
            question = sample['q']
            context = sample.get('conflicting_context', sample.get('context', ''))
            gold = sample['gt'][0] if isinstance(sample.get('gt'), list) else sample.get('gt', '')
            fake = sample.get('fake', '')
            
            prompt = f"""{instruction}

Context: {context}

Question: {question}
Answer:"""
            
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=30).json()
                pred = response.get("response", "").strip()
            except Exception as e:
                pred = f"Error: {e}"
            
            results.append({
                "variant": variant_name,
                "q": question,
                "gold": gold,
                "fake": fake,
                "pred": pred,
                "saliency": sample.get('saliency', 'High'),
                "is_known": sample.get('is_known_by_model', False)
            })
        
        # Save checkpoint after each variant completes
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"   {variant_name} complete. Intermediate save to {OUTPUT_FILE}")
    
    print(f"\nAblation Complete! Total results: {len(results)}")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_ablation()