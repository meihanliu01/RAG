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

# 每个变体使用的样本数 (从 High-Saliency 中选取 high-confidence 样本)
SAMPLES_PER_VARIANT = 100

# ============================================================
# 指令强度变体 (Instruction Variants)
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
    """选取 High-Saliency 中 is_known_by_model=True 的样本 (最强冲突)"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('details', data) if isinstance(data, dict) else data
    
    # 优先选 High + known (模型确实知道正确答案的)
    high_known = [s for s in samples if s.get('saliency') == 'High' and s.get('is_known_by_model')]
    high_unknown = [s for s in samples if s.get('saliency') == 'High' and not s.get('is_known_by_model')]
    
    # 如果 known 不够，补充 unknown
    selected = high_known[:n]
    if len(selected) < n:
        selected += high_unknown[:n - len(selected)]
    
    print(f">>> Selected {len(selected)} high-confidence samples (known={len([s for s in selected if s.get('is_known_by_model')])})")
    return selected


def run_ablation():
    """运行三个指令变体的消融实验"""
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        print("   请先运行完整 pipeline (generate → probe → rag → classifier)")
        return
    
    samples = select_high_confidence_samples(INPUT_FILE, SAMPLES_PER_VARIANT)
    if not samples:
        print("❌ No samples selected.")
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
        
        # 每个变体跑完存一次档
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"   ✅ {variant_name} complete. Intermediate save to {OUTPUT_FILE}")
    
    print(f"\n🚀 Ablation Complete! Total results: {len(results)}")
    print(f"   Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_ablation()