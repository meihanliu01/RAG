import os
import sys
import json
import requests
from tqdm import tqdm

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"
# 输入：parametric probe 的输出（包含 pred_base 字段）
INPUT_FILE = os.path.join(RESULTS_DIR, "llama_parametric_probe_results.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "rag_final_results.json")

def main():
    print(f">>> 1. Loading Augmented Data: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        print("   请先运行: python scripts/inference/parametric_probe.py")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 兼容你之前的结构，数据在 'details' 键下
    samples = data.get('details', data)

    # 统计需要跑的样本数
    to_run = [s for s in samples if not s.get('pred') or s['pred'] == "RE-RUN REQUIRED" or s['pred'].startswith("Error")]
    print(f">>> 2. Starting Inference: {len(to_run)}/{len(samples)} samples need processing...")
    
    for item in tqdm(samples, desc="Generating Predictions"):
        # 跳过已经有有效预测结果的样本
        existing_pred = item.get('pred', '')
        if existing_pred and existing_pred != "RE-RUN REQUIRED" and not existing_pred.startswith("Error"):
            continue

        question = item['q']
        # 使用冲突上下文（gold answer 已被 fake 替换）
        # 优先使用 conflicting_context，其次用 context（原始段落）
        context = item.get('conflicting_context', item.get('context', 'No context provided'))

        # 构造 Prompt (保持和你之前实验一致的格式)
        prompt = f"""Use the following pieces of retrieved context to answer the question. 
If you don't know the answer based on the context, just say you don't know. 
Keep the answer as short as possible.

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
            item['pred'] = response.get("response", "").strip()
        except Exception as e:
            item['pred'] = f"Error: {e}"

    # 保存最终结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"details": samples}, f, indent=4)
    
    print(f"\n🚀 Inference Complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()