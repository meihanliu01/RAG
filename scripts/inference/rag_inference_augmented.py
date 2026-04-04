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
MODEL_NAME = "qwen2.5:7b"
INPUT_FILE = os.path.join(RESULTS_DIR, "rag_labeled_augmented.json")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "qwen_rag_final_results.json")

def main():
    print(f">>> 1. Loading Augmented Data: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    samples = data.get('details', data)

    print(f">>> 2. Starting Inference (Optimized for Extraction)...")
    
    for item in tqdm(samples, desc="Generating Predictions"):
        # 核心改动：只针对 Low-Saliency 样本重新跑，或者你可以去掉这个判断跑全量
        if item.get('saliency') != 'Low' and item.get('pred') != "RE-RUN REQUIRED":
            continue

        question = item['q']
        # 确保使用的是冲突上下文
        context = item.get('fake', "No context provided")

        # --- 优化后的 Prompt：更具攻击性，强制提取 ---
        prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question.
Extract the answer directly from the context snippets. 
If the information is in the context, you MUST provide it. 
Keep the answer extremely short (1-3 words).

Context: {context}

Question: {question}
Answer:"""

        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt, 
            "stream": False, 
            "options": {
                "temperature": 0.0,  # 保持确定性
                "num_predict": 20    # 限制输出长度，防止模型废话
            }
        }
        
        try:
            response_json = requests.post(OLLAMA_URL, json=payload, timeout=30).json()
            # 这里的 response 处理稍微健壮一点
            prediction = response_json.get("response", "").strip()
            
            # 简单清洗：去除末尾句号，统一大小写转换交由 classifier 处理
            item['pred'] = prediction.rstrip('.')
            
        except Exception as e:
            item['pred'] = f"Error: {e}"

    # 保存最终结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 建议保留原始格式，封装进 details
        json.dump({"details": samples}, f, indent=4)
    
    print(f"\n🚀 Inference Complete! Low-Saliency samples updated.")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()