import os
import sys
import json
import requests
from tqdm import tqdm

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"
# 指定你刚才生成的增强数据集
INPUT_FILE = "rag_labeled_augmented.json" 
OUTPUT_FILE = "rag_final_results.json"

def main():
    print(f">>> 1. Loading Augmented Data: {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found.")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 兼容你之前的结构，数据在 'details' 键下
    samples = data.get('details', data)

    print(f">>> 2. Starting Inference for {len(samples)} samples...")
    
    for item in tqdm(samples, desc="Generating Predictions"):
        # 如果不是 Low-Saliency 且已经有结果了，可以跳过以节省时间
        # 但为了保证一致性，建议全部重新跑一遍，或者只跑标注为 "RE-RUN REQUIRED" 的
        if item.get('pred') != "RE-RUN REQUIRED":
            continue

        question = item['q']
        # 这里的 Context 必须使用你构造好的冲突上下文 (fake)
        context = item.get('fake', "No context provided")

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