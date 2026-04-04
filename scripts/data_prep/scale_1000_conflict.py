import json
import requests
import random
import re
from datasets import load_dataset
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

def smart_perturb(text):
    """更智能的实体干扰算法"""
    # 模式 A: 年份替换 (如 1990 -> 2090)
    if re.match(r'^\d{4}$', text):
        return str(int(text) + random.choice([-100, 100, 200]))
    # 模式 B: 百分比/数字替换 (如 50% -> 100%)
    if re.search(r'\d+', text):
        return re.sub(r'\d+', lambda x: str(int(x.group()) * 2 + 1), text)
    # 模式 C: 专有名词 (如 London -> Fake_London)
    return f"alt_{text}"

def main():
    print(">>> Loading SQuAD Validation Set...")
    dataset = load_dataset("squad", split="validation")
    
    # 随机抽取 1000 个样本
    indices = random.sample(range(len(dataset)), 1000)
    test_samples = [dataset[i] for i in indices]

    final_results = []

    print(f">>> Running 1000 Conflicts on {MODEL_NAME}...")
    for item in tqdm(test_samples):
        question = item['question']
        gold = item['answers']['text'][0]
        context = item['context']
        
        # 构造冲突
        fake = smart_perturb(gold)
        conflicting_context = context.replace(gold, fake)

        prompt = f"Background: {conflicting_context}\nQuestion: {question}\nAnswer in one word:"
        
        try:
            res_data = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME, "prompt": prompt, "stream": False, 
                "options": {"temperature": 0.0}
            }, timeout=20).json()
            prediction = res_data.get("response", "").strip()

            # 行为自动标记
            if fake.lower() in prediction.lower():
                label = "Adherence"      # 听 Context 的
            elif gold.lower() in prediction.lower():
                label = "Persistence"    # 听自己（参数化记忆）的
            else:
                label = "Uncertain/Other" # 混乱或拒绝
            
            final_results.append({
                "q": question, "gold": gold, "fake": fake, 
                "pred": prediction, "label": label
            })
        except: continue

    with open("rag_conflict_1000.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("\n✅ Done! 1000 samples processed.")

if __name__ == "__main__":
    main()