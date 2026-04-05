import json
import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

def get_llama_response(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0} 
    }
    response = requests.post(OLLAMA_URL, json=payload).json()
    return response.get("response", "").strip()

def run_conflict_experiment(sample_data):
    # 1. 提取原始数据
    question = sample_data['question']
    original_context = sample_data.get('retrieved_context', "No context provided")
    correct_answer = sample_data['ground_truth'][0]

    # 2. 构造冲突上下文 (以修改日期为例)
    # 假设正确答案是 "1856", 我们将其改为 "2026"
    fake_answer = "2026" 
    conflicting_context = original_context.replace(correct_answer, fake_answer)

    # 3. 构造三种 Prompt
    prompts = {
        "Closed-Book": f"Question: {question}\nAnswer:",
        "Standard-RAG": f"Context: {original_context}\nQuestion: {question}\nAnswer:",
        "Conflicting-RAG": f"Context: {conflicting_context}\nQuestion: {question}\nAnswer:"
    }

    results = {"question": question, "correct_ans": correct_answer, "fake_ans": fake_answer}
    
    print(f"\nTesting Question: {question}")
    for mode, prompt in prompts.items():
        response = get_llama_response(prompt)
        results[mode] = response
        print(f"[{mode}] -> {response}")

    return results

# --- 模拟运行 ---
# 你可以从 rag_k5.json 的 details 中挑选具体的 index
test_case = {
    "question": "In what year was Nikola Tesla born?",
    "retrieved_context": "Nikola Tesla was born an ethnic Serb in the village of Smiljan... on 10 July 1856.",
    "ground_truth": ["1856"]
}

final_res = run_conflict_experiment(test_case)