import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

def test_conflict(case_name, question, original_context, correct_ans, fake_ans):
    conflicting_context = original_context.replace(correct_ans, fake_ans)
    
    prompt = f"""Use the following context to answer the question.
Context: {conflicting_context}

Question: {question}
Answer:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    print(f"\n--- Testing {case_name} ---")
    print(f"Target: Change '{correct_ans}' to '{fake_ans}'")
    response = requests.post(OLLAMA_URL, json=payload).json().get("response", "").strip()
    print(f"Model Response: {response}")
    
    # 判定逻辑
    if fake_ans in response:
        print("Result: 🟢 Followed Context (High Faithfulness)")
    elif correct_ans in response:
        print("Result: 🔴 Stuck to Internal Memory (Knowledge Persistence)")
    else:
        print("Result: 🟡 Uncertain / Refused")

# --- 案例 1：常识 (Tesla) ---
test_conflict(
    "Common Knowledge (Tesla)",
    "In what year was Nikola Tesla born?",
    "Nikola Tesla was born on 10 July 1856.",
    "1856",
    "2026"
)

# --- 案例 2：冷门知识 (Tentilla - 你的数据里有的) ---
test_conflict(
    "Niche Knowledge (Tentilla)",
    "What are the little tentacles that cydippids have called?",
    "Cydippids have little tentacles called tentilla.",
    "tentilla",
    "spaghetti_arms" # 故意改成一个荒诞的词
)