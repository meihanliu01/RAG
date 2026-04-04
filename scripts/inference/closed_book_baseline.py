import os
import sys
import json
import random
import re
import string
from collections import Counter
import requests
from tqdm import tqdm

# Must be at the very top for macOS safety
os.environ["OBJC_DISABLE_INITIALIZE_FOR_SAFETY"] = "YES"

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

print(">>> Script starting: Loading base libraries...", flush=True)

try:
    from datasets import load_dataset
    print(">>> Libraries loaded successfully.", flush=True)
except Exception as e:
    print(f"❌ Crash during import phase: {e}", flush=True)
    sys.exit(1)

# --- Configuration ---
SAMPLE_SIZE = 100
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b" 
SEED = 42

# --- Evaluation Functions ---
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def calculate_f1(prediction, ground_truth):
    p_tokens = normalize_answer(prediction).split()
    g_tokens = normalize_answer(ground_truth).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(p_tokens)
    recall = 1.0 * num_same / len(g_tokens)
    return (2 * precision * recall) / (precision + recall)

def calculate_em(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def get_llama_response(prompt):
    # Added context to the prompt to ensure the model focuses on the question
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Question: {prompt}\nAnswer in as few words as possible:",
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return response.json().get("response", "").strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print(">>> Entering main function...", flush=True)
    
    print(">>> Downloading/Loading SQuAD dataset...", flush=True)
    dataset = load_dataset("squad", split="validation")
    
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), SAMPLE_SIZE)
    test_samples = [dataset[i] for i in indices]
    print(f">>> Successfully sampled {SAMPLE_SIZE} items.", flush=True)

    results = []
    total_em, total_f1 = 0, 0

    print(f">>> Requesting responses from Ollama ({MODEL_NAME})...", flush=True)
    
    for item in tqdm(test_samples, desc="Testing"):
        question = item['question']
        ground_truths = item['answers']['text']
        
        prediction = get_llama_response(question)
        
        # SQuAD evaluation: take the max score among all possible ground truth answers
        em = max(calculate_em(prediction, gt) for gt in ground_truths)
        f1 = max(calculate_f1(prediction, gt) for gt in ground_truths)
        
        total_em += em
        total_f1 += f1
        
        results.append({
            "question": question, 
            "ground_truth": ground_truths, 
            "prediction": prediction, 
            "em": em, 
            "f1": f1
        })

    final_em = total_em / SAMPLE_SIZE
    final_f1 = total_f1 / SAMPLE_SIZE
    
    print(f"\n✅ Experiment Complete!\nEM Score: {final_em:.4f}\nF1 Score: {final_f1:.4f}", flush=True)
    
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump({"em": final_em, "f1": final_f1, "details": results}, f, indent=4)
    print(">>> Results saved to results/baseline_results.json")

if __name__ == "__main__":
    main()