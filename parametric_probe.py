import json
import requests
import os
from tqdm import tqdm

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"  # Ensure this matches your 'ollama list' output
INPUT_FILE = "rag_balanced_2400.json"
OUTPUT_FILE = "llama_parametric_probe_results.json"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file {INPUT_FILE} not found. Please run generate_balanced_data.py first.")
        return

    with open(INPUT_FILE, "r", encoding='utf-8') as f:
        data = json.load(f)
        samples = data.get('details', [])

    print(f">>> 🚀 Starting Parametric Probe (Zero-Context)...")
    print(f">>> Total Samples to Process: {len(samples)}")
    
    # Initialize statistics for each Saliency Tier
    stats = {
        "High": {"correct": 0, "total": 0}, 
        "Medium": {"correct": 0, "total": 0}, 
        "Low": {"correct": 0, "total": 0}
    }

    for item in tqdm(samples):
        question = item['q']
        gold_answers = [str(ans).lower() for ans in item.get('gt', [])]
        tier = item.get('saliency', 'Unknown')
        
        # Professional academic prompt for fact retrieval
        prompt = (
            f"Instruction: Answer the following question in 1-3 words based on your internal knowledge. "
            f"If the answer is unknown, strictly respond with 'I don't know'.\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )

        payload = {
            "model": MODEL_NAME, 
            "prompt": prompt, 
            "stream": False, 
            "options": {
                "temperature": 0.0, 
                "num_predict": 10,  # Limits output length for faster inference
                "stop": ["\n", "Question:"]
            }
        }
        
        try:
            res = requests.post(OLLAMA_URL, json=payload, timeout=30).json()
            pred = res.get("response", "").strip().lower().rstrip('.')
            item['pred_base'] = pred
            
            # Strict matching logic: check if any gold answer exists in model prediction
            is_correct = any(ans in pred for ans in gold_answers)
            item['is_correct_base'] = is_correct
            
            # Update statistics per tier
            if tier in stats:
                stats[tier]["total"] += 1
                if is_correct:
                    stats[tier]["correct"] += 1

        except Exception as e:
            item['pred_base'] = f"Inference Error: {str(e)}"
            item['is_correct_base'] = False

    # Save processed results
    with open(OUTPUT_FILE, "w", encoding='utf-8') as f:
        json.dump({"details": samples}, f, indent=4, ensure_ascii=False)

    # Final summary report
    print(f"\n✅ Task Complete! Results saved to: {OUTPUT_FILE}")
    print("-" * 40)
    print("📊 Zero-Context Accuracy (Parametric Memory Baseline):")
    for tier, data in stats.items():
        accuracy = (data['correct'] / data['total'] * 100) if data['total'] > 0 else 0
        print(f" - {tier} Saliency: {accuracy:.2f}% ({data['correct']}/{data['total']})")
    print("-" * 40)
    print("Logic Note: For your final RAG analysis, focus on samples where 'is_correct_base' is True.")

if __name__ == "__main__":
    main()