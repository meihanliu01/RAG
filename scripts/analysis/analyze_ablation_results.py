import json
import os
from collections import Counter

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def classify_behavior_improved(response):
    response = response.lower()
    
    if "alt_" in response:
        return "Adherence"
    
    hedging_terms = ["not mentioned", "not provided", "however", "conflicting", "uncertain", "i apologize"]
    if any(term in response for term in hedging_terms):
        return "Uncertain"
    

    return "Persistence"

def analyze(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    stats = {
        "neutral": Counter(),
        "strict_context": Counter(),
        "strict_parametric": Counter()
    }
    
    for entry in data:
        variant = entry.get('variant', 'unknown')
        if variant not in stats: continue
        
        behavior = classify_behavior_improved(entry['response'])
        stats[variant][behavior] += 1

    # 打印报表
    print(f"\n{'Variant':<20} | {'Persistence':<12} | {'Adherence':<12} | {'Uncertain':<12}")
    print("-" * 65)
    
    for variant, counts in stats.items():
        total = sum(counts.values())
        if total == 0: continue
        
        p = f"{(counts['Persistence']/total)*100:>5.1f}%"
        a = f"{(counts['Adherence']/total)*100:>5.1f}%"
        u = f"{(counts['Uncertain']/total)*100:>5.1f}%"
        
        print(f"{variant:<20} | {p:<12} | {a:<12} | {u:<12}")

if __name__ == "__main__":
    analyze(os.path.join(RESULTS_DIR, "ablation_results.json"))