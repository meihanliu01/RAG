import json
import os
import re
import string

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def normalize(s):
    """Normalize string: lowercase, remove articles, punctuation (incl. hyphens), extra whitespace."""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Convert hyphens and apostrophes to spaces (e.g., al-banna -> al banna)
    s = s.replace('-', ' ').replace("'", ' ')
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return ' '.join(s.split())

def is_soft_match(pred, target):
    """Soft match: supports substring containment and token overlap."""
    if not pred or not target: return False
    p_norm = normalize(pred)
    t_norm = normalize(target)
    
    # 1. Substring containment (bidirectional)
    if t_norm in p_norm or p_norm in t_norm: 
        return True
    
    # 2. Token overlap matching (handles variants)
    p_tokens = set(p_norm.split())
    t_tokens = set(t_norm.split())
    if not t_tokens: return False
    intersection = p_tokens.intersection(t_tokens)
    
    # Match if >= 50% of target tokens appear in prediction
    return len(intersection) / len(t_tokens) >= 0.5

def classify_behavior(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Support both dict and list formats
    samples = raw_data.get('details', raw_data) if isinstance(raw_data, dict) else raw_data

    print(f">>> Processing {len(samples)} samples...")

    for entry in samples:
        # 1. Extract base fields
        pred_rag = entry.get('pred', '')        # Model output under RAG mode
        pred_base = entry.get('pred_base', '')  # Model output under zero-context mode (Probe)
        
        gt_data = entry.get('gt', entry.get('gold', ''))
        gold = gt_data[0] if isinstance(gt_data, list) else gt_data
        fake = entry.get('fake', '')
        saliency = entry.get('saliency', 'Medium')

        # 2. Determine if model already knew the answer (bidirectional matching)
        # Only samples meeting this condition count toward Base Accuracy
        entry['is_known_by_model'] = is_soft_match(pred_base, gold)

        # 3. RAG behavior classification (based on pred_rag)
        # Unified taxonomy (Table 1): same logic for all saliency tiers
        pred_rag_norm = normalize(pred_rag)
        
        # A. Filter refusal responses (Uncertain)
        if any(msg in pred_rag.lower() for msg in ["don't know", "dont know", "not mentioned", "no information"]) or not pred_rag_norm:
            entry['label'] = "Uncertain/Other"
            
        # B. Unified classification logic (applies to High / Medium / Low)
        elif is_soft_match(pred_rag, gold):
            entry['label'] = "Persistence"
        elif is_soft_match(pred_rag, fake):
            entry['label'] = "Adherence"
        else:
            entry['label'] = "Uncertain/Other"

    # --- Save results ---
    output_json = {"details": samples}
    # Preserve other metadata from original JSON
    if isinstance(raw_data, dict):
        for k, v in raw_data.items():
            if k != 'details': output_json[k] = v

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"Success! Data labeled with enhanced matching logic.")
    print(f"Final labeled file saved to: {output_file}")

if __name__ == "__main__":
    # Input: RAG inference results (contains pred + pred_base fields)
    classify_behavior(os.path.join(RESULTS_DIR, "rag_final_results.json"), os.path.join(RESULTS_DIR, "rag_final_labeled_augmented.json"))