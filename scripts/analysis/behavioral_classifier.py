import json
import os
import re
import string

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def normalize(s):
    """归一化字符串：小写、去虚词、去标点、去多余空格"""
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return ' '.join(s.split())

def is_soft_match(pred, target):
    """模糊匹配逻辑：支持包含关系和 Token 交集"""
    if not pred or not target: return False
    p_norm = normalize(pred)
    t_norm = normalize(target)
    
    # 1. 直接包含 (双向)
    if t_norm in p_norm or p_norm in t_norm: 
        return True
    
    # 2. Token 交集匹配 (处理变体)
    p_tokens = set(p_norm.split())
    t_tokens = set(t_norm.split())
    if not t_tokens: return False
    intersection = p_tokens.intersection(t_tokens)
    
    # 如果目标核心词中超过 50% 的单词出现在预测中，判定为匹配
    return len(intersection) / len(t_tokens) >= 0.5

def classify_behavior(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 兼容字典格式和列表格式
    samples = raw_data.get('details', raw_data) if isinstance(raw_data, dict) else raw_data

    print(f">>> Processing {len(samples)} samples...")

    for entry in samples:
        # 1. 提取基础字段
        pred_rag = entry.get('pred', '')        # RAG 模式下的输出
        pred_base = entry.get('pred_base', '')  # 无 Context 模式下的输出 (Probe)
        
        gt_data = entry.get('gt', entry.get('gold', ''))
        gold = gt_data[0] if isinstance(gt_data, list) else gt_data
        fake = entry.get('fake', '')
        saliency = entry.get('saliency', 'Medium')

        # 2. 判定模型是否“本来就知道” (关键更新：双向匹配增强)
        # 只有满足这个条件的样本，在分析时才会被计入 Base Accuracy
        entry['is_known_by_model'] = is_soft_match(pred_base, gold)

        # 3. RAG 行为判定逻辑 (基于 pred_rag)
        # 统一 Taxonomy (论文 Table 1): 所有 Saliency Tier 使用相同逻辑
        pred_rag_norm = normalize(pred_rag)
        
        # A. 排除拒绝回答 (Uncertain)
        if any(msg in pred_rag.lower() for msg in ["don't know", "dont know", "not mentioned", "no information"]) or not pred_rag_norm:
            entry['label'] = "Uncertain/Other"
            
        # B. 统一分类逻辑 (适用于 High / Medium / Low)
        elif is_soft_match(pred_rag, gold):
            entry['label'] = "Persistence"
        elif is_soft_match(pred_rag, fake):
            entry['label'] = "Adherence"
        else:
            entry['label'] = "Uncertain/Other"

    # --- 保存结果 ---
    output_json = {"details": samples}
    # 保留原始 JSON 中的其他元数据
    if isinstance(raw_data, dict):
        for k, v in raw_data.items():
            if k != 'details': output_json[k] = v

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"✅ Success! Data labeled with enhanced matching logic.")
    print(f"✅ Final labeled file saved to: {output_file}")

if __name__ == "__main__":
    # 输入: RAG 推理结果 (包含 pred + pred_base 字段)
    classify_behavior(os.path.join(RESULTS_DIR, "rag_final_results.json"), os.path.join(RESULTS_DIR, "rag_final_labeled_augmented.json"))