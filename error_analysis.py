import json

def analyze_errors(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    details = data['details']
    total = len(details)
    
    categories = {
        "Verbosity (Too many words)": 0,    # 意思对但话太多导致 EM=0
        "Retrieval Failure (I don't know)": 0, # 搜不到资料
        "Knowledge Conflict / Error": 0,   # 答错了（可能是由于内部知识干扰）
        "Correct (EM=1)": 0
    }

    examples = []

    for item in details:
        pred = item['prediction'].lower()
        gt_list = [g.lower() for g in item['ground_truth']]
        
        if item['em'] == 1:
            categories["Correct (EM=1)"] += 1
        elif "don't know" in pred or "does not mention" in pred:
            categories["Retrieval Failure (I don't know)"] += 1
        elif any(gt in pred for gt in gt_list):
            # 如果标准答案在预测字符串里，但 EM=0，说明是“话太多”
            categories["Verbosity (Too many words)"] += 1
        else:
            categories["Knowledge Conflict / Error"] += 1
            examples.append(item)

    print(f"--- Error Analysis for {file_path} ---")
    for cat, count in categories.items():
        print(f"{cat}: {count} ({count/total*100:.1f}%)")
    
    return examples

# 运行分析
error_samples = analyze_errors("rag_k5.json")