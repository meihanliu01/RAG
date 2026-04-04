import json
import random
import os

def build_balanced_dataset(input_file, output_file, target_per_tier=800):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 兼容你之前的格式
    all_samples = data.get('details', data) if isinstance(data, dict) else data

    # --- 1. 完善分类逻辑 ---
    # 建议：如果你有 Wikipedia 词频数据，在这里合并。
    # 如果没有，至少增加关键词库的覆盖面。
    high_kws = ['president', 'capital', 'war', 'century', 'founder', 'science', 'earth', 'human']
    
    # 准备三个容器
    tiers = {'High': [], 'Medium': [], 'Low': []}

    for entry in all_samples:
        q_text = str(entry.get('q', '')).lower()
        # 这里的逻辑可以根据你的 saliency 评分脚本进行替换
        if any(kw in q_text for kw in high_kws):
            entry['saliency'] = 'High'
        else:
            entry['saliency'] = 'Medium'
        
        # 先把原始数据里的 Medium 和 High 分开
        tiers[entry['saliency']].append(entry)

    # --- 2. 构造真正的 Low-Saliency 样本 ---
    # 严谨做法：Low Saliency 不能靠复制，要靠“长尾实体注入”
    # 你可以准备一个包含几千个长尾实体的 json (如生物学、稀有矿物、偏僻人名)
    long_tail_entities = [
        ("Zfyve26", "Zfyve27"), ("Apicoplast", "Chloroplast-X"), 
        ("Merytre-Hatshepsut", "Hatshepsut-B"), ("Boulengerula", "Boulengerula-Z")
        # 这里建议扩展到 1000 对以上，不要重复使用
    ]
    
    # 构造 Low 桶
    for i in range(target_per_tier):
        # 随机选一个基础样本作为模板（只取其句式）
        template = random.choice(all_samples)
        pair = random.choice(long_tail_entities) # 理想情况是这里有 >800 个不同的 pair
        
        new_entry = template.copy()
        new_entry['q'] = f"What is the primary function of the {pair[0]} in biological systems?"
        new_entry['gt'] = [pair[0]]
        new_entry['fake'] = pair[1]
        new_entry['saliency'] = 'Low'
        tiers['Low'].append(new_entry)

    # --- 3. 等额抽样 (Stratified Sampling) ---
    balanced_data = []
    for tier, samples in tiers.items():
        print(f"Tier {tier} available: {len(samples)}")
        if len(samples) < target_per_tier:
            # 如果样本不足，这里抛出警告，而不是简单复制
            print(f"Warning: {tier} only has {len(samples)}, needed {target_per_tier}")
            # 这种情况下，你可能需要去 SQuAD 原始集里多抓点数据
            balanced_data.extend(samples) 
        else:
            balanced_data.extend(random.sample(samples, target_per_tier))

    # --- 4. 保存 ---
    output_json = {"details": balanced_data}
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=4)
    
    print(f"Success! Total balanced samples: {len(balanced_data)}")
    # 预期输出: Total 2400 (800 High, 800 Medium, 800 Low)

if __name__ == "__main__":
    build_balanced_dataset("rag_final_analyzed_1000.json", "rag_balanced_2400.json")