import json
from collections import Counter

def analyze_1000_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    labels = [item['label'] for item in data]
    stats = Counter(labels)
    
    print(f"{'='*30}")
    print(f"📊 1,000 SAMPLE EMPIRICAL RESULTS")
    print(f"{'='*30}")
    for label, count in stats.items():
        percentage = (count / total) * 100
        print(f"{label:<15}: {count:>4} ({percentage:.2f}%)")
    
    # 挑选两个极端案例用于 Discussion 章节
    print(f"\n💡 Case Study Selection:")
    persistence_cases = [i for i in data if i['label'] == 'Persistence'][:3]
    adherence_cases = [i for i in data if i['label'] == 'Adherence'][:3]
    
    for c in persistence_cases:
        print(f"[REJECTED CONFLICT] Q: {c['q']} | Real: {c['gold']} | Fake: {c['fake']}")
    for c in adherence_cases:
        print(f"[FOLLOWED CONFLICT] Q: {c['q']} | Real: {c['gold']} | Fake: {c['fake']}")

analyze_1000_results("rag_conflict_1000.json")