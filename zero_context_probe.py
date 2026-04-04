import json

with open("rag_final_labeled_augmented.json") as f:
    data = json.load(f)['details']

def compute_stats(group):
    total = len(group)
    p = sum(1 for x in group if x['final_label']=="Persistence") / total * 100
    a = sum(1 for x in group if x['final_label']=="Adherence") / total * 100
    u = sum(1 for x in group if x['final_label']=="Uncertain/Other") / total * 100
    return round(p,1), round(a,1), round(u,1), total

known = [x for x in data if x['is_known_by_model']]
unknown = [x for x in data if not x['is_known_by_model']]

print("Known:", compute_stats(known))
print("Unknown:", compute_stats(unknown))