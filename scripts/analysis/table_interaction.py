import json
import os
from collections import defaultdict

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

INPUT_FILE = os.path.join(RESULTS_DIR, "rag_final_labeled_augmented.json")

SALIENCY_ORDER = ["High", "Medium", "Low"]
LABEL_ORDER = ["Persistence", "Adherence", "Uncertain/Other"]

def pct(count, total):
    return round(count / total * 100, 1) if total > 0 else 0.0

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)["details"]

    # nested dict:
    # stats[saliency][known]["N" / "Persistence" / "Adherence" / "Uncertain/Other"]
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for item in data:
        saliency = item.get("saliency", "Unknown")
        known = "Correct" if item.get("is_known_by_model", False) else "Incorrect"
        label = item.get("label", "Uncertain/Other")

        if label not in LABEL_ORDER:
            label = "Uncertain/Other"

        stats[saliency][known]["N"] += 1
        stats[saliency][known][label] += 1

    print("\nTable Y: Interaction Between Saliency and Parametric Knowledge\n")
    header = f"{'Saliency':<10} {'Probe':<10} {'Persistence (%)':<16} {'Adherence (%)':<15} {'Uncertainty (%)':<17} {'N':<5}"
    print(header)
    print("-" * len(header))

    rows = []

    for sal in SALIENCY_ORDER:
        for known in ["Correct", "Incorrect"]:
            total = stats[sal][known]["N"]
            p = pct(stats[sal][known]["Persistence"], total)
            a = pct(stats[sal][known]["Adherence"], total)
            u = pct(stats[sal][known]["Uncertain/Other"], total)

            row = {
                "saliency": sal,
                "probe": known,
                "persistence": p,
                "adherence": a,
                "uncertainty": u,
                "N": total
            }
            rows.append(row)

            print(f"{sal:<10} {known:<10} {p:<16} {a:<15} {u:<17} {total:<5}")

    # save json version too
    with open(os.path.join(RESULTS_DIR, "table_y_interaction_results.json"), "w") as f:
        json.dump({"rows": rows}, f, indent=4)

    print("\nSaved to table_y_interaction_results.json")

if __name__ == "__main__":
    main()