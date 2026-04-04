import json
import os
import matplotlib.pyplot as plt

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

def load_result(filename, default_k=None):
    if not os.path.exists(filename):
        print(f"⚠️ Warning: {filename} not found, skipping...")
        return None
    with open(filename, 'r') as f:
        data = json.load(f)
        k = data.get('k', default_k)
        return {"k": k, "em": data['em'], "f1": data['f1']}

def main():

    print(">>> Plotting script started...")
    files = [
        (os.path.join(RESULTS_DIR, "baseline_results.json"), 0),
        (os.path.join(DATA_DIR, "rag_k1.json"), 1),
        (os.path.join(DATA_DIR, "rag_k3.json"), 3),
        (os.path.join(DATA_DIR, "rag_k5.json"), 5),
        (os.path.join(DATA_DIR, "rag_k8.json"), 8)
    ]

    results = []
    for filename, k_val in files:
        res = load_result(filename, default_k=k_val)
        if res:
            results.append(res)


    results.sort(key=lambda x: x['k'])

    ks = [r['k'] for r in results]
    ems = [r['em'] for r in results]
    f1s = [r['f1'] for r in results]

    plt.figure(figsize=(10, 6))
    
    # EM 
    plt.plot(ks, ems, marker='o', linestyle='-', linewidth=2, label='EM (Exact Match)', color='#1f77b4')
    # F1
    plt.plot(ks, f1s, marker='s', linestyle='--', linewidth=2, label='F1 Score', color='#ff7f0e')


    max_em_idx = ems.index(max(ems))
    plt.annotate(f'Peak EM: {ems[max_em_idx]:.2f}', 
                 xy=(ks[max_em_idx], ems[max_em_idx]), 
                 xytext=(ks[max_em_idx], ems[max_em_idx]+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.title('RAG Performance vs. Top-K Retrieval', fontsize=14, fontweight='bold')
    plt.xlabel('Top-K (Number of Retrieved Chunks)', fontsize=12)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xticks(ks)
    plt.ylim(0, 0.8)  
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    

    plt.text(0.5, 0.05, "K=0 is Closed-Book Baseline", transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


    plt.tight_layout()
    output_img = os.path.join(FIGURES_DIR, "rag_performance_comparison.png")
    plt.savefig(output_img, dpi=300)
    print(f"✅ Success! Plot saved as '{output_img}'")
    plt.show()
    

if __name__ == "__main__":
    main()