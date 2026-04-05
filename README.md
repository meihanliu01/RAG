# Knowledge Sovereignty vs. Contextual Interference in RAG Systems

**ECE 657D Research Project | University of Waterloo | Winter 2026**

This project investigates how large language models handle knowledge conflicts in Retrieval-Augmented Generation (RAG) systems. We construct 2,100 counterfactual conflicts stratified by entity saliency and measure whether models persist with internal knowledge, adhere to false context, or hedge with uncertainty.

## Key Findings

- **Saliency shapes conflict behavior**: Both Llama-3-8B and Qwen2.5-7B show ~80% persistence for well-known entities but collapse to <5% for obscure ones.
- **Parametric recall correlates with resistance**: Samples where a zero-context probe confirms the model knows the answer show meaningfully higher persistence.
- **Cross-model consistency**: Both models exhibit nearly identical saliency-dependent patterns despite architectural differences.
- **Prompt-level intervention has limited reach**: Instruction wording shifts persistence by at most 7 percentage points for high-confidence facts.

## Project Structure

```
.
├── data/                              # Input data
│   ├── rag_balanced_2400.json           # Main dataset (2,100 stratified samples)
│   ├── chunks.json                      # SQuAD text chunks for vector DB
│   └── squad_faiss.index                # FAISS index for retrieval
├── scripts/
│   ├── data_prep/                     # Data preparation
│   │   ├── generate_balanced_data.py    # Generate saliency-stratified dataset
│   │   └── build_vector_db.py           # Build FAISS vector database
│   ├── inference/                     # Model inference
│   │   ├── parametric_probe.py          # Llama zero-context probe
│   │   ├── rag_pipeline.py              # Llama RAG inference with conflicts
│   │   ├── qwen_full_pipeline.py        # Qwen all-in-one pipeline
│   │   └── ablation_instruction_strength.py  # Instruction ablation study
│   ├── analysis/                      # Result analysis
│   │   ├── behavioral_classifier.py     # Classify responses (Persist/Adhere/Uncertain)
│   │   ├── analyze_saliency.py          # Saliency-stratified analysis
│   │   └── generate_all_tables.py       # Generate all paper tables (2-7)
│   └── plotting/                      # Visualization
│       ├── plot_final_results.py        # Saliency behavior bar chart
│       └── plot_comparison_qwen.py      # Llama vs Qwen comparison plot
├── results/                           # Experiment outputs
│   ├── rag_final_labeled_augmented.json # Llama labeled results (primary)
│   ├── qwen_final_labeled.json          # Qwen labeled results
│   ├── llama_parametric_probe_results.json
│   ├── rag_final_results.json
│   └── ablation_results.json
├── figures/                           # Generated plots
├── docs/                              # Paper and references
│   ├── final_paper.tex
│   └── references.bib
└── README.md
```

## Pipeline

The experiment follows a sequential pipeline:

```
1. generate_balanced_data.py    Generate 2,100 saliency-stratified samples
         |
2. parametric_probe.py          Zero-context probe (Llama)
         |
3. rag_pipeline.py              RAG inference with conflicting context (Llama)
         |
4. behavioral_classifier.py     Classify: Persistence / Adherence / Uncertain
         |
5. generate_all_tables.py       Produce all paper tables

   qwen_full_pipeline.py        Run steps 2-4 for Qwen in one script
   ablation_instruction_strength.py   Instruction variant ablation
```

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) with `llama3:8b` and `qwen2.5:7b` installed

### Installation

```bash
# Clone the repository
git clone https://github.com/meihanliu01/Rag.git
cd Rag

# Create virtual environment
python -m venv rag_env
source rag_env/bin/activate

# Install dependencies
pip install requests tqdm numpy matplotlib datasets sentence-transformers faiss-cpu

# Pull models via Ollama
ollama pull llama3:8b
ollama pull qwen2.5:7b
```

## Usage

### Full Pipeline (Llama)

```bash
# 1. Generate balanced dataset
python scripts/data_prep/generate_balanced_data.py

# 2. Run parametric probe
python scripts/inference/parametric_probe.py

# 3. Run RAG inference with conflicts
python scripts/inference/rag_pipeline.py

# 4. Classify behavioral responses
python scripts/analysis/behavioral_classifier.py

# 5. Generate all tables
python scripts/analysis/generate_all_tables.py
```

### Cross-Model Comparison (Qwen)

```bash
python scripts/inference/qwen_full_pipeline.py
```

### Instruction Ablation

```bash
python scripts/inference/ablation_instruction_strength.py
```

### Generate Plots

```bash
python scripts/plotting/plot_final_results.py
python scripts/plotting/plot_comparison_qwen.py
```

## Method

1. **Counterfactual Injection**: For each SQuAD question, the gold answer in the passage is replaced with a type-consistent counterfactual (years with years, names with names).

2. **Saliency Stratification**: Entities are divided into High (well-known topics), Medium (general SQuAD), and Low (synthetic long-tail entities from biology, chemistry, etc.).

3. **Zero-Context Probe**: Each question is posed without context to verify if the model can recall the answer from parametric memory alone.

4. **Behavioral Classification**: Responses are classified via soft-token matching (>=50% normalized overlap) into Persistence, Adherence, or Uncertain.

## Results Summary

| Tier | Persist (Llama) | Persist (Qwen) | Adhere (Llama) | Adhere (Qwen) |
|------|-----------------|----------------|----------------|----------------|
| High | 79.6% | 78.9% | 10.9% | 10.5% |
| Medium | 77.8% | 77.2% | 11.6% | 12.4% |
| Low | 4.6% | 2.8% | 29.2% | 22.2% |

## Author

Meihan Liu — University of Waterloo (21220972)

## License

This project is for academic purposes (ECE 657D coursework).
