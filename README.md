# Knowledge Sovereignty vs. Contextual Interference in RAG Systems

**ECE 657D Research Project | University of Waterloo | Winter 2026**

This project investigates how large language models handle knowledge conflicts in Retrieval-Augmented Generation (RAG): when the retrieved context contradicts what the model already knows, which one does it follow?

> **v2 (current).** A re-audit of the original (v1) evaluation found that its answer classifier and counterfactual design inflated "persistence". v2 rebuilds the benchmark on PopQA and reverses two of the v1 conclusions. v1 results and the course paper (`docs/final_paper.tex`) are kept unchanged for reference; see [What changed from v1](#what-changed-from-v1).

## Key Findings (v2)

Llama-3-8B and Qwen2.5-7B, 1,800 PopQA questions stratified by Wikipedia popularity, false context built from a real same-type entity:

- **Plausible false context almost always wins.** Models adopt the false answer 86% (Llama) / 92% (Qwen) of the time, and restate the true answer in only 0.1–0.3% of cases.
- **Knowing the answer does not protect the model.** Even on questions it answers correctly with no context, the model adopts the false context 66% (Llama) / 74% (Qwen) of the time.
- **Popularity changes *how* models react, not whether they correct.** For popular entities the model abstains more often ("I don't know": 27% Llama, 15% Qwen) instead of correcting the context; for long-tail entities it adopts the false answer ~95% of the time.
- **Prompt wording matters a lot.** On popular entities, adoption ranges from 40% ("prioritize factual accuracy") to 99% ("only use the context") for Llama. This reverses v1's "prompting has limited reach".
- **Models do read context correctly.** With a true context the same questions are answered correctly 97% / 99% of the time, so adoption is context-following, not noise.

## v2 Results

False context, neutral RAG prompt (`Adhere` = outputs the false answer, `Persist` = outputs the true answer, `Abstain/Other` = refuses or neither):

| Popularity tier | Llama Adhere | Llama Persist | Llama Abstain/Other | Qwen Adhere | Qwen Persist | Qwen Abstain/Other |
|---|---|---|---|---|---|---|
| High (median 33k views/mo) | 72.8% | 0.3% | 26.5% | 84.0% | 0.2% | 15.0% |
| Medium (median 1.3k) | 91.3% | 0.3% | 8.2% | 95.7% | 0.2% | 3.7% |
| Low (median 131) | 94.7% | 0.2% | 5.2% | 96.7% | 0.0% | 3.3% |
| **All (N=1,800)** | **86.3%** | 0.3% | 13.3% | **92.1%** | 0.1% | 7.3% |

Adoption of false context, split by whether the model answers correctly with no context (zero-context probe):

| | Llama | Qwen |
|---|---|---|
| Model knows the answer | 65.5% (N=383) | 73.9% (N=303) |
| Model does not know | 91.9% (N=1,417) | 95.8% (N=1,497) |

Instruction ablation, adoption of false context (100 samples per tier, same samples across prompts):

| Tier | Llama neutral | Llama strict-context | Llama strict-factual | Qwen neutral | Qwen strict-context | Qwen strict-factual |
|---|---|---|---|---|---|---|
| High | 75% | 99% | 40% | 82% | 100% | 62% |
| Medium | 89% | 98% | 81% | 91% | 100% | 81% |
| Low | 95% | 98% | 88% | 97% | 100% | 91% |

Control (true context) accuracy: Llama 96.9%, Qwen 98.8%.

## v2 Method

1. **Data** (`scripts/data_prep/build_popqa_conflicts.py`): questions from [PopQA](https://github.com/AlexTMallen/adaptive-retrieval) (Mallen et al., 2023), 16 relation types (author, director, place of birth, capital, ...).
2. **Popularity tiers**: by the subject's monthly Wikipedia page views, cut by quantile *within each relation type* (Low ≤ 25th pct, Medium 40–60th, High ≥ 80th), 600 per tier with the same relation mix in every tier.
3. **Counterfactuals**: the true answer is replaced by a real object of the same relation (another real city for "place of birth"), excluding all aliases of the true answer.
4. **Three queries per question** (`scripts/inference/run_conflict_eval.py`): no context (probe), false context, and true context (control). Instruction ablation on 100 questions per tier. Ollama, `temperature=0`.
5. **Labeling** (`scripts/analysis/matching_v2.py`): an output is credited to the true or false answer only by the tokens that distinguish the two, with alias lists from PopQA.

### Reproduce v2

```bash
mkdir -p data/popqa
curl -L -o data/popqa/popQA.tsv https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv
python scripts/data_prep/build_popqa_conflicts.py
python scripts/inference/run_conflict_eval.py --model llama3:8b     # resumable, ~6k queries
python scripts/inference/run_conflict_eval.py --model qwen2.5:7b
python scripts/analysis/analyze_v2.py
```

## What changed from v1

| Issue in v1 | Effect | v2 fix |
|---|---|---|
| Classifier checked the true answer first with ≥50% token overlap; fakes were derived from the true answer (`alt_X`, `$3.1/W installed cost`) | Outputs that copied the fake were labeled Persistence (~600 of 2,100 for Llama) | Match only on tokens that differ between true and false answer (`matching_v2.py`) |
| Named-entity fakes were `alt_` + true answer | Models could spot and strip the obvious marker; remaining "persistence" came almost entirely from these | Real same-type entities from PopQA |
| Saliency from question keywords; Low tier was synthetic templates whose "answer" was the entity name itself and whose context was unrelated | Tiers differed in construction, not just popularity; Low tier not interpretable | Wikipedia page views, within-relation quantiles, identical relation mix |
| No true-context control | Could not tell resisting false context from ignoring context | Supporting-context control per question |
| FAISS index built but not used at inference | Context was injected directly | Stated explicitly: this is a controlled-context evaluation, not end-to-end retrieval |

Re-labeling the saved v1 outputs with the fixed classifier (`scripts/analysis/relabel_v2.py`, no new inference) already drops High-tier persistence from ~80% to ~45–49% and shows the "strict context" prompt raising adoption from 35% to 61%, before any dataset changes.

## v1 (original course submission)

The sections below describe the original pipeline and results as submitted in `docs/final_paper.tex`.

### v1 Key Findings (superseded)

- **Saliency shapes conflict behavior**: Both Llama-3-8B and Qwen2.5-7B show ~80% persistence for well-known entities but collapse to <5% for obscure ones.
- **Parametric recall correlates with resistance**: Samples where a zero-context probe confirms the model knows the answer show meaningfully higher persistence.
- **Cross-model consistency**: Both models exhibit nearly identical saliency-dependent patterns despite architectural differences.
- **Prompt-level intervention has limited reach**: Instruction wording shifts persistence by at most 7 percentage points for high-confidence facts.

## Project Structure

```
.
├── data/                              # Input data
│   ├── popqa/popQA.tsv                  # v2: PopQA source (downloaded)
│   ├── popqa_conflicts_v2.json          # v2: 1,800 popularity-stratified conflicts
│   ├── rag_balanced_2400.json           # v1 dataset (2,100 stratified samples)
│   ├── chunks.json                      # SQuAD text chunks for vector DB
│   └── squad_faiss.index                # FAISS index for retrieval
├── scripts/
│   ├── data_prep/                     # Data preparation
│   │   ├── build_popqa_conflicts.py     # v2: build PopQA conflict set
│   │   ├── generate_balanced_data.py    # v1: generate saliency-stratified dataset
│   │   └── build_vector_db.py           # Build FAISS vector database
│   ├── inference/                     # Model inference
│   │   ├── run_conflict_eval.py         # v2: probe + false + true context + ablation
│   │   ├── parametric_probe.py          # Llama zero-context probe
│   │   ├── rag_pipeline.py              # Llama RAG inference with conflicts
│   │   ├── qwen_full_pipeline.py        # Qwen all-in-one pipeline
│   │   └── ablation_instruction_strength.py  # Instruction ablation study
│   ├── analysis/                      # Result analysis
│   │   ├── matching_v2.py               # v2: fixed answer matcher
│   │   ├── relabel_v2.py                # v2 matcher applied to saved v1 outputs
│   │   ├── analyze_v2.py                # v2 result tables
│   │   ├── behavioral_classifier.py     # v1: classify responses (Persist/Adhere/Uncertain)
│   │   ├── analyze_saliency.py          # Saliency-stratified analysis
│   │   └── generate_all_tables.py       # Generate all paper tables (2-7)
│   └── plotting/                      # Visualization
│       ├── plot_final_results.py        # Saliency behavior bar chart
│       └── plot_comparison_qwen.py      # Llama vs Qwen comparison plot
├── results/                           # Experiment outputs
│   ├── v2/llama3_8b.json, v2/qwen2.5_7b.json  # v2 raw outputs + labels
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
pip install requests tqdm numpy pandas matplotlib datasets sentence-transformers faiss-cpu

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

## v1 Results Summary (superseded)

| Tier | Persist (Llama) | Persist (Qwen) | Adhere (Llama) | Adhere (Qwen) |
|------|-----------------|----------------|----------------|----------------|
| High | 79.6% | 78.9% | 10.9% | 10.5% |
| Medium | 77.8% | 77.2% | 11.6% | 12.4% |
| Low | 4.6% | 2.8% | 29.2% | 22.2% |

## Author

Meihan Liu — University of Waterloo (21220972)

## License

This project is for academic purposes (ECE 657D coursework).
