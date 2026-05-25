# PDOS: Persona-Driven Overton Synthesis

Safety in large language models is often reduced to monolithic judgments that fail to capture the range of reasonable viewpoints. We introduce **SOS**, the first safety-focused Overton dataset of 1,770 real-world safety situations annotated across 14 harm categories, and **PDOS**, a lightweight training-free framework that retrieves personas from a synthetic pool and integrates their perspectives to generate unified Overton safety responses. Evaluated across five baselines and ten LLMs, PDOS achieves the highest average NLI coverage of 0.4226.


![SOS Dataset Example]<img width="716" height="366" alt="example" src="https://github.com/user-attachments/assets/4690078e-1294-4cce-a3cc-1fa945c2a4d6" />
)

---

## Pipeline Overview

```
Stage 1: build_persona_pool.py      →  persona_pool.json
Stage 2: persona_retrieval.py       →  retrieval_results.json
Stage 3: commentary_generation.py   →  commentary_results.json
Stage 4: overton_synthesis.py       →  synthesis_results.json
```

### Stage 1 — Persona Pool Construction
Generates 4 diverse ethical personas per seed situation using an LLM, embeds them with `sentence-transformers`, clusters them via agglomerative clustering (cosine, average linkage), and selects one representative per cluster. A lens-completeness audit ensures every harm category has at least 2 personas in the pool.

### Stage 2 — Situation-Aware Persona Retrieval
For each situation in the dataset, retrieves `k=4` personas from the pool using Maximal Marginal Relevance (MMR, λ=0.6) to balance relevance and diversity.

### Stage 3 — Persona-Conditioned Commentary Generation
Each retrieved persona generates a ~180-word moral commentary on its assigned situation, grounded in its core value, harm interpretation, accountability stance, risk tolerance, and stakeholder role.

### Stage 4 — Overton Synthesis
A synthesis model reads all persona commentaries for a situation and produces a single unified ethical assessment identifying shared safety concerns and harms across perspectives. Evaluated across 10 LLMs (5 unaligned + 5 aligned pairs).

---

## Usage

Run each stage sequentially:

```bash
# Stage 1 — Build persona pool from seed situations
python build_persona_pool.py \
    --data_path data/seed_situations.json \
    --output_dir outputs/

# Stage 2 — Retrieve personas for each situation
python persona_retrieval.py \
    --pool_path   outputs/persona_pool.json \
    --sample_path data/dataset.json \
    --output_dir  outputs/retrieval/

# Stage 3 — Generate persona-conditioned commentaries
python commentary_generation.py \
    --input_path outputs/retrieval/retrieval_results.json \
    --output_dir outputs/commentary/

# Stage 4 — Synthesise commentaries into unified assessments
python overton_synthesis.py \
    --input_path  outputs/commentary/commentary_results.json \
    --sample_path data/dataset.json \
    --output_dir  outputs/synthesis/ \
    --model_name  <checkpoint>
```
---

## Models

| Stage | Model |
|---|---|
| Stage 1 (Persona Generation) | `meta-llama/Llama-3.1-70B-Instruct` |
| Stage 1 (Embedding) | `sentence-transformers/all-mpnet-base-v2` |
| Stage 2 (Embedding) | `sentence-transformers/all-mpnet-base-v2` |
| Stage 3 (Commentary) | `Qwen/Qwen2.5-7B-Instruct` |
| Stage 4 (Synthesis) | 10 LLMs — see table below |

### Stage 4 Evaluation Models

| Model | Unaligned | Aligned |
|---|---|---|
| LLaMA2-7B | `meta-llama/Llama-2-7b-hf` | `meta-llama/Llama-2-7b-chat-hf` |
| Gemma-7B | `google/gemma-7b` | `google/gemma-7b-it` |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` |
| LLaMA3-8B | `meta-llama/Meta-Llama-3-8B` | `meta-llama/Meta-Llama-3-8B-Instruct` |
| LLaMA2-13B | `meta-llama/Llama-2-13b-hf` | `meta-llama/Llama-2-13b-chat-hf` |
| Qwen2.5-14B | `Qwen/Qwen2.5-14B` | `Qwen/Qwen2.5-14B-Instruct` |
| Qwen3-14B | `Qwen/Qwen3-14B-Base` | `Qwen/Qwen3-14B` |
| Gemma4-31B | `google/gemma-4-31B` | `google/gemma-4-31B-it` |
| LLaMA3.1-70B | `meta-llama/Llama-3.1-70B` | `meta-llama/Llama-3.1-70B-Instruct` |
| ChatGPT | `davinci-002` | `gpt-3.5-turbo` |

---

## Release Note
This repository is being prepared for public release.
- Evaluation scripts for Overton is built upon the work of [Modular Pluralism](https://github.com/BunsenFeng/modular_pluralism)
