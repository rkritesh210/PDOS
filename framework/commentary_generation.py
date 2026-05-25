# commentary_generation.py — PDOS Pipeline Stage 3
# Generates persona-conditioned moral commentaries (~180 words) for each situation.
# Usage: python commentary_generation.py --input_path outputs/retrieval/retrieval_results.json --output_dir outputs/commentary/ --model_name Qwen/Qwen2.5-7B-Instruct

import argparse
import json
import os
import sys
import time
import random
from collections import defaultdict

SEED     = 42
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
random.seed(SEED)

PROMPT_TEMPLATE = '''\
Situation: "{situation}"
Perspective: {name} — Core Value: {core_value} | Harm: {harm_interpretation} | Accountability: {accountability_stance} | Risk Tolerance: {risk_tolerance} | Role: {stakeholder_role}
Provide a direct moral comment in 180 words that clearly addresses:
- The nature of harm: who is affected and how.
- 2-3 core safety values at stake in this situation.
- Specific rights violated or duties of care neglected.
- Ethical judgment on accountability and what must change.
- Broader societal consequences if this harm goes unaddressed.
Also acknowledge any additional harms this situation involves beyond your primary perspective.
Use normative language (e.g., should, must, ought).
Begin immediately without introduction.\
'''


def parse_args():
    p = argparse.ArgumentParser(description="PDOS — Persona-Conditioned Commentary Generation")
    p.add_argument("--input_path",           default="outputs/retrieval/retrieval_results.json")
    p.add_argument("--output_dir",           default="outputs/commentary/")
    p.add_argument("--model_name",           default=MODEL_ID)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_path):
        print(f"ERROR: Input file not found: {args.input_path}")
        sys.exit(1)

    with open(args.input_path) as f:
        situations = json.load(f)

    for s in situations:
        if "situation_id" not in s:
            s["situation_id"] = s.get("id")
        if "lens_count" not in s:
            s["lens_count"] = len(s.get("safety_lenses", []))
        if "split" not in s:
            s["split"] = "unknown"

    total_personas = sum(len(s["retrieved_personas"]) for s in situations)
    k_per_sit      = total_personas // len(situations) if situations else 0
    print(f"Situations: {len(situations)}  |  Personas/situation: {k_per_sit}  |  Total: {total_personas}")

    flat_items    = []
    conversations = []
    for sit_idx, situation in enumerate(situations):
        for persona in situation["retrieved_personas"]:
            flat_items.append((sit_idx, persona))
            conversations.append([{"role": "user", "content": PROMPT_TEMPLATE.format(
                situation             = situation["situation"],
                name                  = persona["name"],
                core_value            = persona["core_value"],
                harm_interpretation   = persona["harm_interpretation"],
                accountability_stance = persona["accountability_stance"],
                risk_tolerance        = persona["risk_tolerance"],
                stakeholder_role      = persona["stakeholder_role"],
            )}])

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed.")
        sys.exit(1)

    llm = LLM(model=args.model_name, dtype="bfloat16",
               tensor_parallel_size=args.tensor_parallel_size,
               trust_remote_code=False, seed=SEED)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    print(f"Generating {len(conversations)} commentaries...")
    start_time = time.time()
    outputs    = llm.chat(conversations, sampling_params=sampling_params)
    elapsed    = time.time() - start_time
    print(f"Done in {elapsed:.1f}s  ({elapsed/max(1,len(conversations)):.2f}s per commentary)")

    generated_texts  = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]
    sit_commentaries = defaultdict(list)
    for (sit_idx, persona), text in zip(flat_items, generated_texts):
        sit_commentaries[sit_idx].append({
            "persona_id":            persona["persona_id"],
            "persona_name":          persona["name"],
            "core_value":            persona["core_value"],
            "harm_interpretation":   persona["harm_interpretation"],
            "accountability_stance": persona["accountability_stance"],
            "risk_tolerance":        persona["risk_tolerance"],
            "stakeholder_role":      persona["stakeholder_role"],
            "relevance_score":       persona["relevance_score"],
            "mmr_score":             persona["mmr_score"],
            "selection_order":       persona["selection_order"],
            "commentary":            text,
        })

    results = []
    for sit_idx, situation in enumerate(situations):
        results.append({
            "situation_id":  situation["situation_id"],
            "situation":     situation["situation"],
            "safety_lenses": situation["safety_lenses"],
            "lens_count":    situation["lens_count"],
            "split":         situation["split"],
            "commentaries":  sit_commentaries[sit_idx],
        })

    out_path = os.path.join(args.output_dir, "commentary_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    all_commentaries = [c for r in results for c in r["commentaries"]]
    word_counts      = [len(c["commentary"].split()) for c in all_commentaries]
    print(f"Saved: {out_path}")
    print(f"Done. Commentaries: {len(all_commentaries)}  |  Mean words: {sum(word_counts)/len(word_counts):.1f}  |  Short (<30): {sum(1 for w in word_counts if w < 30)}")


if __name__ == "__main__":
    main()
