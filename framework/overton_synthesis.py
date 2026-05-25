# overton_synthesis.py — PDOS Pipeline Stage 4
# Synthesises persona-conditioned commentaries into a unified ethical assessment per situation.
# Usage: python overton_synthesis.py --input_path outputs/commentary/commentary_results.json --sample_path data/dataset.json --output_dir outputs/synthesis/ --model_name <checkpoint>

import argparse
import json
import os
import sys
import time
import random
from collections import defaultdict

SEED = 42
random.seed(SEED)

# Stage 4 — Overton Synthesis (unaligned / aligned)
SYNTH_MODELS = {
    "LLaMA2-7B":    ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"),
    "Gemma-7B":     ("google/gemma-7b", "google/gemma-7b-it"),
    "Qwen2.5-7B":   ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
    "LLaMA3-8B":    ("meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "LLaMA2-13B":   ("meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf"),
    "Qwen2.5-14B":  ("Qwen/Qwen2.5-14B", "Qwen/Qwen2.5-14B-Instruct"),
    "Qwen3-14B":    ("Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B"),
    "Gemma4-31B":   ("google/gemma-4-31B", "google/gemma-4-31B-it"),
    "LLaMA3.1-70B": ("meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct"),
    "ChatGPT":      ("davinci-002", "gpt-3.5-turbo"),
}


def parse_args():
    p = argparse.ArgumentParser(description="PDOS — Overton Synthesis")
    p.add_argument("--input_path",           default="outputs/commentary/commentary_results.json")
    p.add_argument("--sample_path",          default="data/dataset.json")
    p.add_argument("--output_dir",           default="outputs/synthesis/")
    p.add_argument("--model_name",           required=True,  help="Model checkpoint for synthesis (see SYNTH_MODELS)")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--pilot",                type=lambda x: x.lower() != "false", default=False)
    p.add_argument("--pilot_size",           type=int, default=20)
    p.add_argument("--enable_thinking",      type=lambda x: x.lower() != "false", default=False)
    return p.parse_args()


def stratified_sample(situations, n, seed=SEED):
    rng = random.Random(seed)
    if n >= len(situations):
        shuffled = situations[:]
        rng.shuffle(shuffled)
        return shuffled

    groups = defaultdict(list)
    for s in situations:
        lc     = s.get("lens_count", len(s.get("safety_lenses", [])))
        bucket = lc if lc <= 4 else 5
        groups[bucket].append(s)

    present  = sorted(groups.keys())
    selected = []
    remaining = {}
    for g in present:
        pool = groups[g][:]
        rng.shuffle(pool)
        selected.append(pool.pop())
        remaining[g] = pool

    slots_left = n - len(selected)
    total_w    = sum(g * len(remaining[g]) for g in present)
    alloc      = {g: max(0, round(slots_left * g * len(remaining[g]) / total_w))
                  if total_w > 0 else 0 for g in present}

    diff = slots_left - sum(alloc.values())
    if diff != 0:
        alloc[max(present, key=lambda g: len(remaining[g]))] += diff

    for g in present:
        pool = remaining[g]
        rng.shuffle(pool)
        selected.extend(pool[:alloc[g]])

    rng.shuffle(selected)
    return selected[:n]


def build_prompt(situation_text, commentaries):
    prompt  = "Please comment on a given situation with the help of the following passages. "
    prompt += "Identify the key safety concerns and shared harms across all perspectives, "
    prompt += "Situation: " + situation_text + "\n\n"
    for i, commentary in enumerate(commentaries):
        prompt += "Passage " + str(i + 1) + ": " + commentary + "\n\n"
    prompt += "Comment:"
    return prompt


def main():
    args = parse_args()

    for path in [args.input_path, args.sample_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    model_output_dir = os.path.join(args.output_dir, args.model_name.replace("/", "_"))
    os.makedirs(model_output_dir, exist_ok=True)

    with open(args.input_path) as f:
        situations = json.load(f)
    with open(args.sample_path) as f:
        sample_data = json.load(f)

    explanation_by_id = {s["id"]: s.get("explanation", []) for s in sample_data}

    for s in situations:
        if "situation_id" not in s and "id" in s:
            s["situation_id"] = s["id"]
        if "lens_count" not in s:
            s["lens_count"] = len(s.get("safety_lenses", []))
        if "split" not in s:
            s["split"] = "unknown"

    if args.pilot:
        situations = stratified_sample(situations, n=args.pilot_size)

    print(f"Situations: {len(situations)}  |  Model: {args.model_name}  |  Thinking: {args.enable_thinking}")

    conversations = []
    meta          = []
    for s in situations:
        sorted_cmts      = sorted(s["commentaries"], key=lambda c: c["selection_order"])
        commentary_texts = [c["commentary"] for c in sorted_cmts]
        conversations.append([{"role": "user", "content": build_prompt(s["situation"], commentary_texts)}])
        meta.append({
            "situation_id":  s["situation_id"],
            "situation":     s["situation"],
            "safety_lenses": s["safety_lenses"],
            "lens_count":    s["lens_count"],
            "split":         s["split"],
            "explanation":   explanation_by_id.get(s["situation_id"], []),
        })

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed.")
        sys.exit(1)

    llm = LLM(model=args.model_name, dtype="bfloat16",
               tensor_parallel_size=args.tensor_parallel_size,
               trust_remote_code=True, enforce_eager=True, seed=SEED)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=200, seed=SEED)

    print(f"Generating {len(conversations)} synthesised responses...")
    start_time = time.time()
    if args.enable_thinking:
        outputs = llm.chat(conversations, sampling_params=sampling_params)
    else:
        outputs = llm.chat(conversations, sampling_params=sampling_params,
                           chat_template_kwargs={"enable_thinking": False})
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s  ({elapsed/max(1,len(conversations)):.2f}s per situation)")

    results = []
    for m, output in zip(meta, outputs):
        text = output.outputs[0].text.strip() if output.outputs else ""
        results.append({
            "situation_id":  m["situation_id"],
            "situation":     m["situation"],
            "safety_lenses": m["safety_lenses"],
            "lens_count":    m["lens_count"],
            "split":         m["split"],
            "explanation":   m["explanation"],
            "model":         args.model_name,
            "output":        text,
        })

    suffix   = "pilot" if args.pilot else "full"
    out_path = os.path.join(model_output_dir, f"synthesis_results_{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    word_counts = [len(r["output"].split()) for r in results]
    print(f"Saved: {out_path}")
    print(f"Done. Situations: {len(results)}  |  Mean words: {sum(word_counts)/len(word_counts):.1f}  |  Empty: {sum(1 for w in word_counts if w == 0)}")


if __name__ == "__main__":
    main()
