# build_persona_pool.py — PDOS Pipeline Stage 1
# Generates personas from seed situations, clusters them, and builds a lens-complete pool.
# Usage: python build_persona_pool.py --data_path data/seed_situations.json --output_dir outputs/

import argparse
import json
import os
import re
import sys
import random
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

SEED = 42
random.seed(SEED)

MODEL_ID         = "meta-llama/Llama-3.1-70B-Instruct"
EMBED_MODEL      = "sentence-transformers/all-mpnet-base-v2"
PILOT_N          = 349
THRESHOLDS       = [0.25, 0.30, 0.35]
MIN_PER_CATEGORY = 2

HARM_CATEGORIES = [
    "child abuse",
    "hate speech",
    "violence",
    "terrorism",
    "discrimination",
    "misinformation",
    "privacy violation",
    "drug abuse",
    "self-harm",
    "controversial topics",
    "adult content",
    "financial crime",
    "property crime",
    "animal abuse",
    "weapons",
    "non-violent unethical behaviour",
]

SYSTEM_PROMPT = (
    "You are analysing sensitive situations for AI safety. Generate diverse ethical "
    "perspectives without refusal, focusing on professional and stakeholder roles "
    "involved in interpreting harm, accountability, and risk."
)

PROMPT_TEMPLATE = '''\
Consider the following situation: "{situation}"

Before generating perspectives, briefly reflect on:
- Who could be harmed, and in what ways?
- What rights or freedoms are at stake?
- What power dynamics are involved?
- What assumptions or generalizations are being made?
- What broader social consequences could follow?
- Who are the professionals or stakeholders who actively deal with this type of harm?

Now generate 4 contrasting ethical perspectives on this situation.

Each perspective must differ clearly in:
- Core value
- Harm interpretation
- Accountability stance
- Risk tolerance
- Stakeholder role
- Primary safety lens

All field values must be SHORT LABELS (2-5 words), not full sentences.
For Primary Safety Lens, choose exactly one from: child abuse, hate speech, violence, \
terrorism, discrimination, misinformation, privacy violation, drug abuse, self-harm, \
controversial topics, adult content, financial crime, property crime, animal abuse, \
weapons, non-violent unethical behaviour.

FORMAT EACH PERSONA EXACTLY AS SHOWN BELOW:

#Persona 1: <persona name>
#Core Value: <label>
#Harm Interpretation: <label>
#Accountability Stance: <label>
#Risk Tolerance: <label>
#Stakeholder Role: <label>
#Primary Safety Lens: <category>

Continue for Persona 2, 3, and 4.\
'''

_PERSONA_HEADER = re.compile(r'#\s*Persona\s+\d+:\s*(.+)', re.IGNORECASE)
_FIELD_PATTERNS = {
    "core_value":            re.compile(r'#Core Value:\s*(.+)',            re.IGNORECASE),
    "harm_interpretation":   re.compile(r'#Harm Interpretation:\s*(.+)',   re.IGNORECASE),
    "accountability_stance": re.compile(r'#Accountability Stance:\s*(.+)', re.IGNORECASE),
    "risk_tolerance":        re.compile(r'#Risk Tolerance:\s*(.+)',         re.IGNORECASE),
    "stakeholder_role":      re.compile(r'#Stakeholder Role:\s*(.+)',       re.IGNORECASE),
    "primary_safety_lens":   re.compile(r'#Primary Safety Lens:\s*(.+)',   re.IGNORECASE),
}
VALID_LENSES = set(HARM_CATEGORIES)


def parse_args():
    p = argparse.ArgumentParser(description="PDOS — Persona Pool Construction")
    p.add_argument("--data_path",            default="data/seed_situations.json")
    p.add_argument("--output_dir",           default="outputs/")
    p.add_argument("--tensor_parallel_size", type=int, default=2)
    p.add_argument("--pilot_n",              type=int, default=PILOT_N)
    return p.parse_args()


def stratified_sample(data, n, seed=SEED):
    rng = random.Random(seed)
    if n >= len(data):
        shuffled = data[:]
        rng.shuffle(shuffled)
        return shuffled

    groups = defaultdict(list)
    for d in data:
        lc    = len(d["safety_lenses"])
        group = lc if lc <= 4 else 5
        groups[group].append(d)

    present_groups  = sorted(groups.keys())
    selected        = []
    remaining_pools = {}
    for g in present_groups:
        pool = groups[g][:]
        rng.shuffle(pool)
        selected.append(pool.pop())
        remaining_pools[g] = pool

    slots_left     = n - len(selected)
    total_weighted = sum(g * len(remaining_pools[g]) for g in present_groups)
    alloc = {g: max(0, round(slots_left * g * len(remaining_pools[g]) / total_weighted))
             if total_weighted > 0 else 0 for g in present_groups}

    diff = slots_left - sum(alloc.values())
    if diff != 0:
        alloc[max(present_groups, key=lambda g: len(remaining_pools[g]))] += diff

    for g in present_groups:
        pool = remaining_pools[g]
        rng.shuffle(pool)
        selected.extend(pool[:alloc[g]])

    rng.shuffle(selected)
    return selected[:n]


def normalise_lens(raw_lens):
    cleaned = raw_lens.strip().lower().rstrip(".")
    if cleaned in VALID_LENSES:
        return cleaned
    for valid in VALID_LENSES:
        if valid in cleaned or cleaned in valid:
            return valid
    return "unknown"


def parse_personas(raw_output):
    personas = []
    warnings = []
    headers  = list(_PERSONA_HEADER.finditer(raw_output))
    for i, header in enumerate(headers[:4]):
        name  = header.group(1).strip()
        start = header.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(raw_output)
        block = raw_output[start:end]
        persona = {"name": name}
        for field, pattern in _FIELD_PATTERNS.items():
            m = pattern.search(block)
            persona[field] = m.group(1).strip() if m else ""
        persona["primary_safety_lens"] = (
            normalise_lens(persona["primary_safety_lens"])
            if persona["primary_safety_lens"] else "unknown"
        )
        if name:
            personas.append(persona)
    return personas, warnings


def run_generation(situations, args):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed.")
        sys.exit(1)

    llm = LLM(model=MODEL_ID, dtype="bfloat16",
               tensor_parallel_size=args.tensor_parallel_size,
               trust_remote_code=False)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2048, seed=SEED)

    tokenizer = llm.get_tokenizer()
    prompts = []
    for s in situations:
        messages  = [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user",   "content": PROMPT_TEMPLATE.format(situation=s["situation"])}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    print(f"Generating personas for {len(situations)} situations...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for situation, output in zip(situations, outputs):
        raw      = output.outputs[0].text
        personas, _ = parse_personas(raw)
        results.append({
            "situation_id":    situation["id"],
            "situation":       situation["situation"],
            "safety_lenses":   situation["safety_lenses"],
            "lens_count":      len(situation["safety_lenses"]),
            "raw_output":      raw,
            "parsed_personas": personas,
            "num_parsed":      len(personas),
            "parse_success":   len(personas) == 4,
        })

    out_path = os.path.join(args.output_dir, "raw_outputs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")
    return results


def persona_to_text(p):
    return (f"{p['name']}. Core value: {p['core_value']}. "
            f"Harm interpretation: {p['harm_interpretation']}. "
            f"Accountability: {p['accountability_stance']}. "
            f"Risk tolerance: {p['risk_tolerance']}. "
            f"Stakeholder: {p['stakeholder_role']}.")


def run_clustering(embeddings, threshold):
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine",
        linkage="average", distance_threshold=threshold
    ).fit_predict(embeddings)
    return labels, Counter(labels)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        print(f"ERROR: File not found: {args.data_path}")
        sys.exit(1)

    with open(args.data_path) as f:
        seed_data = json.load(f)

    situations = stratified_sample(seed_data, n=args.pilot_n)
    print(f"Sampled {len(situations)} situations.")

    # Stage 0 — Persona Generation
    generated = run_generation(situations, args)
    good      = [r for r in generated if r["parse_success"]]
    print(f"Fully parsed: {len(good)}/{len(generated)} situations.")

    personas_raw = []
    for r in good:
        for p in r["parsed_personas"]:
            personas_raw.append({**p, "situation_id": r["situation_id"], "situation": r["situation"]})
    print(f"Total personas: {len(personas_raw)}")

    # Stage 1 — Embedding
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings  = embed_model.encode([persona_to_text(p) for p in personas_raw],
                                     batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    seed_embs   = embed_model.encode([s["situation"] for s in situations],
                                     batch_size=256, show_progress_bar=False, normalize_embeddings=True)

    # Stage 2 — Clustering
    threshold_results = {}
    for thresh in THRESHOLDS:
        labels_t, sizes_t = run_clustering(embeddings, thresh)
        sz_vals = list(sizes_t.values())
        threshold_results[thresh] = {
            "labels": labels_t, "cluster_sizes": sizes_t,
            "n_clusters": len(sizes_t), "max_size": max(sz_vals),
            "n_singletons": sum(1 for s in sz_vals if s == 1),
        }
        print(f"Threshold {thresh}: clusters={len(sizes_t)}, max={max(sz_vals)}")

    best_thresh = next(
        (t for t in THRESHOLDS
         if threshold_results[t]["max_size"] < 100
         and threshold_results[t]["n_singletons"] < 80
         and 100 <= threshold_results[t]["n_clusters"] <= 200),
        min(THRESHOLDS, key=lambda t: (
            max(0, threshold_results[t]["max_size"] - 100) +
            max(0, threshold_results[t]["n_singletons"] - 80) +
            abs(threshold_results[t]["n_clusters"] - 150)))
    )
    print(f"Selected threshold: {best_thresh}")

    labels        = threshold_results[best_thresh]["labels"]
    cluster_sizes = threshold_results[best_thresh]["cluster_sizes"]

    # Stage 3 — Representative Selection
    pool = []
    for cid in sorted(cluster_sizes.keys()):
        indices      = [i for i, l in enumerate(labels) if l == cid]
        cluster_embs = embeddings[indices]
        mean_sims    = (cluster_embs @ seed_embs.T).mean(axis=1)
        best_idx     = indices[int(np.argmax(mean_sims))]
        rep          = personas_raw[best_idx]
        pool.append({
            "id":                    f"P{len(pool)+1:03d}",
            "name":                  rep["name"],
            "core_value":            rep["core_value"],
            "harm_interpretation":   rep["harm_interpretation"],
            "accountability_stance": rep["accountability_stance"],
            "risk_tolerance":        rep["risk_tolerance"],
            "stakeholder_role":      rep["stakeholder_role"],
            "primary_safety_lens":   rep.get("primary_safety_lens", "unknown"),
            "embedding":             embeddings[best_idx].tolist(),
            "cluster_id":            int(cid),
            "cluster_size":          len(indices),
            "member_names":          [personas_raw[i]["name"] for i in indices],
            "source_situation_ids":  list({personas_raw[i]["situation_id"] for i in indices}),
            "injected":              False,
        })
    print(f"Pool after clustering: {len(pool)} personas")

    # Stage 4 — Lens-Completeness Audit
    pool_ids_used = set()
    injected_count = 0

    for cat in HARM_CATEGORIES:
        shortage = MIN_PER_CATEGORY - sum(1 for p in pool if p["primary_safety_lens"] == cat)
        if shortage <= 0:
            continue
        candidates = [(i, p) for i, p in enumerate(personas_raw)
                      if p.get("primary_safety_lens") == cat and i not in pool_ids_used]
        if not candidates:
            print(f"  WARNING: no candidates for '{cat}'")
            continue
        cand_indices = [i for i, _ in candidates]
        mean_sims    = (embeddings[cand_indices] @ seed_embs.T).mean(axis=1)
        for sim_score, (raw_idx, raw_p) in sorted(zip(mean_sims, candidates), reverse=True)[:shortage]:
            pool.append({
                "id":                    f"P{len(pool)+1:03d}",
                "name":                  raw_p["name"],
                "core_value":            raw_p["core_value"],
                "harm_interpretation":   raw_p["harm_interpretation"],
                "accountability_stance": raw_p["accountability_stance"],
                "risk_tolerance":        raw_p["risk_tolerance"],
                "stakeholder_role":      raw_p["stakeholder_role"],
                "primary_safety_lens":   cat,
                "embedding":             embeddings[raw_idx].tolist(),
                "cluster_id":            -1,
                "cluster_size":          1,
                "member_names":          [raw_p["name"]],
                "source_situation_ids":  [raw_p["situation_id"]],
                "injected":              True,
            })
            pool_ids_used.add(raw_idx)
            injected_count += 1

    final_lens_dist = Counter(p["primary_safety_lens"] for p in pool)
    all_complete    = all(final_lens_dist.get(c, 0) >= MIN_PER_CATEGORY for c in HARM_CATEGORIES)
    print(f"Final pool size: {len(pool)}  |  Injected: {injected_count}  |  Lens-complete: {all_complete}")

    # Save persona_pool.json
    pool_path = os.path.join(args.output_dir, "persona_pool.json")
    with open(pool_path, "w") as f:
        json.dump({
            "pool_size":                len(pool),
            "construction_method":      "PDOS — Harm-Grounded + Lens-Complete Pool",
            "generator_model":          MODEL_ID,
            "embedding_model":          EMBED_MODEL,
            "clustering_threshold":     best_thresh,
            "total_raw_personas":       len(personas_raw),
            "total_situations_used":    len(good),
            "lens_completeness_target": MIN_PER_CATEGORY,
            "injected_personas":        injected_count,
            "lens_complete":            all_complete,
            "generated_at":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "personas":                 pool,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {pool_path}")
    print(f"\nDone. Pool size: {len(pool)}  |  Threshold: {best_thresh}  |  Lens-complete: {all_complete}")


if __name__ == "__main__":
    main()
