# persona_retrieval.py — PDOS Pipeline Stage 2
# Retrieves k diverse personas per situation using Maximal Marginal Relevance (MMR).
# Usage: python persona_retrieval.py --pool_path outputs/persona_pool.json --sample_path data/dataset.json --output_dir outputs/retrieval/

import argparse
import json
import os
import sys
import numpy as np
from collections import Counter
from datetime import datetime

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
LAMBDA      = 0.6
K           = 4


def parse_args():
    p = argparse.ArgumentParser(description="PDOS — Situation-Aware Persona Retrieval")
    p.add_argument("--pool_path",   default="outputs/persona_pool.json")
    p.add_argument("--sample_path", default="data/dataset.json")
    p.add_argument("--output_dir",  default="outputs/retrieval/")
    p.add_argument("--lambda_mmr",  type=float, default=LAMBDA)
    p.add_argument("--k",           type=int,   default=K)
    return p.parse_args()


def mmr_retrieve(situation_emb, pool_embs, pool_personas, k, lam):
    rel_scores       = pool_embs @ situation_emb
    selected_indices = []
    selected_embs    = []
    results          = []
    remaining        = list(range(len(pool_personas)))

    for order in range(1, k + 1):
        best_idx   = None
        best_score = -np.inf
        best_rel   = 0.0
        for i in remaining:
            relevance         = float(rel_scores[i])
            diversity_penalty = float(np.max(pool_embs[i] @ np.array(selected_embs).T)) \
                                if selected_embs else 0.0
            score = lam * relevance - (1 - lam) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx   = i
                best_rel   = relevance
        selected_embs.append(pool_embs[best_idx].tolist())
        remaining.remove(best_idx)
        results.append((best_idx, best_rel, best_score, order))

    return results


def main():
    args = parse_args()

    for path in [args.pool_path, args.sample_path]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.pool_path) as f:
        pool_doc = json.load(f)
    with open(args.sample_path) as f:
        situations = json.load(f)

    for s in situations:
        if "id" not in s:
            s["id"] = s.get("situation_id")
        if "split" not in s:
            s["split"] = "unknown"

    print(f"Pool: {pool_doc['pool_size']} personas  |  Situations: {len(situations)}  |  k={args.k}  |  lambda={args.lambda_mmr}")

    pool_personas   = pool_doc["personas"]
    pool_embeddings = np.array([p["embedding"] for p in pool_personas], dtype=np.float32)
    norms           = np.linalg.norm(pool_embeddings, axis=1, keepdims=True)
    pool_embeddings = pool_embeddings / np.where(norms == 0, 1, norms)

    from sentence_transformers import SentenceTransformer
    model                = SentenceTransformer(EMBED_MODEL)
    situation_embeddings = model.encode([s["situation"] for s in situations],
                                        batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    results = []
    for i, (situation, sit_emb) in enumerate(zip(situations, situation_embeddings)):
        retrieved          = mmr_retrieve(sit_emb, pool_embeddings, pool_personas, k=args.k, lam=args.lambda_mmr)
        retrieved_personas = []
        for pool_idx, rel_score, mmr_score, order in retrieved:
            p = pool_personas[pool_idx]
            retrieved_personas.append({
                "persona_id":            p["id"],
                "name":                  p["name"],
                "core_value":            p["core_value"],
                "harm_interpretation":   p["harm_interpretation"],
                "accountability_stance": p["accountability_stance"],
                "risk_tolerance":        p["risk_tolerance"],
                "stakeholder_role":      p["stakeholder_role"],
                "primary_safety_lens":   p.get("primary_safety_lens", "unknown"),
                "relevance_score":       round(float(rel_score), 6),
                "mmr_score":             round(float(mmr_score), 6),
                "selection_order":       order,
            })
        results.append({
            "situation_id":       situation["id"],
            "situation":          situation["situation"],
            "safety_lenses":      situation["safety_lenses"],
            "lens_count":         len(situation["safety_lenses"]),
            "split":              situation.get("split", "unknown"),
            "retrieved_personas": retrieved_personas,
        })
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1:4d}/{len(situations)}] id={situation['id']}")

    out_path = os.path.join(args.output_dir, "retrieval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"Done. {len(results)} situations processed.")


if __name__ == "__main__":
    main()
