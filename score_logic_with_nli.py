#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score logic self-check pairs with an NLI model and aggregate into LogicScore per your research plan.

Requires:
  pip install transformers torch

Default model:
  roberta-large-mnli  (fast, strong baseline)
  You can also use facebook/bart-large-mnli or deberta-v3-large-mnli if available.

Inputs:
  --pairs data/processed/logic_selfcheck_pairs.jsonl
  --out_dir data/processed
  --model roberta-large-mnli
  --batch_size 16
  --alpha 0.4 --beta 0.3 --gamma 0.3   # weights for Neg/Para/Mod

Outputs:
  {out_dir}/logic_pair_scores.jsonl       # each pair with P(entail), P(contradict), P(neutral)
  {out_dir}/logic_scores_by_id.jsonl      # aggregated NegScore, ParaScore, ModScore, LogicScore per id
"""

import os, sys, json, argparse, math
from collections import defaultdict
import torch

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            # expected keys: id, premise, hypothesis, type
            if not all(k in obj for k in ("id","premise","hypothesis","type")):
                continue
            pairs.append(obj)
    return pairs

def chunked(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf; buf = []
    if buf:
        yield buf

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # pragma: no cover
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/logic_selfcheck_pairs.jsonl")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--model", default="roberta-large-mnli")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4)  # Neg
    ap.add_argument("--beta", type=float, default=0.3)   # Para
    ap.add_argument("--gamma", type=float, default=0.3)  # Mod
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = load_pairs(args.pairs)
    if not pairs:
        print("No pairs found at", args.pairs, file=sys.stderr)
        sys.exit(1)

    device = get_device()

    # lazy import transformers
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print("Please install transformers and torch first: pip install transformers torch", file=sys.stderr)
        raise

    print(f"Loading NLI model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # label mapping depends on model; normalize by sorting logit indices via known label strings if available
    # We'll try to detect entail/neutral/contradiction order from config id2label if present.
    id2label = getattr(model.config, "id2label", None)
    label_map = None
    if id2label:
        inv = {v.lower(): k for k,v in id2label.items()}
        # common keys: entailment, neutral, contradiction
        if all(k in inv for k in ("entailment","neutral","contradiction")):
            label_map = [inv["entailment"], inv["neutral"], inv["contradiction"]]
    # Fallback to RoBERTa-MNLI order: CONTRADICTION, NEUTRAL, ENTAILMENT -> map to [E, N, C]
    if label_map is None:
        label_map = [2,1,0]

    import torch.nn.functional as F

    pair_scores_path = os.path.join(args.out_dir, "logic_pair_scores.jsonl")
    agg_scores_path  = os.path.join(args.out_dir, "logic_scores_by_id.jsonl")

    # score all pairs
    out_f = open(pair_scores_path, "w", encoding="utf-8")

    # storage for aggregation
    by_id = defaultdict(lambda: {
        "neg": [], "para": [], "mod_certain": [], "mod_possible": []
    })

    def write_pair_score(obj, p_ent, p_neu, p_contra):
        rec = {
            "id": obj["id"],
            "type": obj["type"],
            "premise": obj["premise"],
            "hypothesis": obj["hypothesis"],
            "p_entail": float(p_ent),
            "p_neutral": float(p_neu),
            "p_contradict": float(p_contra)
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for batch in chunked(pairs, args.batch_size):
        texts = [(b["premise"], b["hypothesis"]) for b in batch]
        enc = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            # reorder to [E, N, C]
            logits_reordered = torch.stack([logits[:,label_map[0]], logits[:,label_map[1]], logits[:,label_map[2]]], dim=1)
            probs = F.softmax(logits_reordered, dim=-1).cpu().numpy()

        for obj, pr in zip(batch, probs):
            p_ent, p_neu, p_contra = pr.tolist()
            write_pair_score(obj, p_ent, p_neu, p_contra)
            t = obj["type"]
            if t == "negation":
                by_id[obj["id"]]["neg"].append(p_contra)
            elif t == "paraphrase":
                by_id[obj["id"]]["para"].append(p_ent)
            elif t == "modality_certain":
                by_id[obj["id"]]["mod_certain"].append(p_ent)
            elif t == "modality_possible":
                by_id[obj["id"]]["mod_possible"].append(p_ent)

    out_f.close()

    # aggregate per id
    def mean(xs):
        return sum(xs)/len(xs) if xs else 0.0

    with open(agg_scores_path, "w", encoding="utf-8") as fo:
        for sid, comp in by_id.items():
            neg = mean(comp["neg"])
            para = mean(comp["para"])
            mod  = max(0.0, mean(comp["mod_certain"]) - mean(comp["mod_possible"]))
            logic = args.alpha * neg + args.beta * para + args.gamma * mod
            fo.write(json.dumps({
                "id": sid,
                "NegScore": round(neg,6),
                "ParaScore": round(para,6),
                "ModScore": round(mod,6),
                "LogicScore": round(logic,6),
                "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma
            }, ensure_ascii=False) + "\n")

    print("Done.")
    print("Pair scores:", pair_scores_path)
    print("Aggregated scores:", agg_scores_path)

if __name__ == "__main__":
    main()
