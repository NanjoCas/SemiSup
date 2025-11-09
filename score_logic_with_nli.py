#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score logic self-check pairs with an NLI model and aggregate into LogicScore (JSONL outputs).
- Proper JSONL writing (one JSON per line, real "\n")
- CUDA / DirectML / CPU auto-selection
- Optional FP16 inference (--fp16)
- Optional auto batch size tuning (--auto_batch_tune)
- Max sequence length override (--max_length)
- Prints device & effective batch size

Install:
  pip install transformers torch
"""

import os, sys, json, argparse, contextlib
from collections import defaultdict

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # expected keys
            if not all(k in obj for k in ("id","premise","hypothesis","type")):
                continue
            pairs.append(obj)
    return pairs

def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    # optional DirectML
    try:
        import torch_directml  # pragma: no cover
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")

def print_device_info(device):
    try:
        import torch
        if device.type == "cuda":
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using device: {device} | GPU: {name} | Total VRAM: {total:.1f} GB")
        else:
            print(f"Using device: {device}")
    except Exception:
        print(f"Using device: {device}")

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/logic_selfcheck_pairs.jsonl")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--model", default="roberta-large-mnli")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4)  # Neg
    ap.add_argument("--beta", type=float, default=0.3)   # Para
    ap.add_argument("--gamma", type=float, default=0.3)  # Mod
    ap.add_argument("--fp16", action="store_true", help="Use FP16 autocast + model.half() when on CUDA")
    ap.add_argument("--auto_batch_tune", action="store_true", help="Probe max safe batch size without OOM")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = load_pairs(args.pairs)
    if not pairs:
        print("No pairs found at", args.pairs, file=sys.stderr)
        sys.exit(1)

    device = get_device()
    print_device_info(device)

    # TF32 (harmless on T4 which doesn't use TF32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    print(f"Loading NLI model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    if args.fp16 and device.type == "cuda":
        try:
            model.half()
            print("FP16 enabled: model.half()")
        except Exception as e:
            print("FP16 not applied:", e)
    model.eval()

    # figure out label mapping (entail, neutral, contradiction)
    id2label = getattr(model.config, "id2label", None)
    label_map = None
    if id2label:
        inv = {v.lower(): k for k,v in id2label.items()}
        if all(k in inv for k in ("entailment","neutral","contradiction")):
            label_map = [inv["entailment"], inv["neutral"], inv["contradiction"]]
    if label_map is None:
        # common for roberta-large-mnli: 0=CONTRADICTION,1=NEUTRAL,2=ENTAILMENT
        label_map = [2,1,0]

    pair_scores_path = os.path.join(args.out_dir, "logic_pair_scores.jsonl")
    agg_scores_path  = os.path.join(args.out_dir, "logic_scores_by_id.jsonl")

    out_f = open(pair_scores_path, "w", encoding="utf-8")
    by_id = defaultdict(lambda: {"neg": [], "para": [], "mod_certain": [], "mod_possible": []})

    # optional auto batch tuning
    effective_bs = args.batch_size
    if args.auto_batch_tune:
        probe = pairs[:max(args.batch_size, 64)]
        trial = max(8, args.batch_size)
        print("Auto batch tuning... (starting from)", trial)
        while True:
            try:
                texts = [(p["premise"], p["hypothesis"]) for p in probe[:trial]]
                enc = tokenizer.batch_encode_plus(
                    texts, padding=True, truncation=True,
                    max_length=args.max_length, return_tensors="pt"
                )
                enc = {k: v.to(device) for k,v in enc.items()}
                with torch.no_grad():
                    ctx = torch.cuda.amp.autocast(dtype=torch.float16) if (args.fp16 and device.type=="cuda") else contextlib.nullcontext()
                    with ctx:
                        _ = model(**enc).logits
                effective_bs = trial
                trial = min(trial * 2, 1024)
                if trial == effective_bs:
                    break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    break
                else:
                    raise
        print(f"Auto batch tuning done. Effective batch size = {effective_bs}")
    else:
        print(f"Effective batch size = {effective_bs} (no auto tuning)")

    # main scoring loop
    def chunked_dynamic(iterable, n):
        buf = []
        for x in iterable:
            buf.append(x)
            if len(buf) >= n:
                yield buf; buf = []
        if buf:
            yield buf

    for batch in chunked_dynamic(pairs, effective_bs):
        texts = [(b["premise"], b["hypothesis"]) for b in batch]
        enc = tokenizer.batch_encode_plus(
            texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        )
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            ctx = torch.cuda.amp.autocast(dtype=torch.float16) if (args.fp16 and device.type=="cuda") else contextlib.nullcontext()
            with ctx:
                logits = model(**enc).logits
            # reorder to [E, N, C]
            logits_reordered = torch.stack(
                [logits[:,label_map[0]], logits[:,label_map[1]], logits[:,label_map[2]]], dim=1
            )
            probs = torch.softmax(logits_reordered, dim=-1).cpu().numpy()

        for obj, pr in zip(batch, probs):
            p_ent, p_neu, p_contra = pr.tolist()
            # write pair-level score (REAL newline)
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

            # accumulate for aggregation
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

    # aggregate per id (REAL newline)
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
