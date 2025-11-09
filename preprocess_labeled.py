#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess labeled FEVER-style JSONL (labeled.jsonl) for supervised training.
- Does NOT require Wikipedia text.
- Produces claim-only train/val, logic self-check pairs, evidence metadata, class weights, and stats.

Usage:
  python preprocess_labeled.py --input data/raw/labeled.jsonl --val_ratio 0.1 --seed 42 --min_len 5 --max_len 128
"""

import os, sys, json, re, random, hashlib
from collections import Counter, defaultdict
from datetime import datetime
import argparse

ALLOWED_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
ALLOWED_VERIFIABLE = {"VERIFIABLE", "NOT VERIFIABLE"}

CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F\u200B\uFEFF]")
MULTISPACE_RE = re.compile(r"\s+")

def norm_text(s: str) -> str:
    s = s.replace("“","\"").replace("”","\"").replace("‘","'").replace("’","'")
    s = s.replace("–","-").replace("—","-")
    s = CONTROL_CHARS_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def is_printable_ratio_ok(s: str, max_nonprint_ratio=0.05) -> bool:
    total = len(s) or 1
    nonprint = sum(1 for ch in s if not ch.isprintable())
    return (nonprint/total) <= max_nonprint_ratio

def hash_claim(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def parse_evidence_meta(evidence):
    metas = []
    try:
        for group in evidence or []:
            for item in group or []:
                if not item or len(item) < 4:
                    continue
                page = item[2]
                sent_idx = item[3]
                if page is None or sent_idx is None:
                    continue
                metas.append({"page": str(page), "sent_idx": int(sent_idx)})
    except Exception:
        pass
    # deduplicate
    seen = set()
    uniq = []
    for m in metas:
        key = (m["page"], m["sent_idx"])
        if key in seen: continue
        seen.add(key)
        uniq.append(m)
    return uniq

PARA_SWAPS = [
    ("worked with", "collaborated with"),
    ("collaborated with", "worked with"),
    ("television", "TV"),
    ("TV", "television"),
    ("series", "show"),
    ("show", "series"),
    ("American", "US"),
    ("US", "American")
]

def gen_negation(claim: str) -> str:
    tokens = claim.split()
    aux_set = {"am","is","are","was","were","do","does","did","has","have","had","can","could","will","would","shall","should","may","might","must"}
    for i,t in enumerate(tokens):
        if t.lower() in aux_set:
            if i+1 < len(tokens) and tokens[i+1].lower() == "not":
                return claim
            tokens.insert(i+1, "not")
            return " ".join(tokens)
    # fallback
    return "did not " + claim[0].lower() + claim[1:] if claim else "did not"

def gen_paraphrases(claim: str, max_k=2):
    outs = []
    s = claim
    for a,b in PARA_SWAPS:
        if a in s and len(outs) < max_k:
            outs.append(s.replace(a,b,1))
    outs = [o for o in dict.fromkeys(outs) if o != claim]
    return outs[:max_k]

def gen_modality_pairs(claim: str):
    return f"It is certain that {claim}", f"It is possible that {claim}"

def stratified_split(items, labels, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for it, lb in zip(items, labels):
        by_label[lb].append(it)
    train, val = [], []
    for lb, arr in by_label.items():
        rng.shuffle(arr)
        k = max(1, int(round(len(arr)*val_ratio))) if len(arr)>5 else 1 if len(arr)>=2 else 0
        val.extend(arr[:k])
        train.extend(arr[k:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val

def compute_class_weights(items):
    cnt = Counter([r["label"] for r in items])
    total = sum(cnt.values())
    weights = {}
    # inverse frequency normalized to mean=1.0
    inv = {k: total/v for k,v in cnt.items()}
    mean_inv = sum(inv.values())/len(inv) if inv else 1.0
    for k,v in inv.items():
        weights[k] = v/mean_inv
    return cnt, weights

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/labeled.jsonl")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    raw_path = args.input
    interim_dir = os.path.join(args.out_root, "interim")
    processed_dir = os.path.join(args.out_root, "processed")
    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # read & clean
    total, dropped = 0, 0
    drop_reasons = Counter()
    items = []
    seen = set()

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            total += 1
            if not all(k in obj for k in ("id","verifiable","label","claim","evidence")):
                drop_reasons["missing_keys"] += 1; dropped += 1; continue
            if obj["label"] not in ALLOWED_LABELS:
                drop_reasons["bad_label"] += 1; dropped += 1; continue
            if obj["verifiable"] not in ALLOWED_VERIFIABLE:
                drop_reasons["bad_verifiable"] += 1; dropped += 1; continue
            claim = obj.get("claim","")
            if not isinstance(claim, str) or not claim.strip():
                drop_reasons["empty_claim"] += 1; dropped += 1; continue
            claim = norm_text(claim)
            if not is_printable_ratio_ok(claim):
                drop_reasons["nonprint_ratio"] += 1; dropped += 1; continue
            toks = claim.split()
            if len(toks) < args.min_len or len(toks) > args.max_len:
                drop_reasons["length_filter"] += 1; dropped += 1; continue

            h = hashlib.sha256(claim.encode("utf-8")).hexdigest()
            if h in seen:
                drop_reasons["duplicate_claim"] += 1; dropped += 1; continue
            seen.add(h)

            ev_meta = parse_evidence_meta(obj.get("evidence"))
            items.append({
                "id": int(obj["id"]),
                "verifiable": obj["verifiable"],
                "label": obj["label"],
                "claim": claim,
                "evidence_meta": ev_meta
            })

    cleaned_path = os.path.join(interim_dir, "cleaned.jsonl")
    with open(cleaned_path, "w", encoding="utf-8") as fo:
        for r in items:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    # split
    labels = [r["label"] for r in items]
    train_items, val_items = stratified_split(items, labels, args.val_ratio, args.seed)

    split_train_path = os.path.join(interim_dir, "split_train.jsonl")
    split_val_path = os.path.join(interim_dir, "split_val.jsonl")
    with open(split_train_path, "w", encoding="utf-8") as fo:
        for r in train_items: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(split_val_path, "w", encoding="utf-8") as fo:
        for r in val_items: fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    # claim-only supervised
    claim_train_path = os.path.join(processed_dir, "claim_only_train.jsonl")
    claim_val_path   = os.path.join(processed_dir, "claim_only_val.jsonl")
    with open(claim_train_path, "w", encoding="utf-8") as fo:
        for r in train_items:
            fo.write(json.dumps({"id": r["id"], "claim": r["claim"], "label": r["label"]}, ensure_ascii=False) + "\n")
    with open(claim_val_path, "w", encoding="utf-8") as fo:
        for r in val_items:
            fo.write(json.dumps({"id": r["id"], "claim": r["claim"], "label": r["label"]}, ensure_ascii=False) + "\n")

    # evidence metadata (supervised reasoning hook)
    ev_meta_train_path = os.path.join(processed_dir, "evidence_meta_train.jsonl")
    ev_meta_val_path   = os.path.join(processed_dir, "evidence_meta_val.jsonl")
    with open(ev_meta_train_path, "w", encoding="utf-8") as fo:
        for r in train_items:
            fo.write(json.dumps({"id": r["id"], "claim": r["claim"], "label": r["label"], "evidence_meta": r["evidence_meta"]}, ensure_ascii=False) + "\n")
    with open(ev_meta_val_path, "w", encoding="utf-8") as fo:
        for r in val_items:
            fo.write(json.dumps({"id": r["id"], "claim": r["claim"], "label": r["label"], "evidence_meta": r["evidence_meta"]}, ensure_ascii=False) + "\n")

    # logic self-check pairs (for ALL cleaned items)
    logic_pairs_path = os.path.join(processed_dir, "logic_selfcheck_pairs.jsonl")
    cnt_pairs = 0
    with open(logic_pairs_path, "w", encoding="utf-8") as fo:
        for r in items:
            claim = r["claim"]
            neg = gen_negation(claim)
            if neg and neg != claim:
                fo.write(json.dumps({"id": r["id"], "premise": claim, "type": "negation", "hypothesis": neg}, ensure_ascii=False) + "\n")
                cnt_pairs += 1
            for p in gen_paraphrases(claim, max_k=2):
                fo.write(json.dumps({"id": r["id"], "premise": claim, "type": "paraphrase", "hypothesis": p}, ensure_ascii=False) + "\n")
                cnt_pairs += 1
            certain, possible = gen_modality_pairs(claim)
            fo.write(json.dumps({"id": r["id"], "premise": certain, "type": "modality_certain", "hypothesis": claim}, ensure_ascii=False) + "\n")
            fo.write(json.dumps({"id": r["id"], "premise": possible, "type": "modality_possible", "hypothesis": claim}, ensure_ascii=False) + "\n")
            cnt_pairs += 2

    # class weights suggestion
    cls_counts, cls_weights = compute_class_weights(train_items)

    # stats
    stats = {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "input": os.path.abspath(raw_path),
        "total_read": total,
        "kept_cleaned": len(items),
        "dropped": dropped,
        "drop_reasons": dict(drop_reasons),
        "label_distribution_all": Counter([r["label"] for r in items]),
        "label_distribution_train": Counter([r["label"] for r in train_items]),
        "label_distribution_val": Counter([r["label"] for r in val_items]),
        "train_size": len(train_items),
        "val_size": len(val_items),
        "logic_pairs_generated": cnt_pairs,
        "class_weights_suggestion": cls_weights,
        "class_counts_train": cls_counts,
        "settings": {
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "min_len": args.min_len,
            "max_len": args.max_len
        }
    }
    with open(os.path.join(processed_dir, "stats.json"), "w", encoding="utf-8") as fo:
        json.dump(stats, fo, ensure_ascii=False, indent=2)

    print("DONE.")
    print("Interim:")
    print(" -", cleaned_path)
    print(" -", split_train_path)
    print(" -", split_val_path)
    print("Processed:")
    print(" -", claim_train_path)
    print(" -", claim_val_path)
    print(" -", ev_meta_train_path)
    print(" -", ev_meta_val_path)
    print(" -", logic_pairs_path)
    print(" -", os.path.join(processed_dir, "stats.json"))

if __name__ == "__main__":
    main()
