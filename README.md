
---

## 0) Environment

```bash
# Python >= 3.9
pip install -U transformers torch tqdm ujson
# For CUDA on NVIDIA T4 (16GB), install a CUDA build of PyTorch; e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Hardware used: **NVIDIA T4 (16GB)**, FP16 enabled.

---

## 1) Repository layout (key files)

```
SemiSup/
├─ dataprocess.py                 # Split FEVER-like train.jsonl into labeled/unlabeled
├─ preprocess_labeled.py          # Clean + stratified split + claim-only + logic self-check pairs
├─ score_logic_with_nli.py        # NLI scoring for logic pairs + aggregate LogicScore
├─ data/
│  ├─ raw/                        # Place train.jsonl here
│  ├─ interim/                    # Cleaned & split (traceable) variants
│  └─ processed/                  # Model-ready outputs (claim_only_*, logic_*.jsonl, stats)
└─ README.md (this file)
```

---

## 2) Data format (input)

Your **train.jsonl** contains one JSON object per line:
```json
{"id": 75397, "verifiable": "VERIFIABLE", "label": "SUPPORTS",
 "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
 "evidence": [[[92206, 104971, "Nikolaj_Coster-Waldau", 7],
              [92206, 104971, "Fox_Broadcasting_Company", 0]]]}
```
`label` ∈ {`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`} (NEI).

Place the file at:
```
SemiSup/data/raw/train.jsonl
```

---

## 3) Step‑by‑step

### 3.1 Split into labeled / unlabeled — `dataprocess.py`
Random (or stratified) split to create **supervised** and **unsupervised** pools.

**Run:**
```bash
cd SemiSup
python dataprocess.py \
  --input data/raw/train.jsonl \
  --labeled_out data/labeled.jsonl \
  --unlabeled_out data/unlabeled.jsonl \
  --labeled_ratio 0.10 \
  --seed 42
```

**Outputs:**
- `data/labeled.jsonl` — cleaned FEVER subset (10% by default)
- `data/unlabeled.jsonl` — remaining 90% (no labels used downstream until pseudo‑labeling)


---

### 3.2 Preprocess labeled — `preprocess_labeled.py`
Cleans text, deduplicates, length‑filters (default: 5–128 tokens), **stratified train/val split (9:1)**, and creates **logic self‑check pairs** (negation / paraphrase / modality).

**Run:**
```bash
python preprocess_labeled.py \
  --input data/labeled.jsonl \
  --val_ratio 0.1 \
  --seed 42 \
  --min_len 5 \
  --max_len 128
```

**Outputs:**
- `data/interim/cleaned.jsonl` — cleaned, deduped labeled set (with `evidence_meta` only)
- `data/interim/split_train.jsonl`, `data/interim/split_val.jsonl` — traceable splits
- `data/processed/claim_only_train.jsonl`, `data/processed/claim_only_val.jsonl` — **teacher** inputs (`id, claim, label`)
- `data/processed/evidence_meta_train.jsonl`, `data/processed/evidence_meta_val.jsonl` — wiki‑free page/sent meta (for future evidence channel)
- `data/processed/logic_selfcheck_pairs.jsonl` — NLI inputs (`id, premise, hypothesis, type`)
- `data/processed/stats.json` — counts, label distribution, lengths (recommended)


---

### 3.3 Logic scoring (NLI) — `score_logic_with_nli.py`
Scores each logic self‑check pair with an MNLI model (e.g., `roberta-large-mnli`) and aggregates per‑id.

**Run:**
```bash
python score_logic_with_nli.py \
  --pairs data/processed/logic_selfcheck_pairs.jsonl \
  --out_dir data/processed \
  --model roberta-large-mnli \
  --fp16 \
  --auto_batch_tune \
  --max_length 512 \
  --alpha 0.4 --beta 0.3 --gamma 0.3
```

**Outputs:**
- `data/processed/logic_pair_scores.jsonl`  
  Per‑pair NLI probabilities:
  ```json
  {"id":75397,"type":"negation",
   "premise":"Nikolaj ...","hypothesis":"Nikolaj did not ...",
   "p_entail":0.01,"p_neutral":0.05,"p_contradict":0.94}
  ```
- `data/processed/logic_scores_by_id.jsonl`  
  Per‑id aggregated scores:
  ```json
  {"id":75397,"NegScore":0.92,"ParaScore":0.78,"ModScore":0.31,"LogicScore":0.67,
   "alpha":0.4,"beta":0.3,"gamma":0.3}
  ```

**Notes:**
- Uses real newlines in JSONL (each object on its own line).  
- Auto‑selects device (CUDA/DirectML/CPU) and prints GPU name/VRAM.  
- `--fp16` significantly reduces VRAM and boosts throughput on T4.

---


