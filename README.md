
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
python dataprocess.py
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
# Textual Feature Extractor (Teacher Model)
Semi-Supervised Fake News Detection — Feature Extraction Module

This directory contains the implementation of the **Textual Feature Extractor**, which serves as the *Teacher Model* in the semi-supervised misinformation detection framework. It performs supervised pretraining on labeled FEVER claims and generates pseudo-labels and linguistic/logic-aware features for downstream modules such as the RL Selector and Fake News Detector.

## Overview

The extractor is based on **DeBERTa‑v3‑base** and includes:
- **Supervised pretraining** on labeled claims (3-way FEVER labels: SUPPORTS / REFUTES / NOT ENOUGH INFO)
- **Pseudo-label generation** for unlabeled claim texts
- **Semantic confidence features** (p_max, entropy)
- **LogicScore integration** from an external NLI/logic module
- **Linguistic/discourse features** (modality, negation, causal cues, sentiment-like indicators)
- **Fused score S = 0.6 × semantic + 0.3 × logic + 0.1 × linguistic**

This module outputs feature-enriched pseudo-labels used to construct the weakly labeled dataset for semi-supervised training.

---

## Directory Structure

```
textual_feature_extractor.py   # Main script (training, pseudo-labeling, plotting)
data/processed/
    claim_only_train.jsonl     # Labeled training claims
    claim_only_val.jsonl       # Labeled validation claims
    logic_scores_by_id.jsonl   # Precomputed logic features
data/unlabeled.jsonl           # Unlabeled claim dataset
models/extractor/              # Saved teacher model (after pretraining)
```

---

## Key Components

### 1. Supervised Pretraining
The model fine-tunes DeBERTa-v3-base using:
- Cross-entropy loss
- Layer-wise learning rate decay (LLRD)
- Weight decay regularization
- Optional gradient checkpointing (for memory saving)
- Early stopping (based on val macro-F1)
- Automatic batch-size tuning (optional)

Example command:

```bash
python textual_feature_extractor.py
--mode pretrain
--train data/processed/claim_only_train.jsonl
--val   data/processed/claim_only_val.jsonl
--model_name microsoft/deberta-v3-base
--extractor_dir models/extractor
--epochs 5
--auto_batch_tune
--gradient_checkpointing
--patience 2
```

---

### 2. Pseudo-Label Generation
For unlabeled data, the extractor outputs:
- predicted label
- probability distribution for 3 classes
- entropy-based uncertainty
- LogicScore features (NegScore, ParaScore, ModScore, LogicScore)
- Linguistic feature score (LingScore)
- Fused confidence score

Example:

```bash
python textual_feature_extractor.py
--mode generate_pseudo
--extractor_dir models/extractor
--unlabeled data/unlabeled.jsonl
--logic_scores data/processed/logic_scores_by_id.jsonl
--pseudo_out data/processed/pseudo_with_features.jsonl
--auto_batch_tune
```

Output example:

```json
{
  "id": "123",
  "claim": "The flu vaccine can cause the flu.",
  "pseudo_label": "REFUTES",
  "p_supports": 0.02,
  "p_refutes": 0.91,
  "p_nei": 0.07,
  "p_max": 0.91,
  "entropy": 0.21,
  "LogicScore": -0.41,
  "LingScore": 0.32,
  "fused_score": 0.67
}
```

---

## Training Visualization
You can generate loss and metric curves from saved history:

```bash
python textual_feature_extractor.py
--mode plot_history
--extractor_dir models/extractor
--plot_out extractor_training.png
```

This produces:
- `extractor_training_loss.png`
- `extractor_training_metrics.png`

---

## Output Files

| File | Description |
|------|-------------|
| `models/extractor/` | Trained DeBERTa teacher model |
| `pseudo_with_features.jsonl` | Pseudo-labels + semantic/logic/linguistic features |
| `unlabeled_embeddings.pt` | Optional CLS embeddings |
| `training_history.jsonl` | Per-epoch training metrics |

---


## Notes
- This module is **strictly aligned** with your research plan design.
- Loss function and feature computation follow your architecture exactly.
- The extractor is shared by the Fake News Detector and RL Selector through its pretrained encoder.



