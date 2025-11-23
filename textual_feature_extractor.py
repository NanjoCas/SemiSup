#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Textual Feature Extractor for SemiSup Fake News Detection
- 基于 DeBERTa-v3-base 的文本编码器（与 Fake News Detector 共享）
- 在 FEVER 有标签 claim 数据上做 3 类分类预训练 (SUPPORTS / REFUTES / NOT ENOUGH INFO)
- 对 unlabeled claims 进行伪标签预测 + 置信度计算
- 结合逻辑一致性特征 LogicScore（由 NLI/逻辑打分模块生成）
- 从句子中抽取简单的语言/话语特征（modality / negation / causal / sentiment-like）
- 按加权公式融合得到最终 pseudo-label score，用于 RL Selector 和 Fake News Detector

输入文件约定：
- data/processed/claim_only_train.jsonl
- data/processed/claim_only_val.jsonl
- data/unlabeled.jsonl
- data/processed/logic_scores_by_id.jsonl

输出文件：
- models/extractor/  (预训练好的 DeBERTa-v3-base 分类模型，可与 Fake News Detector 共享)
- data/processed/pseudo_with_features.jsonl  (伪标签 + 语义/逻辑/语言特征)
- 可选：data/processed/unlabeled_embeddings.pt  (CLS 向量，供下游使用)
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch import amp  # 新：统一使用 torch.amp 接口
import matplotlib.pyplot as plt  # 新：用于可视化训练曲线

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# --------------------
# 标签定义
# --------------------

LABEL_LIST = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABEL_LIST)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABEL_LIST)}


# --------------------
# 通用工具函数
# --------------------

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception as e:
                print(f"[WARN] Skip bad line in {path}: {e}")
    return data


def save_jsonl(path: str, records: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    熵：衡量预测分布的不确定性，用于伪标签置信度参考。
    """
    p = probs.clamp(min=eps, max=1.0)
    return -(p * p.log()).sum(dim=-1)


# --------------------
# 数据集 & collator
# --------------------

@dataclass
class ClaimExample:
    id: str
    text: str
    label_id: Optional[int]  # labeled: 0..2; unlabeled: None
    raw_label: Optional[str]


class ClaimDataset(Dataset):
    def __init__(self, examples: List[ClaimExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "id": ex.id,
            "text": ex.text,
            "label": -1 if ex.label_id is None else ex.label_id,
        }


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]):
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        ids = [b["id"] for b in batch]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "ids": ids,
        }


# --------------------
# 从 labeled / unlabeled 构造样本
# --------------------

def build_examples_from_labeled(path: str) -> List[ClaimExample]:
    data = load_jsonl(path)
    exs: List[ClaimExample] = []
    for obj in data:
        cid = str(obj.get("id"))
        claim = obj.get("claim")
        label = obj.get("label")
        if claim is None or label is None:
            continue
        label = label.upper()
        if label not in LABEL2ID:
            print(f"[WARN] Unknown label {label} for id={cid}, skip.")
            continue
        exs.append(ClaimExample(
            id=cid,
            text=claim,
            label_id=LABEL2ID[label],
            raw_label=label,
        ))
    print(f"[labeled] Loaded {len(exs)} examples from {path}")
    return exs


def build_examples_from_unlabeled(path: str) -> List[ClaimExample]:
    data = load_jsonl(path)
    exs: List[ClaimExample] = []
    for obj in data:
        cid = str(obj.get("id"))
        claim = obj.get("claim")
        if claim is None:
            continue
        exs.append(ClaimExample(
            id=cid,
            text=claim,
            label_id=None,
            raw_label=None,
        ))
    print(f"[unlabeled] Loaded {len(exs)} examples from {path}")
    return exs


# --------------------
# 简单语言/话语特征 (LingScore)
# --------------------

MODAL_WORDS = [
    "might", "may", "could", "can", "possibly", "perhaps", "likely",
    "unlikely", "seems", "appears", "reportedly",
]
NEGATION_WORDS = [
    "no", "not", "never", "neither", "nor", "n't", "without",
]
CAUSAL_WORDS = [
    "because", "since", "therefore", "thus", "so that", "as a result",
    "due to", "lead to", "result in", "cause", "caused", "causing",
]
POSITIVE_WORDS = ["good", "benefit", "success", "improve", "increase"]
NEGATIVE_WORDS = ["bad", "harm", "risk", "worse", "decrease", "decline"]


def compute_ling_score(text: str) -> float:
    """
    粗糙但实用的 LingScore 计算（0~1），考虑：
    - modality 词（可能/不确定）
    - negation 词（否定）
    - causal 连接词（因果）
    - 简单情感词（正/负）
    """
    t_low = text.lower()

    def count_any(words):
        return sum(1 for w in words if w in t_low)

    n_modal = count_any(MODAL_WORDS)
    n_neg = count_any(NEGATION_WORDS)
    n_causal = count_any(CAUSAL_WORDS)
    n_pos = count_any(POSITIVE_WORDS)
    n_neg_sent = count_any(NEGATIVE_WORDS)

    # 简单归一化：每类最多计 3 个
    modal_score = min(n_modal, 3) / 3.0
    neg_score = min(n_neg, 3) / 3.0
    causal_score = min(n_causal, 3) / 3.0
    sent_score = min(n_pos + n_neg_sent, 3) / 3.0

    ling = 0.25 * modal_score + 0.25 * neg_score + 0.3 * causal_score + 0.2 * sent_score
    return float(max(0.0, min(ling, 1.0)))


# --------------------
# 自动 batch_size 调参
# --------------------

def auto_tune_batch_size(
    model,
    tokenizer,
    examples: List[ClaimExample],
    device,
    max_length: int,
    fp16: bool = True,
    start_bs: int = 16,
    max_bs: int = 256,
    max_probe_samples: int = 256,
) -> int:
    """
    简易自动 batch size 调参：
    - 从 start_bs 开始，每次翻倍
    - 只用前 max_probe_samples 个样本试一轮前向
    - OOM 时退回上一次成功的 batch size
    """
    if not torch.cuda.is_available():
        print("[auto_batch_tune] CUDA 不可用，直接使用 start_bs =", start_bs)
        return start_bs

    print(f"[auto_batch_tune] Start probing from bs={start_bs}, max_bs={max_bs}")
    probe_examples = examples[:max_probe_samples] if len(examples) > max_probe_samples else examples
    bs = start_bs
    last_ok = start_bs

    while bs <= max_bs:
        try:
            ds = ClaimDataset(probe_examples)
            loader = DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                collate_fn=Collator(tokenizer, max_length),
            )
            batch = next(iter(loader))
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                with amp.autocast("cuda", enabled=fp16):
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

            last_ok = bs
            print(f"[auto_batch_tune] bs={bs} OK, try larger...")
            bs *= 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[auto_batch_tune] bs={bs} OOM，退回 last_ok={last_ok}")
                torch.cuda.empty_cache()
                break
            else:
                raise

    print(f"[auto_batch_tune] Effective batch size = {last_ok}")
    return last_ok


# --------------------
# LLRD + Weight Decay 优化器构造
# --------------------

def build_optimizer_with_llrd(
    model,
    base_lr: float,
    weight_decay: float = 0.01,
    lr_layer_decay: float = 0.9,
):
    """
    针对 DeBERTa-v3-base 的 Layer-wise Learning Rate Decay (LLRD) 实现：
    - 越靠近输出的 encoder layer，学习率越高
    - embeddings 学习率最低
    - classifier/pooler 使用 base_lr
    - 全部使用统一的 weight_decay（简化 no_decay 逻辑）
    """
    param_groups = []

    # 确定层数
    encoder_layers = list(model.deberta.encoder.layer)
    num_layers = len(encoder_layers)

    # Embeddings：最小 lr
    embed_lr = base_lr * (lr_layer_decay ** (num_layers + 1))
    param_groups.append({
        "params": model.deberta.embeddings.parameters(),
        "lr": embed_lr,
        "weight_decay": weight_decay,
    })

    # Encoder layers：从底层到顶层逐层递增 lr
    for layer_idx, layer_module in enumerate(encoder_layers):
        # 底层 → lr 最小；顶层 → lr 最大接近 base_lr
        depth = num_layers - 1 - layer_idx
        lr = base_lr * (lr_layer_decay ** depth)
        param_groups.append({
            "params": layer_module.parameters(),
            "lr": lr,
            "weight_decay": weight_decay,
        })

    # Pooler + Classifier：用 base_lr
    head_params = []
    if hasattr(model, "pooler") and model.pooler is not None:
        head_params.extend(list(model.pooler.parameters()))
    if hasattr(model, "classifier") and model.classifier is not None:
        head_params.extend(list(model.classifier.parameters()))
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
        })

    optimizer = AdamW(param_groups, lr=base_lr)  # base_lr 作为默认 fallback
    return optimizer


# --------------------
# 训练 Textual Feature Extractor
# --------------------

def train_extractor(
    train_path: str,
    val_path: str,
    model_name: str,
    output_dir: str,
    max_length: int = 256,
    batch_size: int = 32,
    lr: float = 2e-5,
    num_epochs: int = 3,
    fp16: bool = True,
    seed: int = 42,
    auto_batch_tune_flag: bool = False,
    weight_decay: float = 0.01,
    lr_layer_decay: float = 0.9,
    gradient_checkpointing: bool = False,
    patience: int = 2,
):
    """
    在 claim-only 有标签数据上预训练 DeBERTa-v3-base 分类器（3 类）。
    加入：
    - LLRD + Weight Decay 优化器
    - 可选 Gradient Checkpointing
    - Early stopping + 保存 val_macroF1 最佳模型
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Extractor] Using device: {device}")

    train_exs = build_examples_from_labeled(train_path)
    val_exs = build_examples_from_labeled(val_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if gradient_checkpointing:
        print("[Extractor] Enable gradient checkpointing")
        model.gradient_checkpointing_enable()

    model.to(device)

    # 自动调参 batch_size（只作为上限的一半，防止训练时 OOM）
    if auto_batch_tune_flag:
        tuned_bs = auto_tune_batch_size(
            model=model,
            tokenizer=tokenizer,
            examples=train_exs,
            device=device,
            max_length=max_length,
            fp16=fp16,
            start_bs=batch_size,
            max_bs=256,
            max_probe_samples=256,
        )
        effective_bs = max(8, tuned_bs // 2)
    else:
        effective_bs = batch_size

    print(f"[Extractor] Train batch size = {effective_bs}")

    train_ds = ClaimDataset(train_exs)
    val_ds = ClaimDataset(val_exs)

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_bs,
        shuffle=True,
        collate_fn=Collator(tokenizer, max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(effective_bs, 128),
        shuffle=False,
        collate_fn=Collator(tokenizer, max_length),
    )

    optimizer = build_optimizer_with_llrd(
        model=model,
        base_lr=lr,
        weight_decay=weight_decay,
        lr_layer_decay=lr_layer_decay,
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    scaler = amp.GradScaler("cuda", enabled=fp16)

    # early stopping & best model
    best_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    # 训练记录：用于可视化
    os.makedirs(output_dir, exist_ok=True)
    history = []
    history_path = os.path.join(output_dir, "training_history.jsonl")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train-extractor]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with amp.autocast("cuda", enabled=fp16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n or 1))

        acc, macro_f1 = evaluate(model, val_loader, device, fp16=fp16)
        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] train_loss={avg_train_loss:.4f} "
              f"val_acc={acc:.4f} val_macroF1={macro_f1:.4f}")

        # 记录到 history 并写入 jsonl，方便后续画图
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_acc": float(acc),
            "val_macro_f1": float(macro_f1),
        })
        save_jsonl(history_path, history)
        print(f"[Epoch {epoch+1}] history saved to {history_path}")

        # early stopping & best model save
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[Epoch {epoch+1}] New best model saved to {output_dir}")
        else:
            epochs_no_improve += 1
            print(f"[Epoch {epoch+1}] No improvement for {epochs_no_improve} epoch(s)")

        if patience is not None and epochs_no_improve >= patience:
            print(f"[EarlyStopping] Stop training at epoch {epoch+1}, best_epoch={best_epoch}, best_macroF1={best_f1:.4f}")
            break

    print(f"[Extractor] Training finished. Best epoch = {best_epoch}, best_macroF1 = {best_f1:.4f}")


def evaluate(model, val_loader, device, fp16=True) -> Tuple[float, float]:
    model.eval()
    all_labels = []
    all_preds = []
    import numpy as np

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Eval-extractor"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with amp.autocast("cuda", enabled=fp16):
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            probs = softmax_logits(logits)
            preds = probs.argmax(dim=-1)

            mask = labels != -1
            all_labels.extend(labels[mask].cpu().tolist())
            all_preds.extend(preds[mask].cpu().tolist())

    if not all_labels:
        return 0.0, 0.0

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    acc = (all_labels == all_preds).mean()
    n_classes = len(LABEL_LIST)
    f1s = []
    for c in range(n_classes):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    macro_f1 = float(np.mean(f1s))
    return float(acc), macro_f1


# --------------------
# 训练过程可视化
# --------------------

def plot_training_history(history_path: str, out_png: Optional[str] = None):
    """
    根据 training_history.jsonl 可视化训练过程：
    - 图1：train_loss
    - 图2：val_acc & val_macro_f1
    """
    if not os.path.exists(history_path):
        print(f"[Plot] history file not found: {history_path}")
        return

    history = load_jsonl(history_path)
    if not history:
        print(f"[Plot] empty history: {history_path}")
        return

    epochs = [h.get("epoch") for h in history]
    train_loss = [h.get("train_loss") for h in history]
    val_acc = [h.get("val_acc") for h in history]
    val_macro_f1 = [h.get("val_macro_f1") for h in history]

    # 图1：train loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Extractor Training Loss")
    plt.grid(True)
    if out_png:
        base, ext = os.path.splitext(out_png)
        loss_path = base + "_loss" + (ext or ".png")
        plt.savefig(loss_path, bbox_inches="tight")
        print(f"[Plot] Saved loss curve to {loss_path}")
    else:
        plt.show()

    # 图2：val acc & macro F1
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_acc, marker="o", label="Val Acc")
    plt.plot(epochs, val_macro_f1, marker="s", label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Extractor Validation Metrics")
    plt.legend()
    plt.grid(True)
    if out_png:
        base, ext = os.path.splitext(out_png)
        metric_path = base + "_metrics" + (ext or ".png")
        plt.savefig(metric_path, bbox_inches="tight")
        print(f"[Plot] Saved metrics curve to {metric_path}")
    else:
        plt.show()


# --------------------
# 伪标签 + 特征生成
# --------------------

def load_logic_scores(path: str) -> Dict[str, Dict]:
    """
    读取逻辑打分模块输出的 logic_scores_by_id.jsonl
    每行形如：
    {"id": "123", "NegScore":..., "ParaScore":..., "ModScore":..., "LogicScore":...}
    """
    data = load_jsonl(path)
    by_id = {}
    for obj in data:
        cid = str(obj.get("id"))
        by_id[cid] = obj
    print(f"[logic] Loaded {len(by_id)} logic scores from {path}")
    return by_id


def generate_pseudo_with_features(
    extractor_dir: str,
    unlabeled_path: str,
    logic_scores_path: str,
    out_path: str,
    max_length: int = 256,
    batch_size: int = 32,
    fp16: bool = True,
    w_sem: float = 0.6,
    w_logic: float = 0.3,
    w_ling: float = 0.1,
    save_embeddings_path: Optional[str] = None,
    auto_batch_tune_flag: bool = False,
):
    """
    使用预训练好的 Textual Feature Extractor 为 unlabeled 数据生成：
    - pseudo_label, p_max, entropy
    - LogicScore (来自 logic_scores_by_id.jsonl)
    - LingScore (简单语言/话语特征)
    - 融合后的整体分数 fused_score = w_sem * p_max + w_logic * |LogicScore| + w_ling * LingScore
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PseudoGen] Using device: {device}")
    print(f"[PseudoGen] Weights: w_sem={w_sem}, w_logic={w_logic}, w_ling={w_ling}")

    logic_by_id = load_logic_scores(logic_scores_path)
    unlabeled_exs = build_examples_from_unlabeled(unlabeled_path)

    tokenizer = AutoTokenizer.from_pretrained(extractor_dir)
    model = AutoModelForSequenceClassification.from_pretrained(extractor_dir).to(device)
    model.eval()

    # 推理阶段可以相对 aggressive 一点
    if auto_batch_tune_flag:
        tuned_bs = auto_tune_batch_size(
            model=model,
            tokenizer=tokenizer,
            examples=unlabeled_exs,
            device=device,
            max_length=max_length,
            fp16=fp16,
            start_bs=batch_size,
            max_bs=1024,
            max_probe_samples=256,
        )
        effective_bs = tuned_bs
    else:
        effective_bs = batch_size

    print(f"[PseudoGen] Inference batch size = {effective_bs}")

    ds = ClaimDataset(unlabeled_exs)
    loader = DataLoader(
        ds,
        batch_size=effective_bs,
        shuffle=False,
        collate_fn=Collator(tokenizer, max_length),
    )

    records = []
    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generate pseudo-labels"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["ids"]

            with amp.autocast("cuda", enabled=fp16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                logits = out.logits
                hidden_states = out.hidden_states
                last_hidden = hidden_states[-1]    # (B, seq_len, hidden_size)
                cls_emb = last_hidden[:, 0, :]     # (B, hidden_size)

            probs = softmax_logits(logits)
            p_max, pred_idx = probs.max(dim=-1)
            ent = entropy_from_probs(probs)

            for i, cid in enumerate(ids):
                cid_str = str(cid)
                pmax = float(p_max[i].item())
                ent_i = float(ent[i].item())
                pred = ID2LABEL[int(pred_idx[i].item())]

                logic = logic_by_id.get(cid_str, {})
                logic_score = float(logic.get("LogicScore", 0.0))
                neg_score = float(logic.get("NegScore", 0.0))
                para_score = float(logic.get("ParaScore", 0.0))
                mod_score = float(logic.get("ModScore", 0.0))

                text = next((ex.text for ex in unlabeled_exs if ex.id == cid_str), "")
                ling_score = compute_ling_score(text) if text else 0.0

                fused = w_sem * pmax + w_logic * abs(logic_score) + w_ling * ling_score

                rec = {
                    "id": cid_str,
                    "claim": text,
                    "pseudo_label": pred,
                    "p_supports": float(probs[i, LABEL2ID["SUPPORTS"]].item()),
                    "p_refutes": float(probs[i, LABEL2ID["REFUTES"]].item()),
                    "p_nei": float(probs[i, LABEL2ID["NOT ENOUGH INFO"]].item()),
                    "p_max": pmax,
                    "entropy": ent_i,
                    "LogicScore": logic_score,
                    "NegScore": neg_score,
                    "ParaScore": para_score,
                    "ModScore": mod_score,
                    "LingScore": ling_score,
                    "fused_score": float(fused),
                }
                records.append(rec)

            if save_embeddings_path is not None:
                all_embeddings.append(cls_emb.cpu())
                all_ids.extend(ids)

    save_jsonl(out_path, records)
    print(f"[PseudoGen] Saved {len(records)} pseudo-labeled samples with features to {out_path}")

    if save_embeddings_path is not None and all_embeddings:
        embs = torch.cat(all_embeddings, dim=0)
        os.makedirs(os.path.dirname(save_embeddings_path), exist_ok=True)
        torch.save({"ids": all_ids, "embeddings": embs}, save_embeddings_path)
        print(f"[PseudoGen] Saved embeddings to {save_embeddings_path}")


# --------------------
# CLI
# --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "generate_pseudo", "plot_history"], required=True)

    # 共享参数
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--extractor_dir", default="models/extractor")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto_batch_tune", action="store_true")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--lr_layer_decay", type=float, default=0.9)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--patience", type=int, default=2)

    ap.add_argument("--history_path", default=None,
                    help="Path to training_history.jsonl (default: <extractor_dir>/training_history.jsonl)")
    ap.add_argument("--plot_out", default=None,
                    help="If set, save curves to this PNG path (suffix _loss/_metrics will be added)")

    # 数据路径
    ap.add_argument("--train", default="./data/processed/claim_only_train.jsonl")
    ap.add_argument("--val", default="./data/processed/claim_only_val.jsonl")
    ap.add_argument("--unlabeled", default="./data/unlabeled.jsonl")
    ap.add_argument("--logic_scores", default="./data/processed/logic_scores_by_id.jsonl")

    # 伪标签输出
    ap.add_argument("--pseudo_out", default="./data/processed/pseudo_with_features.jsonl")
    ap.add_argument("--emb_out", default=None)

    # 融合权重
    ap.add_argument("--w_sem", type=float, default=0.6)
    ap.add_argument("--w_logic", type=float, default=0.3)
    ap.add_argument("--w_ling", type=float, default=0.1)

    args = ap.parse_args()

    if args.mode == "pretrain":
        train_extractor(
            train_path=args.train,
            val_path=args.val,
            model_name=args.model_name,
            output_dir=args.extractor_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            fp16=not args.no_fp16,
            seed=args.seed,
            auto_batch_tune_flag=args.auto_batch_tune,
            weight_decay=args.weight_decay,
            lr_layer_decay=args.lr_layer_decay,
            gradient_checkpointing=args.gradient_checkpointing,
            patience=args.patience,
        )
    elif args.mode == "generate_pseudo":
        generate_pseudo_with_features(
            extractor_dir=args.extractor_dir,
            unlabeled_path=args.unlabeled,
            logic_scores_path=args.logic_scores,
            out_path=args.pseudo_out,
            max_length=args.max_length,
            batch_size=args.batch_size,
            fp16=not args.no_fp16,
            w_sem=args.w_sem,
            w_logic=args.w_logic,
            w_ling=args.w_ling,
            save_embeddings_path=args.emb_out,
            auto_batch_tune_flag=args.auto_batch_tune,
        )
    elif args.mode == "plot_history":
        history_path = args.history_path
        if history_path is None:
            history_path = os.path.join(args.extractor_dir, "training_history.jsonl")
        plot_training_history(history_path, out_png=args.plot_out)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()