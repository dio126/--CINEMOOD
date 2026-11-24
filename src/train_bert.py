# -*- coding: utf-8 -*-
"""
BERT 微调训练脚本(方案A:新增 BERT 加入对比,绝不替换现有模型)
================================================================
任务书 3.3"可选高级扩展"。模型:hfl/chinese-roberta-wwm-ext(全词遮蔽 BERT,
中文,102M 参数),本地权重 models/bert-base/(HF 镜像下载,git 忽略)。

公平对比设计(与 Bi-LSTM 同数据、同评估口径):
  - 二分类:直接读 data/processed/{train,val,test}.pkl 的 texts/labels
    (与最佳 Bi-LSTM 完全同一份 ChnSentiCorp 去重后 8:1:1 划分);
  - 五级:读 {split}_5c.pkl(与 5c_bilstm_pretune_h256 同一份 DMSC 5 万划分);
  - 评估复用 evaluate.evaluate_model(二分)/ train_5class.evaluate_5class(五级),
    指标口径与 Bi-LSTM 实验完全一致,结果追加进同一张 summary CSV。

训练配置:maxlen 128 / batch 32 / fp16(RTX 4060 8GB 实测足够)/ AdamW
lr 2e-5 / 线性衰减含 6% warmup / 最多 3 epoch / 早停监控验证指标
(二分=acc,五级=macro-F1,与各自任务的 Bi-LSTM 训练脚本同口径)。

ckpt 落盘遵循现有约定:
  二分类 outputs/<run>_best.pt;五级 outputs/<run>_5c_best.pt;
  config 里 model_type='bert' + model_path(本地权重目录)随 ckpt 记录,
  推理(predict/evaluate)按 config 自动重建并加载 tokenizer,Web 自动发现。

用法:
  python train_bert.py --task binary  --run_name bert_roberta_wwm
  python train_bert.py --task 5class  --run_name bert_roberta_wwm_5c
"""

import argparse
import csv
import json
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

SEED = 42
NUM_CLASSES = {"binary": 2, "5class": 5}
SUMMARY_CSV = {
    "binary": (os.path.join(OUT_DIR, "experiments_summary.csv"), [
        "run_name", "model", "embedding", "lr", "hidden", "dropout", "batch_size",
        "best_val_acc", "test_acc", "test_precision", "test_recall", "test_f1", "收敛epoch"]),
    "5class": (os.path.join(OUT_DIR, "experiments_summary_5class.csv"), [
        "run_name", "model", "embedding", "lr", "hidden", "dropout", "batch_size",
        "best_val_macro_f1", "test_acc", "test_macro_f1", "test_weighted_f1", "收敛epoch"]),
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split: str, task: str):
    """从既有 pkl 读 texts/labels(BERT 与 Bi-LSTM 用同一份划分,公平对比)。"""
    import pickle

    suffix = "_5c" if task == "5class" else ""
    path = os.path.join(PROCESSED_DIR, f"{split}{suffix}.pkl")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return [str(t) for t in d["texts"]], np.asarray(d["labels"]).astype(np.int64)


def encode_texts(tokenizer, texts: list, maxlen: int) -> np.ndarray:
    """文本批量编码为 BERT input_ids([PAD]=0 定长 maxlen,评估侧 mask 自动推导)。"""
    enc = tokenizer(list(texts), truncation=True, max_length=maxlen,
                    padding="max_length", return_attention_mask=False)
    return np.asarray(enc["input_ids"], dtype=np.int64)


@torch.no_grad()
def validate(model, loader, criterion, device, task: str):
    """返回 (平均 loss, 主指标):二分=acc,五级=macro-F1(与 train/train_5class 同口径)。"""
    from sklearn.metrics import f1_score

    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for ids, y in loader:
        ids, y = ids.to(device), y.to(device)
        logits = model(ids)
        total_loss += criterion(logits, y).item() * y.size(0)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    metric = (preds == labels).mean() if task == "binary" else \
        f1_score(labels, preds, average="macro", zero_division=0)
    return total_loss / len(labels), float(metric)


def main():
    parser = argparse.ArgumentParser(description="BERT(Chinese-RoBERTa-wwm-ext)情感分类微调")
    parser.add_argument("--task", choices=["binary", "5class"], required=True)
    parser.add_argument("--run_name", type=str, required=True,
                        help="二分类如 bert_roberta_wwm;五级如 bert_roberta_wwm_5c")
    parser.add_argument("--model_path", type=str, default="models/bert-base",
                        help="BERT 本地权重目录(相对项目根)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--maxlen", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--no_fp16", action="store_true", help="禁用混合精度(OOM 排查用)")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    print(f"[bert] 设备 {device},fp16={use_fp16},种子 {SEED}")

    # ---------- tokenizer + 数据(与 Bi-LSTM 同一份 pkl 划分)----------
    from models.bert import BertClassifier, resolve_bert_dir
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    bert_dir = resolve_bert_dir(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(bert_dir, local_files_only=True)
    num_classes = NUM_CLASSES[args.task]

    loaders = {}
    for split in ("train", "val", "test"):
        texts, labels = load_split(split, args.task)
        ids = encode_texts(tokenizer, texts, args.maxlen)
        ds = TensorDataset(torch.from_numpy(ids), torch.from_numpy(labels))
        bs = args.batch_size if split == "train" else args.batch_size * 2
        loaders[split] = DataLoader(ds, batch_size=bs, shuffle=(split == "train"),
                                    num_workers=0, pin_memory=False)
        print(f"[bert] {split}: {len(labels)} 条 → {len(loaders[split])} 批")

    # ---------- 模型 / 损失 / 优化器 ----------
    model = BertClassifier(num_classes=num_classes, model_path=args.model_path).to(device)
    criterion = nn.CrossEntropyLoss()  # 两任务数据均近似均衡(五级采样即均衡),不加权
    total_steps = len(loaders["train"]) * args.max_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ---------- 训练循环 + 早停 ----------
    key = "acc" if args.task == "binary" else "macro_f1"
    history = {"train_loss": [], f"train_{key}": [], "val_loss": [], f"val_{key}": []}
    best_val, best_epoch, best_state = -1.0, -1, None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        from sklearn.metrics import f1_score
        total_loss, preds_all, labels_all = 0.0, [], []
        for ids, y in loaders["train"]:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_fp16):
                logits = model(ids)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * y.size(0)
            preds_all.append(logits.argmax(dim=-1).detach().cpu().numpy())
            labels_all.append(y.cpu().numpy())
        preds_all, labels_all = np.concatenate(preds_all), np.concatenate(labels_all)
        train_loss = total_loss / len(labels_all)
        train_metric = (preds_all == labels_all).mean() if args.task == "binary" else \
            f1_score(labels_all, preds_all, average="macro", zero_division=0)

        val_loss, val_metric = validate(model, loaders["val"], criterion, device, args.task)
        history["train_loss"].append(float(train_loss))
        history[f"train_{key}"].append(float(train_metric))
        history["val_loss"].append(val_loss)
        history[f"val_{key}"].append(val_metric)
        assert np.isfinite(train_loss) and np.isfinite(val_loss), "loss 出现 NaN/Inf!"
        print(f"[bert] epoch {epoch:02d}/{args.max_epochs} loss={train_loss:.4f} "
              f"train_{key}={train_metric:.4f} val_loss={val_loss:.4f} "
              f"val_{key}={val_metric:.4f} ({time.time() - t0:.0f}s)")

        if val_metric > best_val:
            best_val, best_epoch = val_metric, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[bert] 早停触发:验证 {key} 已 {args.patience} 轮未提升")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history[f"best_val_{key}"] = best_val

    # ---------- 保存产物(命名遵循现有二分类/五级约定,Web 自动发现)----------
    suffix = "" if args.task == "binary" else "_5c"
    ckpt = {
        "model_state": best_state,
        "config": {
            "model_type": "bert",  # 推理/评估据此走 BERT 分支(向后兼容:旧 ckpt 无此键)
            "model_path": args.model_path,   # 本地权重目录(相对项目根,随 ckpt 记录)
            "tokenizer_dir": args.model_path,  # tokenizer 目录(与权重同目录)
            "num_classes": num_classes, "maxlen": args.maxlen,
            "lr": args.lr, "batch_size": args.batch_size, "fp16": use_fp16,
            "weight_decay": 0.01, "task": args.task, "seed": SEED,
        },
        f"best_val_{key}": best_val, "best_epoch": best_epoch,
    }
    torch.save(ckpt, os.path.join(OUT_DIR, f"{args.run_name}{suffix}_best.pt"))
    with open(os.path.join(OUT_DIR, f"{args.run_name}{suffix}_history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 曲线图(复用各自任务的绘图口径:二分画 acc,五级画 macro-F1)
    if args.task == "binary":
        from train import plot_curves
        plot_curves(history, os.path.join(OUT_DIR, f"{args.run_name}_curves.png"), args.run_name)
    else:
        from train_5class import plot_curves
        plot_curves(history, os.path.join(OUT_DIR, f"{args.run_name}_5c_curves.png"), args.run_name)

    # ---------- 测试集评估(复用 Bi-LSTM 的评估函数,同一口径)----------
    if args.task == "binary":
        from evaluate import evaluate_model, plot_confusion
        metrics, _p, _l = evaluate_model(model, loaders["test"], device)
    else:
        from train_5class import evaluate_5class, plot_confusion_5class
        metrics = evaluate_5class(model, loaders["test"], device)
        plot_confusion_5class(metrics["cm"], os.path.join(OUT_DIR, f"{args.run_name}_5c_confusion.png"),
                              args.run_name)
    with open(os.path.join(OUT_DIR, f"{args.run_name}{suffix}_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.task == "binary":
        from evaluate import plot_confusion
        plot_confusion(metrics["cm"], os.path.join(OUT_DIR, f"{args.run_name}_confusion.png"), args.run_name)

    # ---------- 追加 summary CSV(与 Bi-LSTM 实验同表,便于统一对比)----------
    csv_path, columns = SUMMARY_CSV[args.task]
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        if args.task == "binary":
            writer.writerow({
                "run_name": args.run_name, "model": "bert",
                "embedding": "chinese-roberta-wwm-ext",
                "lr": args.lr, "hidden": 768, "dropout": 0.1,
                "batch_size": args.batch_size, "best_val_acc": f"{best_val:.4f}",
                "test_acc": f"{metrics['acc']:.4f}", "test_precision": f"{metrics['precision']:.4f}",
                "test_recall": f"{metrics['recall']:.4f}", "test_f1": f"{metrics['f1']:.4f}",
                "收敛epoch": best_epoch})
        else:
            writer.writerow({
                "run_name": args.run_name, "model": "bert",
                "embedding": "chinese-roberta-wwm-ext",
                "lr": args.lr, "hidden": 768, "dropout": 0.1,
                "batch_size": args.batch_size, "best_val_macro_f1": f"{best_val:.4f}",
                "test_acc": f"{metrics['acc']:.4f}", "test_macro_f1": f"{metrics['macro_f1']:.4f}",
                "test_weighted_f1": f"{metrics['weighted_f1']:.4f}", "收敛epoch": best_epoch})
    print(f"[bert] 完成 {args.run_name}: best_val_{key}={best_val:.4f} → {csv_path}")


if __name__ == "__main__":
    main()
