# -*- coding: utf-8 -*-
"""
训练脚本
========
用法示例:
  python train.py --model bilstm --embedding random --lr 1e-3 --dropout 0.5 \
                  --hidden 128 --batch_size 64 --run_name bilstm_lr1e-3_do0.5

功能:
  - 全局固定随机种子(seed=42:random / numpy / torch),保证实验可复现;
  - CrossEntropyLoss + Adam,每 epoch 记录 train/val 的 loss+acc;
  - 早停(patience,监控验证准确率)并恢复最佳权重;
  - 训练结束自动保存:
      outputs/<run_name>_curves.png    训练 vs 验证损失/准确率曲线(同图对比)
      outputs/<run_name>_history.json  逐 epoch 训练历史
      outputs/<run_name>_best.pt       最佳模型(含模型结构与超参配置)
  - 自动在测试集评估,写 <run_name>_metrics.json、<run_name>_confusion.png,
    并把结果追加到 outputs/experiments_summary.csv 总对比表。
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

# 中文字体设置,防乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
SUMMARY_CSV = os.path.join(OUT_DIR, "experiments_summary.csv")
SUMMARY_COLUMNS = [
    "run_name", "model", "embedding", "lr", "hidden", "dropout", "batch_size",
    "best_val_acc", "test_acc", "test_precision", "test_recall", "test_f1", "收敛epoch",
]


def set_seed(seed: int = SEED):
    """固定全部随机源,保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, vocab_size: int, device):
    """按命令行配置构建模型,并处理预训练词向量加载/冻结。"""
    from models.attn_bilstm import AttnBiLSTMClassifier
    from models.bilstm import BiLSTMClassifier

    pretrained = None
    freeze = False
    if args.embedding in ("pretrained", "pretrained_finetune"):
        # 阶段4:加载词向量矩阵(默认二分类矩阵;--embedding_matrix 可指向
        # 其他任务专属矩阵,如混域的 embedding_matrix_mixed.npy)
        matrix_path = args.embedding_matrix or os.path.join(PROCESSED_DIR, "embedding_matrix.npy")
        if not os.path.exists(matrix_path):
            raise FileNotFoundError(
                f"未找到预训练词向量矩阵 {matrix_path},请先运行 "
                "src/download_wordvec.py 生成后再使用 --embedding pretrained"
            )
        pretrained = torch.tensor(np.load(matrix_path), dtype=torch.float32)
        freeze = args.embedding == "pretrained"  # pretrained=冻结,finetune=微调
        # 预训练矩阵维度优先(如腾讯 200 维),自动覆盖命令行默认的 300
        args.emb_dim = pretrained.shape[1]
        print(f"[train] 已加载预训练词向量矩阵 {tuple(pretrained.shape)},freeze={freeze}")

    common = dict(
        vocab_size=vocab_size,
        embedding_dim=args.emb_dim,
        hidden_dim=args.hidden,
        dropout=args.dropout,
        pretrained_embedding=pretrained,
        freeze_embedding=freeze,
    )
    if args.model == "attn_bilstm":
        model = AttnBiLSTMClassifier(**common)
    else:
        model = BiLSTMClassifier(**common)
    return model.to(device)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证/测试:返回 (平均 loss, 准确率)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for ids, y in loader:
        ids, y = ids.to(device), y.to(device)
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def plot_curves(history: dict, out_png: str, run_name: str):
    """训练 vs 验证 损失/准确率曲线(同一张图的上下两个子图)。"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], "o-", label="训练损失")
    axes[0].plot(epochs, history["val_loss"], "s-", label="验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"损失曲线({run_name})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="训练准确率")
    axes[1].plot(epochs, history["val_acc"], "s-", label="验证准确率")
    best_ep = history["best_epoch"]
    axes[1].axvline(best_ep, color="red", linestyle="--", alpha=0.6, label=f"最佳 epoch={best_ep}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"准确率曲线({run_name})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def append_summary(row: dict):
    """把单个实验结果追加到 outputs/experiments_summary.csv(不存在则建表头)。"""
    exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Bi-LSTM / Attention Bi-LSTM 情感分类训练")
    parser.add_argument("--model", choices=["bilstm", "attn_bilstm"], default="bilstm")
    parser.add_argument("--embedding", choices=["random", "pretrained", "pretrained_finetune"],
                        default="random", help="词向量:随机初始化/预训练冻结/预训练微调")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--emb_dim", type=int, default=300, help="词向量维度(需与预训练矩阵一致)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3, help="早停耐心轮数")
    parser.add_argument("--run_name", type=str, required=True, help="实验名,用于输出文件前缀")
    # ---- 方案B1 混域二分类支持(全部参数默认值=原行为,零回归)----
    parser.add_argument("--data_suffix", type=str, default="",
                        help="数据 pkl 后缀(如混域二分类用 _mix,默认空=原二分类数据)")
    parser.add_argument("--vocab_file", type=str, default="",
                        help="词汇表 json 路径(默认空=data/processed/vocab.json)")
    parser.add_argument("--embedding_matrix", type=str, default="",
                        help="预训练词向量矩阵 npy 路径(默认空=data/processed/embedding_matrix.npy)")
    args = parser.parse_args()

    # ---------- 全局可复现性 ----------
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 设备: {device},种子: {SEED}")

    # ---------- 数据 ----------
    from dataset import build_loaders
    from preprocess import load_vocab

    vocab_file = args.vocab_file or os.path.join(PROCESSED_DIR, "vocab.json")
    vocab = load_vocab(vocab_file)
    train_loader, val_loader, test_loader = build_loaders(args.batch_size, suffix=args.data_suffix)
    print(f"[train] 词表 {len(vocab)}({os.path.basename(vocab_file)}),"
          f"数据后缀 '{args.data_suffix}',训练/验证/测试批次数: "
          f"{len(train_loader)}/{len(val_loader)}/{len(test_loader)}")

    # ---------- 模型 / 损失 / 优化器 ----------
    model = build_model(args, len(vocab), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        # 冻结的 embedding 参数 requires_grad=False,Adam 会自动跳过
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # ---------- 训练循环 + 早停 ----------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = -1.0, -1, None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        # ---- 训练一个 epoch ----
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for ids, y in train_loader:
            ids, y = ids.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(ids)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
        train_loss, train_acc = total_loss / total, correct / total

        # ---- 验证 ----
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"[train] epoch {epoch:02d}/{args.max_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)")

        # ---- 早停逻辑:验证准确率出现新的最佳则记录权重,连续 patience 轮无提升则停 ----
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[train] 早停触发:验证准确率已 {args.patience} 轮未提升")
                break

    # 恢复最佳权重(早停的核心:回滚到验证集表现最好的那一刻)
    assert best_state is not None
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_acc"] = best_val_acc

    # ---------- 保存产物 ----------
    ckpt = {
        "model_state": best_state,
        "config": {  # 记录完整结构超参,evaluate/推理时据此重建模型
            "model": args.model, "embedding": args.embedding,
            "embedding_dim": args.emb_dim, "hidden_dim": args.hidden,
            "dropout": args.dropout, "lr": args.lr,
            "batch_size": args.batch_size, "maxlen": 128,
            # 方案B1:记录数据/词表来源(相对项目根),旧 ckpt 无这些键则按原行为
            "data_suffix": args.data_suffix,
            "vocab_file": os.path.relpath(vocab_file, PROJECT_ROOT),
        },
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
    }
    torch.save(ckpt, os.path.join(OUT_DIR, f"{args.run_name}_best.pt"))
    with open(os.path.join(OUT_DIR, f"{args.run_name}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plot_curves(history, os.path.join(OUT_DIR, f"{args.run_name}_curves.png"), args.run_name)

    # ---------- 测试集评估(复用 evaluate 模块,单一评估代码路径)----------
    from evaluate import evaluate_model, plot_confusion

    metrics, _preds, _labels = evaluate_model(model, test_loader, device)
    with open(os.path.join(OUT_DIR, f"{args.run_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_confusion(metrics["cm"], os.path.join(OUT_DIR, f"{args.run_name}_confusion.png"), args.run_name)

    # ---------- 追加到总对比表 ----------
    append_summary({
        "run_name": args.run_name, "model": args.model, "embedding": args.embedding,
        "lr": args.lr, "hidden": args.hidden, "dropout": args.dropout,
        "batch_size": args.batch_size, "best_val_acc": f"{best_val_acc:.4f}",
        "test_acc": f"{metrics['acc']:.4f}", "test_precision": f"{metrics['precision']:.4f}",
        "test_recall": f"{metrics['recall']:.4f}", "test_f1": f"{metrics['f1']:.4f}",
        "收敛epoch": best_epoch,
    })
    print(f"[train] 完成 {args.run_name}: best_val_acc={best_val_acc:.4f} "
          f"test_f1={metrics['f1']:.4f},结果已写入 {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
