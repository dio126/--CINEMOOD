# -*- coding: utf-8 -*-
"""
模型评估模块
============
- evaluate_model:在给定 DataLoader 上计算 Accuracy / Precision / Recall / F1
  (Precision/Recall/F1 以"正面"为正类,pos_label=1)并返回混淆矩阵;
- 主程序:加载 outputs/<run_name>_best.pt 检查点,在测试集输出
  outputs/<run_name>_metrics.json 与 seaborn 混淆矩阵热力图
  outputs/<run_name>_confusion.png。

该模块被 train.py 复用(每个实验训练结束自动评估),避免两份评估逻辑。
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 中文字体设置,防乱码
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()  # 评估阶段不需要梯度,省显存提速
def evaluate_model(model, loader, device):
    """返回 dict(acc, precision, recall, f1, cm) 与 (all_preds, all_labels)。"""
    model.eval()
    preds, labels = [], []
    for ids, y in loader:
        ids, y = ids.to(device), y.to(device)
        out = model(ids)
        logits = out[0] if isinstance(out, tuple) else out  # 注意力模型返回 (logits, alpha)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = {
        "acc": float(accuracy_score(labels, preds)),
        # 以"正面(1)"为正类计算二分类指标
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
        "cm": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
    return metrics, preds, labels


def plot_confusion(cm, out_png: str, run_name: str = ""):
    """seaborn 混淆矩阵热力图(行=真实,列=预测,0=负面 1=正面)。"""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["负面(0)", "正面(1)"],
        yticklabels=["负面(0)", "正面(1)"],
        ax=ax,
    )
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(f"混淆矩阵 {run_name}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def load_checkpoint_for_eval(run_name: str, device=None):
    """加载检查点并重建模型(供 evaluate 主程序 / attention_viz / predict 复用)。"""
    from dataset import SentimentDataset  # noqa: F401  保持与其他模块一致的导入风格
    from models.attn_bilstm import AttnBiLSTMClassifier
    from models.bilstm import BiLSTMClassifier
    from preprocess import load_vocab

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(OUT_DIR, f"{run_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]  # 训练时保存的模型超参,保证重建结构一致
    vocab = load_vocab()
    if cfg["model"] == "attn_bilstm":
        model = AttnBiLSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        )
    else:
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
        )
    model.load_state_dict(ckpt["model_state"])  # 恢复最佳权重
    model.to(device).eval()
    return model, ckpt, vocab, device


def run_evaluate(run_name: str):
    """主评估流程:加载 ckpt → 测试集指标 → metrics.json + 混淆矩阵图。"""
    from torch.utils.data import DataLoader

    from dataset import SentimentDataset

    model, ckpt, _vocab, device = load_checkpoint_for_eval(run_name)
    test_ds = SentimentDataset("test")
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    metrics, preds, labels = evaluate_model(model, loader, device)
    metrics_json = os.path.join(OUT_DIR, f"{run_name}_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_confusion(metrics["cm"], os.path.join(OUT_DIR, f"{run_name}_confusion.png"), run_name)

    print(f"[evaluate] {run_name} 测试集: acc={metrics['acc']:.4f} "
          f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    print(f"[evaluate] 指标已保存: {metrics_json}")
    return metrics


if __name__ == "__main__":
    import sys

    # 用法: python evaluate.py <run_name>
    if len(sys.argv) != 2:
        print("用法: python evaluate.py <run_name>")
        sys.exit(1)
    run_evaluate(sys.argv[1])
