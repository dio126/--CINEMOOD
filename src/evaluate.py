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
    """加载检查点并重建模型(供 evaluate 主程序 / attention_viz / predict 复用)。

    兼容说明:旧二分类 ckpt 的 config 无 num_classes 字段,默认按 2 类重建
    (行为不变);五级分类 ckpt 保存 num_classes=5,按 5 类重建。
    """
    from dataset import SentimentDataset  # noqa: F401  保持与其他模块一致的导入风格
    from models.attn_bilstm import AttnBiLSTMClassifier
    from models.bilstm import BiLSTMClassifier

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(OUT_DIR, f"{run_name}_best.pt")
    if not os.path.exists(ckpt_path):
        # 五级分类产物为 <run>_5c_best.pt,按文件存在性自动探测任务类型
        alt = os.path.join(OUT_DIR, f"{run_name}_5c_best.pt")
        if os.path.exists(alt):
            ckpt_path = alt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]  # 训练时保存的模型超参,保证重建结构一致
    num_classes = cfg.get("num_classes", 2)  # 向后兼容:旧 ckpt 默认二分类

    # ---- 方案A:BERT 分支(config.model_type='bert';旧 ckpt 无此键 → 原 Bi-LSTM 路径不变)----
    if cfg.get("model_type") == "bert":
        from models.bert import BertClassifier

        model = BertClassifier(num_classes=num_classes, model_path=cfg.get("model_path"))
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model, ckpt, None, device  # BERT 无词表(vocab=None),tokenizer 按 config 加载

    # 词表优先级:ckpt 显式记录的 vocab_file(方案B1 混域)> 按任务类型默认
    # (旧 ckpt 无该键 → 行为不变)
    vocab_file = cfg.get("vocab_file")
    if vocab_file:
        vocab = json.load(open(os.path.join(PROJECT_ROOT, vocab_file), encoding="utf-8"))
    elif num_classes == 5:
        from preprocess_5class import load_vocab_5class as load_vocab
        vocab = load_vocab()
    else:
        from preprocess import load_vocab
        vocab = load_vocab()
    if cfg["model"] == "attn_bilstm":
        model = AttnBiLSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            num_classes=num_classes,
        )
    else:
        model = BiLSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=cfg["embedding_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg["dropout"],
            num_classes=num_classes,
        )
    model.load_state_dict(ckpt["model_state"])  # 恢复最佳权重
    model.to(device).eval()
    return model, ckpt, vocab, device


def run_evaluate(run_name: str):
    """主评估流程:加载 ckpt → 测试集指标 → metrics.json + 混淆矩阵图。"""
    from torch.utils.data import DataLoader

    from dataset import SentimentDataset

    model, ckpt, _vocab, device = load_checkpoint_for_eval(run_name)
    # 数据后缀:ckpt 显式记录(方案B1 混域 _mix)则用之,旧 ckpt 无该键=原行为
    suffix = ckpt["config"].get("data_suffix", "")
    test_ds = SentimentDataset("test", suffix=suffix)
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
