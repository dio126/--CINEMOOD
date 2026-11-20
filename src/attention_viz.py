# -*- coding: utf-8 -*-
"""
注意力可视化(进阶 A)
======================
加载 Bi-LSTM+Attention 最佳检查点,从测试集选取若干条样例评论,
对每条生成"词级注意力热力图"(横轴为分词后的词,颜色深浅表示权重大小),
保存为 outputs/attention_1.png ~ attention_N.png。

选样规则:优先挑长度适中(20~80 个有效词)且模型预测正确的样本,
正负面各取若干条,更有代表性。
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
N_SAMPLES = 5  # 生成 5 条样例(计划要求 4~6 条)


def id2tokens(ids_row: np.ndarray, id2word: dict):
    """整数序列 → 词列表(PAD 之后的部分丢弃,还原真实词序列)。"""
    tokens = []
    for idx in ids_row:
        if idx == 0:  # PAD
            break
        tokens.append(id2word.get(int(idx), "<UNK>"))
    return tokens


def pick_samples(model, test_ds, vocab, device, n=N_SAMPLES):
    """从测试集挑预测正确、长度适中的样本(正负面兼顾)。"""
    id2word = {i: w for w, i in vocab.items()}
    # 随机抽一批候选做预测,再筛选
    rng = np.random.default_rng(42)
    cand_idx = rng.choice(len(test_ds), size=min(400, len(test_ds)), replace=False)
    chosen = []
    with torch.no_grad():
        batch_ids = torch.tensor(np.stack([test_ds.ids[i] for i in cand_idx])).to(device)
        out = model(batch_ids)
        logits, attn = out[0], out[1]
        preds = logits.argmax(dim=-1).cpu().numpy()
        for k, gi in enumerate(cand_idx):
            n_valid = int((test_ds.ids[gi] != 0).sum())
            correct = preds[k] == test_ds.labels[gi]
            # 条件:预测正确 + 长度适中 + 正负面样本均衡
            if correct and 20 <= n_valid <= 80:
                label = int(test_ds.labels[gi])
                if sum(1 for c in chosen if c[2] == label) < (n + 1) // 2 + 1:
                    chosen.append((int(gi), preds[k], label))
            if len(chosen) >= n:
                break
    return chosen[:n], id2word


def run_attention_viz(run_name: str = None):
    from dataset import SentimentDataset
    from evaluate import load_checkpoint_for_eval

    # 默认找注意力模型的最佳 run:命令行传入,或自动扫描 outputs 下 attn 模型
    if run_name is None:
        cands = [p[:-len("_best.pt")] for p in os.listdir(OUT_DIR)
                 if p.endswith("_best.pt") and "attn" in p]
        if not cands:
            raise FileNotFoundError("outputs/ 下未找到注意力模型检查点(*attn*_best.pt)")
        run_name = sorted(cands)[0]
    print(f"[attn_viz] 使用模型: {run_name}")

    model, _ckpt, vocab, device = load_checkpoint_for_eval(run_name)
    test_ds = SentimentDataset("test")
    chosen, id2word = pick_samples(model, test_ds, vocab, device)

    with torch.no_grad():
        for i, (gi, pred, label) in enumerate(chosen, start=1):
            ids_row = test_ds.ids[gi]
            tokens = id2tokens(ids_row, id2word)
            x = torch.tensor(ids_row[None, :]).to(device)
            logits, attn = model(x)
            weights = attn[0].cpu().numpy()[: len(tokens)]  # 截取有效长度对应的权重

            # ---- 绘制 1×L 词级注意力热力图 ----
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.45), 2.6))
            sns.heatmap(
                weights.reshape(1, -1),
                cmap="YlOrRd",
                xticklabels=tokens,
                yticklabels=[],
                cbar=True,
                ax=ax,
            )
            ax.set_xticklabels(tokens, rotation=90, fontsize=9)
            polarity = "正面" if label == 1 else "负面"
            # Top 注意力词:过滤纯单字与 <UNK> 占位词,挑权重最高的多字实词,更有语义可读性
            multi = [
                (w, wt) for w, wt in zip(tokens, weights)
                if len(w) >= 2 and w not in ("<PAD>", "<UNK>")
            ]
            top_words = [w for w, _ in sorted(multi, key=lambda x: -x[1])[:3]] or tokens[:3]
            ax.set_title(
                f"样例{i}:真实 {polarity} | 预测 {'正面' if pred == 1 else '负面'} | "
                f"Top 注意力词: {'、'.join(top_words)}"
            )
            ax.set_xlabel("分词后的词(颜色越深注意力权重越大)")
            fig.tight_layout()
            out_png = os.path.join(OUT_DIR, f"attention_{i}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[attn_viz] 已保存: {out_png}({len(tokens)} 词,{polarity})")


if __name__ == "__main__":
    import sys

    run_attention_viz(sys.argv[1] if len(sys.argv) > 1 else None)
