# -*- coding: utf-8 -*-
"""
探索性数据分析(EDA)
====================
统计 ChnSentiCorp 原始数据:
  - 样本总数、正负样本数与比例;
  - 评论长度分布:原始字符长度 + jieba 分词后词数,两幅直方图;
  - 图保存到 outputs/eda_length_dist.png。

所有 matplotlib 图统一设置中文字体(Microsoft YaHei),防中文乱码。
"""

import os

import matplotlib

matplotlib.use("Agg")  # 无界面环境后台出图,避免在服务器/CI 上弹窗报错
import matplotlib.pyplot as plt
import pandas as pd

# 中文字体设置(Windows 自带微软雅黑),必须放在绘图之前
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "chnsenticorp.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def run_eda():
    # 复用预处理模块的分词逻辑,保证统计口径与建模一致
    from preprocess import clean_text, tokenize

    df = pd.read_csv(RAW_CSV)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    # ---------- 基本统计 ----------
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"[eda] 样本总数: {len(df)}")
    print(f"[eda] 正面样本: {n_pos}({n_pos / len(df):.1%})  负面样本: {n_neg}({n_neg / len(df):.1%})")

    # ---------- 长度统计 ----------
    char_lens = df["text"].str.len()  # 原始字符长度
    print("[eda] 分词中(统计词数)...")
    word_lens = [len(tokenize(t)) for t in df["text"]]  # 分词后词数(含停用词过滤)

    for name, series in (("原始字符长度", char_lens), ("分词后词数", word_lens)):
        s = pd.Series(series)
        print(
            f"[eda] {name}: 均值 {s.mean():.1f} / 中位数 {s.median():.0f} / "
            f"最大 {s.max()} / 最小 {s.min()}"
        )

    # ---------- 绘图:两个子图并排的长度分布直方图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(char_lens.clip(upper=2000), bins=50, color="#4C72B0", edgecolor="white")
    axes[0].set_title("原始评论长度分布(字符数,>2000 截断显示)")
    axes[0].set_xlabel("字符数")
    axes[0].set_ylabel("样本数")

    axes[1].hist(pd.Series(word_lens).clip(upper=512), bins=50, color="#DD8452", edgecolor="white")
    axes[1].axvline(128, color="red", linestyle="--", label="maxlen=128")
    axes[1].set_title("分词后词数分布(>512 截断显示)")
    axes[1].set_xlabel("词数")
    axes[1].set_ylabel("样本数")
    axes[1].legend()

    fig.suptitle(f"ChnSentiCorp 评论长度分布(共 {len(df)} 条,正面 {n_pos} / 负面 {n_neg})")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "eda_length_dist.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[eda] 长度分布图已保存: {out_png}")


if __name__ == "__main__":
    run_eda()
