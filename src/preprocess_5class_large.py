# -*- coding: utf-8 -*-
"""
五级分类扩数据预处理管线(方案B2:破容量上限)
================================================
动机:外部基准诊断显示,5 万条训练的 Bi-LSTM 在同分布未见数据上 exact 与
自测集持平(0.4688 vs 0.4700)——已逼近该数据规模下的容量上限。做法:
DMSC 剩余池分层抽样 30 万(每级 6 万)重训同配置模型,验证"数据规模↑ →
上限↑"。

排除集(全部 exact match,防泄漏):
  dmsc_sampled.csv(原 5 万)→ 六个训练 pkl → 外部基准 30000 + 隐晦 300
  (禁入训,回归测试基准)→ 混域 B1 已抽的 4 万(data/raw/mixed_binary_dmsc.csv)。

产物(带 _5cL 后缀,不触碰既有文件):
  data/processed/{train,val,test}_5cL.pkl、vocab_5class_large.json、
  embedding_matrix_5class_large.npy、wordvec_source_5class_large.json。
"""

import json
import os
import pickle
import random
import time

import numpy as np

from eval_external import (
    MAIN_DATA_CSV,
    SUBTLE_DATA_CSV,
    _dedup_like_training,
    _load_dmsc_pool,
)
from preprocess import SEED, build_vocab, encode_tokens, tokenize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SAMPLED_CSV = os.path.join(RAW_DIR, "dmsc_sampled.csv")
MIX_SOURCE_CSV = os.path.join(RAW_DIR, "mixed_binary_dmsc.csv")  # B1 已抽 4 万

VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab_5class_large.json")
MATRIX_FILE = os.path.join(PROCESSED_DIR, "embedding_matrix_5class_large.npy")
WORDVEC_LOG = os.path.join(PROCESSED_DIR, "wordvec_source_5class_large.json")
SPLIT_PKL = {name: os.path.join(PROCESSED_DIR, f"{name}_5cL.pkl")
             for name in ("train", "val", "test")}

LEN_MIN, LEN_MAX = 5, 300
PER_CLASS_N = 60_000  # 每级 6 万 × 5 = 30 万


def run_preprocess():
    import pandas as pd

    random.seed(SEED)
    np.random.seed(SEED)
    counts = {"seed": SEED, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1) DMSC 剩余池(复用 eval_external 解析与去重)
    pool = _load_dmsc_pool(counts)
    pool = _dedup_like_training(pool, counts)

    # 2) 五重排除:原 5 万 + 六个训练 pkl + 外部基准 30000 + 隐晦 300 + B1 混域 4 万
    excl_sets = {}
    excl_sets["dmsc_sampled.csv(五级训练源,5万)"] = set(
        pd.read_csv(SAMPLED_CSV)["text"].astype(str))
    for name in ("train", "val", "test"):
        with open(os.path.join(PROCESSED_DIR, f"{name}_5c.pkl"), "rb") as f:
            excl_sets[f"{name}_5c.pkl(五级)"] = set(pickle.load(f)["texts"])
    for name in ("train", "val", "test"):
        with open(os.path.join(PROCESSED_DIR, f"{name}.pkl"), "rb") as f:
            excl_sets[f"{name}.pkl(二分类)"] = set(pickle.load(f)["texts"])
    excl_sets["eval_30000_data.csv(外部基准,禁入训)"] = set(
        pd.read_csv(MAIN_DATA_CSV)["text"].astype(str))
    excl_sets["subtle_300_data.csv(隐晦基准,禁入训)"] = set(
        pd.read_csv(SUBTLE_DATA_CSV)["text"].astype(str))
    excl_sets["mixed_binary_dmsc.csv(B1 混域已用)"] = set(
        pd.read_csv(MIX_SOURCE_CSV)["text"].astype(str))

    counts["exclusion_steps"] = []
    for label, s in excl_sets.items():
        hit = int(pool["text"].isin(s).sum())
        pool = pool[~pool["text"].isin(s)].reset_index(drop=True)
        counts["exclusion_steps"].append({"set": label, "size": len(s),
                                          "removed_at_step": hit, "pool_after": int(len(pool))})
        print(f"[5cL] 排除 {label}(集合 {len(s)}):移除 {hit} 条 → 剩余池 {len(pool)}")

    # 3) 长度窗口 + 每级分层 6 万
    in_win = pool["text"].str.len().between(LEN_MIN, LEN_MAX)
    pool = pool[in_win].reset_index(drop=True)
    counts["pool_per_star_in_window"] = {int(k): int(v) for k, v in
                                         pool["star"].value_counts().sort_index().items()}
    parts = []
    for star in (1, 2, 3, 4, 5):
        g = pool[pool["star"] == star]
        assert len(g) >= PER_CLASS_N, f"{star}★ 可用 {len(g)} < 目标 {PER_CLASS_N}"
        parts.append(g.sample(n=PER_CLASS_N, random_state=SEED))
    df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
    df["label"] = df["star"] - 1
    print(f"[5cL] 分层抽样 {len(df)} 条,每级分布 {df['label'].value_counts().sort_index().to_dict()}")

    # 4) 防泄漏自检:抽样与全部排除集交集必须为 0
    picked = set(df["text"])
    inter = {k: len(picked & s) for k, s in excl_sets.items()}
    counts["selfcheck_vs_exclusions"] = inter
    assert all(v == 0 for v in inter.values()), f"泄漏!五级扩数据抽样交集非零:{inter}"
    print(f"[5cL] 防泄漏自检:抽样与全部排除集(含 30000 基准与 B1 已用)交集均为 0 ✓")

    # 5) 分词 → 8:1:1 → 词表(仅训练集)→ 编码保存
    print("[5cL] 分词中(jieba,30 万条,约需几分钟)...")
    texts = df["text"].tolist()
    labels = df["label"].values
    token_lists = [tokenize(t) for t in texts]
    indices = list(range(len(texts)))
    random.shuffle(indices)
    n = len(indices)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    split_idx = {"train": sorted(indices[:n_train]),
                 "val": sorted(indices[n_train:n_train + n_val]),
                 "test": sorted(indices[n_train + n_val:])}

    vocab = build_vocab([token_lists[i] for i in split_idx["train"]])
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[5cL] 词汇表(仅训练集构建):{len(vocab)} → {VOCAB_FILE}")

    for name, idx in split_idx.items():
        ids = np.stack([encode_tokens(token_lists[i], vocab) for i in idx])
        lab = np.asarray(labels)[idx].astype(np.int64)
        with open(SPLIT_PKL[name], "wb") as f:
            pickle.dump({"ids": ids, "labels": lab, "texts": [texts[i] for i in idx]}, f)
        print(f"[5cL] {name}: {len(lab)} 条 → {SPLIT_PKL[name]}")

    counts["final_total"] = int(len(texts))
    counts["splits"] = {k: len(v) for k, v in split_idx.items()}
    with open(os.path.join(PROCESSED_DIR, "5class_large_build_counts.json"), "w",
              encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)


def build_embedding_matrix():
    """按五级大词表重建词向量矩阵(复用 download_wordvec 与腾讯缓存)。"""
    import download_wordvec as dw

    if os.path.exists(MATRIX_FILE):
        print(f"[5cL] {MATRIX_FILE} 已存在,跳过")
        return
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[5cL] 为五级大词表({len(vocab)} 词)构建词向量矩阵(复用腾讯 light 缓存)...")
    matrix, source = dw.try_download_and_extract(vocab, time.time() + 15 * 60)
    if matrix is None:
        raise RuntimeError("五级大词表词向量构建失败")
    np.save(MATRIX_FILE, matrix)
    with open(WORDVEC_LOG, "w", encoding="utf-8") as f:
        json.dump({"source": source, "shape": list(matrix.shape)}, f, ensure_ascii=False)
    print(f"[5cL] 词向量矩阵已保存:{MATRIX_FILE} {matrix.shape}(来源 {source})")


if __name__ == "__main__":
    run_preprocess()
    build_embedding_matrix()
