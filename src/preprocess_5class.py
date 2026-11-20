# -*- coding: utf-8 -*-
"""
五级分类预处理管线(非常负面/负面/中性/正面/非常正面)
========================================================
与二分类预处理的关系:本模块【import 复用】preprocess.py 的
clean_text / tokenize(含停用词过滤与否定词合并)/ encode_tokens / build_vocab,
绝不复制第二份分词逻辑,保证两套任务的文本处理路径完全一致。

差异点(均为五级任务独立产物,不触碰二分类文件):
  - 输入:data/raw/dmsc_sampled.csv(text, star, label),label 0~4;
  - 词汇表:vocab_5class.json(仅由五级训练集构建,防数据泄漏);
  - 划分:8:1:1(seed=42,采样阶段已分层均衡,随机划分即近似分层);
  - 产物:train/val/test 的 *_5c.pkl(含 token id 序列、标签、原文);
  - 预训练词向量:按新词表重建 embedding_matrix_5class.npy
    (复用 download_wordvec.py 的构建函数与磁盘缓存的腾讯 light 词向量,
    含 OOV 同量级初始化与"否定+X"组合式初始化)。
"""

import json
import os
import pickle
import random
import time

import numpy as np

# 复用二分类模块的唯一预处理代码路径(一致性关键)
from preprocess import MAXLEN, SEED, build_vocab, encode_tokens, load_stopwords, tokenize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_CSV = os.path.join(RAW_DIR, "dmsc_sampled.csv")
VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab_5class.json")
MATRIX_FILE = os.path.join(PROCESSED_DIR, "embedding_matrix_5class.npy")
WORDVEC_LOG = os.path.join(PROCESSED_DIR, "wordvec_source_5class.json")
SPLIT_PKL = {name: os.path.join(PROCESSED_DIR, f"{name}_5c.pkl")
             for name in ("train", "val", "test")}

# 五级标签名(0~4),训练评估/推理/界面共用同一映射
CLASS_NAMES = ["非常负面", "负面", "中性", "正面", "非常正面"]


def load_vocab_5class(path: str = None) -> dict:
    """加载五级任务词汇表。"""
    with open(path or VOCAB_FILE, encoding="utf-8") as f:
        return json.load(f)


def run_preprocess():
    """执行五级预处理:分词(复用)→ 8:1:1 划分 → 词表(仅训练集)→ 序列化。"""
    import pandas as pd

    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(RAW_CSV)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)  # 0~4
    print(f"[5c-pre] 样本数 {len(df)},标签分布: {df['label'].value_counts().sort_index().to_dict()}")

    # 1) 分词(与二分类同一份 tokenize:清洗+停用词+否定词合并)
    print("[5c-pre] 分词中(复用 preprocess.tokenize)...")
    token_lists = [tokenize(t) for t in df["text"]]

    # 2) 8:1:1 划分(seed=42;采样阶段五类已均衡,随机划分即可)
    indices = list(range(len(df)))
    random.shuffle(indices)
    n = len(indices)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    split_idx = {
        "train": sorted(indices[:n_train]),
        "val": sorted(indices[n_train:n_train + n_val]),
        "test": sorted(indices[n_train + n_val:]),
    }

    # 3) 词汇表仅由【五级训练集】构建(防数据泄漏,同二分类原则)
    vocab = build_vocab([token_lists[i] for i in split_idx["train"]])
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[5c-pre] 词汇表(仅训练集构建): {len(vocab)} → {VOCAB_FILE}")

    # 4) 编码为定长序列并保存 .pkl(格式与二分类一致,便于复用 Dataset)
    for name, idx in split_idx.items():
        ids = np.stack([encode_tokens(token_lists[i], vocab) for i in idx])
        labels = df["label"].values[idx].astype(np.int64)
        with open(SPLIT_PKL[name], "wb") as f:
            pickle.dump({"ids": ids, "labels": labels,
                         "texts": [df["text"].values[i] for i in idx]}, f)
        dist = np.bincount(labels, minlength=5).tolist()
        print(f"[5c-pre] {name}: {len(labels)} 条,分布 {dist} → {SPLIT_PKL[name]}")


def build_embedding_matrix():
    """按五级词表重建词向量矩阵(复用 download_wordvec 的构建函数与磁盘缓存)。"""
    import download_wordvec as dw

    if os.path.exists(MATRIX_FILE):
        print(f"[5c-pre] {MATRIX_FILE} 已存在,跳过")
        return
    vocab = load_vocab_5class()
    print(f"[5c-pre] 为五级词表({len(vocab)} 词)构建词向量矩阵(复用腾讯 light 缓存)...")
    # 复用二分类词向量脚本的候选源/缓存/初始化逻辑(等量级 OOV + 否定组合式初始化)
    matrix, source = dw.try_download_and_extract(vocab, time.time() + 15 * 60)
    if matrix is None:  # 缓存通常必命中;万一全失败则本地兜底不可用时报错
        raise RuntimeError("五级词向量构建失败(腾讯缓存缺失且在线源不可用)")
    np.save(MATRIX_FILE, matrix)
    with open(WORDVEC_LOG, "w", encoding="utf-8") as f:
        json.dump({"source": source, "shape": list(matrix.shape)}, f, ensure_ascii=False)
    print(f"[5c-pre] 词向量矩阵已保存: {MATRIX_FILE} {matrix.shape}(来源 {source})")


if __name__ == "__main__":
    run_preprocess()
    build_embedding_matrix()
