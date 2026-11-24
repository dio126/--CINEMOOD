# -*- coding: utf-8 -*-
"""
混域二分类预处理管线(方案B1:治跨领域掉分)
================================================
动机:30000 条外部基准诊断显示,Bi-LSTM(仅训练于 ChnSentiCorp 酒店/书/电脑)
在豆瓣电影评论上跨域 Acc 从 0.8896 掉到 0.6780。做法:把 DMSC 剩余池中的
电影二分类样本(4★/5★=正面,1★/2★=负面,等比)混入训练集,让模型见到
电影域的语言风格(反语、行为暗示、口语),提升跨域鲁棒性。

数据构成(seed=42,全程可复现):
  - ChnSentiCorp 全量(与 preprocess.py 完全相同的去重规则,9060 条);
  - DMSC 剩余池抽电影二分类 40000 条(1★/2★/4★/5★ 各 10000)。
    剩余池 = DMSC 全量解析去重后,【四重排除】:
      dmsc_sampled.csv(五级训练源 5 万)→ 六个训练 pkl(五级/二分类)
      → 外部基准 30000(outputs/eval_report/eval_30000_data.csv)
      → 隐晦 300(outputs/eval_report/subtle_300_data.csv)
    后两者是回归测试基准,【绝对禁止入训】,本脚本构建后做交集=0 断言自检。

产物(全部带 _mix 后缀,不触碰既有二分类/五级文件):
  data/processed/{train,val,test}_mix.pkl、vocab_mixed.json、
  embedding_matrix_mixed.npy、wordvec_source_mixed.json;
  采样明细落盘 data/raw/mixed_binary_dmsc.csv(供 B2 扩数据时二次排除,git 忽略)。
"""

import json
import os
import pickle
import random
import time

import numpy as np

# 复用唯一预处理代码路径 + 复用外部测评的 DMSC 解析/去重/排除逻辑,不复制
from eval_external import (
    MAIN_DATA_CSV,
    SUBTLE_DATA_CSV,
    _dedup_like_training,
    _load_dmsc_pool,
)
from preprocess import SEED, build_vocab, encode_tokens, load_stopwords, tokenize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

CHN_CSV = os.path.join(RAW_DIR, "chnsenticorp.csv")
SAMPLED_CSV = os.path.join(RAW_DIR, "dmsc_sampled.csv")
MIX_SOURCE_CSV = os.path.join(RAW_DIR, "mixed_binary_dmsc.csv")  # 采样明细(B2 二次排除用)

VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab_mixed.json")
MATRIX_FILE = os.path.join(PROCESSED_DIR, "embedding_matrix_mixed.npy")
WORDVEC_LOG = os.path.join(PROCESSED_DIR, "wordvec_source_mixed.json")
SPLIT_PKL = {name: os.path.join(PROCESSED_DIR, f"{name}_mix.pkl")
             for name in ("train", "val", "test")}

LEN_MIN, LEN_MAX = 5, 300      # 与外部基准同款长度窗口
PER_STAR_N = 10_000             # 1★/2★/4★/5★ 各 1 万 → DMSC 共 4 万(正负等比)


def _load_chnsenticorp_dedup() -> "tuple[list, list]":
    """ChnSentiCorp 去重(与 preprocess.run_preprocess 完全相同的规则)。"""
    import pandas as pd

    df = pd.read_csv(CHN_CSV)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    label_nunique = df.groupby("text")["label"].transform("nunique")
    conflict = set(df.loc[label_nunique > 1, "text"])
    df = df[(label_nunique == 1) & ~df["text"].isin(conflict)]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    print(f"[mix] ChnSentiCorp 去重后 {len(df)} 条")
    return df["text"].tolist(), df["label"].tolist()


def run_preprocess():
    import pandas as pd

    random.seed(SEED)
    np.random.seed(SEED)
    counts = {"seed": SEED, "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # 1) ChnSentiCorp(酒店/书/电脑域)
    chn_texts, chn_labels = _load_chnsenticorp_dedup()

    # 2) DMSC 剩余池:全量解析 + 同训练规则去重(复用 eval_external 的函数)
    counts["dmsc_zip_rows_read"] = None  # 由 _load_dmsc_pool 填
    pool = _load_dmsc_pool(counts)
    pool = _dedup_like_training(pool, counts)

    # 3) 四重排除:五级训练源 + 六个训练 pkl + 外部基准 30000 + 隐晦 300
    excl_sets = {}
    excl_sets["dmsc_sampled.csv(五级训练源,5万)"] = set(
        pd.read_csv(SAMPLED_CSV)["text"].astype(str))
    for name in ("train", "val", "test"):
        with open(os.path.join(PROCESSED_DIR, f"{name}_5c.pkl"), "rb") as f:
            excl_sets[f"{name}_5c.pkl(五级)"] = set(pickle.load(f)["texts"])
    for name in ("train", "val", "test"):
        with open(os.path.join(PROCESSED_DIR, f"{name}.pkl"), "rb") as f:
            excl_sets[f"{name}.pkl(二分类)"] = set(pickle.load(f)["texts"])
    # 关键铁律:30000 外部基准 + 隐晦 300 必须显式排除(它们同样出自剩余池)
    benchmark_texts = set(pd.read_csv(MAIN_DATA_CSV)["text"].astype(str))
    excl_sets["eval_30000_data.csv(外部基准,禁入训)"] = benchmark_texts
    subtle_texts = set(pd.read_csv(SUBTLE_DATA_CSV)["text"].astype(str))
    excl_sets["subtle_300_data.csv(隐晦基准,禁入训)"] = subtle_texts

    counts["exclusion_steps"] = []
    union = set().union(*excl_sets.values())
    for label, s in excl_sets.items():
        hit = int(pool["text"].isin(s).sum())
        pool = pool[~pool["text"].isin(s)].reset_index(drop=True)
        counts["exclusion_steps"].append({"set": label, "size": len(s),
                                          "removed_at_step": hit, "pool_after": int(len(pool))})
        print(f"[mix] 排除 {label}(集合 {len(s)}):移除 {hit} 条 → 剩余池 {len(pool)}")

    # 4) 星级 1/2/4/5 + 长度窗口 → 每星分层抽样 1 万(seed=42)
    pool = pool[pool["star"].isin([1, 2, 4, 5])]
    in_win = pool["text"].str.len().between(LEN_MIN, LEN_MAX)
    pool = pool[in_win].reset_index(drop=True)
    counts["pool_per_star_in_window"] = {int(k): int(v) for k, v in
                                         pool["star"].value_counts().sort_index().items()}
    parts = []
    for star in (1, 2, 4, 5):
        g = pool[pool["star"] == star]
        assert len(g) >= PER_STAR_N, f"{star}★ 可用 {len(g)} < 目标 {PER_STAR_N}"
        parts.append(g.sample(n=PER_STAR_N, random_state=SEED))
    dmsc = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED)
    dmsc["label"] = (dmsc["star"] >= 4).astype(int)  # 4★/5★=正面(1),1★/2★=负面(0)
    dmsc[["text", "star", "label"]].to_csv(MIX_SOURCE_CSV, index=False, encoding="utf-8-sig")
    print(f"[mix] DMSC 二分类抽样 {len(dmsc)} 条(正负 {dmsc['label'].sum()}/"
          f"{len(dmsc) - dmsc['label'].sum()})→ {MIX_SOURCE_CSV}")

    # 5) 防泄漏自检:抽样结果与【全部排除集】交集必须为 0
    picked = set(dmsc["text"])
    inter = {k: len(picked & s) for k, s in excl_sets.items()}
    counts["selfcheck_dmsc_vs_exclusions"] = inter
    assert all(v == 0 for v in inter.values()), f"泄漏!混域 DMSC 抽样与排除集交集非零:{inter}"
    print(f"[mix] 防泄漏自检:抽样与全部排除集(含 30000 基准)交集均为 0 ✓")

    # 6) 合并两域 → 同文本跨域去重(同标签留 1,标签冲突整组丢弃,同训练规则)
    texts = chn_texts + dmsc["text"].tolist()
    labels = chn_labels + dmsc["label"].tolist()
    merged = pd.DataFrame({"text": texts, "label": labels})
    n_before = len(merged)
    label_nunique = merged.groupby("text")["label"].transform("nunique")
    conflict = set(merged.loc[label_nunique > 1, "text"])
    n_conflict = int(merged["text"].isin(conflict).sum())
    merged = merged[(label_nunique == 1) & ~merged["text"].isin(conflict)]
    merged = merged.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    counts["merged_dedup"] = {"before": int(n_before), "after": int(len(merged)),
                              "conflict_dropped": int(n_conflict)}
    print(f"[mix] 跨域合并去重:{n_before} → {len(merged)}(标签冲突丢弃 {n_conflict})")
    texts, labels = merged["text"].tolist(), merged["label"].values

    # 7) 分词(复用唯一 tokenize 路径)→ 8:1:1(seed=42)
    print("[mix] 分词中(jieba)...")
    token_lists = [tokenize(t) for t in texts]
    indices = list(range(len(texts)))
    random.shuffle(indices)
    n = len(indices)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    split_idx = {"train": sorted(indices[:n_train]),
                 "val": sorted(indices[n_train:n_train + n_val]),
                 "test": sorted(indices[n_train + n_val:])}

    # 8) 词表仅由训练集构建(防数据泄漏,同既有原则)
    vocab = build_vocab([token_lists[i] for i in split_idx["train"]])
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[mix] 词汇表(仅训练集构建):{len(vocab)} → {VOCAB_FILE}")

    # 9) 编码保存(格式与二分类 pkl 完全一致,文件名带 _mix)
    for name, idx in split_idx.items():
        ids = np.stack([encode_tokens(token_lists[i], vocab) for i in idx])
        lab = np.asarray(labels)[idx].astype(np.int64)
        with open(SPLIT_PKL[name], "wb") as f:
            pickle.dump({"ids": ids, "labels": lab, "texts": [texts[i] for i in idx]}, f)
        print(f"[mix] {name}: {len(lab)} 条(正面 {int(lab.sum())} / 负面 {len(lab) - int(lab.sum())})"
              f" → {SPLIT_PKL[name]}")

    counts["final_total"] = int(len(texts))
    counts["splits"] = {k: len(v) for k, v in split_idx.items()}
    with open(os.path.join(PROCESSED_DIR, "mixed_build_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)


def build_embedding_matrix():
    """按混域词表重建词向量矩阵(复用 download_wordvec 的构建函数与腾讯缓存)。"""
    import download_wordvec as dw

    if os.path.exists(MATRIX_FILE):
        print(f"[mix] {MATRIX_FILE} 已存在,跳过")
        return
    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[mix] 为混域词表({len(vocab)} 词)构建词向量矩阵(复用腾讯 light 缓存)...")
    matrix, source = dw.try_download_and_extract(vocab, time.time() + 15 * 60)
    if matrix is None:
        raise RuntimeError("混域词向量构建失败(腾讯缓存缺失且在线源不可用)")
    np.save(MATRIX_FILE, matrix)
    with open(WORDVEC_LOG, "w", encoding="utf-8") as f:
        json.dump({"source": source, "shape": list(matrix.shape)}, f, ensure_ascii=False)
    print(f"[mix] 词向量矩阵已保存:{MATRIX_FILE} {matrix.shape}(来源 {source})")


if __name__ == "__main__":
    run_preprocess()
    build_embedding_matrix()
