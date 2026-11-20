# -*- coding: utf-8 -*-
"""
预训练词向量获取与精简(进阶 B)
================================
目标:为词汇表构造 (vocab_size, dim) 的 embedding 矩阵,
保存到 data/processed/embedding_matrix.npy,并报告词表命中率。

获取策略(总尝试时间预算 ≤ 10 分钟,逐级降级,如实记录最终来源):
  1. 腾讯 AI Lab 200 维中文词向量(精简 small 版,tar.gz,流式解压读取);
  2. fastText 官方中文 300 维词向量(cc.zh.300.vec.gz,Facebook AI,
     同为公认中文预训练词向量,作为"腾讯源不可用"时的替代下载源);
  3. 以上全部失败 → 用 gensim 在本语料上训练 word2vec 300 维替代
     (若 gensim 不可用则报错退出,README 中如实注明替代原因)。

说明:GitHub 上的 Chinese-Word-Vectors 仓库文件托管在百度网盘,无法脚本直连,
     故以 fastText 官方直链作为第二候选,均失败才走本语料自训练。

流式读取:逐行解析词向量文本文件,只保留词汇表命中的词,内存占用小;
未命中词随机初始化 N(0, 0.01);PAD 行置零,UNK 行随机初始化。
"""

import gzip
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.request

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(PROCESSED_DIR, exist_ok=True)

MATRIX_PATH = os.path.join(PROCESSED_DIR, "embedding_matrix.npy")
WORDVEC_SOURCE_LOG = os.path.join(PROCESSED_DIR, "wordvec_source.json")
VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab.json")

TIME_BUDGET_SECONDS = 10 * 60  # 总下载尝试时间预算:10 分钟

# 候选下载源:(名称, URL, 格式)
CANDIDATES = [
    (
        "tencent_ailab_d200_small",
        "https://ai.tencent.com/ailab/nlp/data/tencent-ailab-embedding-zh-d200-v0.1.0-s.txt.tar.gz",
        "tar_gz_txt",
    ),
    (
        "fasttext_cc_zh_300",
        "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz",
        "gz_txt",
    ),
]


def _open_stream(url: str, timeout: int = 60):
    """打开下载流(带 UA),返回响应对象(支持流式读取)。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def _iter_vector_lines_from_gz(url: str):
    """流式逐行产出 (word, np.array) —— 针对 .vec.gz 纯文本格式。"""
    with _open_stream(url) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw_line in io.TextIOWrapper(gz, encoding="utf-8", errors="ignore"):
                parts = raw_line.rstrip().split(" ")
                if len(parts) < 3:
                    continue
                word, vec = parts[0], np.asarray(parts[1:], dtype=np.float32)
                yield word, vec


def _iter_vector_lines_from_tar_gz(url: str):
    """流式解析 tar.gz(内含单个 .txt 词向量文件)。"""
    with _open_stream(url) as resp:
        data_stream = io.BytesIO(resp.read())  # tar 需要随机读,先读入内存(small 版 ~1GB 内)
        with tarfile.open(fileobj=data_stream, mode="r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".txt"))
            f = tar.extractfile(member)
            for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                parts = raw_line.rstrip().split(" ")
                if len(parts) < 3:
                    continue
                word, vec = parts[0], np.asarray(parts[1:], dtype=np.float32)
                yield word, vec


def try_download_and_extract(vocab: dict, deadline: float):
    """按候选顺序尝试下载预训练词向量,返回 (matrix, 来源名) 或 (None, None)。"""
    rng = np.random.default_rng(42)  # 未命中词的随机初始化也可复现
    for name, url, fmt in CANDIDATES:
        if time.time() > deadline:
            print("[wordvec] 下载时间预算已耗尽,停止尝试在线下载")
            return None, None
        try:
            print(f"[wordvec] 尝试下载: {name} ({url})")
            iterator = (
                _iter_vector_lines_from_tar_gz(url) if fmt == "tar_gz_txt"
                else _iter_vector_lines_from_gz(url)
            )
            dim = None
            found = {}  # 词 → 向量(只保留词表命中的)
            for word, vec in iterator:
                if dim is None:
                    dim = vec.shape[0]
                idx = vocab.get(word)
                if idx is not None:
                    found[idx] = vec
                # 提前完成:词表全部命中(罕见)可停止
                if len(found) == len(vocab) - 2:
                    break
            if dim is None or len(found) == 0:
                print(f"[wordvec] {name} 解析失败(未取得任何词表命中)")
                continue
            # 组装矩阵:PAD=0 全零,UNK 与未命中词随机初始化 N(0,0.01)
            matrix = rng.normal(0.0, 0.01, size=(len(vocab), dim)).astype(np.float32)
            matrix[PAD_IDX_LOCAL] = 0.0
            for idx, vec in found.items():
                matrix[idx] = vec
            n_real = len(vocab) - 2  # 去掉 PAD/UNK 两个特殊符号
            hit_rate = len(found) / n_real
            print(f"[wordvec] {name} 加载成功:维度 {dim},词表命中 "
                  f"{len(found)}/{n_real}({hit_rate:.1%})")
            return matrix, name
        except Exception as e:  # noqa: BLE001 网络/解析失败均降级到下一候选
            print(f"[wordvec] {name} 失败: {type(e).__name__}: {e}")
    return None, None


PAD_IDX_LOCAL, UNK_IDX_LOCAL = 0, 1


def train_local_word2vec(vocab: dict, dim: int = 300):
    """兜底方案:用 gensim 在本语料(预处理产物对应的原始文本)上训练 word2vec。"""
    import pandas as pd
    from gensim.models import Word2Vec

    from preprocess import RAW_CSV, tokenize

    print("[wordvec] 在线预训练词向量均不可用,改用 gensim 在本语料上训练 word2vec")
    df = pd.read_csv(RAW_CSV)
    corpus = [tokenize(t) for t in df["text"].astype(str)]
    model = Word2Vec(
        sentences=corpus, vector_size=dim, window=5, min_count=1,
        workers=4, seed=42, epochs=10, sg=1,  # skip-gram
    )
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 0.01, size=(len(vocab), dim)).astype(np.float32)
    matrix[PAD_IDX_LOCAL] = 0.0
    hit = 0
    for word, idx in vocab.items():
        if word in model.wv:
            matrix[idx] = model.wv[word]
            hit += 1
    n_real = len(vocab) - 2
    print(f"[wordvec] 本地 word2vec 训练完成:词表命中 {hit}/{n_real}({hit / n_real:.1%})")
    return matrix, "local_word2vec_gensim"


def main():
    # 已生成则跳过(幂等)
    if os.path.exists(MATRIX_PATH) and os.path.exists(WORDVEC_SOURCE_LOG):
        print(f"[wordvec] {MATRIX_PATH} 已存在,跳过(如需重建请先删除)")
        return

    with open(VOCAB_FILE, encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"[wordvec] 词汇表大小: {len(vocab)}")

    deadline = time.time() + TIME_BUDGET_SECONDS
    matrix, source = try_download_and_extract(vocab, deadline)

    if matrix is None:
        try:
            matrix, source = train_local_word2vec(vocab)
        except ImportError:
            print("[wordvec] gensim 未安装,无法执行本地 word2vec 兜底,词向量获取失败")
            sys.exit(1)

    np.save(MATRIX_PATH, matrix)
    with open(WORDVEC_SOURCE_LOG, "w", encoding="utf-8") as f:
        json.dump({"source": source, "shape": list(matrix.shape)}, f, ensure_ascii=False)
    print(f"[wordvec] 词向量矩阵已保存: {MATRIX_PATH},来源: {source} {matrix.shape}")


if __name__ == "__main__":
    main()
