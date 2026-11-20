# -*- coding: utf-8 -*-
"""
预处理管线(训练与推理共用的唯一代码路径,重点保证一致性)
====================================================================
管线步骤:
  1. 清洗:去除 HTML 标签 / URL / 多余空白;
  2. jieba 分词;
  3. 停用词过滤(哈工大停用词表);
  4. 合并官方 train+dev 后按 8:1:1 重新划分 训练/验证/测试(固定种子 42,可复现);
  5. 词汇表【仅由训练集构建】—— 避免验证/测试集词泄漏到训练特征中(数据泄漏),
     上限 50000 词,特殊符号 <PAD>=0、<UNK>=1;
  6. 文本转整数序列,padding / 截断至 maxlen=128;
  7. 保存 data/processed/ 下 train/val/test 三个 .pkl(含 token id 序列、标签、原文)
     与 vocab.json。

推理(src/predict.py、app/)通过 import 本模块的 clean_text / tokenize /
encode_tokens 复用同一份逻辑,绝不复制第二份分词代码。
"""

import json
import os
import pickle
import random
import re

import numpy as np

# 项目根目录(src/ 的上一级),保证任意工作目录下运行均正确
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_CSV = os.path.join(RAW_DIR, "chnsenticorp.csv")
STOPWORDS_FILE = os.path.join(RAW_DIR, "hit_stopwords.txt")
VOCAB_FILE = os.path.join(PROCESSED_DIR, "vocab.json")

PAD_TOKEN, UNK_TOKEN = "<PAD>", "<UNK>"
PAD_IDX, UNK_IDX = 0, 1
MAX_VOCAB_SIZE = 50000  # 词汇表规模上限
MAXLEN = 128  # 序列统一长度(padding / 截断)
SEED = 42  # 全局固定随机种子,保证划分可复现

# 预处理后产物路径
SPLIT_PKL = {name: os.path.join(PROCESSED_DIR, f"{name}.pkl") for name in ("train", "val", "test")}

# ---------- 正则:清洗规则集中定义,便于维护 ----------
_RE_HTML = re.compile(r"<[^>]+>")  # HTML 标签,如 <br/>
_RE_URL = re.compile(r"https?://\S+|www\.\S+")  # URL
_RE_BLANK = re.compile(r"\s+")  # 连续空白(含全角空格转半角后)
_RE_NONWORD = re.compile(r"^[^\w\u4e00-\u9fff]+$")  # 纯符号(不含中文/字母/数字)的词


def clean_text(text: str) -> str:
    """文本清洗:去 HTML 标签、URL、多余空白。训练与推理共用。"""
    text = str(text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = text.replace("\u3000", " ")  # 全角空格转半角,统一处理
    text = _RE_BLANK.sub(" ", text).strip()
    return text


def load_stopwords(path: str = None) -> set:
    """加载停用词表(每行一个词,# 开头为注释行)。"""
    path = path or STOPWORDS_FILE
    stopwords = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w and not w.startswith("#"):
                    stopwords.add(w)
    else:
        print(f"[preprocess] 警告:停用词文件不存在({path}),将不过滤停用词")
    return stopwords


# 模块级缓存:同一进程内(HF 训练循环、Streamlit 多次推理)避免重复读文件
_STOPWORDS_CACHE = None


def tokenize(text: str) -> list:
    """清洗 + jieba 分词 + 停用词过滤,返回词列表。训练与推理共用(一致性关键)。"""
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        _STOPWORDS_CACHE = load_stopwords()
    import jieba  # 延迟导入:jieba 初始化较慢,首次调用时加载

    text = clean_text(text)
    words = jieba.lcut(text)
    # 过滤:停用词、空串、纯符号词(保留中文、字母、数字)
    return [w for w in words if w and w not in _STOPWORDS_CACHE and not _RE_NONWORD.match(w)]


def build_vocab(token_lists: list, max_size: int = MAX_VOCAB_SIZE) -> dict:
    """由【训练集】词列表构建词汇表:按词频降序,上限 max_size,含 <PAD>/<UNK>。

    注意:仅使用训练集构建,验证/测试集中未登录词一律映射为 <UNK>,
    这是防止数据泄漏的标准做法。
    """
    from collections import Counter

    counter = Counter()
    for words in token_lists:
        counter.update(words)
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, _freq in counter.most_common(max_size - 2):  # 预留 2 个特殊符号
        vocab[word] = len(vocab)
    return vocab


def encode_tokens(words: list, vocab: dict, maxlen: int = MAXLEN) -> np.ndarray:
    """词列表 → 定长整数序列(超长截断,不足右侧补 PAD_IDX)。"""
    ids = [vocab.get(w, UNK_IDX) for w in words[:maxlen]]
    ids += [PAD_IDX] * (maxlen - len(ids))
    return np.asarray(ids, dtype=np.int64)


def run_preprocess():
    """执行完整预处理管线,产出 train/val/test.pkl 与 vocab.json。"""
    import pandas as pd

    # 0) 固定随机种子,保证 8:1:1 划分可复现
    random.seed(SEED)
    np.random.seed(SEED)

    # 1) 读取原始数据
    df = pd.read_csv(RAW_CSV)
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    print(f"[preprocess] 原始样本数: {len(df)},标签分布: {df['label'].value_counts().to_dict()}")

    # 2) 清洗 + 分词 + 停用词过滤(全量一次完成)
    print("[preprocess] 分词中(jieba)...")
    token_lists = [tokenize(t) for t in df["text"]]

    # 3) 合并后打乱,按 8:1:1 划分(原始数据已是官方 train+dev 合并体)
    indices = list(range(len(df)))
    random.shuffle(indices)
    n = len(indices)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    split_idx = {
        "train": sorted(indices[:n_train]),
        "val": sorted(indices[n_train:n_train + n_val]),
        "test": sorted(indices[n_train + n_val:]),
    }

    # 4) 词汇表仅由训练集构建(防数据泄漏,见 build_vocab 注释)
    train_tokens = [token_lists[i] for i in split_idx["train"]]
    vocab = build_vocab(train_tokens)
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"[preprocess] 词汇表大小(仅训练集构建): {len(vocab)} → {VOCAB_FILE}")

    # 5) 编码为定长序列并保存 .pkl(含原文,便于注意力可视化取例)
    for name, idx in split_idx.items():
        ids = np.stack([encode_tokens(token_lists[i], vocab) for i in idx])
        labels = df["label"].values[idx].astype(np.int64)
        data = {
            "ids": ids,  # [N, MAXLEN] int64,已 padding
            "labels": labels,  # [N] int64,1=正面 0=负面
            "texts": [df["text"].values[i] for i in idx],  # 原始文本(清洗前),供可视化
        }
        with open(SPLIT_PKL[name], "wb") as f:
            pickle.dump(data, f)
        pos = int(labels.sum())
        print(
            f"[preprocess] {name}: {len(labels)} 条(正面 {pos} / 负面 {len(labels) - pos})"
            f" → {SPLIT_PKL[name]}"
        )


def load_vocab(path: str = None) -> dict:
    """加载词汇表(word → id)。"""
    with open(path or VOCAB_FILE, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    run_preprocess()
