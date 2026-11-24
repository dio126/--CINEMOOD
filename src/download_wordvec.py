# -*- coding: utf-8 -*-
"""
预训练词向量获取与精简(进阶 B)
================================
目标:为词汇表构造 (vocab_size, dim) 的 embedding 矩阵,
保存到 data/processed/embedding_matrix.npy,并报告词表命中率。

获取策略(总尝试时间预算 ≤ 10 分钟,逐级降级,如实记录最终来源):
  1. 腾讯 AI Lab 200 维中文词向量【精简 light 版】(HF 镜像托管的
     light_Tencent_AILab_ChineseEmbedding.bin,143613 词 × 200 维,word2vec
     二进制格式,流式解析;实测腾讯官网原版与 fastText 直连仅几十~百 KB/s,
     1GB 级文件无法在预算内完成,故首选此镜像精简版);
  2. 腾讯 AI Lab 官网 small 版 tar.gz(若网络允许);
  3. fastText 官方中文 300 维词向量(cc.zh.300.vec.gz);
  4. 以上全部失败 → 用 gensim 在本语料上训练 word2vec 300 维替代
     (若 gensim 不可用则报错退出,README 中如实注明替代原因)。

说明:GitHub 上的 Chinese-Word-Vectors 仓库文件托管在百度网盘,无法脚本直连,
     故以 HF 镜像/官网/fastText 直链作为候选,均失败才走本语料自训练。

流式读取:逐行/逐词解析词向量文件,只保留词汇表命中的词,内存占用小;
未命中词按【命中行同量级标准差】随机初始化(幅值对齐,保证可被微调学习);
PAD 行置零,UNK 行随机初始化。
"""

import gzip
import io
import json
import os
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
# 实测说明:腾讯官网源与 fastText 直连在本机网络下仅 23KB/s~185KB/s,
# 881MB~1.3GB 无法在 10 分钟预算内完成;HF 镜像(hf-mirror.com)实测约 1MB/s,
# 且其托管的 light_Tencent_AILab_ChineseEmbedding.bin(143613 词 × 200 维,116MB)
# 正是"腾讯 AI Lab 200 维中文词向量的精简(light)版",故列为首选。
CANDIDATES = [
    (
        "tencent_ailab_d200_light_hf",
        "https://hf-mirror.com/shibing624/text2vec-word2vec-tencent-chinese/resolve/main/"
        "light_Tencent_AILab_ChineseEmbedding.bin",
        "w2v_bin",
    ),
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


def _download_to_file(url: str, path: str, deadline: float):
    """把 URL 内容流式下载到磁盘文件(256KB 分块写盘,不整读内存)。"""
    total = 0
    with _open_stream(url, timeout=120) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total // (20 * 1024 * 1024) != (total - len(chunk)) // (20 * 1024 * 1024):
                print(f"[wordvec] 已下载 {total // 1048576}MB ...")
                if time.time() > deadline:
                    raise TimeoutError("超过下载时间预算")


# w2v_bin 源的本地磁盘缓存:先落盘再解析,重建词表/词向量矩阵时无需重复下载
W2V_BIN_CACHE = os.path.join(RAW_DIR, "tencent_light_word2vec.bin")


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


def _iter_vector_lines_from_tar_gz(url: str, deadline: float):
    """解析 tar.gz(内含单个 .txt 词向量文件)。

    先把压缩包流式下载到临时文件再解包读取,避免把 GB 级数据整块读进内存;
    用完即删临时文件。
    """
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    try:
        _download_to_file(url, tmp_path, deadline)
        with tarfile.open(tmp_path, mode="r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith(".txt"))
            with tar.extractfile(member) as f:
                for raw_line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                    parts = raw_line.rstrip().split(" ")
                    if len(parts) < 3:
                        continue
                    yield parts[0], np.asarray(parts[1:], dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _iter_vectors_from_w2v_bin(url: str, deadline: float, cache_path: str = W2V_BIN_CACHE):
    """解析 word2vec【二进制】格式(.bin):
    头部为 "<词数> <维度>\\n",之后每条为 "词<空格><dim×4字节 float32>"。

    先把文件流式下载到磁盘缓存(cache_path 已存在则直接复用,重建词表时免重下载),
    再从本地文件按 256KB 块缓冲解析,内存占用小。
    """
    if not os.path.exists(cache_path):
        print(f"[wordvec] 下载到磁盘缓存: {cache_path}")
        tmp = cache_path + ".part"
        try:
            _download_to_file(url, tmp, deadline)
            os.replace(tmp, cache_path)  # 原子改名,避免半截缓存被误用
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    else:
        print(f"[wordvec] 命中磁盘缓存: {cache_path}(跳过下载)")

    with open(cache_path, "rb") as f:
        buf = b""
        chunk_iter = iter(lambda: f.read(256 * 1024), b"")

        def _fill() -> bool:
            nonlocal buf
            nxt = next(chunk_iter, b"")
            buf += nxt
            return len(nxt) > 0

        def _read_line() -> bytes:
            nonlocal buf
            while b"\n" not in buf:
                if not _fill():
                    raise ValueError("文件意外结束(读头部时)")
            line, buf2 = buf.split(b"\n", 1)
            buf = buf2
            return line

        def _read_exact(n: int) -> bytes:
            nonlocal buf
            while len(buf) < n:
                if not _fill():
                    raise ValueError("文件意外结束(读向量时)")
            out, buf2 = buf[:n], buf[n:]
            buf = buf2
            return out

        header = _read_line()
        n_words, dim = map(int, header.split())
        print(f"[wordvec] 二进制头: {n_words} 词 × {dim} 维")
        for i in range(n_words):
            if i % 20000 == 0:  # 进度提示
                print(f"[wordvec] 已解析 {i}/{n_words} 词")
            # 读词:UTF-8 字节直到空格(有的实现词前有换行,一并兼容)
            word_bytes = b""
            while True:
                if not buf and not _fill():
                    break
                j = buf.find(b" ")
                if j == -1:
                    word_bytes += buf
                    buf = b""
                    continue
                word_bytes += buf[:j]
                buf = buf[j + 1:]
                break
            vec = np.frombuffer(_read_exact(dim * 4), dtype="<f4")
            yield word_bytes.decode("utf-8", errors="ignore"), vec


def try_download_and_extract(vocab: dict, deadline: float):
    """按候选顺序尝试下载预训练词向量,返回 (matrix, 来源名) 或 (None, None)。"""
    rng = np.random.default_rng(42)  # 未命中词的随机初始化也可复现
    for name, url, fmt in CANDIDATES:
        if time.time() > deadline:
            print("[wordvec] 下载时间预算已耗尽,停止尝试在线下载")
            return None, None
        try:
            print(f"[wordvec] 尝试下载: {name} ({url})")
            if fmt == "w2v_bin":
                iterator = _iter_vectors_from_w2v_bin(url, deadline)
            elif fmt == "tar_gz_txt":
                iterator = _iter_vector_lines_from_tar_gz(url, deadline)
            else:
                iterator = _iter_vector_lines_from_gz(url)
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
            # 组装矩阵:PAD=0 全零;未命中词(含 UNK)随机初始化。
            # 关键:OOV 行的标准差取【预训练命中行自身的整体标准差】而非固定 0.01——
            # 若 OOV 向量幅值远小于预训练向量,训练/微调时这些词(如合并出的"不满意")
            # 的输入信号近零,LSTM 学不到其语义;等量级初始化后微调可正常学习。
            hit_vectors = np.stack(list(found.values()))
            oov_std = float(hit_vectors.std())
            print(f"[wordvec] 预训练命中行标准差={oov_std:.4f}(OOV 行按同量级初始化)")
            matrix = rng.normal(0.0, oov_std, size=(len(vocab), dim)).astype(np.float32)
            matrix[PAD_IDX_LOCAL] = 0.0
            for idx, vec in found.items():
                matrix[idx] = vec
            # OOV 组合式初始化:预处理阶段会把 否定词+紧邻词 合并成词(如 不满意),
            # 这类词多为 OOV。对其可分解部分 否定词+X 且 X 在预训练词表内的,
            # 用 vec(否定词)+vec(X) 作为初始向量(组合语义,fastText 风格):
            # 所有 不X 词共享 vec(不) 分量,为分类器提供一致的"否定方向"特征,
            # 解决稀有 OOV 合并词在低学习率下学不动的问题。
            n_compositional = 0
            word_by_idx = {i: w for w, i in vocab.items()}
            for idx in range(len(vocab)):
                w = word_by_idx[idx]
                if idx in found or len(w) < 2:
                    continue
                for neg in ("不", "没", "别"):
                    # 守卫:否定词本身可能不在词表(如被停用词过滤/词频低),
                    # 此时无法组合初始化,跳过(修复:旧代码直接 vocab[neg] 触发 KeyError)
                    neg_idx = vocab.get(neg)
                    if neg_idx is None:
                        continue
                    if w.startswith(neg) and len(w) > len(neg) and vocab.get(w[len(neg):]) in found:
                        base_idx = vocab[w[len(neg):]]
                        matrix[idx] = matrix[neg_idx] + matrix[base_idx]
                        n_compositional += 1
                        break
            if n_compositional:
                print(f"[wordvec] OOV 组合式初始化:{n_compositional} 个 否定+X 合并词"
                      f"用 vec(否定词)+vec(X) 初始化")
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
