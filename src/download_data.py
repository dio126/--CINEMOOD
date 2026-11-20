# -*- coding: utf-8 -*-
"""
数据集获取脚本
================
获取 ChnSentiCorp 中文情感分类语料(约 9600+ 条,酒店/书籍/电脑评论,正负二分类标注)。

获取策略(逐级降级,保证可重复运行):
  1. 首选 HuggingFace datasets 在线加载(候选多个 Hub 数据集名,逐个尝试);
  2. 直连失败则设置 HF_ENDPOINT=https://hf-mirror.com 走镜像重试;
  3. 仍失败则从 GitHub 备份源(ChineseNlpCorpus 仓库)下载 CSV,
     读取时做编码容错(utf-8 失败回退 gbk)。

同时下载哈工大停用词表(GitHub goto456/stopwords 仓库 raw 文件),
失败则使用本文件内置的精简中文停用词列表兜底(会在文件头注明来源)。

所有产物保存到 data/raw/,文件已存在则跳过下载(脚本幂等)。
"""

import io
import os
import urllib.request

# 项目根目录(src/ 的上一级),兼容任意工作目录启动
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# 合并后的原始数据文件(列为 label,text)
RAW_CSV = os.path.join(RAW_DIR, "chnsenticorp.csv")
# 停用词文件
STOPWORDS_FILE = os.path.join(RAW_DIR, "hit_stopwords.txt")

# HuggingFace Hub 上的候选数据集名(逐个尝试,均含 text/label 字段)
HF_DATASET_CANDIDATES = ["lansinuote/ChnSentiCorp", "dirtycomputer/ChnSentiCorp"]

# GitHub 备份源(ChineseNlpCorpus 的 ChnSentiCorp 酒店评论全量 CSV,utf-8 编码)
GITHUB_BACKUP_URLS = [
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/"
    "ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
    "https://cdn.jsdelivr.net/gh/SophonPlus/ChineseNlpCorpus@master/datasets/"
    "ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
]

# 哈工大停用词表下载地址(goto456/stopwords 仓库 raw)
STOPWORDS_URLS = [
    "https://raw.githubusercontent.com/goto456/stopwords/master/stopwords/hit_stopwords.txt",
    "https://cdn.jsdelivr.net/gh/goto456/stopwords@master/stopwords/hit_stopwords.txt",
]

# 内置精简中文停用词兜底列表(仅当在线下载全部失败时使用)
BUILTIN_STOPWORDS = """的 了 和 是 就 都 而 及 与 着 或 一个 没有 我们 你们 他们 它们 这个 那个 这些 那些
这样 那样 之 也在 很 有 在 就 就要 要 还 还是 又 也 但 但是 因为 所以 如果 虽然 而且 并且 或者
不 不是 没什么 挺 很 非常 十分 特别 更 最 太 挺 较 比较 些 有点 有些 一点儿 一点 被 把 让 使 使得
对 对于 关于 按 按照 根据 通过 随着 作为 成为 变成 是 不是 于是 然后 接着 最后 首先 其次一直 已经
曾经 正在 将要 刚刚 才 只 只是 仅 仅仅 一般 一样 一下 一直 一直 一齐 一起 上面 下面 前面 后面
里面 外面 什么 怎么 怎样 如何 为什么 哪 哪个 哪些 哪里 谁 多少 几 第 每 各 某 吧 吗 呢 啊 呀 哦
哈 嗯 呵 唉 哇 么 嘛 啦 咯 咯 表示 觉得 感觉 知道 认为 以为 以及 等等 之类 一下 一些 一切 一直
不仅 不但 不过 只是 但是 可是 然而 却 由 由于 因此 之内 之后 之后 之前 之外 之间 之中 上面
""".split()


def _read_csv_bytes_with_encoding_fallback(data: bytes):
    """对 CSV 字节流做编码容错:先按 utf-8 读,失败回退 gbk(老版本数据为 GBK 编码)。"""
    import pandas as pd

    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # 最后手段:忽略非法字符
    return pd.read_csv(io.BytesIO(data), encoding="utf-8", encoding_errors="ignore")


def _download_bytes(url: str, timeout: int = 120) -> bytes:
    """下载 URL 内容并返回字节流(带 UA,避免部分站点拦截脚本请求)。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _try_huggingface(use_mirror: bool):
    """尝试通过 HuggingFace datasets 加载 ChnSentiCorp,返回合并后的 DataFrame 或 None。"""
    if use_mirror:
        # 走 hf-mirror 镜像(在国内直连 HF 经常失败)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    import datasets  # 延迟导入,避免未安装时影响其他功能

    for name in HF_DATASET_CANDIDATES:
        try:
            print(f"[download_data] 尝试 HuggingFace{'(镜像)' if use_mirror else ''} 加载 {name} ...")
            ds = datasets.load_dataset(name)
            frames = []
            # 只合并官方 train + dev(validation),官方 test 不参与,后续按 8:1:1 重新划分
            for split in ("train", "validation", "dev"):
                if split in ds:
                    frames.append(ds[split].to_pandas())
            if not frames:
                continue
            df = __import__("pandas").concat(frames, ignore_index=True)
            # 统一列名:只要 text / label 两列
            df = df[["text", "label"]].dropna().reset_index(drop=True)
            print(f"[download_data] HF 加载成功,合并 train+dev 共 {len(df)} 条")
            return df
        except Exception as e:  # noqa: BLE001 任何网络/解析异常都降级到下一候选
            print(f"[download_data] {name} 加载失败: {type(e).__name__}: {e}")
    return None


def _try_github_backup():
    """从 GitHub 备份源下载 ChnSentiCorp CSV,返回 DataFrame 或 None。"""
    import pandas as pd

    for url in GITHUB_BACKUP_URLS:
        try:
            print(f"[download_data] 尝试 GitHub 备份源: {url}")
            data = _download_bytes(url, timeout=300)
            df = _read_csv_bytes_with_encoding_fallback(data)
            if "text" not in df.columns or "label" not in df.columns:
                print("[download_data] 备份源列名不符合预期,跳过")
                continue
            df = df[["text", "label"]].dropna().reset_index(drop=True)
            print(f"[download_data] GitHub 备份源下载成功,共 {len(df)} 条")
            return df
        except Exception as e:  # noqa: BLE001
            print(f"[download_data] 备份源失败: {type(e).__name__}: {e}")
    return None


def download_corpus() -> str:
    """下载 ChnSentiCorp 语料并保存为 data/raw/chnsenticorp.csv,返回文件路径。"""
    if os.path.exists(RAW_CSV):
        print(f"[download_data] {RAW_CSV} 已存在,跳过下载")
        return RAW_CSV

    df = None
    # 1) HF 直连 → 2) HF 镜像 → 3) GitHub 备份源
    df = _try_huggingface(use_mirror=False)
    if df is None:
        df = _try_huggingface(use_mirror=True)
    if df is None:
        df = _try_github_backup()
    if df is None:
        raise RuntimeError(
            "所有数据源均下载失败,请检查网络后重试,"
            "或手动将 ChnSentiCorp 数据(text,label 两列 CSV)放到 " + RAW_CSV
        )

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"[download_data] 已保存 {len(df)} 条到 {RAW_CSV}")
    print(f"[download_data] 标签分布:\n{df['label'].value_counts()}")
    return RAW_CSV


def download_stopwords() -> str:
    """下载哈工大停用词表;全部失败时写入口兜底精简表并注明。"""
    if os.path.exists(STOPWORDS_FILE):
        print(f"[download_data] {STOPWORDS_FILE} 已存在,跳过下载")
        return STOPWORDS_FILE

    for url in STOPWORDS_URLS:
        try:
            print(f"[download_data] 下载停用词表: {url}")
            data = _download_bytes(url, timeout=60)
            text = None
            for enc in ("utf-8", "gbk"):
                try:
                    text = data.decode(enc)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            if text and len(text.split()) > 100:  # 简单校验:哈工大表应有数千词
                with open(STOPWORDS_FILE, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[download_data] 停用词表已保存({len(text.split())} 词)")
                return STOPWORDS_FILE
        except Exception as e:  # noqa: BLE001
            print(f"[download_data] 停用词下载失败: {type(e).__name__}: {e}")

    # 兜底:内置精简停用词,文件头注明来源与原因
    with open(STOPWORDS_FILE, "w", encoding="utf-8") as f:
        f.write("# 兜底精简中文停用词表(在线下载哈工大停用词表失败时代码内置,来源:本文件内置列表)\n")
        f.write("\n".join(BUILTIN_STOPWORDS))
    print(f"[download_data] 在线停用词表下载失败,已使用内置精简停用词({len(BUILTIN_STOPWORDS)} 词)")
    return STOPWORDS_FILE


if __name__ == "__main__":
    # 主入口:先下载数据集,再下载停用词表(两者均可重复运行,已存在则跳过)
    download_corpus()
    download_stopwords()
