# -*- coding: utf-8 -*-
"""
五级分类数据集获取:豆瓣电影短评 dmsc(带 1-5 星)
==================================================
获取策略(总尝试时间预算 ≤15 分钟,逐级降级,如实记录最终来源):
  1. HF 镜像 BerlinWang/DMSC(DMSC.csv.zip,约 150MB,豆瓣电影短评全量,
     含星级字段)——任务书指定的 XiangPan/dmsc 经查询不存在(Repository
     not found),改用镜像上实际存在的 DMSC 仓库;
  2. 失败则尝试 HF 镜像 Leoooooops/douban_movie_comments;
  3. 再失败则尝试 Hieuman/douban_reviews(zh parquet);
  4. 全部失败则如实报错退出,等待人工指示(禁止用无星级数据凑数)。

处理管线:
  - 列名自适应:自动探测文本列(comment/text/评论)与星级列(star/rating/评分);
    兼容数字星级(1-5)与豆瓣文字评级(力荐/推荐/还行/较差/很差);
  - 星级映射:1★=非常负面(0)、2★=负面(1)、3★=中性(2)、4★=正面(3)、
    5★=非常正面(4);
  - 同文本去重(与二分类同一规则:同文本标签一致保留 1 条,标签冲突整组丢弃);
  - 分层抽样:每类目标 10000 条(共 5 万,控制训练时长),不足的类按最小类
    实际数量对齐,并打印采样前后的真实类别分布。

产物:data/raw/dmsc_sampled.csv(text, star, label 三列),原始 zip 留作
磁盘缓存(data/raw/DMSC.csv.zip)以便复现。脚本幂等,已存在则跳过。
"""

import io
import json
import os
import time
import urllib.request
import zipfile

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

SAMPLED_CSV = os.path.join(RAW_DIR, "dmsc_sampled.csv")   # 最终采样结果
SOURCE_LOG = os.path.join(RAW_DIR, "dmsc_source.json")     # 来源与分布记录
SEED = 42
PER_CLASS_TARGET = 10000          # 每类目标采样数(共 5 万)
TIME_BUDGET_SECONDS = 15 * 60     # 总尝试时间预算

# 候选源:(名称, HF 仓库, 文件名, 格式)
CANDIDATES = [
    ("BerlinWang/DMSC", "DMSC.csv.zip", "csv_zip"),
    ("Leoooooops/douban_movie_comments", "new_data.json", "json"),
    ("Hieuman/douban_reviews", "data/zh-00000-of-00010.parquet", "parquet"),
]

# 豆瓣文字评级 → 数字星级(部分数据集星级以中文词给出)
WORD_STAR_MAP = {"力荐": 5, "推荐": 4, "还行": 3, "较差": 2, "很差": 1}
# 文本列 / 星级列的候选列名(按优先级)
TEXT_COLS = ["comment", "text", "content", "评论", "short_comment"]
STAR_COLS = ["star", "rating", "score", "评分", "星级"]


def _hf_url(repo: str, fname: str) -> str:
    """构造 hf-mirror 直链。"""
    return f"https://hf-mirror.com/datasets/{repo}/resolve/main/{fname}"


def _download_to_file(url: str, path: str, deadline: float):
    """流式下载到磁盘(256KB 分块,不整读内存;超预算抛 TimeoutError)。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    total = 0
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as f:
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (20 * 1024 * 1024) < len(chunk):
                print(f"[dmsc] 已下载 {total // 1048576}MB ...")
                if time.time() > deadline:
                    raise TimeoutError("超过下载时间预算")


def _detect_columns(df: pd.DataFrame):
    """自动探测文本列与星级列,返回 (text_col, star_col) 或 (None, None)。"""
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[c] for c in TEXT_COLS if c in cols), None)
    star_col = next((cols[c] for c in STAR_COLS if c in cols), None)
    return text_col, star_col


def _normalize_star(series: pd.Series) -> pd.Series:
    """星级归一化为 1~5 整数:兼容数字与中文文字评级,非法值置 NaN。"""
    s = series.astype(str).str.strip()
    s = s.map(WORD_STAR_MAP).fillna(s)  # 文字评级先转数字
    return pd.to_numeric(s, errors="coerce")


def _load_candidate(name: str, fname: str, fmt: str, cache_dir: str, deadline: float):
    """下载并加载一个候选数据集,返回 DataFrame(text, star)或 None。"""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        print(f"[dmsc] 尝试候选:{name} ({fname})")
        url = _hf_url(name, fname)
        if fmt == "csv_zip":
            zip_path = os.path.join(cache_dir, os.path.basename(fname))
            if not os.path.exists(zip_path):  # zip 留作磁盘缓存,便于复现
                _download_to_file(url, zip_path, deadline)
            # 直接从 zip 读内嵌 CSV(假设单一 csv 成员)
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f)
        else:
            local = os.path.join(cache_dir, os.path.basename(fname))
            if not os.path.exists(local):
                _download_to_file(url, local, deadline)
            df = pd.read_json(local, lines=True) if fmt == "json" else pd.read_parquet(local)

        text_col, star_col = _detect_columns(df)
        if text_col is None or star_col is None:
            print(f"[dmsc] {name} 未找到文本/星级列(实际列:{list(df.columns)[:12]})")
            return None
        star = _normalize_star(df[star_col])
        valid = star.isin([1, 2, 3, 4, 5]) & df[text_col].notna()
        n_valid = int(valid.sum())
        if n_valid < 5000:  # 星级信息不足,视为不可用
            print(f"[dmsc] {name} 有效星级样本过少({n_valid})")
            return None
        out = pd.DataFrame({"text": df.loc[valid, text_col].astype(str), "star": star[valid].astype(int)})
        print(f"[dmsc] {name} 加载成功:{n_valid} 条有效星级样本")
        print(f"[dmsc] 星级原始分布:{out['star'].value_counts().sort_index().to_dict()}")
        return out
    except Exception as e:  # noqa: BLE001 任何网络/解析失败降级到下一候选
        print(f"[dmsc] {name} 失败:{type(e).__name__}: {e}")
        return None


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """同文本去重(与二分类同一规则):同文本标签一致保留 1 条,冲突整组丢弃。"""
    n_before = len(df)
    label_nunique = df.groupby("text")["star"].transform("nunique")
    conflict_texts = set(df.loc[label_nunique > 1, "text"])
    n_conflict = int(df["text"].isin(conflict_texts).sum())
    df = df[(label_nunique == 1) & ~df["text"].isin(conflict_texts)]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    print(f"[dmsc] 去重:{n_before} → {len(df)}(丢弃标签冲突 {n_conflict} 条,重复保留 1 条)")
    return df


def _stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """分层抽样:每类 PER_CLASS_TARGET 条;不足的类按最小类数量对齐(并如实记录)。"""
    counts = df["star"].value_counts()
    per_class = min(PER_CLASS_TARGET, int(counts.min()))
    if per_class < PER_CLASS_TARGET:
        print(f"[dmsc] 注意:最小类仅 {counts.min()} 条,各类采样数对齐为 {per_class}")
    # 不用 groupby.apply(pandas 3 默认排除分组列),显式逐组采样后拼接
    parts = [g.sample(n=min(per_class, len(g)), random_state=SEED)
             for _, g in df.groupby("star", sort=True)]
    sampled = pd.concat(parts, ignore_index=True)
    print(f"[dmsc] 采样后分布:{sampled['star'].value_counts().sort_index().to_dict()}")
    return sampled


def main():
    if os.path.exists(SAMPLED_CSV):
        print(f"[dmsc] {SAMPLED_CSV} 已存在,跳过(如需重建请先删除)")
        return

    deadline = time.time() + TIME_BUDGET_SECONDS
    df = None
    source = None
    for repo, fname, fmt in CANDIDATES:
        if time.time() > deadline:
            print("[dmsc] 下载时间预算已耗尽")
            break
        df = _load_candidate(repo, fname, fmt, RAW_DIR, deadline)
        if df is not None:
            source = f"{repo}/{fname}"
            break
    if df is None:
        raise RuntimeError(
            "所有 dmsc 候选数据集均获取失败——如实报告并停止,请检查网络后重试,"
            "或人工指定带 1-5 星级的中文短评数据集(禁止用无星级数据凑数)"
        )

    df = _dedup(df)
    df = _stratified_sample(df)
    # 星级 → 五级标签:1★=非常负面(0) … 5★=非常正面(4)
    df["label"] = df["star"] - 1
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # 打乱存储
    df.to_csv(SAMPLED_CSV, index=False, encoding="utf-8")
    with open(SOURCE_LOG, "w", encoding="utf-8") as f:
        json.dump({
            "source": source, "rows": len(df),
            "distribution": df["label"].value_counts().sort_index().to_dict(),
            "per_class_target": PER_CLASS_TARGET, "seed": SEED,
        }, f, ensure_ascii=False, indent=2)
    print(f"[dmsc] 已保存 {len(df)} 条到 {SAMPLED_CSV}(来源:{source})")
    print(f"[dmsc] 五级标签分布(0非常负面~4非常正面):{df['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
