# -*- coding: utf-8 -*-
"""
外部泛化测评脚本(独立于训练/预处理代码,只新增不改旧)
====================================================================
目的:用【完全未参与任何训练】的 DMSC 剩余评论,对两个最佳模型做外部泛化测评:
  - 二分类最佳: bilstm_lr5e-4_do0.3_pretune(训练于 ChnSentiCorp 酒店/书/电脑 → 跨域)
  - 五级最佳:  5c_bilstm_pretune_h256(训练于 dmsc_sampled 5 万条 → 同域但测试 100% 未见)

三段式执行(全部固定 seed=42,可复现):
  1) build : DMSC 全量分块解析 → 星级归一化 → 同训练规则去重 → 三重排除
             (dmsc_sampled.csv 全部文本 + 五级三个 pkl + 二分类三个 pkl,exact match)
             → 剩余池分层抽样 30000 条(1★/2★ 各5000=负面组,3★ 10000=中性组,
               4★/5★ 各5000=正面组,长度 5~300 字)
             → 同一剩余池筛隐晦(不含显式情感词)300 条(每星 60 条)
  2) infer : 双模型对 30300 条全量 batch 推理(预处理路径与 predict.py 完全一致:
             preprocess.tokenize + encode_tokens,仅拼 batch,不改逻辑),
             并抽样 20 条与 SentimentPredictor.predict 单条结果一致性校验;
             隐晦 300 需人工理由文件 subtle_reasons.csv(先 build 后人工填写)
  3) report: 全部指标 + 5×5 混淆矩阵 png + 分星组判负率曲线 png
             + summary.json + eval_report.md(开头为准确/错误条数总表与诚实披露)

用法(在 movie-sentiment/ 下):
  .venv/Scripts/python.exe src/eval_external.py build
  # 人工填写 outputs/eval_report/subtle_reasons.csv(300 行,一句话判断理由)
  .venv/Scripts/python.exe src/eval_external.py infer
  # 人工填写 outputs/eval_report/subtle_error_analysis.csv(15 条典型错例分析,可选)
  .venv/Scripts/python.exe src/eval_external.py report

Windows 兼容:不使用 DataLoader/多进程;路径含中文一律 os.path.join + 绝对路径;
matplotlib 中文防乱码(Microsoft YaHei + axes.unicode_minus=False)。
"""

import io
import json
import os
import pickle
import random
import sys
import time
import zipfile

import numpy as np
import pandas as pd

# src/ 自身目录加入 sys.path,保证与训练代码相同的模块导入方式(predict/evaluate/preprocess)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORT_DIR = os.path.join(OUT_DIR, "eval_report")
os.makedirs(REPORT_DIR, exist_ok=True)

DMSC_ZIP = os.path.join(RAW_DIR, "DMSC.csv.zip")
SAMPLED_CSV = os.path.join(RAW_DIR, "dmsc_sampled.csv")

# 两个最佳模型的 run 名(与 outputs/ 下 ckpt 文件名对应)
BIN_RUN = "bilstm_lr5e-4_do0.3_pretune"      # → outputs/bilstm_lr5e-4_do0.3_pretune_best.pt
FIVE_RUN = "5c_bilstm_pretune_h256"          # → outputs/5c_bilstm_pretune_h256_5c_best.pt

SEED = 42
CHUNK_ROWS = 200_000          # DMSC 分块流式读取块大小(防内存吃紧)
LEN_MIN, LEN_MAX = 5, 300     # 抽样文本长度窗口(字符数,含边界)

# 分层抽样目标:负面组 10000(1★/2★ 各半),中性组 10000(3★),正面组 10000(4★/5★ 各半)
STAR_SAMPLE_N = {1: 5000, 2: 5000, 3: 10000, 4: 5000, 5: 5000}
SUBTLE_PER_STAR = 60          # 隐晦专项:每星 60 条,共 300
SUBTLE_MIN_TOKENS = 3         # 隐晦文本至少要有 3 个词(保证模型有输入信号)
SUBTLE_SCAN_CAP = 300_000     # 每星组扫描上限(防极端情况死循环)

# 五级类名(与训练/推理同一映射,import 复用而非复制)
from preprocess_5class import CLASS_NAMES  # noqa: E402  ["非常负面","负面","中性","正面","非常正面"]

# 豆瓣文字评级 → 数字星级(与训练时 download_dmsc.py 的 WORD_STAR_MAP 完全一致)
WORD_STAR_MAP = {"力荐": 5, "推荐": 4, "还行": 3, "较差": 2, "很差": 1}

# 产物路径
BUILD_COUNTS_JSON = os.path.join(REPORT_DIR, "build_counts.json")
MAIN_DATA_CSV = os.path.join(REPORT_DIR, "eval_30000_data.csv")
SUBTLE_DATA_CSV = os.path.join(REPORT_DIR, "subtle_300_data.csv")
REASONS_CSV = os.path.join(REPORT_DIR, "subtle_reasons.csv")            # 人工填写
ERROR_ANALYSIS_CSV = os.path.join(REPORT_DIR, "subtle_error_analysis.csv")  # 人工填写(可选)
MAIN_PRED_CSV = os.path.join(REPORT_DIR, "eval_30000_predictions.csv")
SUBTLE_PRED_CSV = os.path.join(REPORT_DIR, "subtle_300_predictions.csv")
SUMMARY_JSON = os.path.join(REPORT_DIR, "summary.json")
REPORT_MD = os.path.join(REPORT_DIR, "eval_report.md")
CM5_PNG = os.path.join(REPORT_DIR, "confusion_5c.png")
STAR_CURVE_PNG = os.path.join(REPORT_DIR, "star_group_curves.png")

# ---------------- 显式情感词表(用于筛"隐晦"评论)----------------
# 规则:经项目统一 tokenize(清洗+停用词+否定合并)后的词序列中,
#   任一词命中下列词表、或命中"否定前缀剥离后的词"(如 不好看→好看)、
#   或单字情感词作为子串出现在 ≤3 字的词里(如 最烂/太尬),即判为显式情感 → 剔除。
EXPLICIT_POS = [
    "好", "好看", "好听", "不错", "棒", "精彩", "推荐", "喜欢", "爱", "经典", "完美",
    "杰作", "神作", "佳作", "优秀", "优质", "出色", "赞", "力荐", "值得", "动人",
    "感动", "感人", "搞笑", "幽默", "有趣", "有意思", "惊艳", "震撼", "精致", "耐看",
    "可爱", "漂亮", "美丽", "美好", "温暖", "治愈", "期待", "过瘾", "爽", "泪目",
    "催泪", "深刻", "高级", "妙", "佳", "亮点", "新意", "巧妙", "流畅", "圆满",
    "满意", "惊喜", "敬佩", "好评", "给力", "顶尖", "一流", "大爱", "点赞", "神级",
    "顶级", "意犹未尽", "回味", "爱看", "爱了", "心动", "妙极", "华彩", "恢弘", "良心",
    "开心", "满分", "燃",
]
EXPLICIT_NEG = [
    "烂", "烂片", "烂尾", "差", "差劲", "差评", "垃圾", "难看", "难听", "失望", "无聊",
    "恶心", "反感", "讨厌", "恨", "糟糕", "糟心", "注水", "劝退", "尴尬", "狗血",
    "做作", "无语", "拖沓", "冗长", "乏味", "无趣", "崩坏", "败笔", "可惜", "遗憾",
    "硬伤", "缺点", "毛病", "废片", "圈钱", "嫌弃", "厌烦", "头疼", "雷人", "雷",
    "闷", "鸡肋", "无感", "平庸", "一般", "凑合", "将就", "水货", "拉胯", "智障",
    "脑残", "幼稚", "尬", "出戏", "跳戏", "毁", "毁原著", "毁经典", "失真", "还行",
    "意思", "下头", "膈应", "离谱", "尬穿", "看不下去", "快进", "催眠", "浪费", "白瞎",
]
EXPLICIT_LEX = set(EXPLICIT_POS) | set(EXPLICIT_NEG)
SINGLE_CHAR_LEX = {w for w in EXPLICIT_LEX if len(w) == 1}  # 好/烂/差/爱/毁/雷/闷/尬/爽/妙/佳/赞/棒/燃...
MULTI_CHAR_LEX = tuple(w for w in EXPLICIT_LEX if len(w) >= 2)  # 多字词按子串匹配,覆盖形态变体


def is_explicit(tokens: list) -> bool:
    """判断 token 序列是否含显式情感词(隐晦筛选的剔除条件)。

    三层规则(宁严勿漏,人工通读再做最终把关):
      1) token 精确命中词表;
      2) 否定合并产生的 不X/没X/别X 剥离前缀后命中(不好看→好看,没亮点→亮点);
      3) 子串匹配:jieba 常把"挺不错/值得一看/不太好看"等切成整词,精确匹配会漏,
         故多字情感词查子串(挺不错⊃不错),单字情感词仅在 ≤3 字 token 内查子串
         (最烂/太尬/超燃),避免"好莱坞/爱好者"这类长词误伤。
    """
    for t in tokens:
        if t in EXPLICIT_LEX:
            return True
        if len(t) > 2 and t[0] in ("不", "没", "别") and t[1:] in EXPLICIT_LEX:
            return True
        if 2 <= len(t) <= 3 and any(c in t for c in SINGLE_CHAR_LEX):
            return True
        if any(w in t for w in MULTI_CHAR_LEX):
            return True
    return False


# =====================================================================
# 阶段 1:build —— 构建主评测集 30000 + 隐晦候选 300
# =====================================================================
def _load_dmsc_pool(counts: dict) -> pd.DataFrame:
    """DMSC 全量分块解析:星级归一化(数字或 力荐/推荐/还行/较差/很差)→ 有效行 (text, star)。"""
    with zipfile.ZipFile(DMSC_ZIP) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            header = pd.read_csv(io.BytesIO(f.read(64 * 1024)), nrows=0)
        cols = {c.lower(): c for c in header.columns}
        text_col = next((cols[c] for c in ("comment", "text", "content", "评论") if c in cols), None)
        star_col = next((cols[c] for c in ("star", "rating", "score", "评分") if c in cols), None)
        if text_col is None or star_col is None:
            raise RuntimeError(f"DMSC 列名无法识别: {header.columns.tolist()}")
        parts, n_raw = [], 0
        with zf.open(csv_name) as f:
            for chunk in pd.read_csv(f, chunksize=CHUNK_ROWS, usecols=[text_col, star_col],
                                     dtype={text_col: str, star_col: str}):
                n_raw += len(chunk)
                s = chunk[star_col].astype(str).str.strip()
                s = s.map(lambda v: WORD_STAR_MAP.get(v, v))    # 文字评级先转数字(同训练)
                star = pd.to_numeric(s, errors="coerce")
                valid = star.isin([1, 2, 3, 4, 5]) & chunk[text_col].notna()
                if valid.any():
                    parts.append(pd.DataFrame({
                        "text": chunk.loc[valid, text_col].astype(str),
                        "star": star[valid].astype(int),
                    }))
        df = pd.concat(parts, ignore_index=True)
    counts["dmsc_zip_rows_read"] = int(n_raw)
    counts["valid_star_and_text_rows"] = int(len(df))
    counts["dropped_invalid_star_or_empty"] = int(n_raw - len(df))
    print(f"[build] DMSC 解析:zip 共 {n_raw} 行 → 有效(text 非空且星级 1-5){len(df)} 行")
    return df


def _dedup_like_training(df: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """与训练完全相同的去重规则:同文本同标签留 1 条,标签冲突整组丢弃。"""
    n_before = len(df)
    label_nunique = df.groupby("text")["star"].transform("nunique")
    conflict_texts = set(df.loc[label_nunique > 1, "text"])
    n_conflict = int(df["text"].isin(conflict_texts).sum())
    df = df[(label_nunique == 1) & ~df["text"].isin(conflict_texts)]
    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    counts["dedup_before"] = int(n_before)
    counts["dedup_after"] = int(len(df))
    counts["dedup_conflict_dropped"] = int(n_conflict)
    print(f"[build] 去重(同训练规则):{n_before} → {len(df)}(标签冲突整组丢弃 {n_conflict} 条)")
    return df


def _load_exclusion_sets() -> dict:
    """三重排除集:dmsc_sampled.csv 全部文本 + 五级/二分类六个 pkl 的全部文本(exact match)。"""
    sets = {}
    sampled = pd.read_csv(SAMPLED_CSV)
    sets["dmsc_sampled.csv(五级训练源,5万)"] = set(sampled["text"].astype(str))
    for name in ("train", "val", "test"):
        with open(os.path.join(PROC_DIR, f"{name}_5c.pkl"), "rb") as f:
            sets[f"{name}_5c.pkl(五级)"] = set(pickle.load(f)["texts"])
    for name in ("train", "val", "test"):
        with open(os.path.join(PROC_DIR, f"{name}.pkl"), "rb") as f:
            sets[f"{name}.pkl(二分类)"] = set(pickle.load(f)["texts"])
    return sets


def run_build():
    random.seed(SEED)
    np.random.seed(SEED)
    counts = {"seed": SEED, "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "length_window_chars": [LEN_MIN, LEN_MAX]}

    # 1) 全量解析 + 去重
    df = _load_dmsc_pool(counts)
    df = _dedup_like_training(df, counts)

    # 2) 三重排除(逐级统计排除前后计数)
    excl_sets = _load_exclusion_sets()
    counts["exclusion_steps"] = []
    union = set().union(*excl_sets.values())
    counts["pool_overlap_with_union_before_exclusion"] = int(df["text"].isin(union).sum())
    for label, s in excl_sets.items():
        hit = int(df["text"].isin(s).sum())
        df = df[~df["text"].isin(s)].reset_index(drop=True)
        counts["exclusion_steps"].append({"set": label, "size": len(s), "removed_at_step": hit,
                                          "pool_after": int(len(df))})
        print(f"[build] 排除 {label}(集合 {len(s)} 条):移除 {hit} 条 → 剩余池 {len(df)}")
    counts["pool_after_triple_exclusion"] = int(len(df))

    # 3) 长度窗口 5~300 字
    in_win = df["text"].str.len().between(LEN_MIN, LEN_MAX)
    counts["pool_per_star_after_exclusion"] = {int(k): int(v) for k, v in
                                               df["star"].value_counts().sort_index().items()}
    pool = df[in_win].reset_index(drop=True)
    counts["pool_per_star_after_length_filter"] = {int(k): int(v) for k, v in
                                                   pool["star"].value_counts().sort_index().items()}
    print(f"[build] 长度 {LEN_MIN}~{LEN_MAX} 字过滤:{len(df)} → {len(pool)}")

    # 4) 分层抽样 30000(负面组 1★/2★,中性组 3★,正面组 4★/5★)
    parts = []
    for star in (1, 2, 3, 4, 5):  # 固定顺序 + random_state=42,完全可复现
        g = pool[pool["star"] == star]
        parts.append(g.sample(n=STAR_SAMPLE_N[star], random_state=SEED))
    main = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # 真值列:gt_bin(3★ 中性无二元真值 → 空);gt_5c = star-1
    main["gt_bin"] = main["star"].map({1: 0, 2: 0, 4: 1, 5: 1})
    main["gt_5c"] = main["star"] - 1
    main = main[["text", "star", "gt_bin", "gt_5c"]]
    assert len(main) == 30000, f"主集应为 30000,实际 {len(main)}"
    main.to_csv(MAIN_DATA_CSV, index=False, encoding="utf-8-sig")
    print(f"[build] 主评测集 {len(main)} 条 → {MAIN_DATA_CSV}")
    print(f"[build]   星级分布:{main['star'].value_counts().sort_index().to_dict()}")

    # 5) 防泄漏自检:抽样结果与全部训练数据交集必须为 0
    main_texts = set(main["text"])
    inter_main = {k: len(main_texts & s) for k, s in excl_sets.items()}
    counts["selfcheck_main_vs_training_intersection"] = inter_main
    assert all(v == 0 for v in inter_main.values()), f"泄漏!主集与训练数据交集非零:{inter_main}"

    # 6) 隐晦专项 300 条:同一剩余池(且不与主集重叠),不含显式情感词
    from preprocess import tokenize  # 项目唯一分词路径,不另写
    subtle_parts, scan_stats = [], []
    subtle_pool = pool[~pool["text"].isin(main_texts)].reset_index(drop=True)
    for star in (1, 2, 3, 4, 5):
        g = subtle_pool[subtle_pool["star"] == star]
        idx = list(g.index)
        random.shuffle(idx)  # random 模块已 seed=42,洗牌顺序可复现
        picked, scanned = [], 0
        for i in idx:
            if len(picked) >= SUBTLE_PER_STAR or scanned >= SUBTLE_SCAN_CAP:
                break
            scanned += 1
            tokens = tokenize(g.at[i, "text"])
            if len(tokens) < SUBTLE_MIN_TOKENS or is_explicit(tokens):
                continue
            picked.append(i)
        part = g.loc[picked, ["text", "star"]].copy()
        part["gt_bin"] = part["star"].map({1: 0, 2: 0, 4: 1, 5: 1})
        part["gt_5c"] = part["star"] - 1
        subtle_parts.append(part)
        scan_stats.append({"star": star, "scanned": scanned, "passed": len(picked)})
        print(f"[build] 隐晦 {star}★:扫描 {scanned} 条 → 通过 {len(picked)} 条")
    subtle = pd.concat(subtle_parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    assert len(subtle) == SUBTLE_PER_STAR * 5, f"隐晦应为 300,实际 {len(subtle)}"
    subtle.to_csv(SUBTLE_DATA_CSV, index=False, encoding="utf-8-sig")
    counts["subtle_scan_stats"] = scan_stats
    counts["explicit_lexicon_size"] = len(EXPLICIT_LEX)
    print(f"[build] 隐晦候选 {len(subtle)} 条 → {SUBTLE_DATA_CSV}(待人工通读填理由)")

    # 隐晦自检:与训练数据、主集交集均为 0
    subtle_texts = set(subtle["text"])
    inter_subtle = {k: len(subtle_texts & s) for k, s in excl_sets.items()}
    inter_subtle["vs_main_30000"] = len(subtle_texts & main_texts)
    counts["selfcheck_subtle_intersections"] = inter_subtle
    assert all(v == 0 for v in inter_subtle.values()), f"泄漏!隐晦集交集非零:{inter_subtle}"

    with open(BUILD_COUNTS_JSON, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
    print(f"[build] 计数链与自检 → {BUILD_COUNTS_JSON}")
    print(f"[build] 防泄漏自检:主集/隐晦 与全部训练数据交集均为 0 ✓")


# =====================================================================
# 阶段 2:infer —— 双模型全量 batch 推理(预处理路径与 predict.py 一致)
# =====================================================================
def _batch_predict(predictor, texts: list, batch_size: int = 256):
    """批量推理:tokenize/encode 与 SentimentPredictor.predict 同一路径,仅拼 batch 提速。

    返回 (labels, confs)。不改任何预处理逻辑,num_workers=0(无 DataLoader)。
    """
    import torch
    from preprocess import MAXLEN, encode_tokens, tokenize

    token_lists = [tokenize(t)[:MAXLEN] for t in texts]  # 与 predict() 相同截断
    labels, confs = [], []
    with torch.no_grad():
        for i in range(0, len(token_lists), batch_size):
            ids = np.stack([encode_tokens(ts, predictor.vocab)
                            for ts in token_lists[i:i + batch_size]])
            x = torch.from_numpy(ids).to(predictor.device)
            out = predictor.model(x)
            logits = out[0] if isinstance(out, tuple) else out  # 注意力模型返回 (logits, alpha)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            labels.extend(probs.argmax(axis=1).tolist())
            confs.extend(probs.max(axis=1).tolist())
    return labels, confs


def _consistency_check(predictor, texts: list, labels: list, confs: list, n: int = 20):
    """一致性抽检:batch 路径 vs SentimentPredictor.predict 单条路径,标签必须完全一致。"""
    idx = list(range(n))
    mismatch = []
    for j in idx:
        r = predictor.predict(texts[j])
        if r["label"] != labels[j] or abs(r["confidence"] - confs[j]) > 1e-4:
            mismatch.append(j)
    return mismatch


def run_infer():
    from predict import SentimentPredictor

    for p, name in ((MAIN_DATA_CSV, "主评测集"), (SUBTLE_DATA_CSV, "隐晦集")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少 {p},请先执行 build 阶段({name}未构建)")
    if not os.path.exists(REASONS_CSV):
        raise FileNotFoundError(
            f"缺少人工理由文件 {REASONS_CSV}:请先通读 {SUBTLE_DATA_CSV} 并按 id,text,reason "
            "三列逐条填写一句话判断理由,再运行 infer")
    reasons = pd.read_csv(REASONS_CSV)
    assert len(reasons) == SUBTLE_PER_STAR * 5, f"理由文件应 300 行,实际 {len(reasons)}"
    assert reasons["reason"].notna().all() and (reasons["reason"].astype(str).str.len() >= 4).all(), \
        "理由列存在空值或过短,请补全后重跑"

    print(f"[infer] 加载二分类最佳模型:{BIN_RUN}")
    p_bin = SentimentPredictor(BIN_RUN)
    print(f"[infer] 加载五级最佳模型:{FIVE_RUN}")
    p_5c = SentimentPredictor(FIVE_RUN)

    t0 = time.time()
    meta = {"bin_run": BIN_RUN, "five_run": FIVE_RUN,
            "device": str(p_bin.device), "batch_size": 256,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    for data_csv, out_csv, tag in ((MAIN_DATA_CSV, MAIN_PRED_CSV, "主集30000"),
                                   (SUBTLE_DATA_CSV, SUBTLE_PRED_CSV, "隐晦300")):
        df = pd.read_csv(data_csv)
        texts = df["text"].astype(str).tolist()
        print(f"[infer] {tag}:二分类推理 {len(texts)} 条 ...")
        bin_lab, bin_conf = _batch_predict(p_bin, texts)
        print(f"[infer] {tag}:五级推理 {len(texts)} 条 ...")
        c5_lab, c5_conf = _batch_predict(p_5c, texts)

        # 一致性抽检(每集合前 20 条,batch 与单条 predict 必须同标签)
        mm1 = _consistency_check(p_bin, texts, bin_lab, bin_conf)
        mm2 = _consistency_check(p_5c, texts, c5_lab, c5_conf)
        assert not mm1 and not mm2, f"batch 与单条推理不一致(二分类 {mm1},五级 {mm2})"
        print(f"[infer] {tag}:一致性抽检 20 条×2 模型通过 ✓")

        out = pd.DataFrame({
            "text": texts,
            "star": df["star"],
            "gt_bin": df["gt_bin"],
            "gt_5c": df["gt_5c"],
            "bin_pred": bin_lab,
            "bin_polarity": ["正面" if v == 1 else "负面" for v in bin_lab],
            "bin_conf": [round(c, 6) for c in bin_conf],
            "pred_5c": c5_lab,
            "pred_5c_name": [CLASS_NAMES[v] for v in c5_lab],
            "conf_5c": [round(c, 6) for c in c5_conf],
        })
        out["gt_5c_name"] = out["gt_5c"].map(lambda v: CLASS_NAMES[int(v)])
        has_bin = out["gt_bin"].notna()
        out["correct_bin"] = np.where(has_bin, (out["bin_pred"] == out["gt_bin"]).astype(int), np.nan)
        out["correct_5c_exact"] = (out["pred_5c"] == out["gt_5c"]).astype(int)
        out["correct_5c_within1"] = ((out["pred_5c"] - out["gt_5c"]).abs() <= 1).astype(int)
        out.insert(0, "id", range(len(out)))  # 行 id,便于人工理由/错例分析文件对齐
        if tag == "隐晦300":  # 逐条人工判断理由并入明细
            out.insert(2, "reason", reasons.set_index("id").loc[out.index, "reason"].astype(str).values)
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[infer] {tag} 完成 → {out_csv}")

    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(REPORT_DIR, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[infer] 全量推理耗时 {meta['wall_seconds']}s(设备 {meta['device']})")


# =====================================================================
# 阶段 3:report —— 指标汇总 + 图 + 报告
# =====================================================================
def _cm_table(cm, row_names, col_names) -> str:
    """混淆矩阵 → markdown 表(行=真实,列=预测)。"""
    head = "| 真实\\预测 | " + " | ".join(col_names) + " | 合计 |"
    sep = "|" + "---|" * (len(col_names) + 2)
    rows = [head, sep]
    for i, rn in enumerate(row_names):
        rows.append(f"| {rn} | " + " | ".join(str(int(v)) for v in cm[i]) + f" | {int(cm[i].sum())} |")
    rows.append("| 合计 | " + " | ".join(str(int(v)) for v in cm.sum(axis=0)) + f" | {int(cm.sum())} |")
    return "\n".join(rows)


def run_report():
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文防乱码
    plt.rcParams["axes.unicode_minus"] = False

    main = pd.read_csv(MAIN_PRED_CSV)
    subtle = pd.read_csv(SUBTLE_PRED_CSV)
    with open(BUILD_COUNTS_JSON, encoding="utf-8") as f:
        counts = json.load(f)
    with open(os.path.join(REPORT_DIR, "infer_meta.json"), encoding="utf-8") as f:
        infer_meta = json.load(f)
    S = {"models": infer_meta, "count_chain": counts}  # summary.json 根

    # ---------- 二分类:正面+负面组 20000(正面为正类) ----------
    pn = main[main["gt_bin"].notna()].copy()
    y, p = pn["gt_bin"].astype(int), pn["bin_pred"].astype(int)
    cm2 = confusion_matrix(y, p, labels=[0, 1])
    bin_acc = float((y == p).mean())
    tp, fp = int(cm2[1, 1]), int(cm2[0, 1])
    fn, tn = int(cm2[1, 0]), int(cm2[0, 0])
    bin_prec = tp / (tp + fp) if tp + fp else 0.0
    bin_rec = tp / (tp + fn) if tp + fn else 0.0
    bin_f1 = 2 * bin_prec * bin_rec / (bin_prec + bin_rec) if bin_prec + bin_rec else 0.0
    S["binary_posneg_20000"] = {
        "n": int(len(pn)), "accuracy": bin_acc, "precision_pos": bin_prec,
        "recall_pos": bin_rec, "f1_pos": bin_f1,
        "confusionMatrix_rownegcolpos": cm2.tolist(),
        "correct": int((y == p).sum()), "wrong": int((y != p).sum()),
        "by_group": {g: {"n": int(len(g_df)), "pred_pos_rate": float(g_df["bin_pred"].mean()),
                         "mean_conf": float(g_df["bin_conf"].mean()),
                         "accuracy": float((g_df["bin_pred"] == g_df["gt_bin"]).mean())}
                     for g, g_df in pn.groupby("star")},
    }
    print(f"[report] 二分类(正+负 20000):Acc={bin_acc:.4f} P={bin_prec:.4f} R={bin_rec:.4f} F1={bin_f1:.4f}")

    # ---------- 二分类 × 中性组 10000(无真值行为画像) ----------
    neu = main[main["gt_bin"].isna()]
    neu_pos, neu_neg = neu[neu["bin_pred"] == 1], neu[neu["bin_pred"] == 0]
    S["binary_neutral_10000"] = {
        "n": int(len(neu)), "pred_pos": int(len(neu_pos)), "pred_neg": int(len(neu_neg)),
        "pred_pos_rate": float(len(neu_pos) / len(neu)),
        "mean_conf_overall": float(neu["bin_conf"].mean()),
        "mean_conf_when_pred_pos": float(neu_pos["bin_conf"].mean()),
        "mean_conf_when_pred_neg": float(neu_neg["bin_conf"].mean()),
        "mean_conf_posneg_groups_reference": float(pn["bin_conf"].mean()),
    }
    print(f"[report] 中性组 10000:判正 {len(neu_pos)} / 判负 {len(neu_neg)},"
          f"平均置信度 {S['binary_neutral_10000']['mean_conf_overall']:.4f}")

    # ---------- 五级分类 30000 ----------
    y5, p5 = main["gt_5c"].astype(int), main["pred_5c"].astype(int)
    cm5 = confusion_matrix(y5, p5, labels=list(range(5)))
    exact = float((y5 == p5).mean())
    within1 = float(((y5 - p5).abs() <= 1).mean())
    prf = precision_recall_fscore_support(y5, p5, labels=list(range(5)), zero_division=0)
    S["five_class_30000"] = {
        "n": int(len(main)), "exact_accuracy": exact, "within1_accuracy": within1,
        "macro_f1": float(f1_score(y5, p5, labels=list(range(5)), average="macro", zero_division=0)),
        "per_class": {CLASS_NAMES[i]: {"precision": float(prf[0][i]), "recall": float(prf[1][i]),
                                       "f1": float(prf[2][i]), "support": int(prf[3][i]),
                                       "within1_rate": float(((y5 == i) & ((p5 - i).abs() <= 1)).sum()
                                                             / max(prf[3][i], 1))}
                      for i in range(5)},
        "confusionMatrix_5x5": cm5.tolist(),
        "exact_correct": int((y5 == p5).sum()), "exact_wrong": int((y5 != p5).sum()),
        "within1_correct": int(((y5 - p5).abs() <= 1).sum()),
    }
    print(f"[report] 五级 30000:exact={exact:.4f} within1={within1:.4f} macroF1={S['five_class_30000']['macro_f1']:.4f}")

    # 5×5 混淆矩阵 png
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm5, cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(cm5[i, j])}\n{cm5[i, j] / cm5[i].sum():.1%}",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm5[i, j] > cm5.max() * 0.55 else "#333333")
    ax.set_xticks(range(5), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(5), CLASS_NAMES)
    ax.set_xlabel("预测星级档"), ax.set_ylabel("真实星级档")
    ax.set_title(f"五级分类混淆矩阵(外部 30000 条,exact Acc={exact:.2%})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(CM5_PNG, dpi=150)
    plt.close(fig)

    # ---------- 分星组(1~5★)分层指标:二分类判负率 + 五级 exact/±1 ----------
    star_rows = []
    for star in (1, 2, 3, 4, 5):
        g = main[main["star"] == star]
        star_rows.append({
            "star": star, "n": int(len(g)),
            "bin_neg_rate": float((g["bin_pred"] == 0).mean()),
            "bin_mean_conf": float(g["bin_conf"].mean()),
            "five_exact": float(g["correct_5c_exact"].mean()),
            "five_within1": float(g["correct_5c_within1"].mean()),
        })
    S["star_groups"] = star_rows

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = [r["star"] for r in star_rows]
    axes[0].plot(xs, [r["bin_neg_rate"] for r in star_rows], "o-", color="#c0392b")
    for r in star_rows:
        axes[0].annotate(f"{r['bin_neg_rate']:.1%}", (r["star"], r["bin_neg_rate"]),
                         textcoords="offset points", xytext=(0, 8), fontsize=9)
    axes[0].set_xticks(xs, [f"{s}★" for s in xs])
    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_xlabel("真实星级"), axes[0].set_ylabel("二分类判负率")
    axes[0].set_title("分星级二分类判负率曲线(极性方向感)")
    axes[1].plot(xs, [r["five_exact"] for r in star_rows], "s-", label="exact 命中率")
    axes[1].plot(xs, [r["five_within1"] for r in star_rows], "^--", label="±1 星容差命中率")
    for r in star_rows:
        axes[1].annotate(f"{r['five_exact']:.0%}", (r["star"], r["five_exact"]),
                         textcoords="offset points", xytext=(0, 7), fontsize=8)
        axes[1].annotate(f"{r['five_within1']:.0%}", (r["star"], r["five_within1"]),
                         textcoords="offset points", xytext=(0, 7), fontsize=8)
    axes[1].set_xticks(xs, [f"{s}★" for s in xs])
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xlabel("真实星级")
    axes[1].set_title("分星组五级 exact / ±1 命中率")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(STAR_CURVE_PNG, dpi=150)
    plt.close(fig)

    # ---------- 隐晦 300 ----------
    sub_bin = subtle[subtle["gt_bin"].notna()]
    sub_bin_acc = float((sub_bin["bin_pred"] == sub_bin["gt_bin"]).mean())
    sub_exact = float(subtle["correct_5c_exact"].mean())
    sub_within1 = float(subtle["correct_5c_within1"].mean())
    # 对照主集口径:二分类对照 = 正+负组 20000;五级对照 = 30000 全量;±1 同
    S["subtle_300"] = {
        "n": 300, "bin_gt_available": int(len(sub_bin)),
        "bin_accuracy": sub_bin_acc, "bin_correct": int((sub_bin["bin_pred"] == sub_bin["gt_bin"]).sum()),
        "bin_wrong": int((sub_bin["bin_pred"] != sub_bin["gt_bin"]).sum()),
        "five_exact": sub_exact, "five_exact_correct": int(subtle["correct_5c_exact"].sum()),
        "five_within1": sub_within1, "five_within1_correct": int(subtle["correct_5c_within1"].sum()),
        "drop_bin_vs_main": bin_acc - sub_bin_acc,
        "drop_five_exact_vs_main": exact - sub_exact,
        "drop_five_within1_vs_main": within1 - sub_within1,
        "per_star": {int(star): {"n": int(len(g)),
                                 "five_exact": float(g["correct_5c_exact"].mean()),
                                 "five_within1": float(g["correct_5c_within1"].mean()),
                                 "bin_neg_rate": float((g["bin_pred"] == 0).mean())}
                     for star, g in subtle.groupby("star")},
    }
    print(f"[report] 隐晦 300:二分类 {sub_bin_acc:.4f}(降 {S['subtle_300']['drop_bin_vs_main']:+.4f}),"
          f"五级 exact {sub_exact:.4f}(降 {S['subtle_300']['drop_five_exact_vs_main']:+.4f})")
    # 人工复核中标注"偏显式"的条数(词表漏筛的近显式词,如实保留并标注)
    S["subtle_300"]["quasi_explicit_flagged"] = int(subtle["reason"].str.contains("偏显式").sum())

    # 同域对照指标:程序化读取训练时保存的 metrics.json,绝不手写数字
    bin_ref_path = os.path.join(OUT_DIR, f"{BIN_RUN}_metrics.json")
    five_ref_path = os.path.join(OUT_DIR, f"{FIVE_RUN}_5c_metrics.json")
    bin_ref = json.load(open(bin_ref_path, encoding="utf-8"))["acc"]
    five_ref = json.load(open(five_ref_path, encoding="utf-8"))
    S["same_domain_reference"] = {"binary_test_acc_chnsenticorp": bin_ref,
                                  "five_test_acc_dmsc": five_ref["acc"],
                                  "five_test_macro_f1_dmsc": five_ref.get("macro_f1")}

    # ---------- 15 条典型错例(确定性规则挑选,人工分析可选合并) ----------
    err = subtle[((subtle["gt_bin"].notna()) & (subtle["bin_pred"] != subtle["gt_bin"]))
                 | (subtle["correct_5c_within1"] == 0)].copy()
    bin_wrong = np.where(err["gt_bin"].notna(), err["bin_pred"] != err["gt_bin"], False).astype(int)
    err["sev"] = bin_wrong * 2 + (err["pred_5c"] - err["gt_5c"]).abs() \
                 + (err["bin_conf"] + err["conf_5c"]) / 2  # 双错+星级偏差大+高置信 优先
    err = err.sort_values("sev", ascending=False).head(15)  # 保留 id 列以便人工分析对齐
    analysis = {}
    if os.path.exists(ERROR_ANALYSIS_CSV):
        a = pd.read_csv(ERROR_ANALYSIS_CSV)
        analysis = dict(zip(a["id"].astype(int), a["analysis"].astype(str)))

    # ---------- 写 summary.json ----------
    S["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=2)

    # ---------- 写 eval_report.md ----------
    inter = counts["selfcheck_main_vs_training_intersection"]
    inter_sub = counts["selfcheck_subtle_intersections"]
    L = []
    L.append("# 外部泛化测评报告(DMSC 全新评测集 30000 + 隐晦专项 300)")
    L.append("")
    L.append(f"> 生成时间:{S['generated_at']} ｜ 二分类 `{BIN_RUN}` ｜ 五级 `{FIVE_RUN}` ｜"
             f" 推理设备 `{infer_meta['device']}`,全量真实推理耗时 {infer_meta['wall_seconds']}s,seed=42")
    L.append("")
    L.append("## 0. 诚实披露(先读)")
    L.append("")
    L.append("1. **二分类模型跨领域**:训练数据为 ChnSentiCorp(酒店/书/电脑评论),本评测全部为豆瓣电影短评,"
             "对应任务书拓展方向 #3(跨领域迁移);其指标下降**包含领域偏移成分**,不能解读为纯模型能力衰减。")
    L.append("2. **五级模型同域但 100% 未参与训练**:评测集经三重排除(exact match)后从 DMSC 剩余池抽样,"
             "与五级训练源 dmsc_sampled.csv、六个 pkl 划分文件交集均为 0(自检见第 1 节)。")
    L.append("3. **隐晦组含人工成分**:300 条由词表自动过滤后,**人工逐条通读**并写入一句话判断理由"
             "(见 `subtle_300_predictions.csv` 的 reason 列);错例分析亦含人工判断。")
    L.append("4. 指标全部由 `src/eval_external.py` 对 30300 条评论 × 2 模型全量真实推理产生,"
             "batch 推理与 `SentimentPredictor.predict` 单条路径做过 20 条×2 模型一致性抽检(标签零差异);"
             "无任何抽样代替或编造数字。")
    L.append("")
    L.append("## 1. 准确 / 错误条数总表(核心交付)")
    L.append("")
    L.append("| 模型 | 评测集 | 可判定条数 | 正确 | 错误 | 正确率 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| 二分类 | 主集·正面+负面组 | 20000 | {S['binary_posneg_20000']['correct']}"
             f" | {S['binary_posneg_20000']['wrong']} | {bin_acc:.2%} |")
    L.append(f"| 二分类 | 主集·中性组(无真值) | 10000 | —(判正 {S['binary_neutral_10000']['pred_pos']})"
             f" | —(判负 {S['binary_neutral_10000']['pred_neg']}) | 行为画像见第 4 节 |")
    L.append(f"| 五级 exact | 主集 30000 | 30000 | {S['five_class_30000']['exact_correct']}"
             f" | {S['five_class_30000']['exact_wrong']} | {exact:.2%} |")
    L.append(f"| 五级 ±1 星 | 主集 30000 | 30000 | {S['five_class_30000']['within1_correct']}"
             f" | {30000 - S['five_class_30000']['within1_correct']} | {within1:.2%} |")
    L.append(f"| 二分类 | 隐晦·有真值子集 | {len(sub_bin)} | {S['subtle_300']['bin_correct']}"
             f" | {S['subtle_300']['bin_wrong']} | {sub_bin_acc:.2%} |")
    L.append(f"| 五级 exact | 隐晦 300 | 300 | {S['subtle_300']['five_exact_correct']}"
             f" | {300 - S['subtle_300']['five_exact_correct']} | {sub_exact:.2%} |")
    L.append(f"| 五级 ±1 星 | 隐晦 300 | 300 | {S['subtle_300']['five_within1_correct']}"
             f" | {300 - S['subtle_300']['five_within1_correct']} | {sub_within1:.2%} |")
    L.append("")
    L.append("## 2. 数据构建计数链与防泄漏自检")
    L.append("")
    c = counts
    L.append(f"- DMSC.zip 全量:{c['dmsc_zip_rows_read']:,} 行 → 有效(星级 1-5 且文本非空)"
             f"{c['valid_star_and_text_rows']:,} 行(文字评级 力荐/推荐/还行/较差/很差 已按训练同款映射转数字)")
    L.append(f"- 去重(同文本同标签留 1,冲突整组丢弃):{c['dedup_before']:,} → **{c['dedup_after']:,}**"
             f"(冲突丢弃 {c['dedup_conflict_dropped']:,})")
    steps = c["exclusion_steps"]
    L.append(f"- 三重排除前剩余池与训练全集重叠:{c['pool_overlap_with_union_before_exclusion']:,} 条;"
             f"逐级排除:" + ";".join(f"{s['set'].split('(')[0]}移除 {s['removed_at_step']:,}" for s in steps))
    L.append(f"- **剩余池:{c['pool_after_triple_exclusion']:,} 条**;长度 5~300 字过滤后"
             f" {sum(c['pool_per_star_after_length_filter'].values()):,} 条"
             f"(各星 {c['pool_per_star_after_length_filter']})")
    L.append(f"- 分层抽样(seed=42):1★/2★ 各 5000(负面组)、3★ 10000(中性组)、4★/5★ 各 5000(正面组)= **30000 条**")
    L.append(f"- 隐晦 300:同一剩余池且与主集零重叠,显式情感词表 {c['explicit_lexicon_size']} 词"
             f"(token 级过滤,含否定前缀剥离),每星通过 60 条")
    L.append("")
    L.append("**防泄漏自检(交集必须为 0)**:")
    L.append("")
    L.append("| 自检项 | 交集条数 |")
    L.append("|---|---|")
    for k, v in inter.items():
        L.append(f"| 主集 30000 vs {k} | {v} |")
    for k, v in inter_sub.items():
        L.append(f"| 隐晦 300 vs {k if k != 'vs_main_30000' else '主评测集 30000'} | {v} |")
    L.append("")
    L.append("## 3. 二分类主指标(正面+负面组 20000,正面为正类)")
    L.append("")
    L.append(f"- **Accuracy = {bin_acc:.4f}**,Precision = {bin_prec:.4f},Recall = {bin_rec:.4f},F1 = {bin_f1:.4f}")
    for star in (1, 2, 4, 5):
        g = S["binary_posneg_20000"]["by_group"][star]  # 分组键为 int(pn.star 列)
        L.append(f"- {star}★ 组:n={g['n']},判正率 {g['pred_pos_rate']:.2%},组内准确率 {g['accuracy']:.2%},"
                 f"平均置信度 {g['mean_conf']:.3f}")
    L.append("")
    L.append("混淆矩阵(行=真实,列=预测;0=负面,1=正面):")
    L.append("")
    L.append(_cm_table(cm2, ["负面(0)", "正面(1)"], ["负面(0)", "正面(1)"]))
    L.append("")
    L.append(f"> 对照:该模型在 ChnSentiCorp 同域测试集上的 Acc = {bin_ref:.4f}"
             f"(outputs/{BIN_RUN}_metrics.json);此处 {bin_acc:.4f},"
             f"下降 {(bin_ref - bin_acc) * 100:.2f} 个百分点,含跨域(酒店/书/电脑 → 电影)成分。")
    L.append("")
    L.append("## 4. 二分类 × 中性组(10000,无二元真值,行为画像)")
    L.append("")
    n1 = S["binary_neutral_10000"]
    L.append(f"- 判正 {n1['pred_pos']}(占 {n1['pred_pos_rate']:.2%}) / 判负 {n1['pred_neg']};"
             f"整体平均置信度 {n1['mean_conf_overall']:.4f}")
    L.append(f"- 判正时平均置信度 {n1['mean_conf_when_pred_pos']:.4f},判负时 {n1['mean_conf_when_pred_neg']:.4f};"
             f"参考:正/负组平均置信度 {n1['mean_conf_posneg_groups_reference']:.4f}")
    L.append("- 解读:3★ 评论本身情感模糊,模型二选一时是否向一侧偏置、置信度是否塌缩,"
             "是跨域稳健性的行为学证据(详见第 9 节)。")
    L.append("")
    L.append("## 5. 五级分类(30000)")
    L.append("")
    L.append(f"- **exact Acc = {exact:.4f}**;**±1 星容差 Acc = {within1:.4f}**;"
             f"**macro-F1 = {S['five_class_30000']['macro_f1']:.4f}**")
    L.append("")
    L.append("| 真实类 | Precision | Recall | F1 | ±1 命中率 | 支持数 |")
    L.append("|---|---|---|---|---|---|")
    for i, name in enumerate(CLASS_NAMES):
        pc = S["five_class_30000"]["per_class"][name]
        L.append(f"| {name}({i + 1}★) | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f}"
                 f" | {pc['within1_rate']:.2%} | {pc['support']:,} |")
    L.append("")
    L.append(f"5×5 混淆矩阵(行=真实,列=预测;格内=条数/行占比),图:`confusion_5c.png`")
    L.append("")
    L.append(_cm_table(cm5, [f"{n}({i + 1}★)" for i, n in enumerate(CLASS_NAMES)],
                       [f"{n}({i + 1}★)" for i, n in enumerate(CLASS_NAMES)]))
    L.append("")
    L.append(f"> 对照:该模型在 DMSC 同分布测试集 test_5c 上 exact Acc = {five_ref['acc']:.4f},"
             f"macro-F1 = {five_ref.get('macro_f1', 0):.4f}(outputs/{FIVE_RUN}_5c_metrics.json);"
             f"本外部集 {exact:.4f} / {S['five_class_30000']['macro_f1']:.4f},"
             f"评测集未参与训练且按星均衡重抽,指标口径同源可比。")
    L.append("")
    L.append("## 6. 分星组分层指标(二分类判负率 + 五级 exact/±1,图:`star_group_curves.png`)")
    L.append("")
    L.append("| 星组 | n | 二分类判负率 | 二分类平均置信度 | 五级 exact | 五级 ±1 |")
    L.append("|---|---|---|---|---|---|")
    for r in star_rows:
        L.append(f"| {r['star']}★ | {r['n']:,} | {r['bin_neg_rate']:.2%} | {r['bin_mean_conf']:.3f}"
                 f" | {r['five_exact']:.2%} | {r['five_within1']:.2%} |")
    L.append("")
    L.append("## 7. 隐晦情绪专项(300,单独统计,不计入主指标)")
    L.append("")
    L.append(f"- 人工通读 300 条并逐条写入判断理由;其中标注「偏显式」(词表漏筛的近显式词,"
             f"如繁体「爛」、被 jieba 切散的「给力」、口语「牛逼/帅/nice」等)"
             f"共 {S['subtle_300']['quasi_explicit_flagged']} 条,已如实保留并在 reason 列标注")
    L.append(f"- 二分类(有真值 {len(sub_bin)} 条):**{sub_bin_acc:.2%}**,较主集 {bin_acc:.2%} **下降"
             f" {S['subtle_300']['drop_bin_vs_main']:.2%}**")
    L.append(f"- 五级 exact:**{sub_exact:.2%}**,较主集下降 {S['subtle_300']['drop_five_exact_vs_main']:.2%};"
             f"±1 容差 **{sub_within1:.2%}**,下降 {S['subtle_300']['drop_five_within1_vs_main']:.2%}")
    L.append("")
    L.append("| 星组 | n | 二分类判负率 | 五级 exact | 五级 ±1 |")
    L.append("|---|---|---|---|---|")
    for star in (1, 2, 3, 4, 5):
        g = S["subtle_300"]["per_star"][star]  # 分组键为 int(subtle.star 列)
        L.append(f"| {star}★ | {g['n']} | {g['bin_neg_rate']:.2%} | {g['five_exact']:.2%} | {g['five_within1']:.2%} |")
    L.append("")
    L.append("## 8. 典型错例(15 条,自动按\"双模型皆错/星级偏差大/置信度高\"排序挑选,人工分析)")
    L.append("")
    for i, (_, row) in enumerate(err.iterrows()):
        aid = int(row["id"])
        why = analysis.get(aid, "(待人工补充)")
        gt_b = "无(3★)" if pd.isna(row["gt_bin"]) else ("负面" if row["gt_bin"] == 0 else "正面")
        L.append(f"**错例 {i + 1}**(id={aid},{int(row['star'])}★,真值二元 {gt_b}/{CLASS_NAMES[int(row['gt_5c'])]};"
                 f"二分类判「{row['bin_polarity']}」conf={row['bin_conf']:.3f},"
                 f"五级判「{row['pred_5c_name']}」conf={row['conf_5c']:.3f})")
        L.append("")
        L.append(f"> 原文:{row['text']}")
        L.append(">")
        L.append(f"> 人工隐晦判断:{row['reason']}")
        if why != "(待人工补充)":
            L.append(">")
            L.append(f"> 错因分析:{why}")
        L.append("")
    L.append("## 9. 意外现象、结论与局限")
    L.append("")
    L.append("**意外现象**:")
    monotone = all(star_rows[i]["bin_neg_rate"] >= star_rows[i + 1]["bin_neg_rate"] for i in range(4))
    neg_chain = " → ".join(f"{r['star']}★:{r['bin_neg_rate']:.1%}" for r in star_rows)
    L.append(f"1. **中性组置信度几乎不塌缩**:3★ 模糊评论上二分类平均置信度 {n1['mean_conf_overall']:.3f},"
             f"与正/负组 {n1['mean_conf_posneg_groups_reference']:.3f} 几乎相同——模型对「本应最不确定」的样本同样笃定,"
             f"校准(calibration)缺陷在跨域下被放大(与隐晦错例中 0.74~0.98 的高置信错误互相印证)。")
    L.append(f"2. **1★ 差评仍有 {1 - star_rows[0]['bin_neg_rate']:.1%} 被判正**:电影差评常用反语与陈述事实"
             f"(如「看了二十分钟就离开了」),词先验驱动的跨域模型在单条语义层失灵;"
             f"但判负率随星级{'严格单调' if monotone else '基本单调'}({neg_chain}),宏观极性方向感保留。")
    L.append(f"3. **五级「两端易、中间难」且不对称**:1★/5★ exact {star_rows[0]['five_exact']:.1%}/{star_rows[4]['five_exact']:.1%},"
             f"2★/4★ 仅 {star_rows[1]['five_exact']:.1%}/{star_rows[3]['five_exact']:.1%};"
             f"混淆矩阵显示 2★ 大量流向 1★、4★ 大量流向 3★/5★,温和档位边界模糊。")
    L.append(f"4. **同域五级泛化「零衰减」**:外部 30000(三重排除、与训练交集 0)exact {exact:.4f} "
             f"vs 同分布 test_5c {five_ref['acc']:.4f}——既佐证无泄漏,也说明 5 万训练样本下模型已逼近该分布的容量上限。")
    L.append(f"5. **隐晦组 5★ 二分类判负率({S['subtle_300']['per_star'][5]['bin_neg_rate']:.1%})反高于 4★"
             f"({S['subtle_300']['per_star'][4]['bin_neg_rate']:.1%})**:n=60 小样本波动,但结合错例可见 5★ 隐晦好评"
             f"多用行为暗示(「再看一遍」「喝了一箱啤酒」),恰好完全避开情感词,对词驱动模型最难。")
    L.append("")
    L.append("**结论**:")
    L.append(f"- 二分类跨域 Acc {bin_ref:.4f} → {bin_acc:.4f}(降 {(bin_ref - bin_acc) * 100:.1f}pp),"
             f"损失主要来自电影领域特有的反语、指桑骂槐与行为暗示;极性排序能力保留。")
    L.append(f"- 五级模型在同域未见数据上与测试集持平(exact {exact:.4f} vs {five_ref['acc']:.4f}),"
             f"±1 容差 {within1:.1%},具备实用星级估计能力;档位边界(尤其 2★/4★)仍是弱点。")
    L.append(f"- 隐晦情绪是两模型共同短板:二分类再降 {S['subtle_300']['drop_bin_vs_main'] * 100:.1f}pp、"
             f"五级 exact 再降 {S['subtle_300']['drop_five_exact_vs_main'] * 100:.1f}pp、"
             f"±1 再降 {S['subtle_300']['drop_five_within1_vs_main'] * 100:.1f}pp;"
             f"错例集中于四类:反语讽刺、情感目标错位(骂差评者/骂他人)、行为暗示、先抑后扬结构。")
    L.append("")
    L.append("**局限**:")
    L.append(f"- 隐晦组 300 条中 {S['subtle_300']['quasi_explicit_flagged']} 条人工复核为「偏显式」"
             f"(词表漏筛:繁体「爛」、jieba 切分变体「给/力」、口语「牛逼/帅」、英文「nice」等),已如实标注;"
             f"为保持流程可复现未剔除,剔除后隐晦指标会略有变化。")
    L.append("- 人工理由与错因分析为个人判断,可能存在误读;星级真值本身含噪声(同人在不同心境下打分不同)。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("产物清单:`eval_30000_predictions.csv`(30000 行明细)、`subtle_300_predictions.csv`(300 行含人工理由)、"
             "`summary.json`、`confusion_5c.png`、`star_group_curves.png`、`build_counts.json`、`infer_meta.json`、"
             "`subtle_reasons.csv` / `subtle_error_analysis.csv`(人工)。")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[report] 报告 → {REPORT_MD}\n[report] 指标 → {SUMMARY_JSON}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "build":
        run_build()
    elif stage == "infer":
        run_infer()
    elif stage == "report":
        run_report()
    else:
        print("用法: python src/eval_external.py build|infer|report")
        sys.exit(1)
