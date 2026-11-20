# -*- coding: utf-8 -*-
"""
Streamlit 情感分析 Web 应用(三页面 · 深色影院风)
==================================================
页面 1 情感分析:输入任意中文评论 → 极性结论横幅 + 置信度 + 注意力关键词高亮;
页面 2 实验仪表盘:指标总览卡、训练曲线、混淆矩阵、超参对比总表、模型结构;
页面 3 批量分析:上传 csv 或粘贴多行评论 → 批量预测 + 正负占比条 + 结果下载。

视觉设计:深色影院风(深蓝黑背景 + 金色点缀,契合电影评论主题),
主题色由 .streamlit/config.toml 定义,细节样式由下方 CUSTOM_CSS 覆盖。
推理逻辑与原版完全一致:src/predict.py → 与训练共用 src/preprocess.py。

启动:
  .venv/Scripts/python.exe -m streamlit run app/app.py --server.headless true
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

# 保证能导入 src/ 下的模块(以 app/app.py 启动时 src 不在 sys.path)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ---------- 页面全局配置 ----------
st.set_page_config(
    page_title="中文情感分析系统", page_icon="🎬", layout="wide", initial_sidebar_state="expanded"
)

# ================== 深色影院风:全局自定义样式 ==================
# 说明:仅做视觉覆盖,不改变任何交互逻辑;离线可用(无外部字体/CDN 依赖)
CUSTOM_CSS = """
<style>
/* ===== 背景与侧边栏 ===== */
.stApp {
    background: linear-gradient(160deg, #0b0f19 0%, #101828 55%, #0b0f19 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0b0f19 100%);
    border-right: 1px solid rgba(245, 197, 24, 0.12);
}
h1, h2, h3 { color: #f2f4f8; letter-spacing: 0.5px; }

/* ===== 金色主按钮 / 圆角按钮 ===== */
.stButton > button {
    border-radius: 999px !important;
    font-weight: 700;
    border: 1px solid rgba(245, 197, 24, 0.55) !important;
    color: #f5c518 !important;
    background: rgba(245, 197, 24, 0.06) !important;
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(245, 197, 24, 0.35);
    background: rgba(245, 197, 24, 0.14) !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #f5c518 0%, #e2a80c 100%) !important;
    color: #1a1500 !important;
    box-shadow: 0 4px 18px rgba(245, 197, 24, 0.35);
}

/* ===== 英雄区标题 ===== */
.hero {
    padding: 26px 30px 22px;
    border-radius: 18px;
    background:
        radial-gradient(1200px 200px at 15% -40%, rgba(245,197,24,0.16), transparent 60%),
        linear-gradient(135deg, rgba(20,27,45,0.92), rgba(13,18,32,0.92));
    border: 1px solid rgba(245, 197, 24, 0.22);
    margin-bottom: 18px;
}
.hero-title {
    font-size: 34px; font-weight: 800; letter-spacing: 1px;
    background: linear-gradient(100deg, #ffe082 0%, #f5c518 45%, #ffb300 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    margin: 0;
}
.hero-sub { color: #9aa4b8; margin-top: 8px; font-size: 15px; }

/* ===== 通用卡片 ===== */
.cine-card {
    background: rgba(20, 27, 45, 0.82);
    border: 1px solid rgba(245, 197, 24, 0.14);
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

/* ===== 结论横幅(正/负) ===== */
.verdict {
    display: flex; align-items: center; gap: 20px;
    border-radius: 18px; padding: 20px 26px; margin: 6px 0 14px;
}
.verdict.pos {
    background: linear-gradient(135deg, rgba(76, 175, 80, 0.18), rgba(76, 175, 80, 0.05));
    border: 1px solid rgba(102, 187, 106, 0.5);
}
.verdict.neg {
    background: linear-gradient(135deg, rgba(239, 83, 80, 0.18), rgba(239, 83, 80, 0.05));
    border: 1px solid rgba(239, 130, 128, 0.5);
}
.v-emoji { font-size: 52px; line-height: 1; }
.v-label { font-size: 26px; font-weight: 800; letter-spacing: 1px; }
.verdict.pos .v-label { color: #7bd88f; }
.verdict.neg .v-label { color: #ff8a87; }
.v-sub { color: #9aa4b8; font-size: 14px; margin-top: 4px; }
.v-pct {
    margin-left: auto; font-size: 40px; font-weight: 800;
    font-family: Consolas, "Courier New", monospace;
}
.verdict.pos .v-pct { color: #7bd88f; }
.verdict.neg .v-pct { color: #ff8a87; }

/* ===== 置信度进度条(金色渐变填充) ===== */
.conf-bar {
    height: 14px; border-radius: 999px; overflow: hidden;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(245, 197, 24, 0.2);
}
.conf-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #b8860b, #f5c518 70%, #ffe082);
    box-shadow: 0 0 12px rgba(245, 197, 24, 0.55);
    transition: width 0.6s ease;
}

/* ===== 注意力词高亮(金色深浅 = 权重大小) ===== */
.attn-word {
    display: inline-block; padding: 3px 7px; margin: 3px;
    border-radius: 6px; font-size: 17px; color: #f2f4f8;
}
.attn-chip {
    display: inline-block; padding: 4px 12px; margin: 3px;
    border-radius: 999px; font-size: 14px; font-weight: 700;
    background: rgba(245, 197, 24, 0.12); color: #f5c518;
    border: 1px solid rgba(245, 197, 24, 0.35);
}

/* ===== 指标卡 / expander / tabs ===== */
[data-testid="stMetric"] {
    background: rgba(20, 27, 45, 0.72);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-family: Consolas, "Courier New", monospace; }
[data-testid="stExpander"] {
    border: 1px solid rgba(245, 197, 24, 0.15) !important;
    border-radius: 14px !important;
    background: rgba(16, 22, 38, 0.6);
}
.stTabs [data-baseweb="tab"] { color: #9aa4b8 !important; }
.stTabs [aria-selected="true"] { color: #f5c518 !important; font-weight: 700; }

/* ===== 输入框 ===== */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(10, 14, 24, 0.85) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(245, 197, 24, 0.28) !important;
    color: #eef1f7 !important;
}

/* ===== 页脚 ===== */
.page-footer {
    margin-top: 28px; padding-top: 14px;
    border-top: 1px solid rgba(245, 197, 24, 0.15);
    color: #6b7488; font-size: 12.5px; text-align: center;
}

/* ===== 批量页正负占比条 ===== */
.ratio-bar { display: flex; height: 26px; border-radius: 999px; overflow: hidden; }
.ratio-pos { background: linear-gradient(90deg, #2e7d32, #66bb6a); display: flex; align-items: center; justify-content: center; color: #06130a; font-weight: 800; font-size: 13px; }
.ratio-neg { background: linear-gradient(90deg, #ef5350, #c62828); display: flex; align-items: center; justify-content: center; color: #1a0505; font-weight: 800; font-size: 13px; }
</style>
"""


def inject_css():
    """注入全局自定义样式(每个页面渲染前调用一次即可)。"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title_html: str, sub_html: str):
    """页面英雄区:金色渐变标题 + 副标题说明。"""
    st.markdown(
        f'<div class="hero"><h1 class="hero-title">{title_html}</h1>'
        f'<div class="hero-sub">{sub_html}</div></div>',
        unsafe_allow_html=True,
    )


def page_footer():
    st.markdown(
        '<div class="page-footer">🎬 中文电影评论情感分析系统 · 机器学习课程实践 · '
        "Bi-LSTM / Attention / 腾讯预训练词向量</div>",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def list_model_runs():
    """列出 outputs/ 下所有可用的【二分类】模型 run 名(排除五级 ckpt,防混列)。"""
    if not os.path.isdir(OUT_DIR):
        return []
    return sorted(
        p[:-len("_best.pt")] for p in os.listdir(OUT_DIR)
        if p.endswith("_best.pt") and not p.endswith("_5c_best.pt")
    )


@st.cache_resource(show_spinner=False)
def list_model_runs_5class():
    """列出 outputs/ 下所有可用的【五级分类】模型 run 名(<run>_5c_best.pt)。"""
    if not os.path.isdir(OUT_DIR):
        return []
    return sorted(
        p[:-len("_5c_best.pt")] for p in os.listdir(OUT_DIR)
        if p.endswith("_5c_best.pt")
    )


@st.cache_resource(show_spinner=False)
def get_predictor(run_name: str = None):
    """加载推理器(带缓存;predict.py 按 ckpt 自动识别二分类/五级)。"""
    from predict import SentimentPredictor

    return SentimentPredictor(run_name)


@st.cache_data
def load_summary_csv():
    path = os.path.join(OUT_DIR, "experiments_summary.csv")
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


@st.cache_data
def load_summary_csv_5class():
    """五级分类实验总表(与二分类独立)。"""
    path = os.path.join(OUT_DIR, "experiments_summary_5class.csv")
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


# 五级分类的等级样式:(类名, emoji, 主题色, 英文标注)——横幅/堆叠条共用
LEVEL_STYLES = [
    ("非常负面", "😡", "#b71c1c", "VERY NEGATIVE"),
    ("负面", "😞", "#d32f2f", "NEGATIVE"),
    ("中性", "😐", "#78909c", "NEUTRAL"),
    ("正面", "🙂", "#2e7d32", "POSITIVE"),
    ("非常正面", "🤩", "#00c853", "VERY POSITIVE"),
]


# 快速示例(点击即填入输入框,session key 与 text_area 绑定)
EXAMPLES = {
    "😊 正面示例": "这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!",
    "😞 负面示例": "我不满意这家酒店的服务,房间也不干净,隔音差到一夜没睡好。",
    "🎬 电影评论": "这部电影太精彩了,剧情紧凑不拖沓,演员演技在线,强烈推荐!",
    "⚡ 否定句测试": "总体不满意,不会再来。性价比也不高,感觉被骗了。",
}


# ================== 页面 1:情感分析 ==================
def page_analyze():
    hero("中文电影评论情感分析", "基于深度学习(Bi-LSTM + 注意力机制 + 预训练词向量)· 输入评论,即刻判断情感极性")

    # ---- 示例词条:一键填入 ----
    cols = st.columns(len(EXAMPLES))
    for col, (name, sample) in zip(cols, EXAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state["review_input"] = sample

    text = st.text_area(
        "✍️ 请输入中文评论",
        value=st.session_state.pop("review_input", "这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!"),
        height=120,
    )
    analyze = st.button("🚀 开始分析", type="primary", use_container_width=True)
    if not analyze:
        page_footer()
        return

    text = text.strip()
    if not text:
        st.warning("请输入评论内容")
        page_footer()
        return

    with st.spinner("模型推理中…"):
        predictor = get_predictor(st.session_state.get("selected_run"))
        r = predictor.predict(text)

    # ---- 结论横幅:按任务模式渲染(二分类样式与原版完全一致)----
    if r.get("num_classes") == 5:
        # 五级横幅:等级专属配色(深红/红/灰/绿/亮绿)+ emoji
        name, emoji, color, en = LEVEL_STYLES[r["label"]]
        st.markdown(
            f'<div class="verdict" style="background:linear-gradient(135deg,'
            f'{color}33, {color}0d); border:1px solid {color}99">'
            f'<div class="v-emoji">{emoji}</div>'
            f'<div><div class="v-label" style="color:{color}">{name}评价 {en}</div>'
            f'<div class="v-sub">模型:{predictor.run_name} · 置信度 {r["confidence"]:.1%}</div></div>'
            f'<div class="v-pct" style="color:{color}">{r["confidence"]:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        is_pos = r["label"] == 1
        st.markdown(
            f'<div class="verdict {"pos" if is_pos else "neg"}">'
            f'<div class="v-emoji">{"😊" if is_pos else "😞"}</div>'
            f'<div><div class="v-label">{"正面评价 POSITIVE" if is_pos else "负面评价 NEGATIVE"}</div>'
            f'<div class="v-sub">模型:{predictor.run_name} · 置信度 {r["confidence"]:.1%}</div></div>'
            f'<div class="v-pct">{r["confidence"]:.1%}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- 金色置信度进度条 + 概率明细 ----
    st.markdown(
        f'<div class="conf-bar"><div class="conf-fill" style="width:{min(r["confidence"],1.0)*100:.1f}%"></div></div>',
        unsafe_allow_html=True,
    )
    if r.get("num_classes") == 5:
        # 五级:五个等级概率一行展示
        pcols = st.columns(5)
        for i, ((name, emoji, color, _en), col) in enumerate(zip(LEVEL_STYLES, pcols)):
            col.metric(f"{emoji} {name}", f"{r['probs'][name]:.1%}")
    else:
        c1, c2 = st.columns(2)
        c1.metric("负面概率", f"{r['probs']['负面']:.1%}")
        c2.metric("正面概率", f"{r['probs']['正面']:.1%}")

    # ---- 注意力关键词高亮(金色深浅 = 权重大小)----
    if r["attn_weights"] is not None:
        st.markdown(
            '<div class="cine-card"><h3 style="margin-top:4px">🔍 注意力权重可视化</h3>'
            '<div style="color:#9aa4b8;font-size:13.5px;margin-bottom:10px">'
            "颜色越亮,模型判断时越关注该词</div>",
            unsafe_allow_html=True,
        )
        weights = r["attn_weights"]
        w_max = max(weights) if max(weights) > 0 else 1.0
        spans = []
        for w, weight in zip(r["tokens"], weights):
            alpha = 0.10 + 0.85 * (weight / w_max)  # 权重映射到透明度 0.10~0.95
            spans.append(
                f'<span class="attn-word" style="background-color:rgba(245,197,24,{alpha:.2f})">{w}</span>'
            )
        st.markdown('<div style="line-height:2.3">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
        top = sorted(zip(r["tokens"], weights), key=lambda x: -x[1])[:6]
        chips = " ".join(f'<span class="attn-chip">{w} · {wt:.0%}</span>' for w, wt in top)
        st.markdown(
            f'<div style="margin-top:10px;color:#9aa4b8;font-size:13.5px">注意力最高的词:{chips}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("当前加载的是基础 Bi-LSTM 模型(无注意力层)。可在侧边栏选择 Bi-LSTM+Attention 模型以获得关键词高亮。")
    page_footer()


# ================== 页面 2:实验仪表盘 ==================
def page_dashboard():
    hero("实验仪表盘", "二分类 14 组对比实验 + 五级分类拓展实验 · 训练曲线 / 混淆矩阵 / 超参数对比 / 模型结构")

    df = load_summary_csv()

    # ---- 顶部指标总览卡 ----
    if df is not None and len(df):
        best = df.loc[df["test_f1"].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 最佳测试 F1", f"{best['test_f1']:.4f}")
        m2.metric("🎯 最佳测试准确率", f"{best['test_acc']:.4f}")
        m3.metric("🧪 对比实验组数", f"{len(df)} 组")
        m4.metric("📦 测试集规模", "906 条")
        st.markdown(
            f'<div class="cine-card" style="margin:10px 0 6px;border-color:rgba(245,197,24,0.45)">'
            f'<b style="color:#f5c518">🏅 当前最佳模型:</b>{best["run_name"]}  '
            f'<span style="color:#9aa4b8">(P={best["test_precision"]} · R={best["test_recall"]} · '
            f"F1={best['test_f1']} · Acc={best['test_acc']})</span></div>",
            unsafe_allow_html=True,
        )

    tab_table, tab_curves, tab_attn, tab_5c, tab_model = st.tabs(
        ["📊 超参对比总表", "📈 训练曲线与混淆矩阵", "🔥 注意力热力图", "🌟 五级分类实验", "🧠 模型结构"]
    )

    # ---- Tab 1:对比总表 ----
    with tab_table:
        if df is not None and len(df):
            show = df.copy()
            show.insert(0, "★", "")  # 最佳行打星标记
            show.loc[df["test_f1"].idxmax(), "★"] = "🏆"
            st.dataframe(show, use_container_width=True, hide_index=True, height=420)
            st.caption("★ 列为测试 F1 最高的一组;完整数据见 outputs/experiments_summary.csv")
        else:
            st.warning("尚未找到 outputs/experiments_summary.csv,请先完成训练实验。")

    # ---- Tab 2:训练曲线 / 混淆矩阵(按实验分组)----
    with tab_curves:
        runs = sorted({p[:-len("_curves.png")] for p in os.listdir(OUT_DIR)
                       if p.endswith("_curves.png")}) if os.path.isdir(OUT_DIR) else []
        if runs:
            for run in runs:
                with st.expander(f"🧪 {run}", expanded=(run == runs[0])):
                    c1, c2 = st.columns(2)
                    curves = os.path.join(OUT_DIR, f"{run}_curves.png")
                    conf = os.path.join(OUT_DIR, f"{run}_confusion.png")
                    if os.path.exists(curves):
                        c1.image(curves, caption="训练/验证损失与准确率曲线")
                    if os.path.exists(conf):
                        c2.image(conf, caption="测试集混淆矩阵")
        else:
            st.info("outputs/ 下暂无训练曲线图。")

    # ---- Tab 3:注意力热力图样例 ----
    with tab_attn:
        attn_imgs = sorted(p for p in os.listdir(OUT_DIR)
                           if p.startswith("attention_") and p.endswith(".png")) \
            if os.path.isdir(OUT_DIR) else []
        if attn_imgs:
            a1, a2 = st.columns(2)
            for i, img in enumerate(attn_imgs[:6]):
                (a1 if i % 2 == 0 else a2).image(os.path.join(OUT_DIR, img), caption=img)
        else:
            st.info("暂无注意力热力图。")

    # ---- Tab 4:五级分类拓展实验(独立于二分类 14 组)----
    with tab_5c:
        df5 = load_summary_csv_5class()
        if df5 is not None and len(df5):
            best5 = df5.loc[df5["test_macro_f1"].idxmax()]
            b1, b2, b3 = st.columns(3)
            b1.metric("🏆 最佳测试 macro-F1", f"{best5['test_macro_f1']:.4f}")
            b2.metric("🎯 对应测试准确率", f"{best5['test_acc']:.4f}")
            b3.metric("🧪 实验组数", f"{len(df5)} 组")
            show5 = df5.copy()
            show5.insert(0, "★", "")
            show5.loc[df5["test_macro_f1"].idxmax(), "★"] = "🏆"
            st.dataframe(show5, use_container_width=True, hide_index=True)
            st.caption(
                f"拓展任务:豆瓣电影短评(1-5 星 → 非常负面/负面/中性/正面/非常正面),"
                f"数据 5 万条五类均衡;当前最佳:{best5['run_name']}。"
                "细粒度星级分类难度显著高于二分类(相邻星级语义连续),macro-F1 偏低属正常现象。"
            )
            runs5 = sorted({p[:-len("_5c_curves.png")] for p in os.listdir(OUT_DIR)
                            if p.endswith("_5c_curves.png")}) if os.path.isdir(OUT_DIR) else []
            for run in runs5:
                with st.expander(f"🌟 {run}", expanded=(run == runs5[0])):
                    c1, c2 = st.columns(2)
                    curves = os.path.join(OUT_DIR, f"{run}_5c_curves.png")
                    conf = os.path.join(OUT_DIR, f"{run}_5c_confusion.png")
                    if os.path.exists(curves):
                        c1.image(curves, caption="训练/验证损失与 macro-F1 曲线")
                    if os.path.exists(conf):
                        c2.image(conf, caption="测试集 5×5 混淆矩阵")
        else:
            st.info("尚未运行五级分类实验(运行 src/train_5class.py 后此处展示结果)。")

    # ---- Tab 5:模型结构说明 ----
    with tab_model:
        st.markdown(
            """
**1. 基础模型 Bi-LSTM**(`src/models/bilstm.py`)

```
输入 [B,128] 整数序列
  → Embedding(vocab,300)          # 300 维词向量,可随机初始化/预训练冻结/微调
  → Bi-LSTM(hidden=128,双向)     # 每方向 128 维,拼接 256 维
  → 有效位置 mask 平均池化         # 对非 PAD 位置求平均(优于取最后时间步)
  → Dropout(0.3~0.5)
  → Linear(256→2)                # 正/负两类 logits
```

**2. 进阶模型 Bi-LSTM + Bahdanau Attention**(`src/models/attn_bilstm.py`)

```
Bi-LSTM 输出 [B,128,256]
  → 打分 score_t = vᵀ·tanh(W·h_t)          # 可训练对齐网络
  → PAD 位置打分置 -inf 后 softmax          # 关键:屏蔽填充位,防注意力稀释
  → 加权求和 context = Σ α_t·h_t           # 得到句向量
  → Dropout → Linear(256→2)
```

**3. 训练设置**:CrossEntropyLoss + Adam;全局种子 42;早停 patience=3
并恢复最佳权重;数据 8:1:1 划分,词汇表仅由训练集构建(防数据泄漏)。
"""
        )
    page_footer()


# ================== 页面 3:批量分析 ==================
def page_batch():
    hero("批量情感分析", "上传含 text 列的 CSV,或粘贴多行评论(每行一条),批量预测并可下载结果")

    predictor = get_predictor(st.session_state.get("selected_run"))
    texts = []

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="cine-card"><h3>📤 上传 CSV 文件</h3>', unsafe_allow_html=True)
        up = st.file_uploader("需包含 text 列", type=["csv"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        if up is not None:
            try:
                df = pd.read_csv(up)
                if "text" not in df.columns:
                    st.error(f"CSV 中未找到 text 列,实际列:{list(df.columns)}")
                else:
                    texts = df["text"].dropna().astype(str).tolist()
                    st.info(f"已读取 {len(texts)} 条评论")
            except Exception as e:  # noqa: BLE001
                st.error(f"CSV 解析失败:{e}")
    with right:
        st.markdown('<div class="cine-card"><h3>📋 粘贴多行评论</h3>', unsafe_allow_html=True)
        pasted = st.text_area(
            "每行一条评论",
            value="屏幕效果非常好,系统流畅,值得购买。\n电池太不耐用,半天就没电,后悔买了。",
            height=150, label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 批量预测", type="primary", use_container_width=True) and pasted.strip():
        texts += [line.strip() for line in pasted.splitlines() if line.strip()]

    if not texts:
        st.info("请上传 CSV 或粘贴评论后点击批量预测")
        page_footer()
        return

    with st.spinner("批量推理中…"):
        rows = [predictor.predict(t) | {"text": t[:200]} for t in texts]
    result = pd.DataFrame([
        {"text": r["text"], "情感极性": r["polarity"], "置信度": f"{r['confidence']:.1%}"}
        for r in rows
    ])

    # ---- 统计概览 + 极性占比条(二分类两段条不变;五级为五段堆叠条)----
    st.markdown(f"<h3>📋 预测结果(共 {len(result)} 条)</h3>", unsafe_allow_html=True)
    is_5c = rows and rows[0].get("num_classes") == 5
    if is_5c:
        counts = {name: int((result["情感极性"] == name).sum()) for name, *_ in LEVEL_STYLES}
        segs = "".join(
            f'<div style="width:{counts[name] / len(result) * 100:.1f}%;'
            f'background:{color};display:flex;align-items:center;justify-content:center;'
            f'color:#0d1117;font-weight:800;font-size:12.5px" title="{name} {counts[name]} 条">'
            f'{"%d" % round(counts[name] / len(result) * 100)}%</div>'
            for name, _e, color, _en in LEVEL_STYLES
        )
        st.markdown(f'<div class="ratio-bar">{segs}</div>', unsafe_allow_html=True)
        st.caption(" · ".join(
            f"{name} {counts[name]}" for name, _e, _c, _en in LEVEL_STYLES))
    else:
        n_pos = int((result["情感极性"] == "正面").sum())
        n_neg = len(result) - n_pos
        pct_pos = n_pos / len(result) * 100
        pct_neg = 100 - pct_pos
        st.markdown(
            f'<div class="ratio-bar">'
            f'<div class="ratio-pos" style="width:{pct_pos:.1f}%">😊 正面 {n_pos}({pct_pos:.0f}%)</div>'
            f'<div class="ratio-neg" style="width:{pct_neg:.1f}%">😞 负面 {n_neg}({pct_neg:.0f}%)</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    st.dataframe(result, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ 下载结果 CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    page_footer()


# ================== 主入口:侧边栏导航 ==================
def main():
    st.sidebar.markdown(
        '<div style="padding:6px 2px 2px">'
        '<div style="font-size:21px;font-weight:800;letter-spacing:1px;'
        'background:linear-gradient(100deg,#ffe082,#f5c518,#ffb300);'
        '-webkit-background-clip:text;background-clip:text;color:transparent">🎬 情感分析系统</div>'
        '<div style="color:#7d879c;font-size:12.5px;margin-top:4px">'
        "Bi-LSTM · Attention · 预训练词向量</div></div>",
        unsafe_allow_html=True,
    )

    # ---- 任务模式:二分类(默认,行为与原版完全一致)/ 五级分类 ----
    task_mode = st.sidebar.radio("🎯 任务模式", ["二分类", "五级分类"],
                                 index=0, help="五级分类为拓展任务(豆瓣 1-5 星 → 非常负面~非常正面)")
    st.session_state["task_mode"] = task_mode

    # ---- 模型选择:按任务模式列出对应 ckpt ----
    if task_mode == "二分类":
        runs = list_model_runs()
        default_run = None
        best_log = os.path.join(OUT_DIR, "best_model_info.json")
        if os.path.exists(best_log):
            with open(best_log, encoding="utf-8") as f:
                default_run = json.load(f).get("run_name")
    else:
        runs = list_model_runs_5class()
        # 五级默认:总表中 test_macro_f1 最高的 run
        default_run = None
        df5 = load_summary_csv_5class()
        if df5 is not None and len(df5):
            default_run = df5.loc[df5["test_macro_f1"].idxmax(), "run_name"]
    if default_run not in runs and runs:
        default_run = runs[0]
    selected_run = st.sidebar.selectbox(
        "🧠 选择模型", runs, index=runs.index(default_run) if default_run in runs else 0,
        disabled=not runs,
    )
    if not runs:
        st.sidebar.warning("该模式下暂无模型检查点" +
                           ("(请先运行 train_5class.py)" if task_mode == "五级分类" else ""))
    if selected_run:
        st.sidebar.caption(f"当前模型:{selected_run}")
    st.session_state["selected_run"] = selected_run  # 供各页面获取所选模型

    st.sidebar.divider()
    page = st.sidebar.radio("导航", ["🎭 情感分析", "📊 实验仪表盘", "📦 批量分析"])
    if page == "🎭 情感分析":
        page_analyze()
    elif page == "📊 实验仪表盘":
        page_dashboard()
    else:
        page_batch()

    st.sidebar.divider()
    st.sidebar.caption("课程实践:基于深度学习的文本情感分析")


if __name__ == "__main__":
    main()
