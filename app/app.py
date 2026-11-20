# -*- coding: utf-8 -*-
"""
Streamlit 情感分析 Web 应用(三页面)
=====================================
页面 1 情感分析:输入任意中文评论 → 情感极性 + 置信度进度条 + 注意力关键词高亮;
页面 2 实验仪表盘:训练曲线、混淆矩阵、超参对比总表、模型结构说明;
页面 3 批量分析:上传 csv(text 列)或粘贴多行评论 → 批量预测 → 表格 + csv 下载。

启动:
  .venv/Scripts/python.exe -m streamlit run app/app.py --server.headless true
推理通过 src/predict.py 调用,预处理与训练共用 src/preprocess.py(同一份代码)。
"""

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
st.set_page_config(page_title="中文情感分析系统", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner=False)
def get_predictor():
    """加载推理器(带缓存,避免每次交互都重载模型)。"""
    from predict import SentimentPredictor

    return SentimentPredictor()


@st.cache_data
def load_summary_csv():
    path = os.path.join(OUT_DIR, "experiments_summary.csv")
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


# ================== 页面 1:情感分析 ==================
def page_analyze():
    st.header("单条评论情感分析")
    st.caption("输入任意中文评论,模型输出情感极性、置信度,以及注意力权重最高的关键词高亮。")

    text = st.text_area(
        "请输入中文评论:",
        value="这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!",
        height=120,
    )
    if not st.button("开始分析", type="primary"):
        return

    text = text.strip()
    if not text:
        st.warning("请输入评论内容")
        return

    predictor = get_predictor()
    r = predictor.predict(text)

    # ---- 极性与置信度 ----
    if r["label"] == 1:
        st.success(f"预测结果:**正面** 😊(置信度 {r['confidence']:.1%})")
    else:
        st.error(f"预测结果:**负面** 😞(置信度 {r['confidence']:.1%})")

    st.progress(min(r["confidence"], 1.0), text=f"置信度 {r['confidence']:.1%}")

    col1, col2 = st.columns(2)
    col1.metric("负面概率", f"{r['probs']['负面']:.1%}")
    col2.metric("正面概率", f"{r['probs']['正面']:.1%}")

    # ---- 注意力关键词高亮(按权重深浅着色,权重越高背景越深)----
    if r["attn_weights"] is not None:
        st.subheader("注意力权重可视化(颜色越深,模型越关注该词)")
        weights = r["attn_weights"]
        w_max = max(weights) if max(weights) > 0 else 1.0
        spans = []
        for w, weight in zip(r["tokens"], weights):
            alpha = 0.15 + 0.85 * (weight / w_max)  # 权重映射到透明度 0.15~1.0
            color = f"rgba(255,106,0,{alpha:.2f})"  # 橙色系,深浅表示权重
            spans.append(
                f'<span style="background-color:{color};padding:2px 4px;margin:2px;'
                f'border-radius:4px;font-size:16px">{w}</span>'
            )
        st.markdown(
            '<div style="line-height:2.2">' + " ".join(spans) + "</div>",
            unsafe_allow_html=True,
        )
        top = sorted(zip(r["tokens"], weights), key=lambda x: -x[1])[:8]
        st.write("注意力最高的词:" + "、".join(w for w, _ in top))
    else:
        st.info("当前加载的是基础 Bi-LSTM 模型(无注意力层)。可在侧边栏选择 "
                "Bi-LSTM+Attention 模型的 run 以获得关键词高亮。")


# ================== 页面 2:实验仪表盘 ==================
def page_dashboard():
    st.header("实验仪表盘")
    st.caption("展示各实验的训练曲线、混淆矩阵、超参对比总表与模型结构说明。")

    # ---- 超参对比总表 ----
    df = load_summary_csv()
    if df is not None and len(df):
        st.subheader("超参对比总表(experiments_summary.csv)")
        st.dataframe(df, use_container_width=True)
        best = df.loc[df["test_f1"].idxmax()]
        st.success(
            f"测试 F1 最高的实验:**{best['run_name']}**"
            f"(test_f1={best['test_f1']},test_acc={best['test_acc']})"
        )
    else:
        st.warning("尚未找到 outputs/experiments_summary.csv,请先完成训练实验。")

    # ---- 曲线图 / 混淆矩阵(按实验名分组展示)----
    runs = sorted({p[:-len("_curves.png")] for p in os.listdir(OUT_DIR)
                   if p.endswith("_curves.png")}) if os.path.isdir(OUT_DIR) else []
    if runs:
        st.subheader("训练曲线与混淆矩阵")
        for run in runs:
            with st.expander(f"实验:{run}", expanded=(run == runs[0])):
                c1, c2 = st.columns(2)
                curves = os.path.join(OUT_DIR, f"{run}_curves.png")
                conf = os.path.join(OUT_DIR, f"{run}_confusion.png")
                if os.path.exists(curves):
                    c1.image(curves, caption="训练/验证损失与准确率曲线")
                if os.path.exists(conf):
                    c2.image(conf, caption="测试集混淆矩阵")
    else:
        st.info("outputs/ 下暂无训练曲线图。")

    # ---- 注意力热力图样例 ----
    attn_imgs = sorted(p for p in os.listdir(OUT_DIR)
                       if p.startswith("attention_") and p.endswith(".png")) \
        if os.path.isdir(OUT_DIR) else []
    if attn_imgs:
        with st.expander("注意力热力图样例(词级)"):
            for img in attn_imgs[:6]:
                st.image(os.path.join(OUT_DIR, img), caption=img)

    # ---- 模型结构文字说明 ----
    st.subheader("模型结构说明")
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


# ================== 页面 3:批量分析 ==================
def page_batch():
    st.header("批量情感分析")
    st.caption("上传含 text 列的 CSV 文件,或在文本域粘贴多行评论(每行一条)。")

    predictor = get_predictor()
    texts = []

    up = st.file_uploader("上传 CSV(需包含 text 列)", type=["csv"])
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

    pasted = st.text_area("或粘贴多行评论(每行一条,示例已给出):",
                          value="屏幕效果非常好,系统流畅,值得购买。\n"
                                "电池太不耐用,半天就没电,后悔买了。", height=110)
    if st.button("批量预测", type="primary") and pasted.strip():
        texts += [line.strip() for line in pasted.splitlines() if line.strip()]

    if not texts:
        st.info("请上传 CSV 或粘贴评论后点击批量预测")
        return

    rows = [predictor.predict(t) | {"text": t[:200]} for t in texts]
    result = pd.DataFrame([
        {"text": r["text"], "情感极性": r["polarity"], "置信度": f"{r['confidence']:.1%}"}
        for r in rows
    ])
    st.subheader(f"预测结果(共 {len(result)} 条)")
    st.dataframe(result, use_container_width=True)

    # 统计概览
    c1, c2, c3 = st.columns(3)
    c1.metric("总条数", len(result))
    c2.metric("正面条数", int((result["情感极性"] == "正面").sum()))
    c3.metric("负面条数", int((result["情感极性"] == "负面").sum()))

    st.download_button(
        "下载结果 CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_predictions.csv",
        mime="text/csv",
    )


# ================== 主入口:侧边栏导航 ==================
def main():
    st.sidebar.title("🎬 中文情感分析系统")
    st.sidebar.caption("Bi-LSTM / Bi-LSTM+Attention · ChnSentiCorp")
    page = st.sidebar.radio("导航", ["情感分析", "实验仪表盘", "批量分析"])
    if page == "情感分析":
        page_analyze()
    elif page == "实验仪表盘":
        page_dashboard()
    else:
        page_batch()
    st.sidebar.divider()
    st.sidebar.markdown("课程实践:基于深度学习的文本情感分析")


if __name__ == "__main__":
    main()
