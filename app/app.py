# -*- coding: utf-8 -*-
"""
情绪放映室 · CINEMOOD —— 中文电影评论情感分析系统
====================================================
Editorial Intelligence 视觉系统：暖白纸张质感、墨绿主色、陶土橙点缀，
以编辑排版、清晰网格、充足留白和轻量阴影建立信息层级。
内置 5 套浅色主题，经 query 参数 ?theme= 持久化，导航链接自动携带主题。
标题使用现代中文黑体，关键词以衬线体强调；正文保持高可读性的系统字体。

仅做视觉与导航层，推理逻辑完全不变：src/predict.py → 共用 src/preprocess.py。
导航经 query 参数 ?page= &theme= &task= 路由(task 随链接携带,跨页保持任务模式)。
启动：.venv/Scripts/python.exe -m streamlit run app/app.py --server.headless true
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

# ---------- 页面全局配置(隐藏原生侧边栏,导航全部融入页面) ----------
st.set_page_config(
    page_title="情绪放映室 · CINEMOOD", page_icon="🎬",
    layout="wide", initial_sidebar_state="collapsed",
)

# ================== 主题库（5 套浅色编辑主题） ==================
# p 主色 / c 次级强调色 / a 暖点缀；bg 与 wash 控制页面底色和氛围色。
THEMES = {
    "electric": {
        "label": "雾蓝编辑", "p": "#38687A", "c": "#6F96A3", "a": "#C77752",
        "bg": "#F4F6F4", "wash": "#DDE9EA",
    },
    "aurora": {
        "label": "纸墨鼠尾草", "p": "#365E4B", "c": "#7B8F70", "a": "#C8734B",
        "bg": "#F6F5F0", "wash": "#E4E9DC",
    },
    "blush": {
        "label": "陶土玫瑰", "p": "#8A554D", "c": "#BC8174", "a": "#B68A48",
        "bg": "#F8F4F1", "wash": "#F0DFD8",
    },
    "amber": {
        "label": "燕麦琥珀", "p": "#745F35", "c": "#A48B58", "a": "#B96546",
        "bg": "#F7F4EC", "wash": "#EEE3C9",
    },
    "lava": {
        "label": "落日橙光", "p": "#9B523C", "c": "#CE8060", "a": "#506C5B",
        "bg": "#FAF5EF", "wash": "#F3DFD0",
    },
}
DEFAULT_THEME = "aurora"


def _rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' + 透明度 → 'rgba(r,g,b,a)'(主题样式生成用)。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def theme_style(tid: str) -> str:
    """把当前主题注入统一的设计变量，所有组件共享同一套色彩语义。"""
    t = THEMES[tid]
    p, c, a = t["p"], t["c"], t["a"]
    return f"""
<style>
:root {{
    --accent: {p};
    --accent-2: {c};
    --warm: {a};
    --bg: {t["bg"]};
    --wash: {t["wash"]};
    --accent-soft: {_rgba(p, .09)};
    --accent-softer: {_rgba(p, .055)};
    --accent-line: {_rgba(p, .22)};
    --warm-soft: {_rgba(a, .11)};
}}
.brand-mark {{ background: var(--accent); color: #FFFEFA; }}
.navpill.active {{ background: var(--accent); color: #FFFEFA; }}
.hero-title em {{ color: var(--accent); }}
.hero-kicker, .card-number, .control-label {{ color: var(--accent); }}
.mode-badge .dot, .model-chip .dot {{ background: var(--accent-2); }}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 8px 20px {_rgba(p, .20)} !important;
}}
.conf-fill {{ background: linear-gradient(90deg, var(--accent), var(--accent-2)); }}
.attn-chip {{ background: var(--accent-soft); color: var(--accent); border-color: var(--accent-line); }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; background: var(--surface) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}
.theme-dot.active {{ box-shadow: 0 0 0 2px var(--surface), 0 0 0 4px var(--accent); }}
</style>
"""


# ================== 基础样式：Editorial Intelligence ==================
CUSTOM_CSS = """
<style>
:root {
    --surface: #FFFEFA;
    --surface-2: #F0F1EB;
    --ink: #202723;
    --ink-2: #5F6862;
    --ink-3: #8A918B;
    --line: rgba(42, 57, 48, .13);
    --line-strong: rgba(42, 57, 48, .22);
    --shadow-sm: 0 2px 5px rgba(38, 46, 39, .025), 0 12px 30px rgba(38, 46, 39, .045);
    --shadow-md: 0 18px 50px rgba(38, 46, 39, .075);
}

/* ===== 页面基底与字体 ===== */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg);
    color: var(--ink);
}
.stApp {
    font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
    background-image:
        radial-gradient(circle at 92% 3%, var(--wash) 0, transparent 28rem),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='.025'/%3E%3C/svg%3E");
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container {
    max-width: 1240px;
    padding-top: 1rem !important;
    padding-bottom: 3.5rem !important;
}
h1, h2, h3, h4, p, label { color: var(--ink); }
h1, h2, h3, h4 {
    font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    letter-spacing: -.025em;
}
a { color: var(--accent); }
::selection { background: var(--accent-soft); color: var(--accent); }

/* ===== 顶部导航：纸张感悬浮条 ===== */
.topnav {
    position: sticky;
    top: 12px;
    z-index: 100;
    margin: 0 0 10px;
    padding: 10px 12px;
    display: flex;
    align-items: center;
    gap: 16px;
    background: rgba(255, 254, 250, .90);
    backdrop-filter: blur(18px) saturate(1.1);
    -webkit-backdrop-filter: blur(18px) saturate(1.1);
    border: 1px solid var(--line);
    border-radius: 17px;
    box-shadow: var(--shadow-sm);
}
.brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding-right: 8px;
    text-decoration: none;
    flex: none;
}
.brand, .brand *, .navpill, .navpill * { text-decoration: none !important; }
.brand-mark {
    width: 40px;
    height: 40px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -.5px;
    transition: transform .2s ease;
}
.brand:hover .brand-mark { transform: rotate(-4deg); }
.brand-txt {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 19px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.05;
    letter-spacing: .02em;
}
.brand-txt small {
    display: block;
    margin-top: 5px;
    font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif;
    font-size: 8.5px;
    font-weight: 500;
    letter-spacing: 1.7px;
    color: var(--ink-3);
}
.navpills { display: flex; align-items: center; gap: 3px; }
.navpill {
    padding: 9px 15px;
    border-radius: 10px;
    text-decoration: none;
    white-space: nowrap;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink-2);
    transition: background .18s ease, color .18s ease;
}
.navpill:hover { background: var(--accent-soft); color: var(--accent); }
.navpill.active { box-shadow: 0 3px 10px rgba(31, 45, 37, .10); }
.mode-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-left: auto;
    padding: 7px 10px;
    border-left: 1px solid var(--line);
    white-space: nowrap;
    font-size: 11px;
    font-weight: 600;
    color: var(--ink-2);
}
.mode-badge .dot { width: 6px; height: 6px; border-radius: 50%; }
.theme-dots { display: flex; align-items: center; gap: 8px; padding: 7px 9px; }
.theme-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    cursor: pointer;
    border: 1px solid rgba(32, 39, 35, .10);
    padding: 0;
    text-decoration: none;
    transition: transform .18s ease, box-shadow .18s ease;
}
.theme-dot:hover { transform: scale(1.18); }

/* ===== 顶部导航(Streamlit 容器版:粘性吸顶 + 应用内按钮,整页零刷新) ===== */
.topnav-mark { display: none; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark) {
    position: sticky;
    top: 12px;
    z-index: 100;
    margin: 0 0 10px;
    padding: 10px 12px;
    background: rgba(255, 254, 250, .90);
    backdrop-filter: blur(18px) saturate(1.1);
    -webkit-backdrop-filter: blur(18px) saturate(1.1);
    border: 1px solid var(--line) !important;
    border-radius: 17px !important;
    box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark):hover {
    border-color: var(--line) !important;  /* 抵消卡片悬停效果,导航条保持稳定 */
    box-shadow: var(--shadow-sm) !important;
}
/* 导航按钮 → 胶囊形态(仅作用于顶栏内的按钮) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark) .stButton > button,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark) .stButton > button:hover {
    min-height: 38px;
    padding: 6px 14px;
    border-radius: 10px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .01em;
    transition: background .16s ease, color .16s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark) .stButton > button:hover {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    transform: none !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark)
    .stButton > button[kind="primary"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark)
    .stButton > button[data-testid="stBaseButton-primary"],
div[data-testid="stVerticalBlockBorderWrapper"]:has(.topnav-mark)
    .stButton > button[kind="primary"]:hover {
    background: var(--accent) !important;
    color: #FFFEFA !important;
    box-shadow: 0 3px 10px rgba(31, 45, 37, .10) !important;
}
/* 顶栏右侧(模式徽章 + 主题色点) */
.nav-right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
    flex-wrap: wrap;
}
.nav-right .mode-badge { margin-left: 0; border-left: none; padding-left: 0; }

/* ===== Hero：杂志式标题 + 编辑注记 ===== */
.hero {
    position: relative;
    display: grid;
    grid-template-columns: minmax(0, 1.7fr) minmax(230px, .62fr);
    gap: 54px;
    align-items: end;
    padding: 62px 12px 42px;
    margin-bottom: 26px;
    border-bottom: 1px solid var(--line);
    animation: fadeUp .5s ease both;
}
.hero-copy { min-width: 0; }
.hero-kicker {
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.6px;
    text-transform: uppercase;
}
.hero-kicker::before { content: ""; width: 28px; height: 1px; background: currentColor; }
.hero-title {
    max-width: 780px;
    margin: 18px 0 18px;
    font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", sans-serif !important;
    font-size: clamp(40px, 4.5vw, 62px);
    font-weight: 700;
    letter-spacing: -.065em;
    line-height: 1.18;
    color: var(--ink);
}
.hero-title > a, .hero-title a[href^="#"] { display: none !important; }
.hero-title em {
    position: relative;
    font-family: "Songti SC", "STSong", "SimSun", Georgia, serif;
    font-style: normal;
    font-weight: 500;
    letter-spacing: -.04em;
    white-space: nowrap;
}
.hero-title em::after {
    content: "";
    position: absolute;
    z-index: -1;
    left: -.02em;
    right: -.02em;
    bottom: .04em;
    height: .18em;
    background: var(--accent-soft);
}
.hero-sub {
    max-width: 620px;
    margin: 0;
    color: var(--ink-2);
    font-size: 14px;
    line-height: 1.95;
    letter-spacing: .015em;
}
.hero-note { padding: 4px 0 4px 28px; border-left: 1px solid var(--line-strong); }
.hero-note-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--line);
    color: var(--ink-3);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 10px;
    letter-spacing: 1.5px;
}
.hero-note-title {
    margin: 19px 0 16px;
    color: var(--ink);
    font-family: "Songti SC", "STSong", "SimSun", Georgia, serif;
    font-size: 27px;
    line-height: 1.35;
    letter-spacing: .04em;
}
.hero-note-flow { display: flex; align-items: center; gap: 8px; color: var(--ink-2); font-size: 11px; }
.hero-note-flow i { width: 14px; height: 1px; background: var(--line-strong); }
.hero-note-foot {
    margin-top: 24px;
    color: var(--ink-3);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 9px;
    letter-spacing: 1.4px;
}
.hero-mini { padding-top: 48px; }
.hero-mini .hero-title { font-size: clamp(34px, 3.8vw, 52px); }

/* ===== 卡片与标题系统 ===== */
.bento,
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 254, 250, .88);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow-sm);
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    transition: border-color .2s ease, box-shadow .2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--line-strong);
    box-shadow: var(--shadow-md);
}
.bento { position: relative; padding: 21px 23px; }
.card-heading { display: flex; align-items: flex-start; gap: 12px; margin: 1px 0 17px; }
.card-number {
    padding-top: 2px;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
}
.bento-title { color: var(--ink); font-size: 15px; font-weight: 700; letter-spacing: -.01em; }
.bento-hint { margin-top: 5px; color: var(--ink-3); font-size: 11.5px; line-height: 1.6; }
.control-label {
    margin: 1px 0 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

/* ===== 按钮与输入 ===== */
.stButton > button, .stDownloadButton > button {
    min-height: 40px;
    border-radius: 11px !important;
    border: 1px solid var(--line-strong) !important;
    background: var(--surface) !important;
    color: var(--ink-2) !important;
    font-size: 12.5px;
    font-weight: 600;
    letter-spacing: .01em;
    box-shadow: none !important;
    transition: transform .16s ease, border-color .16s ease, background .16s ease, color .16s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--accent-line) !important;
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}
.stButton > button *, .stDownloadButton > button * { color: inherit !important; }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0) scale(.985); }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    min-height: 45px;
    color: #FFFEFA !important;
    font-weight: 700;
}
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(246, 246, 240, .70) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    color: var(--ink) !important;
    font-family: inherit;
    font-size: 14px;
    line-height: 1.8;
    transition: border-color .18s ease, box-shadow .18s ease, background .18s ease;
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder { color: var(--ink-3) !important; }
[data-testid="stTextArea"] textarea:hover,
[data-testid="stTextInput"] input:hover { border-color: var(--line-strong) !important; }
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    background: var(--surface) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-baseweb="select"] > div {
    background: rgba(246, 246, 240, .70) !important;
    border-radius: 11px !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
}
[data-baseweb="select"] * { color: var(--ink); }
[data-testid="stFileUploaderDropzone"] {
    background: rgba(246, 246, 240, .60);
    border: 1px dashed var(--line-strong);
    border-radius: 12px;
}
[data-testid="stAlert"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    color: var(--ink-2);
}
.model-chip {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 9px;
    color: var(--ink-3);
    font-size: 10px;
    line-height: 1.4;
}
.model-chip .dot { width: 5px; height: 5px; border-radius: 50%; flex: none; }

/* ===== 分析结果 ===== */
.verdict {
    position: relative;
    display: flex;
    align-items: center;
    gap: 19px;
    margin: 22px 0 14px;
    padding: 23px 27px;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow-sm);
    animation: fadeUp .4s ease both;
}
.verdict::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: var(--vc, var(--accent));
}
.v-emoji { font-size: 39px; line-height: 1; }
.v-label {
    color: var(--ink);
    font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: -.03em;
}
.v-sub { margin-top: 6px; color: var(--ink-3); font-size: 11.5px; }
.v-pct {
    margin-left: auto;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 42px;
    font-weight: 400;
    letter-spacing: -.045em;
}
.conf-bar {
    height: 7px;
    margin: 0 3px 18px;
    overflow: hidden;
    background: var(--accent-soft);
    border-radius: 999px;
}
.conf-fill { height: 100%; border-radius: 999px; transition: width .7s cubic-bezier(.2,.7,.3,1); }
.attn-word {
    display: inline-block;
    margin: 3px;
    padding: 3px 7px;
    border-radius: 6px;
    color: #26352D;
    font-size: 15px;
    transition: transform .15s ease;
}
.attn-word:hover { transform: translateY(-2px); }
.attn-chip {
    display: inline-block;
    margin: 3px;
    padding: 4px 10px;
    border: 1px solid var(--accent-line);
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}

/* ===== 指标、表格、选项卡 ===== */
[data-testid="stMetric"] {
    padding: 17px 19px;
    background: rgba(255, 254, 250, .88);
    border: 1px solid var(--line);
    border-radius: 15px;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] p {
    color: var(--ink-3) !important;
    font-size: 11px !important;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 29px;
    letter-spacing: -.03em;
}
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 13px; overflow: hidden; }
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--line) !important;
    border-radius: 13px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    padding: 4px;
    background: var(--accent-softer);
    border: 1px solid var(--line);
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding: 7px 15px;
    border-radius: 8px;
    color: var(--ink-3) !important;
    font-size: 12px;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--accent) !important; }
.stTabs [aria-selected="true"] { box-shadow: 0 2px 8px rgba(38, 46, 39, .06); }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ===== 批量占比与页脚 ===== */
.ratio-bar {
    display: flex;
    height: 29px;
    margin: 8px 0 12px;
    overflow: hidden;
    background: var(--surface-2);
    border-radius: 8px;
}
.ratio-seg {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 0;
    overflow: hidden;
    color: #FFFEFA;
    font-size: 11px;
    font-weight: 700;
    transition: filter .18s ease;
}
.ratio-seg:hover { filter: brightness(1.06); }
.page-footer {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    margin-top: 42px;
    padding: 20px 2px 4px;
    border-top: 1px solid var(--line);
    color: var(--ink-3);
    font-family: Georgia, "Times New Roman", serif;
    font-size: 9px;
    letter-spacing: 1.5px;
    line-height: 1.8;
}
.page-footer .footer-cn {
    font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif;
    letter-spacing: .8px;
}

@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@media (max-width: 980px) {
    .topnav { flex-wrap: wrap; gap: 9px; }
    .navpills { order: 3; width: 100%; overflow-x: auto; }
    .mode-badge { margin-left: auto; }
    .hero { grid-template-columns: 1fr; gap: 28px; padding-top: 42px; }
    .hero-note { display: none; }
}
@media (max-width: 820px) {
    div[data-testid="stHorizontalBlock"]:has(.card-heading) { flex-direction: column; }
    div[data-testid="stHorizontalBlock"]:has(.card-heading) > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
}
@media (max-width: 620px) {
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .topnav { top: 6px; }
    .brand-txt { font-size: 17px; }
    .theme-dots { padding-right: 2px; }
    .mode-badge { display: none; }
    .navpill { padding: 8px 11px; font-size: 12px; }
    .hero { padding: 34px 3px 30px; }
    .hero-title { font-size: 37px; }
    .hero-note { display: none; }
    .verdict { align-items: flex-start; padding: 20px; gap: 12px; }
    .v-label { font-size: 20px; }
    .v-pct { font-size: 31px; }
    .page-footer { flex-direction: column; gap: 4px; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""

def inject_css(theme_id: str):
    """注入基础样式与当前主题。"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(theme_style(theme_id), unsafe_allow_html=True)


# ================== 顶部导航(替代侧边栏) ==================
NAV_ITEMS = [
    ("analyze", "情感分析", "🎭"),
    ("dashboard", "实验仪表盘", "📊"),
    ("batch", "批量分析", "📦"),
]


def _go_page(key: str):
    """导航按钮回调:改 query 参数触发应用内重渲染(SPA 式局部刷新,浏览器整页零刷新)。

    st.query_params 的单键赋值会保留其余参数(theme/task 不丢)。
    """
    st.query_params["page"] = key


def top_nav(active_key: str, theme_id: str):
    """悬浮导航:品牌、页面入口按钮(应用内切换,不整页刷新)、主题色点与任务模式。

    页面切换走 st.query_params(仅重渲染应用,无白屏);主题色点仍是链接式
    (切主题属低频操作,保留整页跳转以换取渐变色点的精致呈现)。
    ?task=(binary|five) 随导航携带:main() 按参数恢复任务模式,
    会话不因页面切换而丢失(批量页与单句分析共用同一套模型配置)。
    """
    task = "five" if st.session_state.get("task_mode", "二分类") == "五级分类" else "binary"
    dots = "".join(
        f'<a class="theme-dot{" active" if tid == theme_id else ""}" '
        f'href="?page={active_key}&theme={tid}&task={task}" target="_self" '
        f'title="{t["label"]}" aria-label="{t["label"]}" '
        f'style="background:linear-gradient(135deg,{t["p"]},{t["c"]})"></a>'
        for tid, t in THEMES.items()
    )
    mode = st.session_state.get("task_mode", "二分类")

    with st.container(border=True):
        # 吸顶标记:CSS 据此把该容器固定为粘性导航条(见 CUSTOM_CSS)
        st.markdown('<i class="topnav-mark"></i>', unsafe_allow_html=True)
        c_brand, c_n1, c_n2, c_n3, c_right = st.columns([1.7, 0.95, 1.1, 0.95, 1.5], gap="small")
        c_brand.markdown(
            '<span class="brand"><span class="brand-mark">CM</span>'
            '<span class="brand-txt">CINEMOOD<small>情绪放映室 · NLP LAB</small></span></span>',
            unsafe_allow_html=True,
        )
        for col, (key, label, _icon) in zip((c_n1, c_n2, c_n3), NAV_ITEMS):
            col.button(
                label, key=f"nav_{key}", width="stretch",
                type="primary" if key == active_key else "secondary",
                on_click=_go_page, args=(key,),
            )
        c_right.markdown(
            f'<div class="nav-right"><div class="mode-badge"><span class="dot"></span>{mode}</div>'
            f'<div class="theme-dots" aria-label="界面配色">{dots}</div></div>',
            unsafe_allow_html=True,
        )


def hero_block(page_key: str, theme_id: str, title_html: str, sub: str, kicker: str, mini: bool = False):
    """编辑式页头：大标题、导言与当前页面的语义注记。"""
    notes = {
        "analyze": ("01 / SEMANTIC LENS", "从文字<br>到情绪", ("输入", "理解", "判断")),
        "dashboard": ("02 / EXPERIMENT LOG", "从实验<br>到证据", ("训练", "评估", "对比")),
        "batch": ("03 / BATCH STUDIO", "从单句<br>到群体", ("导入", "推理", "导出")),
    }
    note_head, note_title, flow = notes.get(page_key, notes["analyze"])
    flow_html = "".join(
        f'<span>{item}</span>' + ('<i></i>' if i < len(flow) - 1 else '')
        for i, item in enumerate(flow)
    )
    st.markdown(
        f'<header class="hero{" hero-mini" if mini else ""}">'
        f'<div class="hero-copy"><div class="hero-kicker">{kicker}</div>'
        f'<h1 class="hero-title">{title_html}</h1>'
        f'<p class="hero-sub">{sub}</p></div>'
        f'<aside class="hero-note"><div class="hero-note-head"><span>{note_head}</span><span>CINEMOOD</span></div>'
        f'<div class="hero-note-title">{note_title}</div>'
        f'<div class="hero-note-flow">{flow_html}</div>'
        f'<div class="hero-note-foot">BI-LSTM · ATTENTION · NLP</div></aside></header>',
        unsafe_allow_html=True,
    )


def page_footer():
    st.markdown(
        '<footer class="page-footer"><span>CINEMOOD / NLP LAB · 2026</span>'
        '<span class="footer-cn">中文评论情感分析 · Bi-LSTM / Attention / 预训练词向量</span></footer>',
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


@st.cache_resource(show_spinner=False, max_entries=4)
def get_predictor(run_name: str = None):
    """加载推理器(LRU 上限 4 个:逐个切换模型不会把全部 23 个检查点驻留内存,
    约 1.2GB 的 BERT 权重也会被及时淘汰;cache key 是 run 名,不影响当前使用)。
    predict.py 按 ckpt 自动识别二分类/五级,推理逻辑不变。"""
    from predict import SentimentPredictor

    return SentimentPredictor(run_name)


# 与 get_predictor 的 LRU(max_entries=4)同步维护的"最近已加载"访问序(进程内),
# 作为轻量探测:不在其中 = 冷加载,需要 spinner 反馈;命中则瞬时返回、零打扰。
_RECENTLY_LOADED: list = []


def load_predictor(run_name: str):
    """取当前模型:冷加载时显示含模型名的 spinner(BERT 附"约 10 秒"提示),缓存命中无感。"""
    cold = run_name not in _RECENTLY_LOADED
    if cold:
        hint = "(BERT 首次加载约 10 秒,请稍候)" if "bert" in str(run_name).lower() else ""
        with st.spinner(f"正在加载模型 {run_name}{hint} …"):
            predictor = get_predictor(run_name)
    else:
        predictor = get_predictor(run_name)
    if run_name in _RECENTLY_LOADED:
        _RECENTLY_LOADED.remove(run_name)
    _RECENTLY_LOADED.insert(0, run_name)
    del _RECENTLY_LOADED[4:]  # 与 cache_resource 的 max_entries 对齐(镜像其淘汰顺序)
    return predictor


@st.cache_data
def load_summary_csv():
    path = os.path.join(OUT_DIR, "experiments_summary.csv")
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


@st.cache_data
def load_summary_csv_5class():
    """五级分类实验总表(与二分类独立)。"""
    path = os.path.join(OUT_DIR, "experiments_summary_5class.csv")
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


def metric_available(df) -> bool:
    """指标总表是否可用来决定默认模型(CSV 缺失时才回退 best_model_info.json)。"""
    return df is not None and len(df) > 0 and "run_name" in df.columns


# 五级分类的等级样式：类名、表情、浅底可读色、柔和底色、英文标注。
LEVEL_STYLES = [
    ("非常负面", "😡", "#AD4F45", "rgba(173,79,69,.09)", "VERY NEGATIVE"),
    ("负面", "😞", "#BC6A48", "rgba(188,106,72,.09)", "NEGATIVE"),
    ("中性", "😐", "#7B817C", "rgba(123,129,124,.09)", "NEUTRAL"),
    ("正面", "🙂", "#4F7B60", "rgba(79,123,96,.09)", "POSITIVE"),
    ("非常正面", "🤩", "#2F6E62", "rgba(47,110,98,.10)", "VERY POSITIVE"),
]
BIN_STYLES = {
    1: ("正面", "😊", "#3F7758", "rgba(63,119,88,.09)", "POSITIVE"),
    0: ("负面", "😞", "#B96747", "rgba(185,103,71,.09)", "NEGATIVE"),
}

# 快速示例:经 on_click 回调直接写入 text_area 的 widget key。
# 回调在 rerun 前执行、widget key 状态优先于 value 参数,示例文本在"开始分析"
# 等后续任意 rerun 中都稳定保留(修复旧版 session_state.pop 导致的回滚)。
EXAMPLES = {
    "正面示例": "这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!",
    "负面示例": "我不满意这家酒店的服务,房间也不干净,隔音差到一夜没睡好。",
    "电影评论": "这部电影太精彩了,剧情紧凑不拖沓,演员演技在线,强烈推荐!",
    "否定句测试": "总体不满意,不会再来。性价比也不高,感觉被骗了。",
}


def _fill_example(sample: str):
    """示例按钮回调:填入示例文本,并让旧结论卡失效(输入变了,旧结论不再成立)。"""
    st.session_state["review_text"] = sample
    st.session_state.pop("analyze_result", None)


def _invalidate_result():
    """模型下拉变化回调:结论卡对应的模型已切换,立即失效。"""
    st.session_state.pop("analyze_result", None)


def _has_valid_tokens(text: str) -> bool:
    """空/纯符号输入防呆:经项目统一 tokenize(与训练同路径)后无有效词语则不进模型,
    避免给出随机极性。分析页 warning 提示,批量页跳过该行并标注"无法判断"。"""
    try:
        from preprocess import tokenize

        return len(tokenize(text)) > 0
    except Exception:  # noqa: BLE001 — 检查自身失败时不拦截,回到原推理路径
        return True


def _theme_c_rgb(theme_id: str) -> str:
    """主题亮色的 'r,g,b' 字符串(注意力词高亮透明度渐变用)。"""
    h = THEMES[theme_id]["c"].lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ================== 控制台(任务模式 + 模型选择,融入页面) ==================
def _switch_task_mode(mode: str):
    """任务模式切换回调(on_click,重跑前执行):单次重跑完成切换,不再显式 st.rerun。

    回调里同步 session 与 query 参数(?task=),按钮点击本身触发的重跑会带着
    新状态渲染——一次点击只渲染一次,页面不再闪两下。
    """
    if st.session_state.get("task_mode") != mode:
        st.session_state["task_mode"] = mode
        st.session_state.pop("analyze_result", None)
        st.query_params["task"] = "five" if mode == "五级分类" else "binary"


# 弱基线(run 名)启发式:不含这些标记的 run 视为随机词向量基线实验。
# CSV 有 embedding 列时以 embedding=random 为准,此规则仅作表缺失/缺行时的回退。
_STRONG_RUN_MARKS = ("pretrained", "pretune", "bert", "mixdom", "5c300k")


def _rank_and_annotate(runs: list, df, metric_col: str, five: bool):
    """模型下拉的排序与显示名:按指标降序(最佳在最上),并加推荐/耗时/基线标注。

    返回 (排序后的 run 列表, 显示名映射)。标注只影响显示,selectbox 提交值
    仍是原始 run 名(Web 自动发现与 CLI 的 run 名不变)。
    """
    metric, emb = {}, {}
    if df is not None and len(df):
        if metric_col in df.columns:
            metric = dict(zip(df["run_name"].astype(str), df[metric_col]))
        if "embedding" in df.columns:
            emb = dict(zip(df["run_name"].astype(str), df["embedding"].astype(str)))
    ranked = sorted(runs, key=lambda r: (-(metric.get(r, float("-inf"))), r))  # 缺指标者排后

    names = {}
    for i, run in enumerate(ranked):
        label = f"★ 推荐 · {run}" if (i == 0 and metric and metric.get(run) is not None) else run
        is_bert = emb.get(run) == "bert" or "bert" in run.lower()
        if is_bert:
            label += "(首次加载约10秒)"
        if five:
            # 五级:random 基线与 5 万条旧模型都标基线;5c300k 扩数据版与 BERT 不标
            is_baseline = not is_bert and not run.startswith("5c300k")
        elif run in emb:
            is_baseline = emb[run] == "random"  # 二分:随机词向量弱基线
        else:
            is_baseline = not any(m in run for m in _STRONG_RUN_MARKS)
        if is_baseline:
            label += "(基线实验)"
        names[run] = label
    return ranked, names


def render_control():
    """任务模式分段与模型选择。"""
    cur_mode = st.session_state.get("task_mode", "二分类")

    st.markdown('<div class="control-label">01 · 任务模式</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    # on_click 回调:一次点击 = 一次重跑(消灭"点击重跑 + st.rerun()"的双重渲染闪烁)
    c1.button("二元情感", key="mode_binary",
              type="primary" if cur_mode == "二分类" else "secondary", width="stretch",
              on_click=_switch_task_mode, args=("二分类",))
    c2.button("五级情感", key="mode_five",
              type="primary" if cur_mode == "五级分类" else "secondary", width="stretch",
              help="拓展任务：豆瓣 1–5 星 → 非常负面至非常正面",
              on_click=_switch_task_mode, args=("五级分类",))
    cur_mode = st.session_state.get("task_mode", "二分类")

    if cur_mode == "二分类":
        # 默认=指标总表 test_f1 最大者(与仪表盘同口径);best_model_info.json 仅作 CSV 缺失时的回退
        runs = list_model_runs()
        df_metric = load_summary_csv()
        ranked, names = _rank_and_annotate(runs, df_metric, "test_f1", five=False)
        default_run = ranked[0] if ranked else None
        if not metric_available(df_metric):
            best_log = os.path.join(OUT_DIR, "best_model_info.json")
            if os.path.exists(best_log):
                with open(best_log, encoding="utf-8") as f:
                    fallback = json.load(f).get("run_name")
                if fallback in runs:
                    default_run = fallback
    else:
        runs = list_model_runs_5class()
        df5 = load_summary_csv_5class()
        ranked, names = _rank_and_annotate(runs, df5, "test_macro_f1", five=True)
        default_run = ranked[0] if ranked else None

    st.markdown('<div class="control-label" style="margin-top:15px">02 · 预测模型</div>',
                unsafe_allow_html=True)
    selected_run = st.selectbox(
        "模型", ranked, index=ranked.index(default_run) if default_run in ranked else 0,
        format_func=lambda r: names.get(r, r),  # 显示名带标注,提交值仍是原始 run 名
        disabled=not runs, label_visibility="collapsed",
        on_change=_invalidate_result,  # 换模型即让旧结论卡失效
    )
    if not runs:
        st.warning("该模式下暂无模型检查点" +
                   ("（请先运行 train_5class.py）" if cur_mode == "五级分类" else ""))
    else:
        st.markdown(
            f'<div class="model-chip"><span class="dot"></span>'
            f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
            f'当前检查点 · {selected_run}</span></div>',
            unsafe_allow_html=True,
        )
    st.session_state["selected_run"] = selected_run


# ================== 页面 1:情感分析 ==================
def page_analyze(theme_id: str):
    hero_block(
        "analyze", theme_id,
        "读懂每一条影评的<em>心绪</em>",
        "当评论流过放映机的光，情绪便有了形状。输入一段中文评论，"
        "即刻得到情感极性、置信度，以及模型目光停留的词。",
        "CINEMOOD · EMOTIONAL INTELLIGENCE",
    )

    # 主输入区与模型控制台：宽主栏 + 窄侧栏，形成明确的任务动线。
    c_input, c_deck = st.columns([3.6, 1.55], gap="large")
    with c_input:
        with st.container(border=True):
            st.markdown(
                '<div class="card-heading"><span class="card-number">01</span><div>'
                '<div class="bento-title">写下此刻的评论</div>'
                '<div class="bento-hint">选择一条快捷示例，或输入任意中文评论</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(EXAMPLES))
            for col, (name, sample) in zip(cols, EXAMPLES.items()):
                col.button(name, key=f"ex_{name}", on_click=_fill_example, args=(sample,),
                           width="stretch")
            # 固定 key 的 text_area:示例经 on_click 回调写入 widget 状态,
            # 点"开始分析"触发的 rerun 不会把输入框回滚成默认文本。
            st.session_state.setdefault(
                "review_text",
                "这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!",
            )
            text = st.text_area(
                "请输入中文评论",
                key="review_text",
                height=140,
                label_visibility="collapsed",
                placeholder="在这里输入评论，模型将分析它的情感倾向……",
            )
            analyze = st.button("开始分析  →", key="btn_analyze", type="primary", width="stretch")
    with c_deck:
        with st.container(border=True):
            st.markdown(
                '<div class="card-heading"><span class="card-number">02</span><div>'
                '<div class="bento-title">模型控制台</div>'
                '<div class="bento-hint">选择任务粒度与已训练的检查点</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            render_control()

    # 预加载当前选中模型:冷加载(BERT 约 10 秒)有含模型名的 spinner 反馈,
    # 缓存命中零打扰;点击"开始分析"后只剩纯推理,显著缩短可感知耗时。
    predictor = load_predictor(st.session_state.get("selected_run"))

    if analyze:
        # 新一次点击先让旧结论失效:输入为空/无有效词语时也不残留上一轮结果卡
        st.session_state.pop("analyze_result", None)
        clicked = (text or "").strip()
        if not clicked:
            st.warning("请输入评论内容")
        elif not _has_valid_tokens(clicked):
            st.warning("输入未包含可分析的有效词语,无法判断")
        else:
            with st.spinner("模型推理中…"):
                r = predictor.predict(clicked)
            # 结果写入 session 并绑定产生它的模式与模型,供下方严格路径渲染
            st.session_state["analyze_result"] = {
                "r": r,
                "run": predictor.run_name,
                "mode": st.session_state.get("task_mode", "二分类"),
            }

    # 严格路径:结论只在"确属当前任务模式 + 当前模型"时渲染;
    # 切模式/换模型(回调已清)后页面立即回到纯输入态,不再残留旧结果卡。
    res = st.session_state.get("analyze_result")
    if res is not None:
        if (res["mode"] == st.session_state.get("task_mode", "二分类")
                and res["run"] == st.session_state.get("selected_run")):
            _render_analysis(predictor, res["r"], theme_id)
        else:
            st.session_state.pop("analyze_result", None)
    page_footer()


def _render_analysis(predictor, r: dict, theme_id: str):
    """渲染一次成功预测:结论卡、置信度、概率明细与注意力高亮。"""
    # ---- 结论卡：情绪颜色只承担语义，不使用发光效果 ----
    if r.get("num_classes") == 5:
        name, emoji, color, glow, en = LEVEL_STYLES[r["label"]]
    else:
        name, emoji, color, glow, en = BIN_STYLES[r["label"]]
    st.markdown(
        f'<div class="verdict" style="--vc:{color};--vg:{glow}">'
        f'<div class="v-emoji">{emoji}</div>'
        f'<div><div class="v-label" style="color:{color}">{name} '
        f'<span style="font-size:11px;letter-spacing:1.5px;opacity:.68;font-family:Georgia,serif;font-style:normal">{en}</span></div>'
        f'<div class="v-sub">模型：{predictor.run_name} · 置信度 {r["confidence"]:.1%}</div></div>'
        f'<div class="v-pct" style="color:{color}">{r["confidence"]:.1%}</div></div>',
        unsafe_allow_html=True,
    )

    # ---- 置信度条 + 概率明细 ----
    st.markdown(
        f'<div class="conf-bar"><div class="conf-fill" style="width:{min(r["confidence"],1.0)*100:.1f}%"></div></div>',
        unsafe_allow_html=True,
    )
    if r.get("num_classes") == 5:
        pcols = st.columns(5)
        for (name, emoji, color, _glow, _en), col in zip(LEVEL_STYLES, pcols):
            col.metric(f"{name}概率", f"{r['probs'][name]:.1%}")  # 标签格式与二元模式统一("XX概率")
    else:
        c1, c2 = st.columns(2)
        c1.metric("负面概率", f"{r['probs']['负面']:.1%}")
        c2.metric("正面概率", f"{r['probs']['正面']:.1%}")

    # ---- 注意力关键词：透明度映射权重，辅助理解模型判断 ----
    if r["attn_weights"] is not None:
        with st.container(border=True):
            st.markdown(
                '<div class="card-heading" style="margin-top:2px"><span class="card-number">03</span><div>'
                '<div class="bento-title">模型的目光 · 注意力权重</div>'
                '<div class="bento-hint">颜色越深，模型判断时越关注该词；悬停词语可查看层次</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            cr = _theme_c_rgb(theme_id)
            weights = r["attn_weights"]
            w_max = max(weights) if max(weights) > 0 else 1.0
            spans = []
            for w, weight in zip(r["tokens"], weights):
                alpha = 0.10 + 0.68 * (weight / w_max)
                spans.append(f'<span class="attn-word" style="background-color:rgba({cr},{alpha:.2f})">{w}</span>')
            st.markdown('<div style="line-height:2.3">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
            top = sorted(zip(r["tokens"], weights), key=lambda x: -x[1])[:6]
            chips = " ".join(f'<span class="attn-chip">{w} · {wt:.0%}</span>' for w, wt in top)
            st.markdown(
                f'<div style="margin-top:10px;color:var(--ink-3);font-size:11.5px">注意力最高的词：{chips}</div>',
                unsafe_allow_html=True,
            )
    elif predictor.is_bert:
        # BERT(含五级)走分类头,无词级注意力权重 —— 与"无注意力层的 Bi-LSTM"区分开
        st.info("当前加载的是 BERT 模型,不提供词级注意力高亮。可在模型控制台选择 Bi-LSTM+Attention 模型以获得关键词高亮。")
    else:
        st.info("当前加载的是基础 Bi-LSTM 模型(无注意力层)。可在模型控制台选择 Bi-LSTM+Attention 模型以获得关键词高亮。")


# ================== 页面 2:实验仪表盘 ==================
def page_dashboard(theme_id: str):
    df = load_summary_csv()
    df5 = load_summary_csv_5class()
    n_bin = len(df) if df is not None else 0
    n_5c = len(df5) if df5 is not None else 0
    if n_bin:  # 实验组数随指标总表动态生成,不再写死
        title_html = f"{n_bin} 组实验的<em>全景</em>"
        sub = (f"二分类 {n_bin} 组对比实验"
               + (f"与五级分类 {n_5c} 组拓展实验" if n_5c else "与五级分类拓展实验")
               + "，集中呈现训练曲线、混淆矩阵、超参数与模型结构。")
    else:
        title_html = "全部实验的<em>全景</em>"
        sub = "二分类对比实验与五级分类拓展实验，集中呈现训练曲线、混淆矩阵、超参数与模型结构。"
    hero_block(
        "dashboard", theme_id,
        title_html,
        sub,
        "EXPERIMENT ARCHIVE · EVIDENCE",
    )

    # ---- 顶部指标 Bento 卡 ----
    if df is not None and len(df):
        best = df.loc[df["test_f1"].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最佳测试 F1", f"{best['test_f1']:.4f}")
        m2.metric("最佳测试准确率", f"{best['test_acc']:.4f}")
        m3.metric("对比实验组数", f"{len(df)} 组")
        m4.metric("测试集规模", "906 条")
        st.markdown(
            f'<div class="bento" style="margin:14px 0 10px;display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
            f'<span class="card-number">BEST</span>'
            f'<b style="color:var(--ink)">当前最佳模型</b>'
            f'<span style="color:var(--accent);font-weight:700">{best["run_name"]}</span>'
            f'<span style="color:var(--ink-3);font-size:11.5px">P={best["test_precision"]} · R={best["test_recall"]} · '
            f"F1={best['test_f1']} · Acc={best['test_acc']}</span></div>",
            unsafe_allow_html=True,
        )

    tab_table, tab_curves, tab_attn, tab_5c, tab_model = st.tabs(
        ["超参对比总表", "训练曲线与混淆矩阵", "注意力热力图", "五级分类实验", "模型结构"]
    )

    # ---- Tab 1:对比总表 ----
    with tab_table:
        if df is not None and len(df):
            show = df.copy()
            show.insert(0, "★", "")  # 最佳行打星标记
            show.loc[df["test_f1"].idxmax(), "★"] = "★"
            st.dataframe(show, width="stretch", hide_index=True, height=420)
            st.caption("★ 列为测试 F1 最高的一组;完整数据见 outputs/experiments_summary.csv")
        else:
            st.warning("尚未找到 outputs/experiments_summary.csv,请先完成训练实验。")

    # ---- Tab 2:训练曲线 / 混淆矩阵(按实验分组) ----
    with tab_curves:
        runs = sorted({p[:-len("_curves.png")] for p in os.listdir(OUT_DIR)
                       if p.endswith("_curves.png")}) if os.path.isdir(OUT_DIR) else []
        if runs:
            for run in runs:
                with st.expander(f"实验 · {run}", expanded=(run == runs[0])):
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

    # ---- Tab 4:五级分类拓展实验(独立于二分类 14 组) ----
    with tab_5c:
        df5 = load_summary_csv_5class()
        if df5 is not None and len(df5):
            best5 = df5.loc[df5["test_macro_f1"].idxmax()]
            b1, b2, b3 = st.columns(3)
            b1.metric("最佳测试 macro-F1", f"{best5['test_macro_f1']:.4f}")
            b2.metric("对应测试准确率", f"{best5['test_acc']:.4f}")
            b3.metric("实验组数", f"{len(df5)} 组")
            show5 = df5.copy()
            show5.insert(0, "★", "")
            show5.loc[df5["test_macro_f1"].idxmax(), "★"] = "★"
            st.dataframe(show5, width="stretch", hide_index=True)
            st.caption(
                f"拓展任务:豆瓣电影短评(1-5 星 → 非常负面/负面/中性/正面/非常正面),"
                f"数据 5 万条五类均衡;当前最佳:{best5['run_name']}。"
                "细粒度星级分类难度显著高于二分类(相邻星级语义连续),macro-F1 偏低属正常现象。"
            )
            runs5 = sorted({p[:-len("_5c_curves.png")] for p in os.listdir(OUT_DIR)
                            if p.endswith("_5c_curves.png")}) if os.path.isdir(OUT_DIR) else []
            for run in runs5:
                with st.expander(f"五级实验 · {run}", expanded=(run == runs5[0])):
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
def page_batch(theme_id: str):
    hero_block(
        "batch", theme_id,
        "千百条心声，<em>一场映毕</em>",
        "上传含 text 列的 CSV，或粘贴多行评论。批量预测、统计占比并下载结果，让分散的表达形成清晰的整体判断。",
        "BATCH STUDIO · COLLECTIVE EMOTION",
    )

    with st.container(border=True):
        st.markdown(
            '<div class="card-heading"><span class="card-number">01</span><div>'
            '<div class="bento-title">任务模式与模型</div>'
            '<div class="bento-hint">批量任务与单句分析共用同一套模型配置</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        cd1, cd2 = st.columns([1.2, 1.8], gap="large")
        with cd1:
            render_control()
        with cd2:
            st.markdown(
                '<div style="color:var(--ink-2);font-size:12px;line-height:2;padding:6px 0">'
                '<b style="color:var(--ink)">批量处理说明</b><br>'
                '支持二分类与五级分类。CSV 文件需包含 <b>text</b> 列；'
                '也可以在下方粘贴区直接输入，每行作为一条独立评论。</div>',
                unsafe_allow_html=True,
            )

    predictor = load_predictor(st.session_state.get("selected_run"))
    texts = []

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(
                '<div class="card-heading"><span class="card-number">02</span><div>'
                '<div class="bento-title">上传 CSV 文件</div>'
                '<div class="bento-hint">文件需包含 text 列，适合较大的评论集合</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            up = st.file_uploader("上传 CSV", type=["csv"], label_visibility="collapsed")
        if up is not None:
            try:
                df = pd.read_csv(up)
                if "text" not in df.columns:
                    st.error(f"CSV 中未找到 text 列，实际列：{list(df.columns)}")
                else:
                    texts = df["text"].dropna().astype(str).tolist()
                    st.info(f"已读取 {len(texts)} 条评论")
            except Exception as e:  # noqa: BLE001
                st.error(f"CSV 解析失败：{e}")
    with right:
        with st.container(border=True):
            st.markdown(
                '<div class="card-heading"><span class="card-number">03</span><div>'
                '<div class="bento-title">粘贴多行评论</div>'
                '<div class="bento-hint">每行一条评论，适合快速批量体验</div>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            pasted = st.text_area(
                "粘贴评论",
                value="屏幕效果非常好,系统流畅,值得购买。\n电池太不耐用,半天就没电,后悔买了。",
                height=150,
                label_visibility="collapsed",
            )
            run_batch = st.button("开始批量预测  →", type="primary", width="stretch")

    if run_batch and pasted.strip():
        texts += [line.strip() for line in pasted.splitlines() if line.strip()]

    if not texts:
        st.info("请上传 CSV 或粘贴评论后点击批量预测")
        page_footer()
        return

    with st.spinner("批量推理中…"):
        rows = []
        for t in texts:
            if _has_valid_tokens(t):
                rows.append(predictor.predict(t) | {"text": t[:200]})
            else:  # 空/纯符号行不进模型,结果表标注"无法判断"
                rows.append({"text": t[:200], "polarity": "无法判断",
                             "confidence": None, "num_classes": predictor.num_classes})
    result = pd.DataFrame([
        {"text": r["text"], "情感极性": r["polarity"],
         "置信度": "—" if r.get("confidence") is None else f"{r['confidence']:.1%}"}
        for r in rows
    ])

    # ---- 统计概览 + 极性占比条(二分类两段;五级五段;占比仅统计可判断的行) ----
    st.markdown(
        '<div class="card-heading" style="margin:28px 0 12px"><span class="card-number">04</span><div>'
        '<div class="bento-title">预测结果</div>'
        '<div class="bento-hint">共 %d 条评论 · 情感分布与逐条明细</div></div></div>' % len(result),
        unsafe_allow_html=True,
    )
    valid = result[result["情感极性"] != "无法判断"]
    n_invalid = len(result) - len(valid)
    is_5c = predictor.num_classes == 5
    if len(valid) and is_5c:
        counts = {name: int((valid["情感极性"] == name).sum()) for name, *_ in LEVEL_STYLES}
        segs = "".join(
            f'<div class="ratio-seg" style="width:{counts[name] / len(valid) * 100:.1f}%;'
            f'background:{color}" title="{name} {counts[name]} 条">'
            f'{"%d%%" % round(counts[name] / len(valid) * 100)}</div>'
            for name, _e, color, _g, _en in LEVEL_STYLES
        )
        st.markdown(f'<div class="ratio-bar">{segs}</div>', unsafe_allow_html=True)
        st.caption(" · ".join(f"{name} {counts[name]}" for name, _e, _c, _g, _en in LEVEL_STYLES))
    elif len(valid):
        n_pos = int((valid["情感极性"] == "正面").sum())
        n_neg = len(valid) - n_pos
        pct_pos = n_pos / len(valid) * 100
        pct_neg = 100 - pct_pos
        st.markdown(
            f'<div class="ratio-bar">'
            f'<div class="ratio-seg" style="width:{pct_pos:.1f}%;background:#4F7B60">正面 {n_pos}（{pct_pos:.0f}%）</div>'
            f'<div class="ratio-seg" style="width:{pct_neg:.1f}%;background:#BC6A48">负面 {n_neg}（{pct_neg:.0f}%）</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    if n_invalid:
        st.caption(f"{n_invalid} 条输入为空或纯符号,未包含可分析的有效词语,已标注为「无法判断」并跳过模型预测。")
    st.dataframe(result, width="stretch", hide_index=True)
    st.download_button(
        "下载预测结果 CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_predictions.csv",
        mime="text/csv",
        width="stretch",
    )
    page_footer()


# ================== 主入口:顶部导航路由(query 参数:page + theme + task) ==================
def main():
    page = st.query_params.get("page", "analyze")
    if page not in {key for key, *_ in NAV_ITEMS}:
        page = "analyze"
    theme_id = st.query_params.get("theme", DEFAULT_THEME)
    if theme_id not in THEMES:
        theme_id = DEFAULT_THEME

    # 初始化任务模式:优先跟随 query 参数(?task=binary|five,导航链接携带,跨页不丢);
    # 未指定时保持当前会话,首次进入默认二分类
    task_param = st.query_params.get("task", "")
    if task_param == "five":
        st.session_state["task_mode"] = "五级分类"
    elif task_param == "binary":
        st.session_state["task_mode"] = "二分类"
    st.session_state.setdefault("task_mode", "二分类")

    inject_css(theme_id)
    top_nav(page, theme_id)
    if page == "analyze":
        page_analyze(theme_id)
    elif page == "dashboard":
        page_dashboard(theme_id)
    else:
        page_batch(theme_id)


if __name__ == "__main__":
    main()
