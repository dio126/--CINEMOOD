# -*- coding: utf-8 -*-
"""
情绪放映室 · CINEMOOD —— 中文电影评论情感分析系统
====================================================
多主题 · 午夜电光风框架,内置 5 套可切换色彩主题(顶部导航色点即点即换):
  electric 午夜电光(午夜黑+电蓝+橙) · aurora 极光绿洲(宝石蓝+海蓝+霓虹绿)
  blush 蓝粉甜梦(深蓝+白+粉) · amber 琥珀暖夜(天蓝+云白+琥珀红)
  lava 熔岩荧光(橙红+雪白+荧光黄)
主题经 query 参数 ?theme= 持久化,导航链接自动携带当前主题。
艺术字体:标题/品牌用 华文行楷/楷体(系统自带,离线可用,优雅回退);
英文眉题用衬线斜体。正文保持系统无衬线保证可读性。
交互特效:鼠标跟随光晕、Bento 卡片 3D 倾斜(同源 iframe 注入 JS)、
流动极光 Hero、渐变流光大标题、悬停辉光、渐入动画。

仅做视觉与导航层,推理逻辑完全不变:src/predict.py → 共用 src/preprocess.py。
启动:.venv/Scripts/python.exe -m streamlit run app/app.py --server.headless true
"""

import json
import os
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

# ================== 主题库(5 套,均基于午夜深底) ==================
# p 主色 / c 亮部 / a 暖点缀;fx1 fx2 为鼠标光晕双色;moods 为三页各自的
# (极光色1, 极光色2, 眉题色, 标题渐变中段色, 标题渐变尾色)
THEMES = {
    "electric": {
        "label": "午夜电光", "p": "#3B82F6", "c": "#22D3EE", "a": "#FB923C",
        "fx1": "rgba(34,211,238,.10)", "fx2": "rgba(59,130,246,.05)",
        "moods": {
            "analyze": ("#2563EB", "#FB923C", "#FB923C", "#22D3EE", "#FB923C"),
            "dashboard": ("#1E40AF", "#34D399", "#34D399", "#34D399", "#93C5FD"),
            "batch": ("#3730A3", "#F472B6", "#F472B6", "#F9A8D4", "#F472B6"),
        },
    },
    "aurora": {
        "label": "极光绿洲", "p": "#14B8A6", "c": "#34D399", "a": "#A3E635",
        "fx1": "rgba(52,211,153,.10)", "fx2": "rgba(20,184,166,.05)",
        "moods": {
            "analyze": ("#0EA5E9", "#34D399", "#A3E635", "#34D399", "#A3E635"),
            "dashboard": ("#1D4ED8", "#A3E635", "#34D399", "#A3E635", "#7DD3FC"),
            "batch": ("#14B8A6", "#F0ABFC", "#F0ABFC", "#F0ABFC", "#A3E635"),
        },
    },
    "blush": {
        "label": "蓝粉甜梦", "p": "#818CF8", "c": "#F9A8D4", "a": "#F472B6",
        "fx1": "rgba(249,168,212,.10)", "fx2": "rgba(129,140,248,.05)",
        "moods": {
            "analyze": ("#3B82F6", "#F472B6", "#F472B6", "#F9A8D4", "#F472B6"),
            "dashboard": ("#6366F1", "#F9A8D4", "#C4B5FD", "#C4B5FD", "#F9A8D4"),
            "batch": ("#818CF8", "#FDA4AF", "#FDA4AF", "#FDA4AF", "#818CF8"),
        },
    },
    "amber": {
        "label": "琥珀暖夜", "p": "#38BDF8", "c": "#FDE68A", "a": "#FB7185",
        "fx1": "rgba(253,230,138,.09)", "fx2": "rgba(56,189,248,.05)",
        "moods": {
            "analyze": ("#0284C7", "#FB7185", "#FDE68A", "#FDE68A", "#FB7185"),
            "dashboard": ("#7DD3FC", "#FDE68A", "#FDE68A", "#7DD3FC", "#FDE68A"),
            "batch": ("#38BDF8", "#FDA4AF", "#FDA4AF", "#FDA4AF", "#38BDF8"),
        },
    },
    "lava": {
        "label": "熔岩荧光", "p": "#F97316", "c": "#FACC15", "a": "#F43F5E",
        "fx1": "rgba(250,204,21,.09)", "fx2": "rgba(249,115,22,.05)",
        "moods": {
            "analyze": ("#EA580C", "#FACC15", "#FACC15", "#FACC15", "#F97316"),
            "dashboard": ("#DC2626", "#FDE047", "#FDE047", "#FDE047", "#FB923C"),
            "batch": ("#F97316", "#FB7185", "#FB7185", "#FDA4AF", "#FACC15"),
        },
    },
}
DEFAULT_THEME = "electric"


def _rgba(hex_color: str, alpha: float) -> str:
    """'#RRGGBB' + 透明度 → 'rgba(r,g,b,a)'(主题样式生成用)。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def theme_style(tid: str) -> str:
    """按主题生成覆盖样式(基础 CUSTOM_CSS 以午夜电光为默认,这里按需覆写)。"""
    t = THEMES[tid]
    p, c, a = t["p"], t["c"], t["a"]
    return f"""
<style>
.brand-mark {{ background: linear-gradient(135deg, {p}, {c}); box-shadow: 0 0 18px {_rgba(c, .35)}; }}
.brand:hover .brand-mark {{ box-shadow: 0 0 26px {_rgba(c, .55)}; }}
.navpill:hover {{ background: {_rgba(p, .14)}; color: {c}; }}
.navpill.active {{ background: linear-gradient(135deg, {p}, {c}); box-shadow: 0 0 20px {_rgba(c, .40)}; }}
.mode-badge {{ background: {_rgba(a, .08)}; border: 1px dashed {_rgba(a, .35)}; }}
.mode-badge .dot {{ background: {a}; box-shadow: 0 0 10px {_rgba(a, .8)}; }}
.stButton > button:hover {{
    border-color: {_rgba(c, .5)} !important; background: {_rgba(c, .10)} !important;
    color: {c} !important; box-shadow: 0 0 22px {_rgba(c, .18)} !important;
}}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {p}, {c}) !important;
    box-shadow: 0 0 24px {_rgba(c, .35)} !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    box-shadow: 0 0 34px {_rgba(c, .55)} !important;
}}
.conf-fill {{
    background: linear-gradient(90deg, {p}, {c}, {c});
    background-size: 200% auto; box-shadow: 0 0 16px {_rgba(c, .5)};
}}
.attn-chip {{ background: {_rgba(c, .10)}; color: {c}; border: 1px solid {_rgba(c, .30)}; }}
.attn-chip:hover {{ background: {_rgba(c, .22)}; box-shadow: 0 0 14px {_rgba(c, .25)}; }}
.attn-word:hover {{ box-shadow: 0 0 12px {_rgba(c, .35)}; }}
[data-testid="stMetricValue"] {{ color: {c}; }}
.stTabs [data-baseweb="tab"]:hover {{ color: {c} !important; background: {_rgba(c, .08)}; }}
.stTabs [aria-selected="true"] {{ color: {c} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: linear-gradient(90deg, {p}, {c}) !important; box-shadow: 0 0 10px {_rgba(c, .6)}; }}
[data-testid="stExpander"]:hover {{ border-color: {_rgba(c, .35)} !important; }}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {{ border-color: {c} !important; box-shadow: 0 0 0 3px {_rgba(c, .12)}, 0 0 18px {_rgba(c, .15)} !important; }}
[data-baseweb="select"] > div:hover {{ border-color: {_rgba(c, .5)} !important; }}
.model-chip {{ background: {_rgba(c, .08)}; border: 1px solid {_rgba(c, .25)}; color: {c}; }}
.model-chip .dot {{ background: {a}; box-shadow: 0 0 8px {_rgba(a, .8)}; }}
.bento:hover {{ border-color: {_rgba(c, .42)}; box-shadow: 0 16px 44px rgba(3,6,14,.5), 0 0 24px {_rgba(c, .10)}; }}
[data-testid="stMetric"]:hover {{ border-color: {_rgba(c, .35)}; }}
#fx-glow {{ background: radial-gradient(circle, {t["fx1"]}, {t["fx2"]} 42%, transparent 70%); }}
.theme-dot.active {{ box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px {c}, 0 0 12px {_rgba(c, .6)}; }}
</style>
"""


# ================== 基础样式(午夜电光为默认主题,可被 theme_style 覆写) ==================
CUSTOM_CSS = """
<style>
:root {
    --bg: #070B14;            /* 午夜黑 */
    --bg-2: #0C1322;
    --ink: #E8EEF9;           /* 主文字 */
    --ink-2: #93A3BC;         /* 次级文字 */
    --ink-3: #5B6B84;         /* 弱化文字 */
    --card: rgba(148,163,184,.055);
    --line: rgba(148,163,184,.16);
}

/* ===== 基础 ===== */
html, body, .stApp { background: var(--bg); }
h1, h2, h3 { color: var(--ink); letter-spacing: .3px; }
/* 细微噪点,增加质感 */
body::after {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 4; opacity: .04;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)' opacity='0.6'/%3E%3C/svg%3E");
}
/* 彻底隐藏原生侧边栏与其开关 */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { padding-top: 1.1rem !important; padding-bottom: 3rem !important; max-width: 1200px; }

/* ===== 艺术字体:标题/品牌用书法体(系统自带,缺失时回退 serif),正文保持无衬线 ===== */
/* !important:覆盖 Streamlit 对 h1 的 "Source Sans" 字体设定 */
h1.hero-title, .brand .brand-txt, .verdict .v-label {
    font-family: "华文行楷", "STXingkai", "华文楷体", "STKaiti", "楷体", "KaiTi", "SimSun", serif !important;
}
.hero-kicker {
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic;
}

/* ===== 顶部导航条(暗色玻璃 · 吸顶) ===== */
.topnav {
    position: sticky; top: 0; z-index: 100;
    margin: 0 -1rem .2rem; padding: 10px 18px;
    display: flex; align-items: center; gap: 16px;
    background: rgba(7,11,20,.78);
    backdrop-filter: blur(16px) saturate(1.3);
    -webkit-backdrop-filter: blur(16px) saturate(1.3);
    border-bottom: 1px solid rgba(148,163,184,.14);
    box-shadow: 0 8px 30px rgba(3,6,14,.5);
}
.brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
.brand-mark {
    width: 40px; height: 40px; border-radius: 12px; font-size: 20px;
    display: flex; align-items: center; justify-content: center;
    transition: transform .25s ease, box-shadow .25s ease;
}
.brand:hover .brand-mark { transform: rotate(-8deg) scale(1.06); }
.brand-txt { font-size: 19px; font-weight: 800; color: var(--ink); line-height: 1.1; }
.brand-txt small { display: block; font-family: Georgia, serif; font-size: 9px; font-weight: 700; letter-spacing: 3.5px; color: var(--ink-3); }
.navpills { display: flex; gap: 6px; padding: 4px; border-radius: 999px; background: rgba(148,163,184,.07); border: 1px solid var(--line); }
.navpill {
    padding: 8px 18px; border-radius: 999px; text-decoration: none;
    font-size: 14px; font-weight: 700; color: var(--ink-2);
    transition: all .2s ease;
}
.navpill:hover { transform: translateY(-1px); }
.mode-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 700; color: var(--ink-2);
    margin-left: auto;
}
.mode-badge .dot { width: 8px; height: 8px; border-radius: 50%; }

/* ===== 主题色点(点击切换配色) ===== */
.theme-dots { display: flex; align-items: center; gap: 7px; padding: 6px 10px; border-radius: 999px; background: rgba(148,163,184,.07); border: 1px solid var(--line); }
.theme-dot {
    width: 16px; height: 16px; border-radius: 50%; cursor: pointer;
    border: none; padding: 0; text-decoration: none;
    transition: transform .18s ease, box-shadow .18s ease;
}
.theme-dot:hover { transform: scale(1.25); }

/* ===== Hero 区:流动极光 + 大字渐变 ===== */
.hero { position: relative; overflow: hidden; padding: 46px 34px 36px; margin-bottom: 8px; }
.aurora { position: absolute; border-radius: 50%; filter: blur(80px); opacity: .45; pointer-events: none; }
.aurora-1 { width: 460px; height: 460px; right: -90px; top: -180px; animation: drift1 14s ease-in-out infinite alternate; }
.aurora-2 { width: 340px; height: 340px; left: -70px; bottom: -160px; animation: drift2 11s ease-in-out infinite alternate; }
@keyframes drift1 { from { transform: translate(0,0) scale(1); } to { transform: translate(-40px,30px) scale(1.12); } }
@keyframes drift2 { from { transform: translate(0,0) scale(1.05); } to { transform: translate(30px,-26px) scale(.95); } }
.hero-kicker { font-size: 12px; font-weight: 800; letter-spacing: 5px; animation: fadeUp .5s ease both; }
.hero-title {
    font-size: clamp(32px, 4vw, 46px); font-weight: 900; letter-spacing: 2px; line-height: 1.3;
    margin: 12px 0 10px;
    background: linear-gradient(100deg, var(--t1, #E8EEF9) 30%, var(--t2, #22D3EE) 70%, var(--t3, #FB923C));
    -webkit-background-clip: text; background-clip: text; color: transparent;
    background-size: 200% auto; animation: fadeUp .55s ease .05s both, sheen 6s linear infinite;
    text-shadow: none;
}
@keyframes sheen { to { background-position: 200% center; } }
.hero-sub { color: var(--ink-2); font-size: 15px; line-height: 1.8; max-width: 660px; animation: fadeUp .6s ease .1s both; }
.hero-mini { padding: 28px 30px 20px; }
.hero-mini .hero-title { font-size: clamp(24px, 3vw, 34px); }

/* ===== Bento 卡片(暗色玻璃 · 悬停辉光) ===== */
.bento {
    position: relative;
    background: var(--card);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 10px 34px rgba(3,6,14,.35);
    transition: border-color .25s ease, box-shadow .25s ease;
    animation: fadeUp .55s ease both;
}
.bento-title { font-size: 15px; font-weight: 800; color: var(--ink); margin-bottom: 4px; }
.bento-hint { font-size: 12.5px; color: var(--ink-3); margin-bottom: 12px; }

/* ===== 通用按钮:胶囊 + 辉光 + 按下态 ===== */
.stButton > button {
    border-radius: 999px !important;
    border: 1px solid var(--line) !important;
    background: rgba(148,163,184,.07) !important;
    color: var(--ink-2) !important;
    font-weight: 700; letter-spacing: .3px;
    box-shadow: none;
    transition: all .18s ease !important;
}
.stButton > button:hover { transform: translateY(-2px); }
.stButton > button:active { transform: translateY(0) scale(.97); }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    color: #04101E !important;
    border: 1px solid transparent !important;
}

/* ===== 输入框 ===== */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: rgba(7,11,20,.75) !important;
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important; font-size: 15px;
    transition: border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stTextArea"] textarea:hover { border-color: rgba(148,163,184,.35) !important; }

/* ===== 结论卡(暗玻璃 · 等级色辉光) ===== */
.verdict {
    position: relative;
    display: flex; align-items: center; gap: 20px;
    border-radius: 20px; padding: 22px 28px 22px 32px; margin: 4px 0 14px;
    background: rgba(12,19,34,.82);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--line);
    box-shadow: 0 14px 40px rgba(3,6,14,.5), 0 0 30px var(--vg, rgba(34,211,238,.12));
    overflow: hidden;
    animation: fadeUp .5s ease both;
}
.verdict::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--vc, #22D3EE); box-shadow: 0 0 14px var(--vc, #22D3EE); }
.v-emoji { font-size: 50px; line-height: 1; transition: transform .2s ease; filter: drop-shadow(0 0 12px var(--vg, rgba(34,211,238,.4))); }
.verdict:hover .v-emoji { transform: scale(1.12) rotate(-6deg); }
.v-label { font-size: 27px; font-weight: 900; letter-spacing: 2px; }
.v-sub { color: var(--ink-3); font-size: 12.5px; margin-top: 5px; }
.v-pct {
    margin-left: auto; font-size: 40px; font-weight: 900; letter-spacing: -1px;
    font-family: Consolas, "Courier New", monospace;
}

/* ===== 置信度条(流光) ===== */
.conf-bar { height: 12px; border-radius: 999px; overflow: hidden; background: rgba(148,163,184,.10); border: 1px solid var(--line); }
.conf-fill {
    height: 100%; border-radius: 999px;
    background-size: 200% auto;
    animation: flow 2.4s linear infinite;
    transition: width .8s cubic-bezier(.2,.7,.3,1);
}
@keyframes flow { to { background-position: 200% center; } }

/* ===== 注意力词高亮(主题亮色深浅 · 悬停放大) ===== */
.attn-word {
    display: inline-block; padding: 3px 8px; margin: 3px;
    border-radius: 8px; font-size: 16.5px; color: var(--ink);
    transition: transform .15s ease, box-shadow .15s ease;
}
.attn-word:hover { transform: scale(1.14); }
.attn-chip {
    display: inline-block; padding: 5px 13px; margin: 3px;
    border-radius: 999px; font-size: 13px; font-weight: 700;
    transition: all .18s ease;
}
.attn-chip:hover { transform: translateY(-2px); }

/* ===== 指标卡 ===== */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--line); border-radius: 18px;
    padding: 16px 18px;
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
    transition: all .22s ease;
    animation: fadeUp .5s ease both;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); }
[data-testid="stMetricLabel"] p { font-size: 12px !important; color: var(--ink-3) !important; }
[data-testid="stMetricValue"] { font-family: Consolas, "Courier New", monospace; }

/* ===== expander / tabs ===== */
[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: 16px !important;
    background: var(--card); transition: border-color .2s ease;
}
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--ink-3) !important; font-weight: 700; padding: 8px 16px; border-radius: 10px 10px 0 0; }
.stTabs [data-baseweb="tab-highlight"] { height: 3px; border-radius: 3px 3px 0 0; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ===== 下拉框(模型选择) ===== */
[data-baseweb="select"] > div {
    background: rgba(7,11,20,.75) !important;
    border-radius: 12px !important; border-color: var(--line) !important;
    transition: border-color .18s ease, box-shadow .18s ease;
}
.model-chip {
    display: inline-flex; align-items: center; gap: 7px;
    margin-top: 4px; padding: 5px 12px;
    border-radius: 999px;
    font-size: 11.5px; max-width: 100%; font-weight: 700;
}
.model-chip .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }

/* ===== 页脚 ===== */
.page-footer {
    margin-top: 36px; padding: 18px 0 6px; border-top: 1px solid var(--line);
    color: var(--ink-3); font-size: 12.5px; text-align: center; letter-spacing: 2px;
    font-family: Georgia, serif; font-style: italic;
}
.page-footer .dot { margin: 0 8px; }

/* ===== 批量页极性占比条 ===== */
.ratio-bar { display: flex; height: 30px; border-radius: 999px; overflow: hidden; border: 1px solid var(--line); }
.ratio-seg {
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 12px; min-width: 0; overflow: hidden; color: #04101E;
    transition: filter .18s ease;
}
.ratio-seg:hover { filter: brightness(1.25); }

/* ===== 渐入动画 ===== */
@keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
</style>
"""

# ================== 鼠标特效:光晕跟随 + 卡片 3D 倾斜 ==================
# Streamlit 的 st.markdown 会剥离 <script>,因此用 components.html 的同源 iframe
# 注入 JS 操作父页面 DOM;跨域或环境不支持时静默降级为纯 CSS 悬停效果。
# 光晕颜色由 #fx-glow 规则控制(theme_style 会按主题覆写)。
FX_JS = """
<script>
(function () {
  try {
    const doc = parent.document;
    if (doc.getElementById("fx-glow")) return;  // 防止重跑后重复注入

    const style = doc.createElement("style");
    style.textContent = [
      "#fx-glow{position:fixed;width:560px;height:560px;border-radius:50%;pointer-events:none;",
      "z-index:3;transform:translate(-50%,-50%);mix-blend-mode:screen;",
      "background:radial-gradient(circle,rgba(34,211,238,.10),rgba(59,130,246,.05) 42%,transparent 70%);}",
    ].join("");
    doc.head.appendChild(style);
    const glow = doc.createElement("div");
    glow.id = "fx-glow";
    doc.body.appendChild(glow);
    let gx = innerWidth / 2, gy = innerHeight / 2, cx = gx, cy = gy;
    doc.addEventListener("mousemove", (e) => { gx = e.clientX; gy = e.clientY; });
    (function loop() {  // 缓动跟随,更顺滑
      cx += (gx - cx) * 0.12; cy += (gy - cy) * 0.12;
      glow.style.left = cx + "px"; glow.style.top = cy + "px";
      requestAnimationFrame(loop);
    })();

    // Bento 卡片 3D 倾斜(事件委托,Streamlit 重跑后自动对新卡片生效)
    doc.addEventListener("mousemove", (e) => {
      const card = e.target && e.target.closest ? e.target.closest(".bento") : null;
      doc.querySelectorAll(".bento.fx-tilt").forEach((el) => {
        if (el !== card) { el.style.transform = ""; el.classList.remove("fx-tilt"); }
      });
      if (!card) return;
      const r = card.getBoundingClientRect();
      const rx = ((e.clientY - r.top) / r.height - 0.5) * -5;
      const ry = ((e.clientX - r.left) / r.width - 0.5) * 5;
      card.classList.add("fx-tilt");
      card.style.transform =
        "perspective(950px) rotateX(" + rx.toFixed(2) + "deg) rotateY(" + ry.toFixed(2) + "deg) translateY(-3px)";
    });
  } catch (err) { /* 同源不可用等场景:静默降级 */ }
})();
</script>
"""


def inject_css(theme_id: str):
    """注入基础样式 + 当前主题覆盖样式 + 鼠标特效。"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(theme_style(theme_id), unsafe_allow_html=True)
    components.html(FX_JS, height=0, scrolling=False)


# ================== 顶部导航(替代侧边栏) ==================
NAV_ITEMS = [
    ("analyze", "情感分析", "🎭"),
    ("dashboard", "实验仪表盘", "📊"),
    ("batch", "批量分析", "📦"),
]


def top_nav(active_key: str, theme_id: str):
    """吸顶玻璃导航条:品牌 + 胶囊导航 + 主题色点 + 任务模式徽章。

    导航与主题均走 query 参数并互相携带(?page=x&theme=y),
    显式 target="_self" 以抵消 Streamlit 对 <a> 强加的 target="_blank"。
    """
    pills = "".join(
        f'<a class="navpill{" active" if key == active_key else ""}" '
        f'href="?page={key}&theme={theme_id}" target="_self">{label}</a>'
        for key, label, _icon in NAV_ITEMS
    )
    dots = "".join(
        f'<a class="theme-dot{" active" if tid == theme_id else ""}" '
        f'href="?page={active_key}&theme={tid}" target="_self" '
        f'title="{t["label"]}" style="background:linear-gradient(135deg,{t["p"]},{t["c"]})"></a>'
        for tid, t in THEMES.items()
    )
    mode = st.session_state.get("task_mode", "二分类")
    st.markdown(
        f'<nav class="topnav">'
        f'<a class="brand" href="?page=analyze&theme={theme_id}" target="_self">'
        f'<span class="brand-mark">🎬</span>'
        f'<span class="brand-txt">情绪放映室<small>CINEMOOD</small></span></a>'
        f'<div class="navpills">{pills}</div>'
        f'<div class="theme-dots">{dots}</div>'
        f'<div class="mode-badge"><span class="dot"></span>{mode}</div></nav>',
        unsafe_allow_html=True,
    )


def hero_block(page_key: str, theme_id: str, title_html: str, sub: str, kicker: str, mini: bool = False):
    """页面 Hero 区:流动极光 + 书法渐变大标题(每页 × 每主题独立配色)。"""
    mood = THEMES[theme_id]["moods"].get(page_key, THEMES[theme_id]["moods"]["analyze"])
    c1, c2, k, t2, t3 = mood
    st.markdown(
        f'<header class="hero{" hero-mini" if mini else ""}">'
        f'<div class="aurora aurora-1" style="background:radial-gradient(circle,{c1} 0%,transparent 65%)"></div>'
        f'<div class="aurora aurora-2" style="background:radial-gradient(circle,{c2} 0%,transparent 65%)"></div>'
        f'<div class="hero-kicker" style="color:{k}">{kicker}</div>'
        f'<h1 class="hero-title" style="--t2:{t2};--t3:{t3}">{title_html}</h1>'
        f'<p class="hero-sub">{sub}</p></header>',
        unsafe_allow_html=True,
    )


def page_footer():
    st.markdown(
        '<div class="page-footer">情绪放映室 · CINEMOOD<span class="dot">◆</span>'
        '中文电影评论情感分析<span class="dot">◆</span>Bi-LSTM / Attention / 预训练词向量</div>',
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


# 五级分类的等级样式:(类名, emoji, 主题色, 辉光色, 英文标注)——结论卡/堆叠条/概率卡共用
LEVEL_STYLES = [
    ("非常负面", "😡", "#F87171", "rgba(248,113,113,.25)", "VERY NEGATIVE"),
    ("负面", "😞", "#FB923C", "rgba(251,146,60,.25)", "NEGATIVE"),
    ("中性", "😐", "#94A3B8", "rgba(148,163,184,.22)", "NEUTRAL"),
    ("正面", "🙂", "#34D399", "rgba(52,211,153,.25)", "POSITIVE"),
    ("非常正面", "🤩", "#22D3EE", "rgba(34,211,238,.28)", "VERY POSITIVE"),
]
BIN_STYLES = {
    1: ("正面", "😊", "#34D399", "rgba(52,211,153,.25)", "POSITIVE"),
    0: ("负面", "😞", "#FB923C", "rgba(251,146,60,.25)", "NEGATIVE"),
}

# 快速示例(点击即填入输入框,session key 与 text_area 绑定)
EXAMPLES = {
    "😊 正面示例": "这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!",
    "😞 负面示例": "我不满意这家酒店的服务,房间也不干净,隔音差到一夜没睡好。",
    "🎬 电影评论": "这部电影太精彩了,剧情紧凑不拖沓,演员演技在线,强烈推荐!",
    "⚡ 否定句测试": "总体不满意,不会再来。性价比也不高,感觉被骗了。",
}


def _theme_c_rgb(theme_id: str) -> str:
    """主题亮色的 'r,g,b' 字符串(注意力词高亮透明度渐变用)。"""
    h = THEMES[theme_id]["c"].lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


# ================== 控制台(任务模式 + 模型选择,融入页面) ==================
def render_control():
    """任务模式分段 + 模型选择(嵌入页面控制台卡)。"""
    cur_mode = st.session_state.get("task_mode", "二分类")

    # ---- 任务模式:两个胶囊按钮组成的分段控件 ----
    c1, c2 = st.columns(2)
    if c1.button("⚖️ 二元 正/负", type="primary" if cur_mode == "二分类" else "secondary",
                 use_container_width=True):
        if cur_mode != "二分类":
            st.session_state["task_mode"] = "二分类"
            st.rerun()
    if c2.button("⭐ 五级 细粒度", type="primary" if cur_mode == "五级分类" else "secondary",
                 use_container_width=True, help="拓展任务:豆瓣 1-5 星 → 非常负面~非常正面"):
        if cur_mode != "五级分类":
            st.session_state["task_mode"] = "五级分类"
            st.rerun()
    cur_mode = st.session_state.get("task_mode", "二分类")

    # ---- 模型选择:按任务模式列出对应 ckpt,默认取该模式最佳 ----
    if cur_mode == "二分类":
        runs = list_model_runs()
        default_run = None
        best_log = os.path.join(OUT_DIR, "best_model_info.json")
        if os.path.exists(best_log):
            with open(best_log, encoding="utf-8") as f:
                default_run = json.load(f).get("run_name")
    else:
        runs = list_model_runs_5class()
        default_run = None
        df5 = load_summary_csv_5class()
        if df5 is not None and len(df5):
            default_run = df5.loc[df5["test_macro_f1"].idxmax(), "run_name"]
    if default_run not in runs and runs:
        default_run = runs[0]
    selected_run = st.selectbox(
        "模型", runs, index=runs.index(default_run) if default_run in runs else 0,
        disabled=not runs, label_visibility="collapsed",
    )
    if not runs:
        st.warning("该模式下暂无模型检查点" +
                   ("(请先运行 train_5class.py)" if cur_mode == "五级分类" else ""))
    if selected_run:
        st.markdown(
            f'<div class="model-chip"><span class="dot"></span>'
            f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{selected_run}</span></div>',
            unsafe_allow_html=True,
        )
    st.session_state["selected_run"] = selected_run  # 供各页面获取所选模型


# ================== 页面 1:情感分析 ==================
def page_analyze(theme_id: str):
    hero_block(
        "analyze", theme_id,
        "读懂每一条影评的<em>心绪</em>",
        "当评论流过放映机的光,情绪便有了形状 · 输入一段中文评论,"
        "即刻得到情感极性、置信度,以及模型目光停留的词。",
        "CINEMOOD · MACHINE LEARNING",
    )

    # ---- Bento 网格:左侧输入主卡 + 右侧模型控制台卡 ----
    c_input, c_deck = st.columns([5, 2], gap="large")
    with c_input:
        st.markdown('<div class="bento"><div class="bento-title">✍️ 输入评论</div>'
                    '<div class="bento-hint">试试下面的快捷示例,或直接输入任意中文评论</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for col, (name, sample) in zip(cols, EXAMPLES.items()):
            if col.button(name, use_container_width=True):
                st.session_state["review_input"] = sample
        text = st.text_area(
            "请输入中文评论",
            value=st.session_state.pop("review_input",
                                       "这家酒店位置很好,房间干净整洁,服务态度也特别热情,下次还会再来!"),
            height=110, label_visibility="collapsed",
        )
        analyze = st.button("🚀 开始分析", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c_deck:
        st.markdown('<div class="bento"><div class="bento-title">🎛️ 模型控制台</div>'
                    '<div class="bento-hint">切换任务模式与模型</div>',
                    unsafe_allow_html=True)
        render_control()
        st.markdown("</div>", unsafe_allow_html=True)

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

    # ---- 结论卡:按任务模式渲染(暗玻璃 + 等级色辉光) ----
    if r.get("num_classes") == 5:
        name, emoji, color, glow, en = LEVEL_STYLES[r["label"]]
    else:
        name, emoji, color, glow, en = BIN_STYLES[r["label"]]
    st.markdown(
        f'<div class="verdict" style="--vc:{color};--vg:{glow}">'
        f'<div class="v-emoji">{emoji}</div>'
        f'<div><div class="v-label" style="color:{color}">{name} '
        f'<span style="font-size:14px;letter-spacing:2px;opacity:.75;font-family:Georgia,serif;font-style:italic">{en}</span></div>'
        f'<div class="v-sub">模型:{predictor.run_name} · 置信度 {r["confidence"]:.1%}</div></div>'
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
            col.metric(f"{emoji} {name}", f"{r['probs'][name]:.1%}")
    else:
        c1, c2 = st.columns(2)
        c1.metric("😞 负面概率", f"{r['probs']['负面']:.1%}")
        c2.metric("😊 正面概率", f"{r['probs']['正面']:.1%}")

    # ---- 注意力关键词高亮(主题亮色深浅 = 权重大小) ----
    if r["attn_weights"] is not None:
        st.markdown(
            '<div class="bento" style="margin-top:16px"><div class="bento-title">🔍 模型的目光 · 注意力权重</div>'
            '<div class="bento-hint">颜色越亮,模型判断时越关注该词;悬停单词可放大</div>',
            unsafe_allow_html=True,
        )
        cr = _theme_c_rgb(theme_id)
        weights = r["attn_weights"]
        w_max = max(weights) if max(weights) > 0 else 1.0
        spans = []
        for w, weight in zip(r["tokens"], weights):
            alpha = 0.10 + 0.68 * (weight / w_max)  # 权重映射到透明度 0.10~0.78
            spans.append(f'<span class="attn-word" style="background-color:rgba({cr},{alpha:.2f})">{w}</span>')
        st.markdown('<div style="line-height:2.3">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
        top = sorted(zip(r["tokens"], weights), key=lambda x: -x[1])[:6]
        chips = " ".join(f'<span class="attn-chip">{w} · {wt:.0%}</span>' for w, wt in top)
        st.markdown(
            f'<div style="margin-top:10px;color:var(--ink-3);font-size:12.5px">注意力最高的词:{chips}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("当前加载的是基础 Bi-LSTM 模型(无注意力层)。可在模型控制台选择 Bi-LSTM+Attention 模型以获得关键词高亮。")
    page_footer()


# ================== 页面 2:实验仪表盘 ==================
def page_dashboard(theme_id: str):
    hero_block(
        "dashboard", theme_id,
        "十四组实验的<em>全景</em>",
        "二分类 14 组对比实验 + 五级分类拓展实验 · 训练曲线 / 混淆矩阵 / 超参数对比 / 模型结构。",
        "EXPERIMENTS · GEMSTONE",
    )

    df = load_summary_csv()

    # ---- 顶部指标 Bento 卡 ----
    if df is not None and len(df):
        best = df.loc[df["test_f1"].idxmax()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🏆 最佳测试 F1", f"{best['test_f1']:.4f}")
        m2.metric("🎯 最佳测试准确率", f"{best['test_acc']:.4f}")
        m3.metric("🧪 对比实验组数", f"{len(df)} 组")
        m4.metric("📦 测试集规模", "906 条")
        st.markdown(
            f'<div class="bento" style="margin:12px 0 8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'<span style="font-size:22px">🏅</span>'
            f'<b style="color:var(--ink)">当前最佳模型</b>'
            f'<span style="color:var(--ink);font-weight:800">{best["run_name"]}</span>'
            f'<span style="color:var(--ink-3);font-size:13px">(P={best["test_precision"]} · R={best["test_recall"]} · '
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

    # ---- Tab 2:训练曲线 / 混淆矩阵(按实验分组) ----
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

    # ---- Tab 4:五级分类拓展实验(独立于二分类 14 组) ----
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
def page_batch(theme_id: str):
    hero_block(
        "batch", theme_id,
        "千百条心声,<em>一场映毕</em>",
        "上传含 text 列的 CSV,或粘贴多行评论(每行一条),批量预测、统计占比并下载结果。",
        "BATCH · DEEP BLUE",
    )

    # ---- 顶部:任务模式与模型(紧凑控制行) ----
    st.markdown('<div class="bento" style="padding:14px 18px">'
                '<div class="bento-title" style="margin-bottom:6px">🎛️ 任务模式与模型</div>',
                unsafe_allow_html=True)
    cd1, cd2 = st.columns([1, 2])
    with cd1:
        render_control()
    with cd2:
        st.markdown(
            '<div style="color:var(--ink-3);font-size:12.5px;line-height:1.9;padding-top:4px">'
            "💡 批量任务同样支持二分类与五级分类;上传 CSV 需包含 <b>text</b> 列,"
            "或直接在右侧粘贴区每行一条评论。</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    predictor = get_predictor(st.session_state.get("selected_run"))
    texts = []

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="bento"><div class="bento-title">📤 上传 CSV 文件</div>'
                    '<div class="bento-hint">需包含 text 列</div>',
                    unsafe_allow_html=True)
        up = st.file_uploader("上传 CSV", type=["csv"], label_visibility="collapsed")
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
        st.markdown('<div class="bento"><div class="bento-title">📋 粘贴多行评论</div>'
                    '<div class="bento-hint">每行一条评论</div>',
                    unsafe_allow_html=True)
        pasted = st.text_area(
            "粘贴评论",
            value="屏幕效果非常好,系统流畅,值得购买。\n电池太不耐用,半天就没电,后悔买了。",
            height=150, label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        run_batch = st.button("🚀 批量预测", type="primary", use_container_width=True)

    if run_batch and pasted.strip():
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

    # ---- 统计概览 + 极性占比条(二分类两段;五级五段) ----
    st.markdown(
        "<h3 style='font-size:18px'>📋 预测结果"
        "<span style='color:var(--ink-3);font-weight:400;font-size:14px'>(共 %d 条)</span></h3>" % len(result),
        unsafe_allow_html=True,
    )
    is_5c = rows and rows[0].get("num_classes") == 5
    if is_5c:
        counts = {name: int((result["情感极性"] == name).sum()) for name, *_ in LEVEL_STYLES}
        segs = "".join(
            f'<div class="ratio-seg" style="width:{counts[name] / len(result) * 100:.1f}%;'
            f'background:{color}" title="{name} {counts[name]} 条">'
            f'{"%d%%" % round(counts[name] / len(result) * 100)}</div>'
            for name, _e, color, _g, _en in LEVEL_STYLES
        )
        st.markdown(f'<div class="ratio-bar">{segs}</div>', unsafe_allow_html=True)
        st.caption(" · ".join(f"{name} {counts[name]}" for name, _e, _c, _g, _en in LEVEL_STYLES))
    else:
        n_pos = int((result["情感极性"] == "正面").sum())
        n_neg = len(result) - n_pos
        pct_pos = n_pos / len(result) * 100
        pct_neg = 100 - pct_pos
        st.markdown(
            f'<div class="ratio-bar">'
            f'<div class="ratio-seg" style="width:{pct_pos:.1f}%;background:#34D399">😊 正面 {n_pos}({pct_pos:.0f}%)</div>'
            f'<div class="ratio-seg" style="width:{pct_neg:.1f}%;background:#FB923C">😞 负面 {n_neg}({pct_neg:.0f}%)</div>'
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


# ================== 主入口:顶部导航路由(query 参数:page + theme) ==================
def main():
    page = st.query_params.get("page", "analyze")
    if page not in {key for key, *_ in NAV_ITEMS}:
        page = "analyze"
    theme_id = st.query_params.get("theme", DEFAULT_THEME)
    if theme_id not in THEMES:
        theme_id = DEFAULT_THEME

    # 初始化任务模式(session 首次进入默认二分类)
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
