# -*- coding: utf-8 -*-
"""
病毒性肺炎重症概率计算器 (Streamlit Web 应用)
================================================================
核心功能（3 个 Tab）：
  1. Nomogram (Logistic)：13 项实验室指标 → 重症概率 + 风险层化 + 评分卡 + 横向贡献条形图
  2. SHAP 解释器 (XGBoost)：相同输入 → 概率 + 单例 SHAP 瀑布图 + 全局 vs 个例对比
  3. 双模型对比：并排两模型概率 + 一致性提示 + 临床决策建议

数据基础：树兰 + 市中 + 流感 三中心 642 例多中心训练数据
模型：sklearn Pipeline (IterativeImputer → StandardScaler → LogisticRegression / XGBClassifier)
启动：streamlit run nomogram_streamlit_app.py --server.port 8501
"""

import base64
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# 支持 SHAP 的本地工具：从同级 assets 导入图标 SVG 字典
sys.path.insert(0, str(Path(__file__).parent / "assets"))
from icons import GROUP_ICONS, ICON_NOMOGRAM, ICON_SHAP, ICON_COMPARE  # noqa: E402

warnings.filterwarnings("ignore")

# matplotlib 中文字体（macOS 与 Linux 通用降级）
plt.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Source Han Sans CN",
    "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 路径与全局常量
# ============================================================
# 模型路径：优先用 app 目录下的 models/（部署模式），回退到项目 03_models/（本地开发模式）
APP_DIR = Path(__file__).parent
ASSETS_DIR = APP_DIR / "assets"

_LOCAL_MODELS = APP_DIR / "models"
_DEV_MODELS = APP_DIR.parents[1] / "03_models" if len(APP_DIR.parents) >= 2 else APP_DIR / "models"
MODEL_DIR = _LOCAL_MODELS if (_LOCAL_MODELS / "Logistic.pkl").exists() else _DEV_MODELS

LOGISTIC_PATH = MODEL_DIR / "Logistic.pkl"
XGBOOST_PATH = MODEL_DIR / "XGBoost.pkl"
SPEC_PATH = ASSETS_DIR / "logistic_spec.json"
BANNER_PATH = ASSETS_DIR / "hero_banner.svg"

# 13 个特征按器官系统分组（每组对应 sidebar 一张折叠卡片）
FEATURE_GROUPS = [
    {"key": "basic",  "icon_key": "basic",  "title": "基本信息",  "color": "#5b8def",
     "features": ["age"]},
    {"key": "blood",  "icon_key": "blood",  "title": "血常规",   "color": "#e74c3c",
     "features": ["wbc", "lymph_pct", "neut_pct"]},
    {"key": "inflam", "icon_key": "inflam", "title": "炎症指标", "color": "#f39c12",
     "features": ["crp", "pct"]},
    {"key": "liver",  "icon_key": "liver",  "title": "肝功能",   "color": "#27ae60",
     "features": ["alt", "ast", "tb"]},
    {"key": "kidney", "icon_key": "kidney", "title": "肾功能",   "color": "#16a085",
     "features": ["cr", "bun"]},
    {"key": "heart",  "icon_key": "heart",  "title": "心肌酶/代谢", "color": "#9b59b6",
     "features": ["ldh", "ck"]},
]

# 风险层化阈值（基于训练集 Youden 指数与临床三档划分）
RISK_BANDS = [
    {"max": 0.30, "label": "低风险",   "color": "#27ae60", "bg": "#e8f8f0",
     "advice": "常规监测，按门诊/普通病房路径管理；建议 24–48h 复评血气/影像"},
    {"max": 0.70, "label": "中等风险", "color": "#f39c12", "bg": "#fff5e6",
     "advice": "加强床旁监护与吸氧支持；建议 12h 内复评 SpO₂、呼吸频率与炎症指标变化"},
    {"max": 1.01, "label": "高风险",   "color": "#e74c3c", "bg": "#fde8e8",
     "advice": "立即升级至重症监护层级；评估高流量氧疗/无创通气；多学科会诊"},
]


# ============================================================
# 数据/模型加载（缓存）
# ============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """一次性加载两个 sklearn Pipeline 模型，全局缓存避免重复 I/O"""
    log_model = joblib.load(LOGISTIC_PATH)
    xgb_model = joblib.load(XGBOOST_PATH)
    return log_model, xgb_model


@st.cache_data(show_spinner=False)
def load_spec():
    """加载变量元数据（中文标签、单位、训练集分布、Logistic 系数）"""
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_banner_b64():
    """SVG hero banner → base64 嵌入 HTML，避免外部资源依赖"""
    svg_text = BANNER_PATH.read_text(encoding="utf-8")
    return base64.b64encode(svg_text.encode("utf-8")).decode("ascii")


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_xgb_model):
    """SHAP TreeExplainer 一次构建，重复使用"""
    import shap
    clf = _xgb_model.named_steps["clf"]
    return shap.TreeExplainer(clf)


# ============================================================
# CSS 样式注入（医学专业蓝绿配色 + 卡片化 + 阴影 + 圆角）
# ============================================================
def inject_css():
    st.markdown(
        """
<style>
    /* 主题色变量 */
    :root {
        --primary-deep: #0b3d91;
        --primary-mid: #1a6db0;
        --primary-light: #16a085;
        --bg-soft: #f6f9fc;
        --card-bg: #ffffff;
        --text-main: #2c3e50;
        --text-muted: #7f8c8d;
        --border-soft: #e0e6ed;
        --shadow-card: 0 2px 12px rgba(11, 61, 145, 0.08);
        --shadow-hover: 0 6px 24px rgba(11, 61, 145, 0.14);
    }

    /* 全局背景：极淡渐变 */
    .stApp {
        background: linear-gradient(180deg, #f0f5fb 0%, #f6f9fc 30%, #ffffff 100%);
    }

    /* 主容器留白 */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* 隐藏 streamlit 默认顶栏（让 hero 占满）*/
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* Hero banner 容器 */
    .hero-wrap {
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0 0 16px 16px;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }
    .hero-wrap img { width: 100%; display: block; }

    /* Sidebar 样式覆盖 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Sidebar 分组卡片 */
    .group-card {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 10px;
        border: 1px solid var(--border-soft);
        transition: box-shadow 0.2s ease;
    }
    .group-card:hover { box-shadow: var(--shadow-card); }
    .group-header {
        display: flex; align-items: center; gap: 8px;
        font-size: 14px; font-weight: 700;
        margin-bottom: 8px; padding-bottom: 6px;
        border-bottom: 2px solid;
    }
    .group-header svg { width: 18px; height: 18px; flex-shrink: 0; }

    /* 主区卡片 */
    .panel-card {
        background: var(--card-bg);
        border-radius: 14px;
        padding: 22px 26px;
        box-shadow: var(--shadow-card);
        border: 1px solid var(--border-soft);
        margin-bottom: 16px;
    }
    .panel-title {
        font-size: 17px; font-weight: 700; color: var(--text-main);
        margin-bottom: 14px; padding-bottom: 10px;
        border-bottom: 2px solid #ecf0f6;
        display: flex; align-items: center; gap: 8px;
    }

    /* 大概率显示卡（环形仪表样式） */
    .prob-hero {
        text-align: center;
        padding: 32px 16px 24px 16px;
        border-radius: 14px;
        margin-bottom: 14px;
    }
    .prob-value {
        font-size: 64px; font-weight: 800;
        line-height: 1; letter-spacing: -2px;
        margin: 8px 0;
    }
    .prob-label {
        font-size: 13px; color: var(--text-muted);
        text-transform: uppercase; letter-spacing: 2px;
    }
    .risk-badge {
        display: inline-block; padding: 6px 18px;
        border-radius: 20px; font-weight: 700; font-size: 14px;
        margin-top: 10px;
    }
    .risk-advice {
        font-size: 13px; margin-top: 14px;
        padding: 10px 14px; border-radius: 8px;
        text-align: left; line-height: 1.6;
    }

    /* 评分卡表格 */
    .score-table {
        width: 100%; border-collapse: collapse;
        font-size: 13px;
    }
    .score-table th {
        background: #f6f9fc; padding: 10px 12px;
        text-align: left; font-weight: 600; color: var(--text-main);
        border-bottom: 2px solid var(--border-soft);
    }
    .score-table td {
        padding: 8px 12px; border-bottom: 1px solid #f0f4f8;
    }
    .score-table tr:hover { background: #fafcfe; }

    /* Tab 样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #ffffff;
        padding: 6px;
        border-radius: 12px;
        box-shadow: var(--shadow-card);
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px; padding: 0 22px;
        background: transparent; border-radius: 8px;
        font-weight: 600; color: var(--text-muted);
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-mid), var(--primary-light)) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(26, 109, 176, 0.3);
    }

    /* 数字输入框 */
    .stNumberInput input {
        border-radius: 8px !important;
        border: 1.5px solid var(--border-soft) !important;
        font-size: 14px !important;
    }
    .stNumberInput input:focus {
        border-color: var(--primary-mid) !important;
        box-shadow: 0 0 0 3px rgba(26, 109, 176, 0.15) !important;
    }

    /* 按钮 */
    .stButton button {
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, var(--primary-mid), var(--primary-light));
        color: white;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26, 109, 176, 0.3);
    }

    /* Footer */
    .footer {
        margin-top: 32px; padding: 20px;
        background: #ffffff; border-radius: 12px;
        font-size: 12px; color: var(--text-muted);
        line-height: 1.7; text-align: center;
        border: 1px solid var(--border-soft);
    }
    .footer strong { color: var(--text-main); }

    /* 一致性提示徽章 */
    .agree-good { background: #e8f8f0; color: #1e7e4f; padding: 14px 18px; border-radius: 10px; border-left: 4px solid #27ae60; }
    .agree-warn { background: #fff5e6; color: #b06b00; padding: 14px 18px; border-radius: 10px; border-left: 4px solid #f39c12; }
    .agree-bad  { background: #fde8e8; color: #a52828; padding: 14px 18px; border-radius: 10px; border-left: 4px solid #e74c3c; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 渲染：顶部 hero banner
# ============================================================
def render_hero():
    b64 = load_banner_b64()
    st.markdown(
        f'<div class="hero-wrap"><img src="data:image/svg+xml;base64,{b64}" alt="hero"/></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 渲染：sidebar 输入面板（按器官系统分组）
# ============================================================
def render_sidebar(spec):
    """构建 sidebar 输入面板，返回 {feature: value} 字典"""
    meta = spec["variable_meta"]
    inputs = {}

    with st.sidebar:
        st.markdown(
            "<div style='padding: 8px 4px 16px 4px;'>"
            "<div style='font-size:18px; font-weight:800; color:#0b3d91;'>📋 患者实验室指标录入</div>"
            "<div style='font-size:12px; color:#7f8c8d; margin-top:4px;'>入院 24h 内首次实验室结果</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        # 快速操作按钮组
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 重置默认", use_container_width=True):
                for f in meta:
                    st.session_state[f"input_{f}"] = float(meta[f]["median"])
                st.rerun()
        with col_b:
            preset = st.selectbox(
                "示例患者",
                ["自定义", "重症典型", "轻症典型"],
                label_visibility="collapsed",
            )
        if preset != "自定义":
            apply_preset(preset, meta)

        st.markdown("---")

        # 按器官系统分组渲染
        for grp in FEATURE_GROUPS:
            icon_svg = GROUP_ICONS[grp["icon_key"]]
            color = grp["color"]
            st.markdown(
                f'<div class="group-card" style="border-left: 4px solid {color};">'
                f'<div class="group-header" style="color:{color}; border-bottom-color:{color}33;">'
                f'<span style="color:{color};">{icon_svg}</span>{grp["title"]}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            for f in grp["features"]:
                m = meta[f]
                key = f"input_{f}"
                if key not in st.session_state:
                    st.session_state[key] = float(m["median"])
                inputs[f] = st.number_input(
                    label=f"{m['label_zh']} ({m['unit']})",
                    min_value=float(m["min_clinical"]),
                    max_value=float(m["max_clinical"]),
                    value=float(st.session_state[key]),
                    step=0.1 if m["median"] < 10 else 1.0,
                    key=key,
                    help=f"训练集分布：中位 {m['median']:.2f}，5–95 百分位 [{m['p5']:.2f}, {m['p95']:.2f}]",
                )
    return inputs


def apply_preset(name, meta):
    """加载示例患者数据（重症 / 轻症典型值）"""
    presets = {
        "重症典型": {
            "age": 78.0, "wbc": 11.5, "lymph_pct": 6.0, "neut_pct": 88.0,
            "crp": 145.0, "pct": 1.8, "alt": 65.0, "ast": 95.0, "tb": 22.0,
            "cr": 145.0, "bun": 14.5, "ldh": 460.0, "ck": 320.0,
        },
        "轻症典型": {
            "age": 38.0, "wbc": 5.2, "lymph_pct": 28.0, "neut_pct": 62.0,
            "crp": 8.0, "pct": 0.08, "alt": 18.0, "ast": 22.0, "tb": 8.5,
            "cr": 65.0, "bun": 4.2, "ldh": 195.0, "ck": 75.0,
        },
    }
    for f, v in presets[name].items():
        st.session_state[f"input_{f}"] = float(v)


# ============================================================
# 工具函数：风险层化、概率推断
# ============================================================
def risk_band(prob):
    """根据概率返回 (label, color, bg, advice) 四元组"""
    for b in RISK_BANDS:
        if prob < b["max"]:
            return b
    return RISK_BANDS[-1]


def predict_proba(model, inputs, feature_order):
    """构造 1×N DataFrame → 调用 sklearn Pipeline → 返回 [0,1] 概率"""
    X = pd.DataFrame([[inputs[f] for f in feature_order]], columns=feature_order)
    return float(model.predict_proba(X)[0, 1])


# ============================================================
# 渲染：概率 hero 卡片（统一组件，3 个 Tab 复用）
# ============================================================
def render_prob_card(prob, model_name="Logistic"):
    band = risk_band(prob)
    st.markdown(
        f'<div class="prob-hero" style="background: linear-gradient(135deg, {band["bg"]} 0%, #ffffff 100%);">'
        f'<div class="prob-label">{model_name} 模型预测重症概率</div>'
        f'<div class="prob-value" style="color:{band["color"]};">{prob*100:.1f}%</div>'
        f'<span class="risk-badge" style="background:{band["color"]}; color:white;">'
        f'{band["label"]}</span>'
        f'<div class="risk-advice" style="background:{band["bg"]}; color:{band["color"]}; border-left:4px solid {band["color"]};">'
        f'<strong>临床建议：</strong>{band["advice"]}'
        f'</div></div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Tab 1：Nomogram (Logistic) — 概率 + 评分卡 + 横向贡献条形图
# ============================================================
def render_tab_nomogram(log_model, inputs, spec):
    features = spec["features"]
    coef = np.array(spec["coef"])
    mu = np.array(spec["scaler_mean"])
    sd = np.array(spec["scaler_scale"])
    meta = spec["variable_meta"]

    prob = predict_proba(log_model, inputs, features)

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_prob_card(prob, "Logistic")

        # 评分卡：每变量对 logit 的贡献 = coef × (x - mu) / sd
        x_arr = np.array([inputs[f] for f in features])
        contrib = coef * (x_arr - mu) / sd
        score_rows = sorted(
            [(meta[f]["label_zh"], inputs[f], meta[f]["unit"], contrib[i])
             for i, f in enumerate(features)],
            key=lambda r: abs(r[3]), reverse=True,
        )
        st.markdown('<div class="panel-title">📊 评分卡（按 |贡献度| 降序）</div>', unsafe_allow_html=True)
        rows_html = "".join(
            f'<tr><td>{name}</td><td style="text-align:right; font-family:monospace;">{val:.2f} {unit}</td>'
            f'<td style="text-align:right; font-weight:700; color:{"#e74c3c" if c > 0 else "#27ae60"};">'
            f'{"↑" if c > 0 else "↓"} {c:+.3f}</td></tr>'
            for name, val, unit, c in score_rows
        )
        st.markdown(
            f'<table class="score-table"><thead><tr><th>变量</th><th style="text-align:right;">实测值</th>'
            f'<th style="text-align:right;">logit 贡献</th></tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 个体化贡献分解（Nomogram 风格）</div>', unsafe_allow_html=True)

        # 横向条形图：贡献度可视化（matplotlib 画布全英文，避免云端缺中文字体）
        labels_en = [meta[f]["label_en"] for f in features]
        order = np.argsort(contrib)
        c_sorted = contrib[order]
        labels_sorted = [labels_en[i] for i in order]
        colors = ["#27ae60" if c < 0 else "#e74c3c" for c in c_sorted]

        fig, ax = plt.subplots(figsize=(8, 5.6), dpi=110)
        bars = ax.barh(range(len(c_sorted)), c_sorted, color=colors,
                       edgecolor="white", linewidth=1.5, alpha=0.92)
        ax.set_yticks(range(len(c_sorted)))
        ax.set_yticklabels(labels_sorted, fontsize=10.5)
        ax.axvline(0, color="#555", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xlabel(r"Logit Contribution ($\beta$ $\times$ z-score)", fontsize=10.5)
        ax.set_title(f"Intercept = {spec['intercept']:+.3f}    Final logit = {np.log(prob/(1-prob)):+.3f}",
                     fontsize=11, color="#2c3e50", pad=12)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#bdc3c7")
        ax.spines["bottom"].set_color("#bdc3c7")
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        for bar, val in zip(bars, c_sorted):
            ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                    f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right",
                    fontsize=9, color="#34495e", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "💡 **解读**：红色（向右）= 该指标推高重症风险；绿色（向左）= 该指标降低重症风险。"
            "条形长度 = 该变量在标准化尺度下对 logit 的偏移量。"
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Tab 2：SHAP 解释器 (XGBoost)
# ============================================================
def render_tab_shap(xgb_model, inputs, spec):
    import shap

    features = spec["features"]
    meta = spec["variable_meta"]

    prob = predict_proba(xgb_model, inputs, features)
    explainer = get_shap_explainer(xgb_model)

    # SHAP 需要在 imputer + scaler 处理后的尺度上计算
    X = pd.DataFrame([[inputs[f] for f in features]], columns=features)
    X_imp = xgb_model.named_steps["imputer"].transform(X)
    X_scaled = xgb_model.named_steps["scaler"].transform(X_imp)

    shap_values = explainer.shap_values(X_scaled)
    # 兼容 binary classification 不同返回结构
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    sv_arr = np.asarray(shap_values).squeeze()
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)) and not np.isscalar(expected):
        expected = float(np.array(expected).flatten()[-1])
    else:
        expected = float(expected)

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_prob_card(prob, "XGBoost")

        st.markdown('<div class="panel-title">🔬 SHAP 单例解释（个例 vs 平均）</div>', unsafe_allow_html=True)

        # 标签中文化的特征顺序 + 当前样例 SHAP 值
        order = np.argsort(np.abs(sv_arr))[::-1]
        rows_html = "".join(
            f'<tr><td>{meta[features[i]]["label_zh"]}</td>'
            f'<td style="text-align:right; font-family:monospace;">{inputs[features[i]]:.2f} {meta[features[i]]["unit"]}</td>'
            f'<td style="text-align:right; font-weight:700; color:{"#e74c3c" if sv_arr[i] > 0 else "#27ae60"};">'
            f'{"↑" if sv_arr[i] > 0 else "↓"} {sv_arr[i]:+.3f}</td></tr>'
            for i in order[:13]
        )
        st.markdown(
            f'<table class="score-table"><thead><tr><th>变量</th>'
            f'<th style="text-align:right;">实测值</th><th style="text-align:right;">SHAP 值</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"📌 平均预测基线 (E[f(x)]) = {expected:+.3f}；"
            f"个例预测 logit = {expected + sv_arr.sum():+.3f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📈 SHAP 瀑布图 (Waterfall) — 个例分解</div>', unsafe_allow_html=True)

        # 自绘瀑布图（避开 shap.plots.waterfall 对中文/字体的兼容问题；标签全英文）
        order2 = np.argsort(np.abs(sv_arr))[::-1]
        plot_n = min(13, len(order2))
        sel = order2[:plot_n][::-1]   # 从下往上：贡献小→大
        labels = [meta[features[i]]["label_en"] for i in sel]
        vals = sv_arr[sel]
        colors = ["#27ae60" if v < 0 else "#e74c3c" for v in vals]

        fig, ax = plt.subplots(figsize=(8, 5.6), dpi=110)

        # 累计基线：从 expected 开始累加，画阶梯
        cum = expected
        positions = np.arange(plot_n)
        for i, (v, c, lab) in enumerate(zip(vals, colors, labels)):
            ax.barh(i, v, left=cum, color=c, edgecolor="white", linewidth=1.5, alpha=0.92)
            ax.text(cum + v + (0.015 if v >= 0 else -0.015), i,
                    f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right",
                    fontsize=9, color="#34495e", fontweight="bold")
            cum += v
        ax.axvline(expected, color="#7f8c8d", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(expected, plot_n - 0.3, f"  E[f(x)]={expected:.2f}",
                color="#7f8c8d", fontsize=9.5, ha="left")
        ax.axvline(cum, color="#0b3d91", linewidth=1.5, alpha=0.85)
        ax.text(cum, -0.7, f"f(x)={cum:.2f}", color="#0b3d91",
                fontsize=10, fontweight="bold", ha="center")

        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.set_xlabel("Logit (Cumulative)", fontsize=10.5)
        ax.set_title("XGBoost SHAP Waterfall (sorted by |SHAP|)", fontsize=11, color="#2c3e50", pad=12)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#bdc3c7")
        ax.spines["bottom"].set_color("#bdc3c7")
        ax.grid(axis="x", alpha=0.25, linestyle=":")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "💡 **解读**：从平均基线 E[f(x)] 出发，每个变量按其 SHAP 值推高（红）或拉低（绿）logit，"
            "最终落到本患者预测值 f(x)。这是一种**模型决策的逐步可视化**。"
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Tab 3：双模型对比 + 一致性提示
# ============================================================
def render_tab_compare(log_model, xgb_model, inputs, spec):
    features = spec["features"]
    p_log = predict_proba(log_model, inputs, features)
    p_xgb = predict_proba(xgb_model, inputs, features)
    band_log = risk_band(p_log)
    band_xgb = risk_band(p_xgb)

    col_l, col_m, col_r = st.columns([1, 0.15, 1], gap="medium")

    with col_l:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_prob_card(p_log, "Logistic (传统统计)")
        st.markdown(
            "<div style='font-size:12.5px; color:#7f8c8d; line-height:1.7;'>"
            "<strong>模型特点</strong>：基于 13 个变量的标准多因素回归，"
            "可读 OR、临床透明、易写入指南；对**线性、加性**关系敏感。"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m:
        st.markdown(
            "<div style='display:flex; align-items:center; justify-content:center; "
            "height:200px; font-size:36px; color:#bdc3c7; font-weight:300;'>VS</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        render_prob_card(p_xgb, "XGBoost (机器学习)")
        st.markdown(
            "<div style='font-size:12.5px; color:#7f8c8d; line-height:1.7;'>"
            "<strong>模型特点</strong>：梯度提升树集成，"
            "可捕获**非线性与交互效应**；个体化解释依赖 SHAP；适合复杂样本异质性。"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # 一致性提示
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🤝 双模型一致性评估</div>', unsafe_allow_html=True)

    diff = abs(p_log - p_xgb)
    same_band = band_log["label"] == band_xgb["label"]

    if same_band and diff < 0.10:
        cls, icon, title, body = (
            "agree-good", "✅", "高度一致",
            f"两模型预测概率差异 {diff*100:.1f}%，且同属 **{band_log['label']}** 区段。"
            "可作为强证据支持当前临床决策。"
        )
    elif same_band:
        cls, icon, title, body = (
            "agree-warn", "⚠️", "区段一致但数值有差",
            f"两模型同属 **{band_log['label']}** 区段，但概率差异 {diff*100:.1f}%。"
            "建议参考评分卡/SHAP 解释，理解差异来源（多为非线性效应被 XGBoost 捕获）。"
        )
    else:
        cls, icon, title, body = (
            "agree-bad", "🚨", "模型分歧",
            f"Logistic 判为 **{band_log['label']}** ({p_log*100:.1f}%)，"
            f"XGBoost 判为 **{band_xgb['label']}** ({p_xgb*100:.1f}%)。"
            "建议结合临床动态指标与影像学复评，**人工复核优先**；"
            "可能存在异常值、缺失或非线性区间，模型外推不稳定。"
        )
    st.markdown(
        f'<div class="{cls}"><strong>{icon} {title}</strong><br>{body}</div>',
        unsafe_allow_html=True,
    )

    # 双模型并排小条形对比
    fig, ax = plt.subplots(figsize=(9, 1.6), dpi=110)
    ax.barh([0], [p_log], color="#1a6db0", height=0.42, label=f"Logistic {p_log*100:.1f}%")
    ax.barh([1], [p_xgb], color="#16a085", height=0.42, label=f"XGBoost  {p_xgb*100:.1f}%")
    for thr, lbl, col in [(0.30, "Low / Mid", "#f39c12"), (0.70, "Mid / High", "#e74c3c")]:
        ax.axvline(thr, color=col, linewidth=1.2, linestyle="--", alpha=0.6)
        ax.text(thr, 1.65, lbl, color=col, fontsize=8.5, ha="center")
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Logistic", "XGBoost"], fontsize=10.5)
    ax.set_xlabel("Predicted Severity Probability", fontsize=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bdc3c7")
    ax.tick_params(left=False)
    ax.text(p_log + 0.012, 0, f"{p_log*100:.1f}%", va="center", fontsize=10, color="#1a6db0", fontweight="bold")
    ax.text(p_xgb + 0.012, 1, f"{p_xgb*100:.1f}%", va="center", fontsize=10, color="#16a085", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 页脚
# ============================================================
def render_footer():
    st.markdown(
        '<div class="footer">'
        '<strong>病毒性肺炎重症概率计算器 v1.0</strong> &nbsp;|&nbsp; '
        '基于杭州市科技发展计划项目「急性呼吸道病毒感染重症病例预警模型构建」(2023–2025)<br>'
        '训练数据：树兰医院 + 杭州市第一人民医院 + 杭州市流感监测网络，多中心 642 例 (新冠 525 + 流感 117)<br>'
        '<br>'
        '<em>⚠️ 临床免责声明：本工具为辅助决策参考，<strong>不能替代医师诊断</strong>。'
        '所有诊疗决策应结合患者完整病史、体征、动态实验室与影像学综合判断。</em>'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# 主入口
# ============================================================
def main():
    st.set_page_config(
        page_title="病毒性肺炎重症风险计算器",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_hero()

    spec = load_spec()
    log_model, xgb_model = load_models()
    inputs = render_sidebar(spec)

    tab1, tab2, tab3 = st.tabs([
        f"🩺 列线图 (Logistic)",
        f"🔬 SHAP 解释器 (XGBoost)",
        f"⚖️ 双模型对比",
    ])

    with tab1:
        render_tab_nomogram(log_model, inputs, spec)
    with tab2:
        render_tab_shap(xgb_model, inputs, spec)
    with tab3:
        render_tab_compare(log_model, xgb_model, inputs, spec)

    render_footer()


if __name__ == "__main__":
    main()
