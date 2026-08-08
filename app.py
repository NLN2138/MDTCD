import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import spacy
from spacy import displacy
import re
import json
import pandas as pd

# 載入核心計算模組
from core.syntactic_engine import calculate_mdd_and_memory_load, calculate_l2sca_approximations

# -----------------------------------------------------------------------------
# 1. 頁面配置與 Session State 初始化
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TW-EFL MDTCD 診斷系統", 
    page_icon="🇹🇼",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 初始化歷程紀錄
if "history" not in st.session_state:
    st.session_state.history = []

# 自訂 CSS 樣式
st.markdown("""
<style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    
    .header-container {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 1rem 2rem;
        margin: -1rem -2rem 1.5rem -2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border-bottom: 3px solid #3b82f6;
    }
    .header-title { font-size: 1.85rem !important; font-weight: 800 !important; color: #ffffff !important; margin: 0 !important; }
    .header-subtitle { font-size: 0.95rem !important; color: #94a3b8 !important; margin-top: 4px !important; }

    [data-testid="stMetricValue"] { font-size: 2.1rem !important; font-weight: 700 !important; color: #1e293b; }
    
    .custom-card { background-color: #ffffff; border-radius: 12px; padding: 1.25rem; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1rem; }
    .warning-box { background-color: #fffbebf5; border-left: 5px solid #f59e0b; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }
    .error-box { background-color: #fef2f2; border-left: 5px solid #ef4444; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }
    .success-box { background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 1rem; border-radius: 6px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 頂部固定大標題 (Sticky Header)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🇹🇼 台灣英語教材多維度複雜度自動化診斷系統 (MDTCD)</div>
    <div class="header-subtitle">Multi-Dimensional Textbook Complexity Diagnostics System | 融合 SLA、依存語法與 XAI 可解釋性 AI 原則</div>
</div>
""", unsafe_allow_html=True)

# 標竿資料庫 (Ting 2024 數據)
NORMS = {
    "高中五年級/高二 (Ting 2024)": {"MLS": 17.97, "CNP/C": 1.14, "MDD_target": 2.2},
    "學測優秀作文 (GSAT)": {"MLS": 19.28, "CNP/C": 1.03, "MDD_target": 2.4},
    "真實學術論文 (RA)": {"MLS": 27.31, "CNP/C": 2.32, "MDD_target": 3.2}
}

# -----------------------------------------------------------------------------
# 3. 側邊欄：標竿設定與系統參數可調區
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 標竿對照設定")
target_level = st.sidebar.selectbox("選擇目標語體 / 年級階梯", options=list(NORMS.keys()), index=0)
current_norm = NORMS[target_level]

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 系統參數自訂區")

# 1. MDD 門檻自訂（預設 Liu et al. 2017 臨界值 3.0）
mdd_threshold = st.sidebar.slider(
    "MDD 認知超載臨界值 (Liu et al., 2017)",
    min_value=1.5, max_value=4.5, value=3.0, step=0.1,
    help="預設為 3.0。若單句/全篇 MDD 超過此數值，大腦工作記憶解讀負擔將顯著上升。"
)

# 2. Cowan 工作記憶 4-chunk 超載門檻自訂
arcs_threshold = st.sidebar.slider(
    "未閉合依存弧超載門檻 (Cowan, 2001)",
    min_value=2, max_value=7, value=4, step=1,
    help="預設為 4 條弧線。當單字上空跨越的依存弧 ≥ 4 時，觸發工作記憶超載預警。"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 當前對照基準")
st.sidebar.write(f"• **MLS (句長)**: {current_norm['MLS']}")
st.sidebar.write(f"• **CNP/C (名詞組)**: {current_norm['CNP/C']}")
st.sidebar.write(f"• **MDD (基準)**: {current_norm['MDD_target']}")

# -----------------------------------------------------------------------------
# 4. 輸入區（支援單篇與批次檔案上傳）
# -----------------------------------------------------------------------------
default_sample = (
    "Title: The Digital Footprint of Modern Society.\n\n"
    "In today's interconnected world, almost every online action leaves a digital trace that reflects our personal habits, interests, and behaviors.\n"
    "As teenagers navigate various social media platforms, they often share personal opinions and life moments without realizing the potential consequences of their online presence.\n\n"
    "According to recent educational studies on digital literacy, internet users who regularly broadcast their daily activities to the public tend to expose themselves to potential privacy risks.\n"
    "Furthermore, algorithms designed by large technology corporations analyze these massive amounts of user data to deliver targeted advertisements, which subtly influences individual decision-making processes.\n"
    "Therefore, developing critical thinking skills regarding digital privacy has become an essential responsibility for high school students in the twenty-first century."
)

st.markdown("### 📝 教材文本輸入與批次分析")

input_tab1, input_tab2 = st.tabs(["✍️ 單篇文本分析", "📁 批次檔案上傳 (.txt)"])

active_text = ""
file_name_label = "單篇文章"

with input_tab1:
    user_input = st.text_area(
        "請貼入英文課文或教材（若未貼入，將自動以預設的高二範文進行診斷）：",
        height=140, value=default_sample
    )
    cleaned_text = re.sub(r'([a-zA-Z0-9])\n([a-zA-Z0-9])', r'\1. \2', user_input.strip())
    active_text = cleaned_text if cleaned_text else default_sample

with input_tab2:
    uploaded_files = st.file_uploader("上傳多個課文 .txt 檔案進行批次檢測", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        selected_file = st.selectbox("選擇要預覽與診斷的檔案：", options=[f.name for f in uploaded_files])
        for f in uploaded_files:
            if f.name == selected_file:
                active_text = f.read().decode("utf-8")
                file_name_label = f.name
                break

analyze_btn = st.button("🚀 開始多維度自動診斷", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# 5. 核心運算與結果呈現
# -----------------------------------------------------------------------------
if analyze_btn or active_text:
    mdd_res = calculate_mdd_and_memory_load(active_text)
    l2sca_res = calculate_l2sca_approximations(active_text)
    
    # 覆蓋自訂超載判斷 (基於使用者設定的 mdd_threshold)
    is_custom_overloaded = mdd_res["mdd"] >= mdd_threshold

    # 自動記錄至歷史診斷數據 (Session Log)
    log_entry = {
        "檔名/篇名": file_name_label,
        "MLS": l2sca_res["MLS"],
        "CNP/C": l2sca_res["CNP/C"],
        "MDD": mdd_res["mdd"],
        "超載點數": len(mdd_res["overload_spans"])
    }
    if not any(d["檔名/篇名"] == file_name_label and d["MDD"] == mdd_res["mdd"] for d in st.session_state.history):
        st.session_state.history.append(log_entry)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 頂部 Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("平均句子長度 (MLS)", l2sca_res["MLS"], delta=round(l2sca_res["MLS"] - current_norm["MLS"], 2))
    col2.metric("複雜名詞片語 (CNP/C)", l2sca_res["CNP/C"], delta=round(l2sca_res["CNP/C"] - current_norm["CNP/C"], 2))
    col3.metric("平均依存距離 (MDD)", mdd_res["mdd"], delta=round(mdd_res["mdd"] - current_norm["MDD_target"], 2), delta_color="inverse")
    col4.metric("工作記憶超載點", f"{len(mdd_res['overload_spans'])} 處", delta=f"自訂門檻 MDD>{mdd_threshold}", delta_color="inverse")

    st.markdown("---")

    # XAI 三頁籤 + 系統功能頁籤
    tab1, tab2, tab3, tab4 = st.tabs([
        "💡 為什麼 (Why)", 
        "📊 如何計算與依存弧線 (How)", 
        "🛠️ 三維度診斷與改寫建議 (Recommendations)",
        "⚙️ 系統功能與報告匯出 (System Tools)"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: 為什麼 (Why)
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown(f"#### 📌 當前標竿對照：**{target_level}** | 自訂 MDD 超載門檻：**{mdd_threshold}**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🌿 句法與認知負荷診斷")
            if is_custom_overloaded:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>⚠️ 認知加工負荷偏高</strong><br>
                    全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>（已超過您設定的超載門檻 {mdd_threshold}）。<br>
                    大腦在處理長距離依存關係時需要更多工作記憶。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ 認知加工負荷適中</strong><br>
                    全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>，低於自訂超載門檻 ({mdd_threshold})。
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🔤 片語凝聚度 (Phrasal Style) 診斷")
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                st.markdown(f"""
                <div class="warning-box">
                    <strong>ℹ️ 片語凝聚度提醒</strong><br>
                    當前文本 CNP/C 為 <strong>{l2sca_res['CNP/C']}</strong>，低於標竿值 ({current_norm['CNP/C']})。<br>
                    建議增加名詞片語後置修飾語（Noun Post-modifiers）。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ 片語凝聚度達標</strong><br>
                    CNP/C 為 <strong>{l2sca_res['CNP/C']}</strong>，達到標竿標準 ({current_norm['CNP/C']})。
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: 如何計算 (How) - 包含帶箭頭的依存句法圖
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### 📊 指標對比與依存語法樹 (Dependency Arc Diagram)")
        t2_c1, t2_c2 = st.columns([1, 1])
        with t2_c1:
            st.markdown("#### 1. 當前數值 vs. 標竿數值對比")
            fig = go.Figure(data=[
                go.Bar(name='當前文本', x=['MLS (句長)', 'CNP/C (名詞組)', 'MDD (依存距離)'], y=[l2sca_res['MLS'], l2sca_res['CNP/C'], mdd_res['mdd']], marker_color='#3b82f6'),
                go.Bar(name=f'標竿 ({target_level.split()[0]})', x=['MLS (句長)', 'CNP/C (名詞组)', 'MDD (依存距離)'], y=[current_norm['MLS'], current_norm['CNP/C'], current_norm
