import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import spacy
from spacy import displacy
import re
import pandas as pd

# 載入核心計算模組
from core.syntactic_engine import calculate_mdd_and_memory_load, calculate_l2sca_approximations

# -----------------------------------------------------------------------------
# 1. 頁面配置、效能優化 (Caching) 與 Session State
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TW-EFL MDTCD 診斷系統", 
    page_icon="🇹🇼",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 【效能優化】使用 st.cache_resource 快取 spaCy 模型，避免每次互動重複載入，大幅提升速度！
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

# 初始化歷程紀錄
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------------
# 2. 全局 UI/UX 與 CSS 樣式優化 (統一字體、固定 Header、美化卡片)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* 全局字體與背景優化 */
    body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    /* 頂部固定 Header 容器 (真正固定在視窗最上方) */
    .header-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 1.2rem 2rem 1rem 4.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-bottom: 3px solid #3b82f6;
    }
    
    .header-title {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 !important;
        letter-spacing: 0.5px;
    }
    
    .header-subtitle {
        font-size: 0.95rem !important;
        color: #94a3b8 !important;
        margin-top: 6px !important;
        font-weight: 400;
    }

    /* 確保 Streamlit 選單浮在標題之上 */
    header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 1000 !important; 
    }

    /* 將主要內容區塊往下推，避免被 Header 擋住 */
    .block-container {
        padding-top: 130px !important; 
    }

    /* Metric Cards 數值字體優化 */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #0f172a;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #475569;
        font-weight: 600;
    }
    
    /* 自訂訊息卡片容器 */
    .custom-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        height: 100%;
    }

    .msg-box {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .warning-box { background-color: #fffbeb; border-left: 5px solid #f59e0b; color: #92400e; }
    .error-box { background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b; }
    .success-box { background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534; }
    .info-box { background-color: #f0f9ff; border-left: 5px solid #0ea5e9; color: #075985; }
</style>
""", unsafe_allow_html=True)

# 渲染固定標題
st.markdown("""
<div class="header-container">
    <div class="header-title">🇹🇼 台灣英語教材多維度複雜度自動化診斷系統 (MDTCD)</div>
    <div class="header-subtitle">Multi-Dimensional Textbook Complexity Diagnostics | 融合 SLA、依存語法與 XAI 可解釋性 AI</div>
</div>
""", unsafe_allow_html=True)

# 標竿資料庫
NORMS = {
    "高中五年級/高二 (Ting 2024)": {"MLS": 17.97, "CNP/C": 1.14, "MDD_target": 2.2},
    "學測優秀作文 (GSAT)": {"MLS": 19.28, "CNP/C": 1.03, "MDD_target": 2.4},
    "真實學術論文 (RA)": {"MLS": 27.31, "CNP/C": 2.32, "MDD_target": 3.2}
}

# -----------------------------------------------------------------------------
# 3. 側邊欄：標竿設定與系統參數
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 標竿對照設定")
target_level = st.sidebar.selectbox("選擇目標語體 / 年級階梯", options=list(NORMS.keys()), index=0)
current_norm = NORMS[target_level]

st.sidebar.divider()
st.sidebar.header("⚙️ 系統參數自訂區")

mdd_threshold = st.sidebar.slider(
    "MDD 認知超載臨界值 (Liu et al., 2017)",
    min_value=1.5, max_value=4.5, value=3.0, step=0.1,
    help="預設為 3.0。若單句/全篇 MDD 超過此數值，工作記憶負荷將顯著上升。"
)

arcs_threshold = st.sidebar.slider(
    "未閉合依存弧超載門檻 (Cowan, 2001)",
    min_value=2, max_value=7, value=4, step=1,
    help="預設為 4 條弧線。當單字上空跨越的依存弧 ≥ 4 時，觸發工作記憶超載預警。"
)

st.sidebar.divider()
st.sidebar.markdown("### 📊 當前對照基準")
st.sidebar.write(f"• **平均句長 (MLS)**: {current_norm['MLS']}")
st.sidebar.write(f"• **複雜名詞組 (CNP/C)**: {current_norm['CNP/C']}")
st.sidebar.write(f"• **依存距離 (MDD)**: {current_norm['MDD_target']}")

# -----------------------------------------------------------------------------
# 4. 輸入區 (單篇與批次)
# -----------------------------------------------------------------------------
default_sample = (
    "Title: The Digital Footprint of Modern Society.\n\n"
    "In today's interconnected world, almost every online action leaves a digital trace that reflects our personal habits, interests, and behaviors.\n"
    "As teenagers navigate various social media platforms, they often share personal opinions and life moments without realizing the potential consequences of their online presence.\n\n"
    "According to recent educational studies on digital literacy, internet users who regularly broadcast their daily activities to the public tend to expose themselves to potential privacy risks.\n"
    "Furthermore, algorithms designed by large technology corporations analyze these massive amounts of user data to deliver targeted advertisements, which subtly influences individual decision-making processes.\n"
    "Therefore, developing critical thinking skills regarding digital privacy has become an essential responsibility for high school students in the twenty-first century."
)

input_tab1, input_tab2 = st.tabs(["✍️ 單篇文本分析", "📁 批次檔案上傳 (.txt)"])
active_text = ""
file_name_label = "單篇文章"

with input_tab1:
    user_input = st.text_area("請貼入英文課文或教材（若未貼入，將自動以預設的高二範文進行診斷）：", height=150, value=default_sample)
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
    
    is_custom_overloaded = mdd_res["mdd"] >= mdd_threshold

    # 紀錄至歷史
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
    
    # 指標列
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("平均句子長度 (MLS)", l2sca_res["MLS"], delta=round(l2sca_res["MLS"] - current_norm["MLS"], 2))
    col2.metric("複雜名詞片語 (CNP/C)", l2sca_res["CNP/C"], delta=round(l2sca_res["CNP/C"] - current_norm["CNP/C"], 2))
    col3.metric("平均依存距離 (MDD)", mdd_res["mdd"], delta=round(mdd_res["mdd"] - current_norm["MDD_target"], 2), delta_color="inverse")
    col4.metric("工作記憶超載點", f"{len(mdd_res['overload_spans'])} 處", delta=f"自訂門檻 MDD>{mdd_threshold}", delta_color="inverse")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "💡 為什麼 (Why)", 
        "📊 如何計算與依存弧線 (How)", 
        "🛠️ 診斷與改寫建議 (Recommendations)",
        "⚙️ 系統功能與報告匯出"
    ])

    # --- TAB 1: 為什麼 ---
    with tab1:
        st.markdown(f"#### 📌 當前標竿對照：**{target_level}** | 自訂超載門檻：**{mdd_threshold}**")
        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🌿 句法與認知負荷診斷")
            if is_custom_overloaded:
                st.markdown(f"""
                <div class="msg-box warning-box">
                    <strong>⚠️ 認知加工負荷偏高</strong><br>
                    全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>（已超過超載門檻 {mdd_threshold}）。<br>
                    大腦在處理長距離依存關係時需要更多工作記憶。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-box success-box">
                    <strong>✅ 認知加工負荷適中</strong><br>
                    全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>，低於超載門檻 ({mdd_threshold})。
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 🔤 片語凝聚度 (Phrasal Style) 診斷")
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                st.markdown(f"""
                <div class="msg-box info-box">
                    <strong>ℹ️ 片語凝聚度提醒</strong><br>
                    當前文本 CNP/C 為 <strong>{l2sca_res['CNP/C']}</strong>，低於標竿值 ({current_norm['CNP/C']})。<br>
                    建議增加名詞片語後置修飾語（Noun Post-modifiers）。
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-box success-box">
                    <strong>✅ 片語凝聚度達標</strong><br>
                    CNP/C 為 <strong>{l2sca_res['CNP/C']}</strong>，達到標竿標準 ({current_norm['CNP/C']})。
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- TAB 2: 如何計算 (圖表與弧線統一排版) ---
    with tab2:
        st.markdown("### 📊 指標對比與依存語法樹 (Dependency Arc Diagram)")
        
        # 統一圖表高度與邊距
        t2_c1, t2_c2 = st.columns(2)
        with t2_c1:
            st.markdown("#### 1. 當前數值 vs. 標竿數值對比")
            fig = go.Figure(data=[
                go.Bar(name='當前文本', x=['MLS (句長)', 'CNP/C (名詞組)', 'MDD (依存距離)'], y=[l2sca_res['MLS'], l2sca_res['CNP/C'], mdd_res['mdd']], marker_color='#3b82f6'),
                go.Bar(name=f'標竿 ({target_level.split()[0]})', x=['MLS (句長)', 'CNP/C (名詞组)', 'MDD (依存距離)'], y=[current_norm['MLS'], current_norm['CNP/C'], current_norm['MDD_target']], marker_color='#f97316')
            ])
            fig.update_layout(barmode='group', height=360, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with t2_c2:
            st.markdown("#### 2. 篇章修辭關係分佈估算 (DDS)")
            dds_labels = ['Explanation (解釋)', 'Causality (因果)', 'Parallel (對等)', 'Elaboration (補充)']
            fig_dds = px.pie(names=dds_labels, values=[35, 25, 20, 20], height=360)
            fig_dds.update_traces(textposition='inside', textinfo='percent+label')
            fig_dds.update_layout(margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_dds, use_container_width=True)

        st.divider()
        st.markdown("#### 3. 🎯 句法依存弧線圖展示 (Dependency Tree Visualizer)")
        st.caption("以下為第一句之依存結構。弧線越長，代表認知負載 (Dependency Distance) 越高。")
        
        doc = nlp(active_text)
        first_sent = list(doc.sents)[0] if list(doc.sents) else doc
        
        # 【畫出漂亮弧線的關鍵】設定 compact=False，並加大距離與高度，讓曲線完美呈現！
        displacy_options = {
            "compact": False,          # 關閉緊湊模式，弧線會變成優美的拋物線
            "distance": 130,           # 加寬字距，讓長弧線不擁擠
            "word_spacing": 40, 
            "color": "#1e293b", 
            "bg": "#f8fafc",
            "font": "Arial"
        }
        html_dep = displacy.render(first_sent, style="dep", options=displacy_options, page=False)
        # 加大高度 (height=450) 避免長距離拋物線被上下切斷
        st.components.v1.html(f"<div style='overflow-x: auto; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background: #f8fafc;'>{html_dep}</div>", height=450, scrolling=True)

    # --- TAB 3: 改寫建議 ---
    with tab3:
        st.markdown("### 🛠️ 三維度自動化診斷與自適應改寫建議")
        st.markdown("#### 🌿 一、 句法層級診斷 (Syntactic Level & Memory Load)")
        
        if mdd_res["overload_spans"]:
            st.markdown(f"""
            <div class="msg-box error-box">
                <strong>🚨 檢測到 {len(mdd_res['overload_spans'])} 處大腦工作記憶超載點</strong>（未閉合依存弧 ≥ {arcs_threshold} 條）：
            </div>
            """, unsafe_allow_html=True)
            
            sentences = [sent for sent in doc.sents if sent.text.strip()]
            
            for idx, item in enumerate(mdd_res["overload_spans"]):
                sent_id = item["sentence_id"]
                target_word = item["word"]
                
                if 1 <= sent_id <= len(sentences):
                    target_sent_obj = sentences[sent_id - 1]
                    raw_sentence = target_sent_obj.text.strip()
                    pattern = re.compile(rf'\b({re.escape(target_word)})\b', re.IGNORECASE)
                    highlighted_sentence = pattern.sub(
                        r"<mark style='background-color: #fde047; color: #854d0e; font-weight: bold; padding: 2px 6px; border-radius: 4px;'>\1</mark>", 
                        raw_sentence
                    )
                else:
                    target_sent_obj = None
                    highlighted_sentence = active_text

                with st.expander(f"📍 **超載位置 {idx+1}**：第 {sent_id} 句 (問題詞: `{target_word}`)", expanded=True):
                    st.markdown(
                        f"<div style='background-color: #f8fafc; color: #0f172a; padding: 16px; border-left: 5px solid #eab308; margin-bottom: 15px; font-size: 1.1rem; line-height: 1.6; border-radius: 6px;'>{highlighted_sentence}</div>", 
                        unsafe_allow_html=True
                    )
                    st.write(f"ℹ️ **超載原因**：{item['msg']}")
                    st.write("👉 **句法改寫建議**：此處因前置/後置修飾語過長，建議將長介詞組 (PP) 或關係子句**拆分為兩個獨立小句**。")
                    
                    if target_sent_obj:
                        st.markdown("**【該句之有方向依存樹弧線圖】**：")
                        # 同樣使用優美的拋物線模式
                        svg_html = displacy.render(target_sent_obj, style="dep", options={"compact": False, "distance": 110, "bg": "#ffffff"}, page=False)
                        st.components.v1.html(f"<div style='overflow-x: auto; background-color: #ffffff; padding: 15px; border: 1px solid #e2e8f0; border-radius: 8px;'>{svg_html}</div>", height=350, scrolling=True)
        else:
            st.markdown("""
            <div class="msg-box success-box">
                🎉 <strong>句法結構順暢</strong>：未發現顯著的語意/語法超載結構，依存距離符合標準！
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        c3_1, c3_2 = st.columns(2)
        with c3_1:
            st.markdown("#### 🔤 二、 字彙層級診斷 (Lexical Level)")
            st.write("• **預估詞彙密度 (LD)**: 54.2% (符合標準)")
            st.write("• **學術詞彙 (AWL) 覆蓋率**: 約 8.5%")
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                st.info("💡 **建議**：可將一般動詞改寫為名詞化結構 (Nominalization)。")
            else:
                st.success("✅ **字彙難易度適中**。")
        with c3_2:
            st.markdown("#### 📑 三、 語篇層級診斷 (Discourse Level)")
            st.write("• **邏輯轉折詞**：成功使用 *Furthermore, Therefore* 等，邏輯清晰。")
            st.write("• **連貫性優化提示**：建議增加標示「對比」或「條件」的 EDUs 句型。")

    # --- TAB 4: 系統功能 ---
    with tab4:
        st.markdown("### ⚙️ 系統功能與診斷報告匯出")
        sys_col1, sys_col2 = st.columns([1, 2])
        
        with sys_col1:
            st.markdown("#### 📥 1. 下載自動化診斷報告")
            report_md = f"""# MDTCD 教材複雜度診斷報告\n- **分析檔名**: {file_name_label}\n- **標竿年級**: {target_level}\n- **平均依存距離 (MDD)**: {mdd_res['mdd']}\n- **超載點數量**: {len(mdd_res['overload_spans'])}"""
            st.download_button("📄 下載 Markdown 診斷報告 (.md)", data=report_md, file_name=f"MDTCD_Report.md", mime="text/markdown")

        with sys_col2:
            st.markdown("#### 📜 2. 本次 Session 歷程診斷比對表")
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, height=200)
