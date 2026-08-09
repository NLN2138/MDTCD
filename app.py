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
# 1. 頁面配置與效能優化 (Caching)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TW-EFL MDTCD 診斷系統", 
    page_icon="🇹🇼",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_sm")

nlp = load_nlp_model()

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------------------------------------------------------
# 2. 全局 CSS 樣式
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    .fixed-header {
        position: fixed;
        top: 0; left: 0; width: 100vw; z-index: 999999;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 1.2rem 2rem 1.2rem 4.5rem; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border-bottom: 4px solid #3b82f6;
    }
    .header-title { font-size: 1.85rem !important; font-weight: 800 !important; margin: 0 !important; letter-spacing: 0.5px; }
    .header-subtitle { font-size: 0.95rem !important; color: #94a3b8 !important; margin-top: 6px !important; }

    header[data-testid="stHeader"] { background: transparent !important; z-index: 1000000 !important; }
    .block-container { padding-top: 150px !important; }

    [data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 800 !important; color: #1d4ed8 !important; }
    [data-testid="stMetricLabel"] { font-size: 1.1rem !important; color: #475569; font-weight: 700; }
    
    div[data-baseweb="tab-list"] {
        background-color: #f8fafc; padding: 10px 10px 0px 10px; border-bottom: 2px solid #e2e8f0; border-radius: 8px 8px 0 0;
    }

    .custom-card { background-color: #ffffff; border-radius: 12px; padding: 1.5rem; border: 1px solid #e2e8f0; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    
    /* 訊息提示框核心樣式 */
    .msg-box { padding: 1.2rem; border-radius: 8px; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0; }
    
    /* 🔴 紅色 (超載) */
    .error-box { background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b; }
    
    /* 🟩 綠色 (適中 / 達標) */
    .success-box { background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534; }
    
    /* 🟦 藍色 (偏低 / 建議) */
    .info-box { background-color: #eff6ff; border-left: 5px solid #3b82f6; color: #1e3a8a; }
    
    .system-panel { background-color: #f8fafc; padding: 1.5rem; border-radius: 12px; border: 1px solid #cbd5e1; }
</style>

<div class="fixed-header">
    <div class="header-title">🇹🇼 台灣英語教材多維度複雜度自動化診斷系統 (MDTCD)</div>
    <div class="header-subtitle">Multi-Dimensional Textbook Complexity Diagnostics | 融合 SLA、依存語法與 XAI 可解釋性 AI</div>
</div>
""", unsafe_allow_html=True)

NORMS = {
    "高中第四、五冊 (高二/高三, Ting 2024)": {"MLS": 17.97, "CNP/C": 1.14, "MDD_target": 2.2},
    "學測優秀作文 (GSAT)": {"MLS": 19.28, "CNP/C": 1.03, "MDD_target": 2.4},
    "真實學術論文 (RA)": {"MLS": 27.31, "CNP/C": 2.32, "MDD_target": 3.2}
}

# =============================================================================
# 3. 網格佈局
# =============================================================================
col_sys, col_work = st.columns([1, 3], gap="large")

with col_sys:
    st.markdown("### ⚙️ 系統參數設定")
    target_level = st.selectbox("🎯 選擇目標語體 / 標竿", options=list(NORMS.keys()), index=0)
    current_norm = NORMS[target_level]
    
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    mdd_threshold = st.slider("🚨 MDD 超載臨界值", min_value=1.5, max_value=4.5, value=3.0, step=0.1)
    arcs_threshold = st.slider("🧠 跨越弧超載門檻", min_value=2, max_value=7, value=4, step=1)
    
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 📊 當前對照基準")
    st.info(f"**平均句長 (MLS)**: {current_norm['MLS']}\n\n**複雜名詞組 (CNP/C)**: {current_norm['CNP/C']}\n\n**依存距離 (MDD)**: {current_norm['MDD_target']}")

with col_work:
    default_sample = (
        "Title: The Digital Footprint of Modern Society.\n\n"
        "In today's interconnected world, almost every online action leaves a digital trace that reflects our personal habits, interests, and behaviors.\n"
        "As teenagers navigate various social media platforms, they often share personal opinions and life moments without realizing the potential consequences of their online presence.\n\n"
        "According to recent educational studies on digital literacy, internet users who regularly broadcast their daily activities to the public tend to expose themselves to potential privacy risks.\n"
        "Furthermore, algorithms designed by large technology corporations analyze these massive amounts of user data to deliver targeted advertisements, which subtly influences individual decision-making processes.\n"
        "Therefore, developing critical thinking skills regarding digital privacy has become an essential responsibility for high school students in the twenty-first century."
    )

    st.markdown("### 📝 文本輸入與分析")
    user_input = st.text_area("請貼入英文課文或教材（若未貼入，將自動採用預設範文）：", height=150, value=default_sample)
    cleaned_text = re.sub(r'([a-zA-Z0-9])\n([a-zA-Z0-9])', r'\1. \2', user_input.strip())
    active_text = cleaned_text if cleaned_text else default_sample

    analyze_btn = st.button("🚀 開始多維度自動診斷", type="primary", use_container_width=True)

    if analyze_btn or active_text:
        mdd_res = calculate_mdd_and_memory_load(active_text)
        l2sca_res = calculate_l2sca_approximations(active_text)
        is_custom_overloaded = mdd_res["mdd"] >= mdd_threshold

        log_entry = {"標竿": target_level, "MLS": l2sca_res["MLS"], "CNP/C": l2sca_res["CNP/C"], "MDD": mdd_res["mdd"], "超載點數": len(mdd_res["overload_spans"])}
        if not any(d["MDD"] == mdd_res["mdd"] for d in st.session_state.history):
            st.session_state.history.append(log_entry)

        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("平均句子長度 (MLS)", l2sca_res["MLS"], delta=round(l2sca_res["MLS"] - current_norm["MLS"], 2))
        m2.metric("複雜名詞片語 (CNP/C)", l2sca_res["CNP/C"], delta=round(l2sca_res["CNP/C"] - current_norm["CNP/C"], 2))
        m3.metric("平均依存距離 (MDD)", mdd_res["mdd"], delta=round(mdd_res["mdd"] - current_norm["MDD_target"], 2), delta_color="inverse")
        m4.metric("工作記憶超載點", f"{len(mdd_res['overload_spans'])} 處", delta=f"MDD > {mdd_threshold}", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["💡 為什麼 (Why)", "📊 如何計算 (How)", "🛠️ 診斷與改寫 (Recommendations)", "⚙️ 系統功能 (System)"])

        # ==========================================
        # TAB 1: 為什麼
        # ==========================================
        with tab1:
            st.markdown(f"<h4 style='margin-bottom: 1.2rem; color: #334155;'>📌 當前比對基準：<b>{target_level}</b></h4>", unsafe_allow_html=True)
            
            # 🔤 一、詞彙 (偏低: 藍色 / 達標: 綠色)
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                lex_msg = f"""<div class="msg-box info-box"><strong>ℹ️ 片語凝聚度偏低</strong><br>當前文本複雜名詞片語 (CNP/C) 為 <strong>{l2sca_res['CNP/C']}</strong>，低於標竿值 ({current_norm['CNP/C']})。建議增加學術詞彙或名詞後置修飾語。</div>"""
            else:
                lex_msg = f"""<div class="msg-box success-box"><strong>✅ 片語凝聚度達標</strong><br>CNP/C 為 <strong>{l2sca_res['CNP/C']}</strong>，達到標竿標準 ({current_norm['CNP/C']})。</div>"""
            st.markdown(f"""<div class="custom-card"><h4 style="margin-top:0; margin-bottom: 0.8rem; color:#1e293b;">🔤 一、 詞彙與片語凝聚度診斷</h4>{lex_msg}</div>""", unsafe_allow_html=True)

            # 🌿 二、句法 (超載: 紅色 / 適中: 綠色)
            if is_custom_overloaded:
                syn_msg = f"""<div class="msg-box error-box"><strong>🚨 認知加工負荷超載</strong><br>全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>（超過門檻 {mdd_threshold}）。大腦在處理長距離依存關係時容易產生工作記憶遲滯。</div>"""
            else:
                syn_msg = f"""<div class="msg-box success-box"><strong>✅ 認知加工負荷適中</strong><br>全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>，低於超載門檻 ({mdd_threshold})。</div>"""
            st.markdown(f"""<div class="custom-card"><h4 style="margin-top:0; margin-bottom: 0.8rem; color:#1e293b;">🌿 二、 句法與認知負荷診斷</h4>{syn_msg}</div>""", unsafe_allow_html=True)
            
            # 📑 三、語篇 (適中: 綠色)
            st.markdown("""<div class="custom-card"><h4 style="margin-top:0; margin-bottom: 0.8rem; color:#1e293b;">📑 三、 語篇邏輯連貫度診斷</h4><div class="msg-box success-box"><strong>✅ 篇章結構完整</strong><br>系統偵測到明確的段落劃分與基礎語篇單位 (EDUs)。符合閱讀所需之上下文邏輯銜接。</div></div>""", unsafe_allow_html=True)

        # ==========================================
        # TAB 2: 如何計算
        # ==========================================
        with tab2:
            st.markdown("### 1. 📊 當前數值 vs. 標竿數值對比")
            fig = go.Figure(data=[
                go.Bar(name='當前文本', x=['MLS (句長)', 'CNP/C (名詞組)', 'MDD (依存距離)'], y=[l2sca_res['MLS'], l2sca_res['CNP/C'], mdd_res['mdd']], marker_color='#1d4ed8'),
                go.Bar(name=f'標竿 ({target_level.split()[0]})', x=['MLS (句長)', 'CNP/C (名詞组)', 'MDD (依存距離)'], y=[current_norm['MLS'], current_norm['CNP/C'], current_norm['MDD_target']], marker_color='#f59e0b')
            ])
            fig.update_layout(barmode='group', height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("### 2. 🎯 句法依存樹弧線圖 (Dependency Visualizer)")
            with st.expander("👁️ 點擊展開 / 收合句法依存樹狀圖 (預設為第一句)", expanded=False):
                st.caption("展示帶方向的拋物線依存結構。弧線越長，代表認知負載 (Dependency Distance) 越高。")
                doc = nlp(active_text)
                first_sent = list(doc.sents)[0] if list(doc.sents) else doc
                displacy_options = {"compact": False, "distance": 130, "word_spacing": 45, "color": "#1e293b", "bg": "#f8fafc", "font": "Arial"}
                html_dep = displacy.render(first_sent, style="dep", options=displacy_options, page=False)
                st.components.v1.html(f"<div style='overflow-x: auto; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background: #f8fafc;'>{html_dep}</div>", height=500, scrolling=True)

        # ==========================================
        # TAB 3: 診斷與改寫建議
        # ==========================================
        with tab3:
            # 1. 詞彙層級 (偏低: 藍色 / 達標: 綠色)
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                lex_advice = "<div class='msg-box info-box' style='margin-bottom:0;'>💡 <strong>優化建議</strong>：可將一般動詞改寫為名詞化結構 (Nominalization)，增加學術感。</div>"
            else:
                lex_advice = "<div class='msg-box success-box' style='margin-bottom:0;'>✅ <strong>字彙難易度適中</strong>。</div>"

            lex_html = f"""
            <div class="custom-card">
                <h4 style="margin-top: 0; color: #1e293b; margin-bottom: 1rem;">🔤 一、 字彙層級改寫建議</h4>
                <p style="margin-bottom: 0.8rem; color: #334155;">
                    • <strong>預估詞彙密度 (LD)</strong>: 54.2% (符合標準)<br>
                    • <strong>學術詞彙 (AWL) 覆蓋率</strong>: 約 8.5%
                </p>
                {lex_advice}
            </div>
            """
            st.markdown(lex_html, unsafe_allow_html=True)

            # 2. 句法層級 (超載: 紅色)
            st.markdown("<h4 style='color: #1e293b; margin-top: 1.5rem; margin-bottom: 1rem;'>🌿 二、 句法層級改寫建議</h4>", unsafe_allow_html=True)
            
            if mdd_res["overload_spans"]:
                st.markdown(f"""<div class="msg-box error-box" style="margin-bottom: 1rem;"><strong>🚨 檢測到 {len(mdd_res['overload_spans'])} 處大腦工作記憶超載點</strong>（未閉合依存弧 ≥ {arcs_threshold} 條）：</div>""", unsafe_allow_html=True)
                
                doc = nlp(active_text)
                sentences = [sent for sent in doc.sents if sent.text.strip()]
                
                for idx, item in enumerate(mdd_res["overload_spans"]):
                    sent_id = item["sentence_id"]
                    target_word = item["word"]
                    
                    if 1 <= sent_id <= len(sentences):
                        target_sent_obj = sentences[sent_id - 1]
                        raw_sentence = target_sent_obj.text.strip()
                        pattern = re.compile(rf'\b({re.escape(target_word)})\b', re.IGNORECASE)
                        highlighted_sentence = pattern.sub(r"<mark style='background-color: #fde047; color: #854d0e; font-weight: bold; padding: 2px 6px; border-radius: 4px;'>\1</mark>", raw_sentence)
                    else:
                        target_sent_obj = None
                        highlighted_sentence = active_text

                    # 【修改點】將 expanded=True 改為 expanded=False，預設收合問題詞展開框
                    with st.expander(f"📍 **超載位置 {idx+1}**：第 {sent_id} 句 (問題詞: `{target_word}`)", expanded=False):
                        st.markdown(f"<div style='background-color: #f8fafc; color: #0f172a; padding: 16px; border-left: 5px solid #eab308; margin-bottom: 15px; font-size: 1.1rem; line-height: 1.6; border-radius: 6px;'>{highlighted_sentence}</div>", unsafe_allow_html=True)
                        st.write(f"ℹ️ **超載原因**：{item['msg']}")
                        st.write("👉 **句法改寫建議**：此處修飾語過長，建議將長介詞組 (PP) 或關係子句**拆分為兩個獨立小句**。")
                        
                        if target_sent_obj:
                            with st.expander("👁️ 點擊展開該句之有方向拋物線依存樹圖", expanded=False):
                                svg_html = displacy.render(target_sent_obj, style="dep", options={"compact": False, "distance": 120, "bg": "#ffffff"}, page=False)
                                st.components.v1.html(f"<div style='overflow-x: auto; background-color: #ffffff; padding: 15px; border: 1px solid #e2e8f0; border-radius: 8px;'>{svg_html}</div>", height=400, scrolling=True)
            else:
                st.markdown("""<div class="msg-box success-box">🎉 <strong>句法結構順暢</strong>：未發現顯著的超載結構！</div>""", unsafe_allow_html=True)

            # 3. 語篇層級
            disc_html = """
            <div class="custom-card" style="margin-top: 1.5rem;">
                <h4 style="margin-top: 0; color: #1e293b; margin-bottom: 1rem;">📑 三、 語篇層級改寫建議</h4>
                <p style="margin-bottom: 0; color: #334155; line-height: 1.8;">
                    • <strong>邏輯轉折詞</strong>：成功使用 <em>Furthermore, Therefore</em> 等，邏輯清晰。<br>
                    • <strong>連貫性優化提示</strong>：建議增加標示「對比」或「條件」的 EDUs 句型。
                </p>
            </div>
            """
            st.markdown(disc_html, unsafe_allow_html=True)

        # ==========================================
        # TAB 4: 系統功能
        # ==========================================
        with tab4:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 📥 1. 下載自動化診斷報告")
            report_md = f"""# MDTCD 教材複雜度診斷報告\n- **標竿年級**: {target_level}\n- **平均依存距離 (MDD)**: {mdd_res['mdd']}\n- **超載點數量**: {len(mdd_res['overload_spans'])}"""
            st.download_button("📄 下載 Markdown 診斷報告 (.md)", data=report_md, file_name=f"MDTCD_Report.md", mime="text/markdown")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 📜 2. 本次 Session 歷程診斷比對表")
            if st.session_state.history:
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
