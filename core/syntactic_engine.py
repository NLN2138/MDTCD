import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import spacy
from spacy import displacy
import re
import pandas as pd
import os
import textwrap
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 載入本機 .env 檔案（如果有的話）
load_dotenv()

# 安全取得 OpenAI API Key
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    pass

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key) if api_key else None

# -----------------------------------------------------------------------------
# 1. 載入核心計算模組
# -----------------------------------------------------------------------------
from core.syntactic_engine import (
    calculate_mdd_and_memory_load, 
    calculate_l2sca_approximations,
    analyze_discourse_markers
)

# -----------------------------------------------------------------------------
# 2. 頁面配置與效能優化 (Caching)
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

# -----------------------------------------------------------------------------
# 3. 全局 CSS 樣式
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    /* 1. 頂部固定標題 */
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

    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 800 !important; color: #1d4ed8 !important; }
    [data-testid="stMetricLabel"] { font-size: 1.05rem !important; color: #475569; font-weight: 700; }
    
    div[data-baseweb="tab-list"] {
        background-color: #f8fafc; padding: 10px 10px 0px 10px; border-bottom: 2px solid #e2e8f0; border-radius: 8px 8px 0 0;
    }

    /* 卡片與提示框樣式 */
    .custom-card { background-color: #ffffff; border-radius: 12px; padding: 1.5rem; border: 1px solid #e2e8f0; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .msg-box { padding: 1.2rem; border-radius: 8px; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0; }
    .error-box { background-color: #fef2f2; border-left: 5px solid #ef4444; color: #991b1b; }
    .success-box { background-color: #f0fdf4; border-left: 5px solid #22c55e; color: #166534; }
    .info-box { background-color: #eff6ff; border-left: 5px solid #3b82f6; color: #1e3a8a; }

    /* 左側固定 */
    [data-testid="column"]:first-of-type {
        position: -webkit-sticky;
        position: sticky;
        top: 160px;
        height: calc(100vh - 180px); 
        overflow-y: auto; 
        padding-right: 1rem;
    }
    [data-testid="column"]:first-of-type::-webkit-scrollbar { width: 0px; background: transparent; }
</style>

<div class="fixed-header">
    <div class="header-title">🇹🇼 台灣英語教材多維度複雜度自動化診斷系統 (MDTCD)</div>
    <div class="header-subtitle">Multi-Dimensional Textbook Complexity Diagnostics | 融合 SLA、依存語法與 XAI 可解釋性 AI</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. 標竿基準資料庫
# -----------------------------------------------------------------------------
NORMS = {
    "高中第四、五冊 (高二/高三)": {"MLS": 17.97, "CNP/C": 1.14, "MDD_target": 2.2, "DCD": 2.10},
    "學測優秀作文 (GSAT)": {"MLS": 19.28, "CNP/C": 1.03, "MDD_target": 2.4, "DCD": 2.50},
    "真實學術論文 (RA)": {"MLS": 27.31, "CNP/C": 2.32, "MDD_target": 3.2, "DCD": 3.10}
}

# -----------------------------------------------------------------------------
# 5. OpenAI 智慧改寫函式
# -----------------------------------------------------------------------------
def call_openai_rewriter(overloaded_sentence, target_word, cross_count):
    if not client:
        return "⚠️ 尚未設定 OpenAI API Key。請透過 .env 檔案或 Streamlit Secrets 設定 OPENAI_API_KEY。"
    
    prompt = f"""
    You are an experienced EFL (English as a Foreign Language) junior and senior high school English teacher and corpus linguist in Taiwan. 
    In the following sentence, the word '{target_word}' is structurally overloaded. There are {cross_count} syntactic dependency arcs crossing over it simultaneously, which creates a high storage cost and working memory bottleneck for readers.
    
    Sentence: "{overloaded_sentence}"
    
    Please provide actual, ready-to-use **English rewritten sentences** suitable for Taiwanese high school students, along with brief explanations in Traditional Chinese (台灣繁體中文):
    
    1. **拆句建議 (Split Option)**: 
       - 提供改寫後的**實際英文句子**（將長句拆為兩個簡短的獨立小句，消除跨越弧線）。
       - 附帶對應的中文翻譯。
    
    2. **結構簡化 (Phrasal Option)**: 
       - 提供改寫後的**實際英文句子**（精簡過長的介詞組 PP 或關係子句，維持單句但結構更流暢）。
       - 附帶對應的中文翻譯。
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional EFL teacher and material designer who provides concise, natural English sentence revisions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI 調用發生錯誤：{str(e)}"

# =============================================================================
# 6. 網格佈局 (左系統、右作業)
# =============================================================================
col_sys, col_work = st.columns([1, 3], gap="large")

with col_sys:
    st.markdown("### ⚙️ 系統參數設定")
    target_level = st.selectbox("🎯 選擇目標語體 / 標竿", options=list(NORMS.keys()), index=0)
    current_norm = NORMS[target_level]
    
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    mdd_threshold = st.slider("🚨 全篇 MDD 超載臨界值", min_value=1.5, max_value=4.5, value=3.0, step=0.1)
    
    # 決定有幾條線跨越才發出單句改寫警報
    arcs_threshold = st.slider("🧠 跨越弧(Storage Cost) 超載門檻", min_value=2, max_value=7, value=4, step=1, 
                               help="基於 DLT 依存局部性理論：單字上方若同時跨越多條語法弧線，大腦儲存成本將急遽上升。")
    
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 📊 當前對照基準")
    st.info(f"**平均句長 (MLS)**: {current_norm['MLS']}\n\n**複雜名詞組 (CNP/C)**: {current_norm['CNP/C']}\n\n**依存距離 (MDD)**: {current_norm['MDD_target']}\n\n**語篇密度 (DCD)**: {current_norm['DCD']}%")

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
        # 動態計算核心指標：傳入 arcs_threshold 抓取跨越弧
        mdd_res = calculate_mdd_and_memory_load(active_text, arcs_threshold=arcs_threshold)
        l2sca_res = calculate_l2sca_approximations(active_text)
        disc_res = analyze_discourse_markers(active_text)
        
        is_custom_overloaded = mdd_res["mdd"] >= mdd_threshold
        current_time = datetime.now().strftime("%H:%M:%S")

        st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        
        # 呈現 4 大核心指標
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("平均句子長度 (MLS)", l2sca_res["MLS"], delta=round(l2sca_res["MLS"] - current_norm["MLS"], 2))
        m2.metric("複雜名詞片語 (CNP/C)", l2sca_res["CNP/C"], delta=round(l2sca_res["CNP/C"] - current_norm["CNP/C"], 2))
        m3.metric("平均依存距離 (MDD)", mdd_res["mdd"], delta=round(mdd_res["mdd"] - current_norm["MDD_target"], 2), delta_color="inverse")
        
        # 第 4 指標：顯示抓出幾個大腦瓶頸點
        overload_count = len(mdd_res['overload_spans'])
        m4.metric("跨越弧超載瓶頸點", f"{overload_count} 處", delta=f"超過 {arcs_threshold} 條", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["💡 為什麼 (Why)", "📊 如何計算 (How)", "🛠️ 診斷與改寫 (Recommendations)", "⚙️ 系統功能 (System)"])

        # ==========================================
        # TAB 1: 為什麼
        # ==========================================
        with tab1:
            st.markdown(f"<h4 style='margin-bottom: 1.2rem; color: #334155;'>📌 當前比對基準：<b>{target_level}</b></h4>", unsafe_allow_html=True)
            
            # 1. 詞彙
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                lex_msg = f"""<div class="msg-box info-box"><strong>ℹ️ 片語凝聚度偏低</strong><br>當前文本複雜名詞片語 (CNP/C) 為 <strong>{l2sca_res['CNP/C']}</strong>，低於標竿值 ({current_norm['CNP/C']})。建議增加學術詞彙或名詞後置修飾語。</div>"""
            else:
                lex_msg = f"""<div class="msg-box success-box"><strong>✅ 片語凝聚度達標</strong><br>CNP/C 為 <strong>{l2sca_res['CNP/C']}</strong>，達到標竿標準 ({current_norm['CNP/C']})。</div>"""
            st.markdown(f"""<div class="custom-card"><h4 style="margin-top:0; margin-bottom: 0.8rem; color:#1e293b;">🔤 一、 詞彙與片語凝聚度診斷</h4>{lex_msg}</div>""", unsafe_allow_html=True)

            # 2. 句法
            if is_custom_overloaded:
                syn_msg = f"""<div class="msg-box error-box"><strong>🚨 全篇認知加工負荷偏高</strong><br>全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>（超過門檻 {mdd_threshold}）。這代表文章整體而言充滿長距離結構。</div>"""
            else:
                syn_msg = f"""<div class="msg-box success-box"><strong>✅ 全篇認知加工負荷適中</strong><br>全篇 MDD 為 <strong>{mdd_res['mdd']}</strong>，低於警示門檻 ({mdd_threshold})。</div>"""
            st.markdown(f"""<div class="custom-card"><h4 style="margin-top:0; margin-bottom: 0.8rem; color:#1e293b;">🌿 二、 句法與認知負荷診斷</h4>{syn_msg}</div>""", unsafe_allow_html=True)
            
            # 3. 篇章
            if disc_res["discourse_density"] < current_norm["DCD"]:
                disc_msg = f"""<div class="msg-box info-box"><strong>ℹ️ 邏輯銜接密度較低</strong><br>當前每百字邏輯銜接詞為 <strong>{disc_res['discourse_density']} 個</strong>，低於標竿 ({current_norm['DCD']} 個)。建議補強因果或轉折連接詞以提升流暢度。</div>"""
            else:
                disc_msg = f"""<div class="msg-box success-box"><strong>✅ 語篇連貫度良好</strong><br>當前每百字邏輯銜接詞為 <strong>{disc_res['discourse_density']} 個</strong>，已達到或超過標竿 ({current_norm['DCD']} 個)。共偵測到 {disc_res['total_markers']} 個銜接標記。</div>"""
            st.markdown(f"""<div class="custom-card"><h4 style="margin-top:0; margin-bottom: 0.8rem; color:#1e293b;">📑 三、 語篇邏輯連貫度診斷</h4>{disc_msg}</div>""", unsafe_allow_html=True)

        # ==========================================
        # TAB 2: 如何計算
        # ==========================================
        with tab2:
            st.markdown("### 1. 📊 多維度診斷指標 (詞彙、句法、篇章) vs. 標竿對比圖")
            
            categories = ['詞彙: CNP/C<br>(名詞組複雜度)', '句法: MLS<br>(平均句子長度)', '句法: MDD<br>(平均依存距離)', '篇章: DCD<br>(每百字銜接詞數)']
            
            current_values = [
                l2sca_res['CNP/C'], 
                l2sca_res['MLS'], 
                mdd_res['mdd'], 
                disc_res['discourse_density']
            ]
            
            norm_values = [
                current_norm['CNP/C'], 
                current_norm['MLS'], 
                current_norm['MDD_target'], 
                current_norm['DCD']
            ]

            fig = go.Figure(data=[
                go.Bar(
                    name='當前文本分析值', x=categories, y=current_values, 
                    text=current_values, textposition='auto', marker_color='#1d4ed8'
                ),
                go.Bar(
                    name=f'標竿對照值 ({target_level.split()[0]})', x=categories, y=norm_values, 
                    text=norm_values, textposition='auto', marker_color='#f59e0b'
                )
            ])
            
            fig.update_layout(
                barmode='group', height=420, margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="指標數值", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("### 2. 📑 篇章邏輯銜接詞成分統計")
            col_d1, col_d2 = st.columns([1, 1])
            with col_d1:
                df_cat = pd.DataFrame(list(disc_res["category_counts"].items()), columns=["邏輯類別", "出現次數"])
                fig_cat = px.bar(df_cat, x="邏輯類別", y="出現次數", color="邏輯類別", title="語篇標記分類分布")
                fig_cat.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_cat, use_container_width=True)
            with col_d2:
                st.markdown("**🔍 本文偵測到的邏輯銜接詞：**")
                if disc_res["found_markers"]:
                    df_found = pd.DataFrame(disc_res["found_markers"])
                    df_found.columns = ["銜接詞 (Marker)", "邏輯類別 (Category)", "出現頻次 (Count)"]
                    st.dataframe(df_found, use_container_width=True, height=250)
                else:
                    st.info("💡 當前文本中未偵測到常見的語篇邏輯銜接詞。")

            st.markdown("<hr>", unsafe_allow_html=True)

            st.markdown("### 3. 🎯 句法依存樹弧線圖 (Dependency Visualizer)")
            with st.expander("👁️ 點擊展開 / 收合句法依存樹狀圖 (預設為第一句)", expanded=False):
                st.caption("展示帶方向的拋物線依存結構。弧線越長、跨越越密集，代表認知負載 (Storage Cost) 越高。")
                doc = nlp(active_text)
                first_sent = list(doc.sents)[0] if list(doc.sents) else doc
                displacy_options = {"compact": False, "distance": 130, "word_spacing": 45, "color": "#1e293b", "bg": "#f8fafc", "font": "Arial"}
                html_dep = displacy.render(first_sent, style="dep", options=displacy_options, page=False)
                st.components.v1.html(f"<div style='overflow-x: auto; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; background: #f8fafc;'>{html_dep}</div>", height=500, scrolling=True)

        # ==========================================
        # TAB 3: 診斷與改寫建議
        # ==========================================
        with tab3:
            # 一、 詞彙層級
            if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
                lex_advice = "<div class='msg-box info-box' style='margin-bottom:0;'>💡 <strong>優化建議</strong>：可將一般動詞改寫為名詞化結構 (Nominalization)，增加學術感與語句凝聚度。</div>"
            else:
                lex_advice = "<div class='msg-box success-box' style='margin-bottom:0;'>✅ <strong>字彙難易度適中</strong>。</div>"

            lex_html = textwrap.dedent(f"""
            <div class="custom-card">
                <h4 style="margin-top: 0; color: #1e293b; margin-bottom: 1rem;">🔤 一、 字彙層級改寫建議</h4>
                <p style="margin-bottom: 0.8rem; color: #334155;">
                    • <strong>複雜名詞組現況</strong>: {l2sca_res['CNP/C']} (標竿為 {current_norm['CNP/C']})
                </p>
                {lex_advice}
            </div>
            """)
            st.markdown(lex_html, unsafe_allow_html=True)

            # 二、 句法層級 (依跨越弧線數定位)
            st.markdown("<h4 style='color: #1e293b; margin-top: 1.5rem; margin-bottom: 1rem;'>🌿 二、 句法層級改寫建議（動態定位跨越弧瓶頸）</h4>", unsafe_allow_html=True)
            
            if mdd_res["overload_spans"]:
                total_overloads = len(mdd_res['overload_spans'])
                st.markdown(f"""<div class="msg-box error-box" style="margin-bottom: 1rem;"><strong>🚨 檢測到 {total_overloads} 處大腦工作記憶「儲存成本 (Storage Cost)」超載瓶頸</strong>（基於資源保護，系統限制每篇文章<strong>最多可使用 AI 索取 3 句</strong>實際改寫範例）：</div>""", unsafe_allow_html=True)
                
                doc = nlp(active_text)
                sentences = [sent for sent in doc.sents if sent.text.strip()]
                
                limited_spans = mdd_res["overload_spans"][:3]
                
                for idx, item in enumerate(limited_spans):
                    sent_id = item["sentence_id"]
                    target_word = item["word"]
                    cross_count = item["cross_count"]
                    
                    if 1 <= sent_id <= len(sentences):
                        target_sent_obj = sentences[sent_id - 1]
                        raw_sentence = target_sent_obj.text.strip()
                        pattern = re.compile(rf'\b({re.escape(target_word)})\b', re.IGNORECASE)
                        highlighted_sentence = pattern.sub(r"<mark style='background-color: #fde047; color: #854d0e; font-weight: bold; padding: 2px 6px; border-radius: 2px;'>\1</mark>", raw_sentence)
                    else:
                        target_sent_obj = None
                        highlighted_sentence = active_text

                    with st.expander(f"📍 **瓶頸位置 {idx+1} / {len(limited_spans)}**：第 {sent_id} 句 (阻塞字: `{target_word}` / 跨越弧: {cross_count} 條)", expanded=False):
                        
                        st.markdown(f"##### 1️⃣ 瓶頸位置與基礎分析")
                        st.markdown(f"<div style='background-color: #f8fafc; color: #0f172a; padding: 16px; border-left: 5px solid #eab308; margin-bottom: 15px; font-size: 1.1rem; line-height: 1.6; border-radius: 6px;'>{highlighted_sentence}</div>", unsafe_allow_html=True)
                        st.write(f"ℹ️ **診斷原因**：{item['msg']}")
                        st.write("👉 **自動改寫提示**：該字上方跨越過多未完成的語法關係，建議呼叫 AI 助教進行拆句或修飾語精簡。")
                        
                        st.markdown("<hr style='margin: 1rem 0; border: none; border-top: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)

                        st.markdown(f"##### 2️⃣ 依存樹圖分析 (視覺化跨越弧)")
                        if target_sent_obj:
                            with st
