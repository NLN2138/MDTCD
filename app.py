import streamlit as st
import plotly.graph_objects as go
from core.syntactic_engine import calculate_mdd_and_memory_load, calculate_l2sca_approximations

st.set_page_config(page_title="TW-EFL MDTCD 診斷系統", layout="wide")

st.title("🇹🇼 台灣中小學英語教材多維度複雜度自動化診斷系統 (MDTCD)")

# 常模對照資料庫 (對齊 Ting 2024 實證數據)
NORMS = {
    "高中五年級/高二 (Ting 2024)": {"MLS": 17.97, "CNP/C": 1.14, "MDD_target": 2.2},
    "學測優秀作文 (GSAT)": {"MLS": 19.28, "CNP/C": 1.03, "MDD_target": 2.4},
    "真實學術論文 (RA)": {"MLS": 27.31, "CNP/C": 2.32, "MDD_target": 3.2}
}

# 1. 側邊欄：目標基準設定
st.sidebar.header("🎯 教材與年級設定")
target_level = st.sidebar.selectbox(
    "選擇對照標竿年級 / 語體",
    options=list(NORMS.keys()),
    index=0
)

# 取得當前選中的標竿數據
current_norm = NORMS[target_level]

# 2. 文字輸入區
default_sample = "The English textbooks used in senior high schools contain complex syntactic structures that challenge students in their learning process."
user_text = st.text_area("請貼入英文課文或教材文本：", height=120, value=default_sample)

# 3. 即時運算 (Reactive - 只要文字或側邊欄選單改變，自動重新計算)
if user_text.strip():
    mdd_res = calculate_mdd_and_memory_load(user_text)
    l2sca_res = calculate_l2sca_approximations(user_text)

    st.markdown("---")
    
    # 頂部 Metric Cards (動態計算 Delta 差值)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "平均句子長度 (MLS)", 
        l2sca_res["MLS"], 
        delta=round(l2sca_res["MLS"] - current_norm["MLS"], 2)
    )
    col2.metric(
        "複雜名詞片語 (CNP/C)", 
        l2sca_res["CNP/C"], 
        delta=round(l2sca_res["CNP/C"] - current_norm["CNP/C"], 2)
    )
    col3.metric(
        "平均依存距離 (MDD)", 
        mdd_res["mdd"], 
        delta=round(mdd_res["mdd"] - current_norm["MDD_target"], 2), 
        delta_color="inverse"
    )
    col4.metric(
        "工作記憶超載點", 
        f"{len(mdd_res['overload_spans'])} 處"
    )

    st.markdown("---")

    # Khosravi et al. (2022) 三頁籤 XAI 介面
    tab1, tab2, tab3 = st.tabs(["💡 頁籤一：為什麼 (Why)", "📊 頁籤二：如何計算 (How)", "🛠️ 頁籤三：改寫建議 (Recommendations)"])

    with tab1:
        st.subheader("診斷理由與常模對照")
        st.info(f"📌 當前對照標竿：**{target_level}** (基準值：MLS={current_norm['MLS']}, CNP/C={current_norm['CNP/C']}, MDD={current_norm['MDD_target']})")
        
        if mdd_res["is_overloaded"]:
            st.warning(f"⚠️ **認知加工負荷過高**：全篇 MDD 為 {mdd_res['mdd']}（超過安全閾值 3.0）。大腦在解讀長距離依存關係時容易產生工作記憶遲滯。")
        else:
            st.success("✅ **認知加工負荷適中**：依存距離符合學習者工作記憶可承受範圍。")

        if l2sca_res["CNP/C"] < current_norm["CNP/C"]:
            st.warning(f"ℹ️ **片語凝聚度提醒**：當前文本 CNP/C 為 {l2sca_res['CNP/C']}，低於標竿值 ({current_norm['CNP/C']})。若目標為銜接該語體，建議增加名詞片語修飾結構。")
        else:
            st.success(f"✅ **片語凝聚度達標**：CNP/C 為 {l2sca_res['CNP/C']}，高於或平於標竿值 ({current_norm['CNP/C']})。")

    with tab2:
        st.subheader("結構與數值詳細分析")
        fig = go.Figure(data=[
            go.Bar(name='當前文本', x=['MLS', 'CNP/C', 'MDD'], y=[l2sca_res['MLS'], l2sca_res['CNP/C'], mdd_res['mdd']], marker_color='#1f77b4'),
            go.Bar(name=f'標竿 ({target_level.split()[0]})', x=['MLS', 'CNP/C', 'MDD'], y=[current_norm['MLS'], current_norm['CNP/C'], current_norm['MDD_target']], marker_color='#ff7f0e')
        ])
        fig.update_layout(barmode='group', title_text=f'當前文本 vs. {target_level} 數值對比', height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Span-level 高亮與自適應修改建議")
        if mdd_res["overload_spans"]:
            for item in mdd_res["overload_spans"]:
                st.error(f"📍 **句子 {item['sentence_id']}** - {item['msg']}")
                st.markdown("👉 **建議**：嘗試將前置/後置介詞組 (PP) 拆分為獨立小句，或調整修飾語位置以縮短詞與 Head 之間的距離。")
        else:
            st.success("🎉 未發現顯著的語意/語法超載結構，教材編寫流暢！")
else:
    st.warning("請在上方輸入框貼入英文課文進行分析。")
