import streamlit as st
import pandas as pd
import os
import glob
import re
import uuid
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler

# ==========================================
# 1. UI 設定:v40.1 修復版
# ==========================================
st.set_page_config(page_title="頂級AI智能助理", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    /* 全域字體設定 */
    .stApp, .stMarkdown, .stText, p, div { 
        font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif !important; 
        color: #2c3e50 !important;
        font-size: 16px !important;
        line-height: 1.8 !important;
    }
    
    /* 修復行內代碼樣式 */
    code {
        color: #2c3e50 !important;
        background-color: #f8f9fa !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid #e9ecef !important;
    }
    
    /* 側邊欄 */
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 2px solid #dee2e6;
    }
    
    /* 聊天氣泡 - 固定不可捲動 */
    .stChatMessage { 
        padding: 1.8rem; 
        border-radius: 16px; 
        margin-bottom: 1.2rem; 
        border: none; 
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        white-space: pre-wrap !important;
        max-height: none !important;
        overflow: visible !important;
    }
    .stChatMessage[data-testid="chat-message-user"] { 
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 6px solid #ff9800;
    }
    .stChatMessage[data-testid="chat-message-assistant"] { 
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 6px solid #2196f3;
    }
    
    /* 強制內容自動換行 */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        white-space: pre-wrap !important;
        max-width: 100% !important;
    }
    
    /* 輸入框 */
    .main .block-container { padding-bottom: 140px !important; }
    .stChatInput { max-width: 1000px; margin: 0 auto; }
    .stChatInput textarea { 
        background-color: #ffffff !important; 
        border: 2px solid #dee2e6 !important; 
        border-radius: 50px !important; 
        padding: 18px 35px !important; 
        font-size: 16px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stChatInput textarea:focus { 
        border-color: #2196f3 !important; 
        box-shadow: 0 12px 40px rgba(33,150,243,0.25) !important;
        transform: translateY(-3px);
    }
    
    /* 標題 */
    .main-title { 
        font-size: 2.8rem; 
        font-weight: 900; 
        text-align: center; 
        margin-bottom: 50px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 數據卡片 */
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px; 
        padding: 25px; 
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
        text-align: center; 
        transition: all 0.3s;
        color: white !important;
    }
    .metric-box:hover { 
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 35px rgba(102,126,234,0.4);
    }
    .metric-num { 
        font-size: 2.5rem; 
        font-weight: 900; 
        color: #ffffff !important;
        margin: 12px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .metric-desc { 
        color: #f0f0f0 !important;
        font-size: 0.95rem; 
        font-weight: 700;
        text-transform: uppercase; 
        letter-spacing: 1.5px;
    }

    /* 狀態顯示器 */
    [data-testid="stStatusWidget"] { 
        border: 2px solid #dee2e6;
        border-radius: 12px;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* 按鈕美化 */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心狀態管理
# ==========================================
keys = ["logged_in", "api_key", "chat_sessions", "current_session_id", 
        "generated_chart", "df_profile", "status_msg", "df", "data_years", 
        "file_count", "all_columns", "data_summary"]
for k in keys:
    if k not in st.session_state: 
        st.session_state[k] = None

if st.session_state.logged_in is None: 
    st.session_state.logged_in = False
if st.session_state.chat_sessions is None: 
    st.session_state.chat_sessions = {}
if st.session_state.data_years is None: 
    st.session_state.data_years = []

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        'title': f"對話-{datetime.now().strftime('%H:%M')}",
        'messages': [{
            "role": "assistant", 
            "content": "🤖 **頂級 AI 助理已就緒** (v40.1)\n\n我已完成系統初始化,可以為您提供:\n\n**✓** 精準數據分析\n**✓** 智能圖表生成  \n**✓** 商業洞察建議\n**✓** 多維度數據探索\n\n請隨時告訴我您需要什麼分析!", 
            "chart": None
        }]
    }
    st.session_state.current_session_id = new_id
    return new_id

if not st.session_state.chat_sessions: 
    create_new_session()

def validate_login(key):
    return key.strip().startswith("AIza") and not re.search(r'[\u4e00-\u9fff]', key)

# ==========================================
# 3. 強化數據處理核心
# ==========================================
def generate_deep_profile(df):
    """生成超詳細的數據檔案"""
    profile = []
    
    # 基本信息
    profile.append(f"資料總筆數: {len(df):,} 筆")
    profile.append(f"欄位總數: {len(df.columns)} 個")
    
    # 時間範圍
    if 'Year' in df.columns:
        years = sorted(df['Year'].unique().tolist())
        profile.append(f"涵蓋年份: {', '.join(map(str, years))}")
    if 'Month' in df.columns:
        months = sorted(df['Month'].dropna().unique().tolist())
        profile.append(f"涵蓋月份: {', '.join(map(str, months))}")
    
    # 數值欄位統計
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        profile.append("\n關鍵數值欄位:")
        for col in numeric_cols[:5]:
            total = df[col].sum()
            avg = df[col].mean()
            profile.append(f"  - {col}: 總計 {total:,.0f} | 平均 {avg:,.0f}")
    
    # 類別欄位範例
    profile.append("\n類別欄位範例:")
    categorical_priority = ['業務員名稱', '客戶供應商簡稱', '產品代號', '規格']
    for col in categorical_priority:
        if col in df.columns:
            unique_count = df[col].nunique()
            examples = df[col].dropna().unique()[:4].tolist()
            examples_str = ', '.join(map(str, examples))
            profile.append(f"  - {col} ({unique_count} 種): {examples_str}...")
    
    return "\n".join(profile)

@st.cache_data(show_spinner=False, ttl=600)
def load_data():
    """超強數據載入系統 - 支援所有 XLSX 檔案"""
    files = glob.glob(os.path.join("data", "*.xlsx")) + glob.glob(os.path.join("data", "*.csv"))
    
    if not files: 
        return None, "❌ 未發現數據文件", "", [], 0, []
    
    all_dataframes = []
    
    for file_path in files:
        try:
            # 讀取所有工作表
            if file_path.endswith('.xlsx'):
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    df.columns = df.columns.str.strip()
                    
                    # 智能數據清洗
                    for col in df.columns:
                        # 處理數值欄位
                        if df[col].dtype == 'object':
                            try:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass
                        
                        # 清理字串欄位
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
                    
                    # 日期處理
                    date_columns = [c for c in df.columns if '日期' in c or 'date' in c.lower()]
                    for date_col in date_columns:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        if df[date_col].notna().any():
                            df['Year'] = df[date_col].dt.year
                            df['Month'] = df[date_col].dt.month
                            df['Quarter'] = df[date_col].dt.quarter
                    
                    all_dataframes.append(df)
            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                all_dataframes.append(df)
                
        except Exception as e:
            st.warning(f"⚠️ 讀取 {os.path.basename(file_path)} 時發生錯誤: {str(e)}")
            continue
    
    if not all_dataframes:
        return None, "❌ 所有文件讀取失敗", "", [], 0, []
    
    # 合併所有數據
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # 移除完全重複的行
    df_combined = df_combined.drop_duplicates()
    
    # 統計年份
    years_list = []
    if 'Year' in df_combined.columns:
        df_combined = df_combined[df_combined['Year'].notna()]
        years_list = sorted(df_combined['Year'].astype(int).unique().tolist())
    
    # 生成檔案
    profile = generate_deep_profile(df_combined)
    all_cols = df_combined.columns.tolist()
    
    return df_combined, "✅ 數據已載入", profile, years_list, len(files), all_cols

# ==========================================
# 4. 超級 Agent 配置 (修復 f-string 格式化錯誤)
# ==========================================
def get_super_agent(df, df_profile, api_key):
    """打造超越 Gemini 的智能助理"""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        temperature=0.1,
        google_api_key=api_key
    )
    
    # 使用字符串拼接代替 f-string 避免格式化錯誤
    prefix_prompt = """
你是一位世界頂級的商業智能分析師，具備以下核心能力:

## 你的使命
提供精準、可操作、有洞察力的數據分析，幫助企業做出更好的決策。

## 當前數據概況
""" + str(df_profile) + """

## 絕對規則 (CRITICAL RULES)

### 禁止使用變數佔位符
- 錯誤示例: "業績為 [sales_amount]" 或 "客戶為 {{customer_name}}"
- 正確示例: "業績為 NT$ 1,500,000" 或 "客戶為 ABC公司"

### 最終回答禁止程式碼
- Final Answer 必須是純文字報告
- 不得包含任何 Python 語法、變數、或程式碼區塊
- 使用粗體強調重點，不要用 code 格式

### 數據精確度
- 所有數字必須來自實際計算結果
- 四捨五入到適當位數 (金額到元，百分比到小數點後1位)
- 必須標註貨幣單位 (NT$) 和數量單位

### 圖表生成規範
當需要視覺化時:
- 使用 Plotly 生成互動式圖表
- 圖表必須包含清晰的標題、軸標籤、圖例
- 顏色選擇要專業且易於區分
- 自動儲存圖表到 st.session_state.generated_chart

## 分析方法論

### 商業分析框架
1. 趨勢分析: 識別時間序列中的模式和異常
2. 比較分析: 跨類別、時期、區域的對比
3. 佔比分析: 計算貢獻度和市場份額
4. 排名分析: Top/Bottom N 的識別
5. 關聯分析: 找出變數間的相關性

### 特殊分析情境
- 瞎忙型客戶: 下單次數 > 10 且 AOV < 平均值
- 業績衰退: 將年度分為 H1 (1-6月) 和 H2 (7-12月)，比較增長率
- 產品組合: 分析SKU貢獻度和長尾效應
- 客戶生命週期: 新客/舊客/流失客分析

## 輸出格式標準

### 結構化報告範本

主題標題

核心發現:
[用 2-3 句話總結最重要的洞察]

詳細數據:
- 指標1: 數值 (增長/下降 X%)
- 指標2: 數值 (說明)
[最多列出 5 個關鍵指標]

戰略建議:
1. 短期行動: [具體可執行的建議]
2. 中期優化: [改進方向]
3. 長期規劃: [策略思考]

風險提示:
[如有需要，指出潛在問題]

## 圖表生成範例

當用戶要求圖表時，使用 Plotly 創建專業圖表並存儲到 st.session_state.generated_chart

## 智能推理流程
1. 理解意圖: 準確解讀用戶問題的核心需求
2. 數據探索: 檢查相關欄位和數據質量
3. 執行分析: 使用 Pandas 進行計算
4. 驗證結果: 確保數字邏輯正確
5. 生成洞察: 提供商業價值的解讀
6. 格式化輸出: 按照標準範本呈現

## 性能優化
- 對大型數據集使用向量化操作
- 避免不必要的重複計算
- 合理使用 groupby 和 pivot_table
- 當數據量 > 100,000 筆時先採樣分析

現在，請開始你的分析任務。記住: 你的目標是提供比 Gemini 更準確、更有洞察力、更實用的分析結果!
"""
    
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=80,
        max_execution_time=900,
        agent_executor_kwargs={
            "handle_parsing_errors": True
        },
        prefix=prefix_prompt,
        number_of_head_rows=10
    )

# ==========================================
# 5. 主介面
# ==========================================
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<br><br><div class='main-title'>🤖 頂級 AI 智能助理</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#666; font-size:1.1rem; margin-bottom:30px;'>超越 Gemini 的企業級數據分析系統</p>", unsafe_allow_html=True)
        
        key = st.text_input("🔑 Google API Key", type="password", placeholder="請輸入您的 API Key...")
        
        if st.button("🚀 啟動智能系統", use_container_width=True):
            if validate_login(key):
                st.session_state.api_key = key
                st.session_state.logged_in = True
                st.rerun()
            else: 
                st.error("❌ 無效的 API Key，請檢查格式")

else:
    # 側邊欄
    with st.sidebar:
        st.markdown("### 🎛️ 控制面板")
        
        if st.button("➕ 開啟新對話", use_container_width=True): 
            create_new_session()
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔄 重新載入數據", use_container_width=True):
            st.cache_data.clear()
            st.session_state.df = None
            st.rerun()
        
        st.divider()
        
        # 數據載入
        if st.session_state.df is None:
            with st.spinner("🔍 正在掃描並載入所有數據文件..."):
                result = load_data()
                st.session_state.df = result[0]
                st.session_state.status_msg = result[1]
                st.session_state.df_profile = result[2]
                st.session_state.data_years = result[3]
                st.session_state.file_count = result[4]
                st.session_state.all_columns = result[5]
                
                if st.session_state.df is None:
                    st.error("❌ 無法載入數據，請檢查 data 資料夾")
                    st.stop()
        
        # 數據儀表板
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-desc">數據總筆數</div>
            <div class="metric-num">{len(st.session_state.df):,}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-desc">數據欄位數</div>
            <div class="metric-num">{len(st.session_state.all_columns)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"📡 狀態: {st.session_state.status_msg}")
        st.caption(f"📁 已載入 {st.session_state.file_count} 個文件")
        
        if st.session_state.data_years:
            st.caption(f"📅 年份範圍: {min(st.session_state.data_years)} - {max(st.session_state.data_years)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📋 查看數據欄位"):
            for i, col in enumerate(st.session_state.all_columns, 1):
                st.caption(f"{i}. {col}")
        
        st.divider()
        
        if st.button("🚪 系統登出", key="logout"): 
            st.session_state.logged_in = False
            st.rerun()

    # 主要對話區
    current_id = st.session_state.current_session_id
    current_messages = st.session_state.chat_sessions[current_id]['messages']

    st.markdown("<div class='main-title'>🤖 頂級 AI 智能助理</div>", unsafe_allow_html=True)

    # 顯示對話歷史
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)

    # 用戶輸入
    if user_input := st.chat_input("💬 請輸入您的問題或分析需求..."):
        # 更新對話標題
        if len(current_messages) == 1:
            title = (user_input[:15] + "...") if len(user_input) > 15 else user_input
            st.session_state.chat_sessions[current_id]['title'] = title
        
        # 顯示用戶訊息
        st.chat_message("user").markdown(user_input)
        current_messages.append({"role": "user", "content": user_input, "chart": None})
        
        # 創建 Agent
        agent = get_super_agent(
            st.session_state.df, 
            st.session_state.df_profile, 
            st.session_state.api_key
        )
        
        # 構建帶有上下文的查詢
        context_messages = current_messages[-8:]
        conversation_history = "\n".join([
            f"{'用戶' if m['role'] == 'user' else 'AI'}: {m['content'][:100]}" 
            for m in context_messages
        ])
        
        full_query = f"""
對話歷史:
{conversation_history}

當前問題:
{user_input}

任務要求:
請根據上述對話歷史和當前問題，提供精準、專業的分析結果。
記住要遵循所有輸出格式規範，並確保數據準確無誤。
"""

        # AI 分析
        with st.chat_message("assistant"):
            st.session_state.generated_chart = None
            
            with st.status("🧠 AI 正在進行深度分析...", expanded=True) as status:
                try:
                    st_cb = StreamlitCallbackHandler(
                        st.container(), 
                        expand_new_thoughts=True,
                        collapse_completed_thoughts=True
                    )
                    
                    response = agent.run(full_query, callbacks=[st_cb])
                    
                    status.update(
                        label="✅ 分析完成", 
                        state="complete", 
                        expanded=False
                    )
                    
                except Exception as e:
                    status.update(
                        label="⚠️ 分析過程中遇到問題", 
                        state="error"
                    )
                    st.error(f"錯誤詳情: {str(e)}")
                    response = f"抱歉，分析過程中遇到技術問題。\n\n錯誤信息: {str(e)}\n\n請嘗試:\n1. 重新表述您的問題\n2. 確認數據欄位名稱是否正確\n3. 簡化查詢條件"
            
            # 顯示回應
            if 'response' in locals():
                st.markdown(response)
                
                # 顯示圖表
                chart_obj = st.session_state.generated_chart
                if chart_obj is not None:
                    st.plotly_chart(chart_obj, use_container_width=True)
                
                # 保存到對話歷史
                current_messages.append({
                    "role": "assistant", 
                    "content": response, 
                    "chart": chart_obj
                })
