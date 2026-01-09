import os
import re
import time
import uuid
import json
import logging
import glob
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from sqlalchemy import create_engine, text
import pandas as pd
import httpx

import matplotlib
# 設定 Matplotlib 後端為 Agg (防止伺服器繪圖錯誤)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO

from google import genai
from google.genai import types

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart ERP Bot", version="Final_Drive_Integrated")

# =========================
# 資料庫連線
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./erp.db"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# =========================
# LINE & Gemini 設定
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Google Drive Excel 連結 (從環境變數讀取)
SALES_EXCEL_URL = os.getenv("SALES_EXCEL_URL", "")
PURCHASE_EXCEL_URL = os.getenv("PURCHASE_EXCEL_URL", "")

if not LINE_CHANNEL_ACCESS_TOKEN:
    logger.warning("⚠️ LINE_CHANNEL_ACCESS_TOKEN 未設定")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY 未設定")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# 記憶體存儲
# =========================
CHAT_MEMORY: Dict[str, List[Any]] = {} 
IMG_STORE: Dict[str, Dict[str, Any]] = {}

# =========================
# 📥 Google Drive 下載與資料匯入邏輯
# =========================
def get_drive_id(url: str) -> str:
    """從 Google Drive 連結提取 File ID"""
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/spreadsheets/d/([a-zA-Z0-9_-]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return ""

def download_file_from_google_drive(id: str, destination: str):
    """下載 Google Drive 檔案 (支援大檔案確認)"""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    
    logger.info(f"正在下載檔案 ID: {id} 到 {destination}...")
    try:
        response = session.get(URL, params={'id': id}, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            
        if response.status_code == 200:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
            logger.info(f"✅ 下載成功: {destination}")
            return True
        else:
            logger.error(f"❌ 下載失敗，狀態碼: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 下載發生錯誤: {str(e)}")
        return False

def import_data_to_db():
    """下載並匯入資料到資料庫"""
    logger.info("🔄 開始執行資料初始化程序...")
    
    # 1. 下載檔案
    sales_file = "sales_data.xlsx"
    purchase_file = "purchase_data.xlsx"
    
    has_sales = False
    has_purchase = False
    
    # 處理銷售檔案
    if SALES_EXCEL_URL:
        file_id = get_drive_id(SALES_EXCEL_URL)
        if file_id and download_file_from_google_drive(file_id, sales_file):
            has_sales = True
    else:
        # 如果沒設定 URL，檢查是否有本地檔案
        if os.path.exists(sales_file): has_sales = True
        elif glob.glob("sales*.xlsx"): 
            sales_file = glob.glob("sales*.xlsx")[0]
            has_sales = True

    # 處理採購檔案
    if PURCHASE_EXCEL_URL:
        file_id = get_drive_id(PURCHASE_EXCEL_URL)
        if file_id and download_file_from_google_drive(file_id, purchase_file):
            has_purchase = True
    else:
        if os.path.exists(purchase_file): has_purchase = True
        elif glob.glob("purchase*.xlsx"): 
            purchase_file = glob.glob("purchase*.xlsx")[0]
            has_purchase = True
            
    # 2. 讀取並匯入資料庫
    try:
        # --- 匯入 Sales ---
        if has_sales:
            logger.info(f"正在讀取銷售 Excel: {sales_file}")
            # 讀取所有 sheet
            xls = pd.read_excel(sales_file, sheet_name=None)
            all_sales = []
            
            for sheet_name, df in xls.items():
                logger.info(f"  - 處理分頁: {sheet_name}")
                # 檢查必要欄位 (根據您提供的 CSV 欄位名稱)
                # 欄位: 日期(轉換), 客戶供應商簡稱, 品名, 數量, 進銷明細未稅金額
                if '日期(轉換)' in df.columns and '進銷明細未稅金額' in df.columns:
                    clean_df = pd.DataFrame({
                        'date': pd.to_datetime(df['日期(轉換)'], errors='coerce'),
                        'customer': df['客戶供應商簡稱'],
                        'product': df['品名'], # 銷售檔通常叫 '品名'
                        'quantity': pd.to_numeric(df['數量'], errors='coerce').fillna(0),
                        'amount': pd.to_numeric(df['進銷明細未稅金額'], errors='coerce').fillna(0)
                    })
                    # 移除日期無效的資料
                    clean_df = clean_df.dropna(subset=['date'])
                    clean_df['year'] = clean_df['date'].dt.year
                    clean_df['date'] = clean_df['date'].dt.strftime('%Y-%m-%d')
                    all_sales.append(clean_df)
            
            if all_sales:
                final_sales = pd.concat(all_sales, ignore_index=True)
                final_sales.to_sql('sales', engine, if_exists='replace', index=False)
                logger.info(f"✅ Sales 資料匯入完成，共 {len(final_sales)} 筆")
            else:
                logger.warning("⚠️ Sales Excel 中找不到符合格式的分頁")
        else:
            logger.warning("⚠️ 無法找到或下載 Sales 檔案")

        # --- 匯入 Purchase ---
        if has_purchase:
            logger.info(f"正在讀取採購 Excel: {purchase_file}")
            xls = pd.read_excel(purchase_file, sheet_name=None)
            all_purchase = []
            
            for sheet_name, df in xls.items():
                logger.info(f"  - 處理分頁: {sheet_name}")
                # 欄位: 日期(轉換), 客戶供應商簡稱, 對方品名/品名備註, 數量, 進銷明細未稅金額
                if '日期(轉換)' in df.columns and '進銷明細未稅金額' in df.columns:
                    # 採購檔的品名欄位可能不同，嘗試找 '對方品名/品名備註'
                    prod_col = '對方品名/品名備註' if '對方品名/品名備註' in df.columns else '品名'
                    
                    clean_df = pd.DataFrame({
                        'date': pd.to_datetime(df['日期(轉換)'], errors='coerce'),
                        'supplier': df['客戶供應商簡稱'],
                        'product': df[prod_col],
                        'quantity': pd.to_numeric(df['數量'], errors='coerce').fillna(0),
                        'amount': pd.to_numeric(df['進銷明細未稅金額'], errors='coerce').fillna(0)
                    })
                    clean_df = clean_df.dropna(subset=['date'])
                    clean_df['year'] = clean_df['date'].dt.year
                    clean_df['date'] = clean_df['date'].dt.strftime('%Y-%m-%d')
                    all_purchase.append(clean_df)
            
            if all_purchase:
                final_purchase = pd.concat(all_purchase, ignore_index=True)
                final_purchase.to_sql('purchase', engine, if_exists='replace', index=False)
                logger.info(f"✅ Purchase 資料匯入完成，共 {len(final_purchase)} 筆")
            else:
                logger.warning("⚠️ Purchase Excel 中找不到符合格式的分頁")
        else:
            logger.warning("⚠️ 無法找到或下載 Purchase 檔案")

    except Exception as e:
        logger.error(f"❌ 資料匯入嚴重錯誤: {str(e)}")

# =========================
# 工具函數
# =========================
def execute_sql_query(sql: str) -> str:
    """【工具】執行 SQL SELECT 查詢 sales 或 purchase 表。"""
    logger.info(f"執行 SQL: {sql}")
    
    # 清洗 SQL
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql_lower = sql.lower()
    if not sql_lower.startswith("select"):
        return "錯誤：只允許 SELECT 查詢。"
    
    # 檢查是否嘗試修改資料
    if any(k in sql_lower for k in ['drop', 'delete', 'update', 'insert', 'alter']):
        return "錯誤：禁止修改資料庫。"
    
    try:
        with engine.connect() as conn:
            # 檢查 Table 是否存在 (防止 'no such table' 錯誤)
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            table_names = [t[0] for t in tables]
            
            # 如果 SQL 裡提到的表不存在，回傳友善錯誤
            if 'sales' in sql_lower and 'sales' not in table_names:
                return "系統錯誤：銷售資料表 (sales) 尚未建立，請聯繫管理員檢查資料匯入狀況。"
            if 'purchase' in sql_lower and 'purchase' not in table_names:
                return "系統錯誤：採購資料表 (purchase) 尚未建立。"

            df = pd.read_sql(text(sql), conn)
            
            if df.empty: 
                return "查詢成功但沒有找到資料。請嘗試放寬條件或確認關鍵字。"
            
            # 轉字串避免 JSON 錯誤
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].astype(str)
            
            if len(df) > 100:
                logger.info(f"結果過多 ({len(df)})，僅回傳前 100 筆")
                df = df.head(100)
                
            return df.to_json(orient="records", force_ascii=False, date_format='iso')
    except Exception as e:
        logger.error(f"SQL 執行錯誤: {str(e)}")
        return f"SQL Error: {str(e)}"

def create_chart(title: str, chart_type: str, data_json: str, x_key: str, y_key: str) -> str:
    """【工具】繪製圖表 (bar/line/pie)。"""
    logger.info(f"繪製圖表: {title}")
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        if df.empty: return "無資料繪圖"
        
        if x_key not in df.columns or y_key not in df.columns:
            return f"欄位錯誤: {x_key} 或 {y_key} 不存在"
        
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)
        
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        if chart_type == "line": plt.plot(df[x_key], df[y_key], marker='o')
        elif chart_type == "bar": plt.bar(df[x_key], df[y_key], color='steelblue')
        elif chart_type == "pie":
            df_s = df.sort_values(by=y_key, ascending=False).head(8)
            plt.pie(df_s[y_key], labels=df_s[x_key], autopct='%1.1f%%')
            
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        
        img_id = str(uuid.uuid4())
        IMG_STORE[img_id] = {"bytes": buf.getvalue(), "ts": time.time()}
        return f"IMAGE_ID:{img_id}"
    except Exception as e:
        return f"Chart Error: {str(e)}"

def get_database_schema() -> str:
    """【工具】取得資料表結構"""
    try:
        with engine.connect() as conn:
            tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            summary = {}
            for t in tables:
                t_name = t[0]
                cols = conn.execute(text(f"SELECT * FROM {t_name} LIMIT 1")).keys()
                count = conn.execute(text(f"SELECT COUNT(*) FROM {t_name}")).scalar()
                summary[t_name] = {'columns': list(cols), 'count': count}
            return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# 工具列表
# =========================
tools_list = [execute_sql_query, create_chart, get_database_schema]

# =========================
# 系統提示詞
# =========================
SYSTEM_PROMPT = """你是一個專業的 ERP 數據助理。
請根據資料庫中的 `sales` (銷售) 與 `purchase` (採購) 資料表回答問題。

## 重要指令
1. **直接回答**：不要自我介紹，不要說「我是小智」，直接針對問題提供數據或圖表。
2. **模糊搜尋**：用戶輸入的關鍵字可能會有錯字，請使用 `LIKE` 進行模糊比對。
   - 例如：用戶查 "ipone" -> SQL 用 `product LIKE '%iPhone%'`
   - 例如：用戶查 "華碩" -> SQL 用 `customer LIKE '%華碩%'`
3. **資料表結構**：
   - sales: date, customer, product, quantity, amount, year
   - purchase: date, supplier, product, quantity, amount, year

## SQL 規則
- 查詢總額使用 `SUM(amount)`
- 查詢銷量使用 `SUM(quantity)`
- 若查無資料，請嘗試放寬條件 (例如移除年份限制或簡化關鍵字)
"""

# =========================
# Agent 處理邏輯
# =========================
async def agent_process(user_id: str, text: str, base_url: str):
    if not client: return {"text": "API Key 未設定"}
    
    history = CHAT_MEMORY.get(user_id, [])
    user_message = types.Content(role="user", parts=[types.Part(text=text)])
    contents = history + [user_message]
    
    config = types.GenerateContentConfig(
        tools=tools_list,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3
    )
    
    final_text = "抱歉，無法處理。"
    image_url = None
    
    try:
        # 使用 gemini-flash-latest (對應 1.5 Flash)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=contents,
            config=config
        )
        
        # 處理 Function Call
        if response.candidates:
            candidate = response.candidates[0]
            # 迴圈處理工具呼叫 (支援多輪)
            for _ in range(5): # 最多 5 輪
                has_tool = False
                for part in candidate.content.parts:
                    if part.function_call:
                        has_tool = True
                        fc = part.function_call
                        logger.info(f"Tool Call: {fc.name}")
                        
                        res = ""
                        if fc.name == "execute_sql_query":
                            res = execute_sql_query(fc.args.get("sql", ""))
                        elif fc.name == "create_chart":
                            chart_res = create_chart(
                                fc.args.get("title", ""),
                                fc.args.get("chart_type", "bar"),
                                fc.args.get("data_json", "[]"),
                                fc.args.get("x_key", ""),
                                fc.args.get("y_key", "")
                            )
                            if "IMAGE_ID" in chart_res:
                                img_id = chart_res.split(":")[1]
                                image_url = f"{base_url}/img/{img_id}"
                                res = "圖表已生成"
                            else: res = chart_res
                        elif fc.name == "get_database_schema":
                            res = get_database_schema()
                        
                        # 回傳工具結果
                        contents.append(candidate.content)
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part(
                                function_response=types.FunctionResponse(
                                    name=fc.name,
                                    response={"result": res}
                                )
                            )]
                        ))
                        
                        # 再次呼叫模型取得文字回應
                        response = client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=contents,
                            config=config
                        )
                        candidate = response.candidates[0]
                        break # 跳出 parts 迴圈，處理新的 response
                
                if not has_tool:
                    final_text = response.text
                    break

        CHAT_MEMORY[user_id] = contents[-20:]
        return {"text": final_text, "image": image_url}
        
    except Exception as e:
        logger.error(f"Agent Error: {e}")
        return {"text": f"發生錯誤: {str(e)}"}

# =========================
# API 端點
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "ERP Bot"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/img/{img_id}")
def get_img(img_id: str):
    if img_id not in IMG_STORE: raise HTTPException(404, "Not Found")
    return Response(content=IMG_STORE[img_id]["bytes"], media_type="image/png")

@app.post("/line/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    
    # 驗證簽名 (若有設定 SECRET)
    if LINE_CHANNEL_SECRET:
        import hmac, hashlib, base64
        hash_val = hmac.new(LINE_CHANNEL_SECRET.encode('utf-8'), body, hashlib.sha256).digest()
        expected = base64.b64encode(hash_val).decode('utf-8')
        if signature != expected: raise HTTPException(400, "Invalid Signature")

    try:
        events = json.loads(body.decode("utf-8")).get("events", [])
    except: return {"ok": False}
    
    base_url = f"https://{request.headers.get('host', 'localhost')}"
    
    for event in events:
        if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
            user_id = event["source"]["userId"]
            text = event["message"]["text"]
            reply_token = event["replyToken"]
            background_tasks.add_task(handle_message, user_id, text, reply_token, base_url)
            
    return {"ok": True}

async def handle_message(user_id: str, text: str, reply_token: str, base_url: str):
    try:
        if text.lower() in ['/reset', '清除']:
            CHAT_MEMORY.pop(user_id, None)
            await reply_line(reply_token, "記憶已清除", None)
            return

        result = await agent_process(user_id, text, base_url)
        await reply_line(reply_token, result.get("text"), result.get("image"))
    except Exception as e:
        logger.error(f"Handle Error: {e}")
        await reply_line(reply_token, "系統忙碌中", None)

async def reply_line(token: str, text: Optional[str], img_url: Optional[str]):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    messages = []
    if img_url: messages.append({"type": "image", "originalContentUrl": img_url, "previewImageUrl": img_url})
    if text: messages.append({"type": "text", "text": text[:4999]})
    if not messages: messages.append({"type": "text", "text": "..."})
    
    async with httpx.AsyncClient() as c:
        await c.post("https://api.line.me/v2/bot/message/reply", headers=headers, json={"replyToken": token, "messages": messages})

@app.on_event("startup")
async def startup():
    """啟動時自動下載並匯入資料"""
    try:
        import_data_to_db()
    except Exception as e:
        logger.error(f"Startup Error: {e}")