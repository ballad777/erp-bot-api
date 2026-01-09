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
from sqlalchemy import create_engine, text, inspect
import pandas as pd
import httpx

# ❌ 移除 Matplotlib (不再需要繪圖)

from google import genai
from google.genai import types

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart ERP Bot", version="Text_Analysis_Only")

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

# Google Drive Excel 連結
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
# 移除 IMG_STORE (不再需要存圖片)

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
    """下載 Google Drive 檔案"""
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
    
    sales_file = "sales_data.xlsx"
    purchase_file = "purchase_data.xlsx"
    has_sales = False
    has_purchase = False
    
    # 下載 Sales
    if SALES_EXCEL_URL:
        if download_file_from_google_drive(get_drive_id(SALES_EXCEL_URL), sales_file):
            has_sales = True
    elif os.path.exists(sales_file): has_sales = True

    # 下載 Purchase
    if PURCHASE_EXCEL_URL:
        if download_file_from_google_drive(get_drive_id(PURCHASE_EXCEL_URL), purchase_file):
            has_purchase = True
    elif os.path.exists(purchase_file): has_purchase = True
            
    try:
        # 處理 Sales
        if has_sales:
            logger.info(f"正在讀取銷售 Excel: {sales_file}")
            xls = pd.read_excel(sales_file, sheet_name=None)
            all_sales = []
            for sheet_name, df in xls.items():
                df.columns = df.columns.str.strip() # 去除欄位空白
                # 檢查關鍵欄位
                if '日期(轉換)' in df.columns and '進銷明細未稅金額' in df.columns:
                    clean_df = pd.DataFrame({
                        'date': pd.to_datetime(df['日期(轉換)'], errors='coerce'),
                        'customer': df['客戶供應商簡稱'],
                        'product': df['品名'],
                        'quantity': pd.to_numeric(df['數量'], errors='coerce').fillna(0),
                        'amount': pd.to_numeric(df['進銷明細未稅金額'], errors='coerce').fillna(0)
                    })
                    clean_df = clean_df.dropna(subset=['date'])
                    clean_df['year'] = clean_df['date'].dt.year
                    clean_df['date'] = clean_df['date'].dt.strftime('%Y-%m-%d')
                    all_sales.append(clean_df)
            
            if all_sales:
                final_sales = pd.concat(all_sales, ignore_index=True)
                final_sales.to_sql('sales', engine, if_exists='replace', index=False)
                logger.info(f"✅ Sales 資料匯入完成，共 {len(final_sales)} 筆")

        # 處理 Purchase
        if has_purchase:
            logger.info(f"正在讀取採購 Excel: {purchase_file}")
            xls = pd.read_excel(purchase_file, sheet_name=None)
            all_purchase = []
            for sheet_name, df in xls.items():
                df.columns = df.columns.str.strip()
                if '日期(轉換)' in df.columns and '進銷明細未稅金額' in df.columns:
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

    except Exception as e:
        logger.error(f"❌ 資料匯入嚴重錯誤: {str(e)}")

# =========================
# 工具函數
# =========================
def execute_sql_query(sql: str) -> str:
    """【工具】執行 SQL SELECT 查詢 sales 或 purchase 表。"""
    logger.info(f"執行 SQL: {sql}")
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    if not sql.lower().startswith("select"): return "錯誤：只允許 SELECT 查詢。"
    if any(k in sql.lower() for k in ['drop', 'delete', 'update', 'insert', 'alter']):
        return "錯誤：禁止修改資料庫。"
    
    try:
        insp = inspect(engine)
        table_names = insp.get_table_names()
        
        if 'sales' in sql.lower() and 'sales' not in table_names:
            return "系統錯誤：銷售資料表 (sales) 尚未建立，請確認資料是否已匯入。"
        if 'purchase' in sql.lower() and 'purchase' not in table_names:
            return "系統錯誤：採購資料表 (purchase) 尚未建立。"

        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
            if df.empty: return "查無資料。"
            
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].astype(str)
            
            # 限制回傳筆數，避免 JSON 過大
            if len(df) > 50:
                logger.info(f"結果過多 ({len(df)})，僅回傳前 50 筆")
                df = df.head(50)
                
            return df.to_json(orient="records", force_ascii=False, date_format='iso')
    except Exception as e:
        return f"SQL Error: {str(e)}"

def get_database_schema() -> str:
    """【工具】取得資料表結構"""
    try:
        insp = inspect(engine)
        table_names = insp.get_table_names()
        summary = {}
        with engine.connect() as conn:
            for t_name in table_names:
                if t_name not in ['sales', 'purchase']: continue
                cols = conn.execute(text(f"SELECT * FROM {t_name} LIMIT 1")).keys()
                count = conn.execute(text(f"SELECT COUNT(*) FROM {t_name}")).scalar()
                summary[t_name] = {'columns': list(cols), 'count': count}
        return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

# ❌ 移除 create_chart 工具
tools_list = [execute_sql_query, get_database_schema]

# =========================
# 系統提示詞 (極簡風格調教)
# =========================
SYSTEM_PROMPT = """你是一個專業、俐落的 ERP 商業分析師。
請根據資料庫中的 `sales` (銷售) 與 `purchase` (採購) 資料表回答問題。

## ⚠️ 回答風格規範 (Violations will be punished)
1. **嚴禁使用 Markdown 格式**：
   - 絕對不要使用米字號 `*` 或 `**`。
   - 絕對不要使用井字號 `#` 做標題。
   - 請使用純文字，用換行或連字號 `-` 來條列重點。
   
2. **專注文字分析**：
   - 用戶**不需要圖表**。
   - 請消化數據後，用文字提供「洞察 (Insights)」。
   - 例如：不要只列出數字，要告訴用戶「跟去年比成長了多少」或「哪個客戶佔比最高」。

3. **回答精簡扼要**：
   - 除非用戶要求「詳細清單」，否則預設只給總結數據。
   - 不要把 JSON 資料直接貼出來。

4. **專注當下**：
   - 只回答用戶最新一次輸入的問題，忽略無關的歷史對話。

5. **模糊搜尋**：
   - 用戶打錯字或打簡稱（如 "ipone", "華碩"），請自動用 `LIKE` 修正查詢。

## 資料表結構
- `sales` (銷售): date, customer, product, quantity, amount, year
- `purchase` (採購): date, supplier, product, quantity, amount, year
"""

# =========================
# Agent 處理邏輯
# =========================
async def agent_process(user_id: str, text: str, base_url: str):
    if not client: return {"text": "API Key 未設定"}
    
    # 只取最近 2 輪對話，保持對話乾淨
    history = CHAT_MEMORY.get(user_id, [])[-2:] 
    
    user_message = types.Content(role="user", parts=[types.Part(text=text)])
    contents = history + [user_message]
    
    config = types.GenerateContentConfig(
        tools=tools_list,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2 # 低溫，讓回答更收斂
    )
    
    final_text = "抱歉，無法處理。"
    
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=contents,
            config=config
        )
        
        if response.candidates:
            candidate = response.candidates[0]
            for _ in range(5): 
                has_tool = False
                for part in candidate.content.parts:
                    if part.function_call:
                        has_tool = True
                        fc = part.function_call
                        logger.info(f"Tool Call: {fc.name}")
                        
                        res = ""
                        if fc.name == "execute_sql_query":
                            res = execute_sql_query(fc.args.get("sql", ""))
                        elif fc.name == "get_database_schema":
                            res = get_database_schema()
                        
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
                        
                        response = client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=contents,
                            config=config
                        )
                        candidate = response.candidates[0]
                        break 
                
                if not has_tool:
                    final_text = response.text
                    break

        # 更新記憶
        CHAT_MEMORY[user_id] = contents[-4:]
        
        # 再次過濾米字號 (雙重保險)
        final_text = final_text.replace("*", "").replace("#", "")
        
        return {"text": final_text, "image": None}
        
    except Exception as e:
        logger.error(f"Agent Error: {e}")
        return {"text": f"發生錯誤: {str(e)}"}

# =========================
# API 端點
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "ERP Bot (Text Only)"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/line/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    
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
        await reply_line(reply_token, result.get("text"), None)
    except Exception as e:
        logger.error(f"Handle Error: {e}")
        await reply_line(reply_token, "系統忙碌中", None)

async def reply_line(token: str, text: Optional[str], img_url: Optional[str]):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    messages = []
    # 這裡已經不需要 img_url 了，但為了相容性保留參數
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