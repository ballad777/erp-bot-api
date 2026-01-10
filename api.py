import os
import re
import json
import time
import hmac
import base64
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO

import requests
import pandas as pd
import httpx

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from google import genai
from google.genai import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("erp_ultra_pro")

# =========================================================
# App
# =========================================================
app = FastAPI(title="ERP Bot Ultra PRO", version="3.2_Fixed")

# =========================================================
# Environment
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# 支援多種 URL 格式
SALES_SHEET_URL = os.getenv("SALES_EXCEL_URL", "")
PURCHASE_SHEET_URL = os.getenv("PURCHASE_EXCEL_URL", "")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "secret123")
AUTO_IMPORT_ON_STARTUP = os.getenv("AUTO_IMPORT_ON_STARTUP", "0") == "1"

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "40"))

# =========================================================
# Globals
# =========================================================
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

RATE_STORE: Dict[str, List[float]] = {}
CHAT_MEMORY: Dict[str, List[Any]] = {}
IMG_STORE: Dict[str, Dict[str, Any]] = {}
DB_READY = False

# =========================================================
# Time utils
# =========================================================
def now_taipei() -> datetime:
    return datetime.utcnow() + timedelta(hours=8)

# =========================================================
# Security utils
# =========================================================
def verify_line_signature(body: bytes, signature: str):
    if not LINE_CHANNEL_SECRET:
        logger.warning("⚠️ LINE_CHANNEL_SECRET 未設定，跳過簽名驗證")
        return
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    if not hmac.compare_digest(signature.strip(), expected):
        logger.error("❌ LINE 簽名驗證失敗")
        raise HTTPException(400, "Invalid Signature")

def require_admin(request: Request):
    if not ADMIN_TOKEN:
        raise HTTPException(500, "ADMIN_TOKEN not set")
    token = request.headers.get("X-Admin-Token", "")
    if not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(401, "Unauthorized")

def rate_limit_ok(user_id: str) -> bool:
    now = time.time()
    window_start = now - 60
    ts = RATE_STORE.get(user_id, [])
    ts = [t for t in ts if t >= window_start]
    if len(ts) >= RATE_LIMIT_PER_MIN:
        RATE_STORE[user_id] = ts
        return False
    ts.append(now)
    RATE_STORE[user_id] = ts
    return True

# =========================================================
# DB init
# =========================================================
def ensure_tables():
    """建立資料表和索引"""
    logger.info("🔧 檢查資料表...")
    
    with engine.begin() as conn:
        # Sales 表
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sales (
            date TEXT,
            customer TEXT,
            product TEXT,
            quantity REAL,
            amount REAL,
            year INTEGER
        );
        """))
        
        # Purchase 表
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS purchase (
            date TEXT,
            supplier TEXT,
            product TEXT,
            quantity REAL,
            amount REAL,
            year INTEGER
        );
        """))
        
        # 嘗試建立索引（忽略錯誤）
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sales_year ON sales(year);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_purchase_date ON purchase(date);"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_purchase_year ON purchase(year);"))
        except Exception as e:
            logger.warning(f"索引建立失敗（可忽略）: {e}")
    
    logger.info("✅ 資料表檢查完成")

def require_db_ready():
    global DB_READY
    if not DB_READY:
        ensure_tables()
        DB_READY = True

def table_counts() -> Dict[str, int]:
    """取得各表筆數"""
    insp = inspect(engine)
    names = set(insp.get_table_names())
    out = {"sales": 0, "purchase": 0}
    
    with engine.connect() as conn:
        for t in out.keys():
            if t in names:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                    out[t] = int(result.scalar() or 0)
                except:
                    out[t] = 0
    
    return out

# =========================================================
# Google Sheet 下載（改進版）
# =========================================================
def extract_sheet_id(url: str) -> Optional[str]:
    """從各種 Google Sheet URL 格式提取 ID"""
    patterns = [
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

def download_google_sheet_xlsx(sheet_url: str, max_retries: int = 3) -> Optional[BytesIO]:
    """
    下載 Google Sheet 為 Excel 格式
    回傳 BytesIO 物件或 None（失敗）
    """
    sheet_id = extract_sheet_id(sheet_url)
    if not sheet_id:
        logger.error(f"❌ 無法從 URL 提取 Sheet ID: {sheet_url}")
        return None
    
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
    params = {"format": "xlsx"}
    
    logger.info(f"📥 下載 Google Sheet: {sheet_id}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                export_url,
                params=params,
                timeout=60,
                allow_redirects=True
            )
            
            # 檢查狀態碼
            if response.status_code != 200:
                logger.error(f"❌ 下載失敗，狀態碼: {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            # 檢查內容類型
            content_type = response.headers.get('Content-Type', '')
            logger.info(f"Content-Type: {content_type}")
            
            # 檢查是否為 Excel 格式
            content = response.content
            
            # Excel 檔案應該以 PK 開頭（ZIP 格式標記）
            if not content.startswith(b'PK'):
                logger.error("❌ 回應不是 Excel 格式")
                logger.error(f"前 200 字元: {content[:200]}")
                
                # 如果是 HTML，可能是權限問題
                if b'<html' in content[:500].lower():
                    logger.error("❌ 收到 HTML 回應，可能是權限問題")
                    logger.error("請確認：")
                    logger.error("1. Google Sheet 已設定為「知道連結的人」可以檢視")
                    logger.error("2. URL 格式正確")
                    return None
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            logger.info(f"✅ 下載成功，大小: {len(content)} bytes")
            return BytesIO(content)
            
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ 下載超時 (嘗試 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ 下載錯誤: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    return None

# =========================================================
# ETL normalize（彈性版本）
# =========================================================
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """從候選欄位名稱中找到第一個存在的欄位"""
    df.columns = df.columns.astype(str).str.strip()
    for candidate in candidates:
        for col in df.columns:
            if candidate in col:
                return col
    return None

def normalize_sales_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """標準化銷售資料（彈性欄位匹配）"""
    logger.info(f"處理銷售資料，欄位: {list(df.columns)}")
    
    # 彈性尋找欄位
    date_col = find_column(df, ["日期(轉換)", "日期", "Date", "date", "交易日期"])
    customer_col = find_column(df, ["客戶供應商簡稱", "客戶簡稱", "客戶", "Customer", "客戶名稱"])
    product_col = find_column(df, ["品名", "產品", "Product", "品號", "產品代號"])
    quantity_col = find_column(df, ["數量", "Quantity", "銷售數量"])
    amount_col = find_column(df, ["進銷明細未稅金額", "未稅金額", "金額", "Amount", "含稅金額"])
    
    # 檢查必要欄位
    if not all([date_col, customer_col, product_col]):
        logger.warning(f"⚠️ 缺少必要欄位: date={date_col}, customer={customer_col}, product={product_col}")
        return None
    
    # 建立標準化 DataFrame
    clean = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "customer": df[customer_col].astype(str).str.strip(),
        "product": df[product_col].astype(str).str.strip(),
        "quantity": pd.to_numeric(df[quantity_col], errors="coerce").fillna(0) if quantity_col else 0,
        "amount": pd.to_numeric(df[amount_col], errors="coerce").fillna(0) if amount_col else 0,
    }).dropna(subset=["date"])
    
    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    
    logger.info(f"✅ 標準化完成，保留 {len(clean)} 筆資料")
    return clean

def normalize_purchase_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """標準化採購資料（彈性欄位匹配）"""
    logger.info(f"處理採購資料，欄位: {list(df.columns)}")
    
    # 彈性尋找欄位
    date_col = find_column(df, ["日期(轉換)", "日期", "Date", "date", "交易日期"])
    supplier_col = find_column(df, ["客戶供應商簡稱", "供應商", "Supplier", "廠商", "廠商名稱"])
    product_col = find_column(df, ["對方品名/品名備註", "品名", "產品", "Product", "品號"])
    quantity_col = find_column(df, ["數量", "Quantity", "採購數量"])
    amount_col = find_column(df, ["進銷明細未稅金額", "未稅金額", "金額", "Amount", "含稅金額"])
    
    # 檢查必要欄位
    if not all([date_col, supplier_col, product_col]):
        logger.warning(f"⚠️ 缺少必要欄位: date={date_col}, supplier={supplier_col}, product={product_col}")
        return None
    
    # 建立標準化 DataFrame
    clean = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "supplier": df[supplier_col].astype(str).str.strip(),
        "product": df[product_col].astype(str).str.strip(),
        "quantity": pd.to_numeric(df[quantity_col], errors="coerce").fillna(0) if quantity_col else 0,
        "amount": pd.to_numeric(df[amount_col], errors="coerce").fillna(0) if amount_col else 0,
    }).dropna(subset=["date"])
    
    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    
    logger.info(f"✅ 標準化完成，保留 {len(clean)} 筆資料")
    return clean

# =========================================================
# Upsert（插入或忽略）
# =========================================================
def bulk_insert(table: str, df: pd.DataFrame) -> int:
    """批次插入資料"""
    if df.empty:
        return 0
    
    try:
        # 使用 pandas 直接寫入
        rows_before = 0
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            rows_before = result.scalar() or 0
        
        # 寫入資料（append 模式）
        df.to_sql(table, engine, if_exists='append', index=False)
        
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            rows_after = result.scalar() or 0
        
        inserted = rows_after - rows_before
        logger.info(f"✅ {table} 寫入完成: {inserted} 筆新資料")
        return inserted
        
    except Exception as e:
        logger.error(f"❌ {table} 寫入失敗: {str(e)}")
        return 0

def import_data_to_db() -> Dict[str, Any]:
    """從 Google Sheet 匯入資料"""
    require_db_ready()
    
    logger.info("🔄 開始資料匯入...")
    t0 = time.time()
    before = table_counts()
    msgs = []
    inserted = {"sales": 0, "purchase": 0}
    
    # 匯入銷售資料
    if SALES_SHEET_URL:
        logger.info("📊 處理銷售資料...")
        excel_bytes = download_google_sheet_xlsx(SALES_SHEET_URL)
        
        if excel_bytes:
            try:
                xls = pd.read_excel(excel_bytes, sheet_name=None)
                logger.info(f"找到 {len(xls)} 個工作表")
                
                dfs = []
                for sheet_name, df in xls.items():
                    logger.info(f"處理工作表: {sheet_name}")
                    normalized = normalize_sales_df(df)
                    if normalized is not None and len(normalized) > 0:
                        dfs.append(normalized)
                
                if dfs:
                    final = pd.concat(dfs, ignore_index=True)
                    # 移除重複資料
                    final = final.drop_duplicates(subset=['date', 'customer', 'product', 'amount'], keep='first')
                    inserted["sales"] = bulk_insert("sales", final)
                    msgs.append(f"✅ sales: 處理 {len(final)} 筆，新增 {inserted['sales']} 筆")
                else:
                    msgs.append("⚠️ sales: 沒有符合格式的工作表")
            except Exception as e:
                logger.error(f"❌ sales 處理錯誤: {str(e)}", exc_info=True)
                msgs.append(f"❌ sales: 處理失敗 - {str(e)}")
        else:
            msgs.append("❌ sales: 下載失敗")
    else:
        msgs.append("ℹ️ sales: 未設定 SALES_EXCEL_URL")
    
    # 匯入採購資料
    if PURCHASE_SHEET_URL:
        logger.info("📦 處理採購資料...")
        excel_bytes = download_google_sheet_xlsx(PURCHASE_SHEET_URL)
        
        if excel_bytes:
            try:
                xls = pd.read_excel(excel_bytes, sheet_name=None)
                logger.info(f"找到 {len(xls)} 個工作表")
                
                dfs = []
                for sheet_name, df in xls.items():
                    logger.info(f"處理工作表: {sheet_name}")
                    normalized = normalize_purchase_df(df)
                    if normalized is not None and len(normalized) > 0:
                        dfs.append(normalized)
                
                if dfs:
                    final = pd.concat(dfs, ignore_index=True)
                    # 移除重複資料
                    final = final.drop_duplicates(subset=['date', 'supplier', 'product', 'amount'], keep='first')
                    inserted["purchase"] = bulk_insert("purchase", final)
                    msgs.append(f"✅ purchase: 處理 {len(final)} 筆，新增 {inserted['purchase']} 筆")
                else:
                    msgs.append("⚠️ purchase: 沒有符合格式的工作表")
            except Exception as e:
                logger.error(f"❌ purchase 處理錯誤: {str(e)}", exc_info=True)
                msgs.append(f"❌ purchase: 處理失敗 - {str(e)}")
        else:
            msgs.append("❌ purchase: 下載失敗")
    else:
        msgs.append("ℹ️ purchase: 未設定 PURCHASE_EXCEL_URL")
    
    after = table_counts()
    cost = time.time() - t0
    
    result = {
        "ok": True,
        "before": before,
        "after": after,
        "inserted": inserted,
        "seconds": round(cost, 2),
        "messages": msgs
    }
    
    logger.info(f"✨ 資料匯入完成: {result}")
    return result

# =========================================================
# AI 查詢處理（Gemini Function Calling）
# =========================================================
def execute_sql_query(sql: str) -> str:
    """執行 SQL 查詢"""
    logger.info(f"🔍 執行 SQL: {sql[:200]}")
    
    if not sql.strip().lower().startswith("select"):
        return "錯誤：只允許 SELECT 查詢"
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
            if df.empty:
                return "查詢成功但無資料"
            
            # 限制筆數
            if len(df) > 50:
                df = df.head(50)
            
            return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        logger.error(f"❌ SQL 錯誤: {str(e)}")
        return f"SQL 錯誤: {str(e)}"

def create_chart(title: str, chart_type: str, data_json: str, x_key: str, y_key: str) -> str:
    """生成圖表"""
    logger.info(f"📊 繪製圖表: {title} ({chart_type})")
    
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        if df.empty:
            return "無資料繪圖"
        
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)
        
        if chart_type == "line":
            plt.plot(df[x_key], df[y_key], marker='o', linewidth=2)
        elif chart_type == "bar":
            plt.bar(df[x_key], df[y_key])
            plt.xticks(rotation=45, ha='right')
        elif chart_type == "pie":
            plt.pie(df[y_key], labels=df[x_key], autopct='%1.1f%%')
        
        plt.title(title)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        
        img_id = f"chart_{int(time.time())}_{hash(title) % 10000}"
        IMG_STORE[img_id] = {"bytes": buf.getvalue(), "ts": time.time()}
        
        logger.info(f"✅ 圖表生成: {img_id}")
        return f"IMAGE_ID:{img_id}"
        
    except Exception as e:
        logger.error(f"❌ 圖表錯誤: {str(e)}")
        return f"圖表錯誤: {str(e)}"

# Gemini 工具定義
tools_list = [execute_sql_query, create_chart]
google_search = {"google_search": {}}

# 系統提示詞
SYSTEM_PROMPT = """你是專業的 ERP 資料分析助理「小智」。

## 📊 可用資料表
- **sales** (銷售): date, customer, product, quantity, amount, year
- **purchase** (採購): date, supplier, product, quantity, amount, year

## 🎯 你的任務
1. 理解用戶問題
2. 使用 execute_sql_query 查詢資料
3. 用 create_chart 視覺化（需要時）
4. 提供專業分析和建議

## 💡 查詢範例
```sql
-- 2024年總銷售額
SELECT SUM(amount) as total FROM sales WHERE year = 2024;

-- 前10大客戶
SELECT customer, SUM(amount) as total 
FROM sales 
GROUP BY customer 
ORDER BY total DESC 
LIMIT 10;
```

## 📈 繪圖流程
1. 先用 SQL 查詢資料
2. 將查詢結果的 JSON 傳給 create_chart
3. chart_type: "line"(趨勢), "bar"(比較), "pie"(佔比)

## 原則
- 主動提供洞察，不只回答問題
- 數據說話，用實際數字支持觀點
- 建議視覺化時主動繪圖
- 用繁體中文，友善專業"""

async def agent_process(user_id: str, text: str, base_url: str) -> Dict[str, Any]:
    """AI Agent 處理用戶訊息"""
    if not client:
        return {"text": "❌ Gemini API 未設定"}
    
    logger.info(f"🤖 處理訊息: {text}")
    
    history = CHAT_MEMORY.get(user_id, [])
    
    try:
        # 構建對話
        contents = history + [text]
        
        config = types.GenerateContentConfig(
            tools=tools_list + [google_search],
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7
        )
        
        final_text = ""
        image_url = None
        max_turns = 5
        
        for turn in range(max_turns):
            logger.info(f"🔄 第 {turn + 1} 輪處理")
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
            
            if not response.candidates:
                final_text = "抱歉，無法處理此請求"
                break
            
            candidate = response.candidates[0]
            content = candidate.content
            
            # 檢查工具調用
            has_tool = any(hasattr(p, 'function_call') for p in content.parts)
            
            if has_tool:
                # 處理工具調用
                tool_responses = []
                
                for part in content.parts:
                    if not hasattr(part, 'function_call'):
                        continue
                    
                    fc = part.function_call
                    logger.info(f"🔧 調用工具: {fc.name}")
                    
                    if fc.name == "execute_sql_query":
                        result = execute_sql_query(fc.args.get("sql", ""))
                    elif fc.name == "create_chart":
                        result = create_chart(
                            fc.args.get("title", "圖表"),
                            fc.args.get("chart_type", "bar"),
                            fc.args.get("data_json", "[]"),
                            fc.args.get("x_key", ""),
                            fc.args.get("y_key", "")
                        )
                        if "IMAGE_ID:" in result:
                            img_id = result.split(":")[1]
                            image_url = f"{base_url}/img/{img_id}"
                            result = "圖表已生成"
                    else:
                        result = f"未知工具: {fc.name}"
                    
                    tool_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": result}
                            )
                        )
                    )
                
                # 加入對話
                contents.append(content)
                contents.append(types.Content(role="user", parts=tool_responses))
            else:
                # 沒有工具調用，取得最終回應
                final_text = response.text
                break
        
        # 更新記憶
        CHAT_MEMORY[user_id] = contents[-10:]
        
        return {
            "text": final_text or "處理完成",
            "image": image_url
        }
        
    except Exception as e:
        logger.error(f"❌ Agent 錯誤: {str(e)}", exc_info=True)
        return {"text": f"系統錯誤: {str(e)}"}

# =========================================================
# LINE 回覆
# =========================================================
async def reply_line(reply_token: str, text_out: Optional[str], img_url: Optional[str] = None):
    """回覆 LINE 訊息"""
    if not LINE_CHANNEL_ACCESS_TOKEN:
        logger.warning("⚠️ LINE_CHANNEL_ACCESS_TOKEN 未設定")
        return
    
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    messages = []
    
    if img_url:
        messages.append({
            "type": "image",
            "originalContentUrl": img_url,
            "previewImageUrl": img_url
        })
    
    if text_out:
        messages.append({
            "type": "text",
            "text": text_out[:4999]
        })
    
    if not messages:
        messages.append({"type": "text", "text": "處理完成"})
    
    payload = {
        "replyToken": reply_token,
        "messages": messages
    }
    
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            response = await c.post(
                "https://api.line.me/v2/bot/message/reply",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                logger.info("✅ LINE 訊息已送出")
            else:
                logger.error(f"❌ LINE API 錯誤: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ 發送訊息失敗: {str(e)}")

# =========================================================
# Routes
# =========================================================
@app.get("/")
def root():
    """根路徑"""
    return {
        "status": "ok",
        "service": "ERP Bot Ultra PRO",
        "version": "3.2",
        "timestamp": now_taipei().isoformat()
    }

@app.get("/health")
def health():
    """健康檢查"""
    require_db_ready()
    
    checks = {
        "database": False,
        "gemini": bool(client),
        "line": bool(LINE_CHANNEL_ACCESS_TOKEN),
        "sales_url": bool(SALES_SHEET_URL),
        "purchase_url": bool(PURCHASE_SHEET_URL)
    }
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except:
        pass
    
    counts = table_counts()
    
    return {
        "status": "healthy" if checks["database"] and checks["gemini"] and checks["line"] else "degraded",
        "checks": checks,
        "counts": counts,
        "timestamp": now_taipei().isoformat()
    }

@app.get("/img/{img_id}")
def get_img(img_id: str):
    """取得圖片"""
    if img_id not in IMG_STORE:
        raise HTTPException(404, "圖片不存在")
    
    return Response(
        content=IMG_STORE[img_id]["bytes"],
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"}
    )

@app.post("/admin/reload_sync")
def admin_reload_sync(request: Request):
    """管理員：手動同步資料"""
    require_admin(request)
    return import_data_to_db()

@app.post("/admin/clear_data")
def admin_clear_data(request: Request):
    """管理員：清空資料表"""
    require_admin(request)
    
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM sales"))
        conn.execute(text("DELETE FROM purchase"))
    
    return {"ok": True, "message": "資料已清空"}

@app.post("/line/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """LINE Webhook"""
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    
    # 驗證簽名
    verify_line_signature(body, signature)
    
    # 解析事件
    try:
        events = json.loads(body.decode("utf-8")).get("events", [])
    except Exception as e:
        logger.error(f"❌ JSON 解析失敗: {str(e)}")
        return {"ok": False}
    
    base_url = f"https://{request.headers.get('host', 'localhost')}"
    
    for event in events:
        event_type = event.get("type")
        logger.info(f"📨 收到事件: {event_type}")
        
        # 訊息事件
        if event_type == "message":
            message = event.get("message", {})
            
            if message.get("type") == "text":
                user_id = event["source"]["userId"]
                user_text = message["text"]
                reply_token = event["replyToken"]
                
                logger.info(f"👤 用戶 {user_id}: {user_text}")
                
                # 檢查限流
                if not rate_limit_ok(user_id):
                    background_tasks.add_task(
                        reply_line,
                        reply_token,
                        "請稍後再試（請求過於頻繁）",
                        None
                    )
                    continue
                
                # 特殊指令
                if user_text.strip().lower() in ["/reset", "/清除", "清除"]:
                    CHAT_MEMORY.pop(user_id, None)
                    background_tasks.add_task(
                        reply_line,
                        reply_token,
                        "✅ 對話記憶已清除",
                        None
                    )
                    continue
                
                if user_text.strip().lower() in ["/help", "/說明", "說明"]:
                    help_text = """🤖 ERP 助理使用說明

📊 **查詢功能**
• 直接問問題，例如：
  - 2024年總銷售額？
  - 前10大客戶是誰？
  - 採購趨勢如何？

📈 **視覺化功能**
• 要求繪圖，例如：
  - 畫出月銷售趨勢
  - 顯示產品佔比
  - 比較各年度業績

⚙️ **指令**
/清除 - 清除對話記憶
/說明 - 顯示此說明

💡 **提示**
我會主動提供分析和建議！"""
                    background_tasks.add_task(
                        reply_line,
                        reply_token,
                        help_text,
                        None
                    )
                    continue
                
                # AI 處理
                background_tasks.add_task(
                    handle_message,
                    user_id,
                    user_text,
                    reply_token,
                    base_url
                )
        
        # 追蹤事件（加入好友）
        elif event_type == "follow":
            reply_token = event["replyToken"]
            welcome = """👋 歡迎使用 ERP 智能助理！

我可以幫你：
📊 查詢銷售和採購數據
📈 生成圖表和趨勢分析
💡 提供商業洞察

試試問我：
• 「2024年總銷售額？」
• 「前10大客戶？」
• 「畫出銷售趨勢圖」

輸入 /說明 查看更多功能！"""
            
            background_tasks.add_task(reply_line, reply_token, welcome, None)
    
    return {"ok": True}

async def handle_message(user_id: str, user_text: str, reply_token: str, base_url: str):
    """處理訊息（背景任務）"""
    try:
        # 檢查資料庫是否有資料
        counts = table_counts()
        if counts["sales"] == 0 and counts["purchase"] == 0:
            await reply_line(
                reply_token,
                "⚠️ 資料庫目前沒有資料\n\n請管理員先同步 Google Sheet 資料。",
                None
            )
            return
        
        # AI 處理
        result = await agent_process(user_id, user_text, base_url)
        await reply_line(reply_token, result.get("text"), result.get("image"))
        
    except Exception as e:
        logger.error(f"❌ 處理訊息錯誤: {str(e)}", exc_info=True)
        await reply_line(reply_token, f"系統錯誤，請稍後再試", None)

# =========================================================
# 啟動事件
# =========================================================
@app.on_event("startup")
async def startup():
    """應用啟動"""
    global DB_READY
    
    logger.info("🚀 應用啟動中...")
    
    # 檢查環境變數
    logger.info(f"DATABASE_URL: {'✅' if DATABASE_URL else '❌'}")
    logger.info(f"GEMINI_API_KEY: {'✅' if GEMINI_API_KEY else '❌'}")
    logger.info(f"LINE_CHANNEL_ACCESS_TOKEN: {'✅' if LINE_CHANNEL_ACCESS_TOKEN else '❌'}")
    logger.info(f"LINE_CHANNEL_SECRET: {'✅' if LINE_CHANNEL_SECRET else '❌'}")
    logger.info(f"SALES_EXCEL_URL: {'✅' if SALES_SHEET_URL else '❌'}")
    logger.info(f"PURCHASE_EXCEL_URL: {'✅' if PURCHASE_SHEET_URL else '❌'}")
    
    # 建立資料表
    try:
        ensure_tables()
        DB_READY = True
        logger.info("✅ 資料庫初始化完成")
    except Exception as e:
        logger.error(f"❌ 資料庫初始化失敗: {str(e)}")
    
    # 自動匯入
    if AUTO_IMPORT_ON_STARTUP:
        logger.info("🔄 啟動時自動匯入資料...")
        try:
            result = import_data_to_db()
            logger.info(f"✅ 自動匯入完成: {result}")
        except Exception as e:
            logger.error(f"❌ 自動匯入失敗: {str(e)}")
    
    logger.info("✨ 應用啟動完成！")

@app.on_event("shutdown")
async def shutdown():
    """應用關閉"""
    logger.info("👋 應用關閉中...")
    IMG_STORE.clear()
    CHAT_MEMORY.clear()
    logger.info("✅ 清理完成")