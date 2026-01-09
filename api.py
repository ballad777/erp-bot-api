import os
import time
import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from sqlalchemy import create_engine, text
import pandas as pd
import httpx

import matplotlib
# ✅ 設定 Matplotlib 後端為 Agg (防止伺服器繪圖錯誤)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO

from google import genai
from google.genai import types

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart ERP Bot", version="Excel_Only_Fuzzy_Match")

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
# 工具函數 (Python Functions - 處理 Excel 資料庫)
# =========================
def execute_sql_query(sql: str) -> str:
    """【工具】執行 SQL SELECT 查詢 sales 或 purchase 表。"""
    logger.info(f"執行 SQL: {sql}")
    
    # 安全檢查
    sql_lower = sql.strip().lower()
    if not sql_lower.startswith("select"):
        return "錯誤：只允許 SELECT 查詢。"
    
    # 防止危險操作
    dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'truncate', 'alter']
    if any(keyword in sql_lower for keyword in dangerous_keywords):
        return "錯誤：不允許修改資料的操作。"
    
    try:
        with engine.connect() as conn:
            # 執行查詢
            df = pd.read_sql(text(sql), conn)
            
            if df.empty: 
                return "查詢成功但沒有找到符合的資料 (No Data Found)。"
            
            # 處理日期時間欄位，轉成字串
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].astype(str)
            
            # 限制回傳筆數 (如果超過 100 筆，只回傳前 100 筆並提示)
            if len(df) > 100:
                logger.info(f"結果筆數過多 ({len(df)})，僅回傳前 100 筆")
                df = df.head(100)
                
            return df.to_json(orient="records", force_ascii=False, date_format='iso')
    except Exception as e:
        logger.error(f"SQL 執行錯誤: {str(e)}")
        # 回傳錯誤訊息給 AI，讓 AI 知道 SQL 寫錯了，它可以嘗試修正
        return f"SQL Execution Error: {str(e)}"

def create_chart(title: str, chart_type: str, data_json: str, x_key: str, y_key: str) -> str:
    """【工具】繪製圖表。"""
    logger.info(f"繪製圖表: {title} ({chart_type})")
    
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        if df.empty: 
            return "無資料可繪圖。"
        
        if x_key not in df.columns or y_key not in df.columns:
            return f"錯誤：找不到欄位 {x_key} 或 {y_key}，現有欄位: {list(df.columns)}"
        
        # 數值轉換
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)
        
        # 繪圖設定
        plt.figure(figsize=(10, 6))
        # 設定通用中文字型
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        if chart_type == "line": 
            plt.plot(df[x_key], df[y_key], marker='o', linewidth=2)
        elif chart_type == "bar": 
            plt.bar(df[x_key], df[y_key], color='steelblue')
            plt.xticks(rotation=45, ha='right')
        elif chart_type == "pie":
            df_s = df.sort_values(by=y_key, ascending=False).head(8)
            plt.pie(df_s[y_key], labels=df_s[x_key], autopct='%1.1f%%', startangle=90)
        else:
            return f"不支援的圖表類型: {chart_type}"
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches='tight')
        plt.close()
        
        img_id = str(uuid.uuid4())
        IMG_STORE[img_id] = {
            "bytes": buf.getvalue(), 
            "ts": time.time(),
            "title": title
        }
        
        logger.info(f"圖表生成成功: {img_id}")
        return f"IMAGE_ID:{img_id}"
    except json.JSONDecodeError as e:
        return f"JSON 格式錯誤: {str(e)}"
    except Exception as e:
        logger.error(f"圖表生成錯誤: {str(e)}")
        return f"Chart Error: {str(e)}"

def get_database_schema() -> str:
    """【工具】取得資料庫結構資訊。"""
    try:
        with engine.connect() as conn:
            # 取得 Sales 表資訊
            sales_cols = conn.execute(text("SELECT * FROM sales LIMIT 1")).keys()
            sales_count = conn.execute(text("SELECT COUNT(*) FROM sales")).scalar()
            
            # 取得 Purchase 表資訊
            purchase_cols = conn.execute(text("SELECT * FROM purchase LIMIT 1")).keys()
            purchase_count = conn.execute(text("SELECT COUNT(*) FROM purchase")).scalar()
            
            # 讓 AI 知道欄位名稱和資料量，方便它寫 SQL
            return json.dumps({
                "database_summary": {
                    "sales_table": {
                        "description": "銷售資料表",
                        "columns": list(sales_cols),
                        "total_rows": sales_count,
                        "example_columns": ["date", "customer", "product", "quantity", "amount", "year"]
                    },
                    "purchase_table": {
                        "description": "採購資料表",
                        "columns": list(purchase_cols),
                        "total_rows": purchase_count,
                        "example_columns": ["date", "supplier", "product", "quantity", "amount", "year"]
                    }
                }
            }, ensure_ascii=False)
    except Exception as e:
        return f"Schema Error: {str(e)}"

# =========================
# 工具列表 (只保留 Excel 相關)
# =========================
tools_list = [execute_sql_query, create_chart, get_database_schema]

# =========================
# 系統提示詞 (強調模糊比對與糾錯)
# =========================
SYSTEM_PROMPT = """你是一個極度聰明的 ERP 數據助理，名字是「小智」。
你的任務是查詢資料庫並回答用戶關於「銷售 (sales)」與「採購 (purchase)」的問題。

## 🧠 你的核心能力：模糊比對與糾錯
用戶輸入的查詢可能會有錯字、簡寫或模糊不清，你必須**先推測用戶的真實意圖**，再撰寫 SQL。

1. **自動修正錯字**：
   - 如果用戶輸入 "ipone"，你要知道他在查 "iPhone"，SQL 請用 `WHERE product LIKE '%iPhone%'`。
   - 如果用戶輸入 "Samung"，你要修正為 "Samsung"。
   - 如果用戶輸入 "電腦"，SQL 請用 `LIKE '%電腦%'` 或 `LIKE '%PC%'` (根據你對產品的理解)。

2. **模糊查詢**：
   - 除非用戶指定確切名稱，否則查詢文字欄位時，請一律使用 `LIKE %關鍵字%`。
   - 範例：查 "華碩"，SQL 應為 `WHERE customer LIKE '%華碩%' OR product LIKE '%華碩%'`。

3. **資料表結構**：
   - **sales (銷售)**: date, customer, product, quantity, amount, year
   - **purchase (採購)**: date, supplier, product, quantity, amount, year

## 📝 SQL 撰寫規則
- 只使用 SELECT。
- 字串比對一律加上單引號，例如 `product = 'iPhone 15'`。
- 日期格式通常為 'YYYY-MM-DD'。
- 如果用戶問「總額」或「多少錢」，請使用 `SUM(amount)`。
- 如果用戶問「銷量」或「多少個」，請使用 `SUM(quantity)`。

## 🚫 限制
- **絕對不要使用 Google 搜尋**，你只能查資料庫。
- 如果資料庫查不到，請嘗試放寬 SQL 條件 (例如把 `AND` 改成 `OR`，或是減少 WHERE 條件) 再查一次。

記住：你的目標是**無論用戶怎麼問、字怎麼打，都要盡力從資料庫挖出相關的資料**！
"""

# =========================
# Agent 處理邏輯
# =========================
async def agent_process(user_id: str, text: str, base_url: str, max_turns: int = 5):
    """處理對話"""
    if not client: 
        return {"text": "❌ Gemini API Key 未設定，請檢查環境變數"}
    
    history = CHAT_MEMORY.get(user_id, [])
    
    try:
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=text)]
        )
        
        contents = history + [user_message]
        
        config = types.GenerateContentConfig(
            tools=tools_list, 
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4  # 降低隨機性，讓 SQL 更精確
        )
        
        final_text = ""
        image_url = None
        turn = 0
        
        while turn < max_turns:
            turn += 1
            logger.info(f"Agent 第 {turn} 輪處理")
            
            # ✅ 使用您日誌中出現過的可用模型
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=contents,
                config=config
            )
            
            if not response.candidates:
                final_text = "抱歉，我無法處理這個請求。"
                break
            
            candidate = response.candidates[0]
            content = candidate.content
            
            # 檢查 Function Call
            has_function_call = any(
                part.function_call for part in content.parts if hasattr(part, 'function_call')
            )
            
            if has_function_call:
                function_responses = []
                
                for part in content.parts:
                    if not hasattr(part, 'function_call'):
                        continue
                        
                    fc = part.function_call
                    logger.info(f"調用工具: {fc.name} | 參數: {fc.args}")
                    
                    tool_result = ""
                    
                    if fc.name == "execute_sql_query":
                        tool_result = execute_sql_query(fc.args.get("sql", ""))
                    elif fc.name == "create_chart":
                        chart_res = create_chart(
                            fc.args.get("title", "圖表"),
                            fc.args.get("chart_type", "bar"),
                            fc.args.get("data_json", "[]"),
                            fc.args.get("x_key", ""),
                            fc.args.get("y_key", "")
                        )
                        if "IMAGE_ID" in chart_res:
                            img_id = chart_res.split(":")[1]
                            image_url = f"{base_url}/img/{img_id}"
                            tool_result = "圖表已成功生成！"
                        else:
                            tool_result = chart_res
                    elif fc.name == "get_database_schema":
                        tool_result = get_database_schema()
                    else:
                        tool_result = f"未知工具: {fc.name}"
                    
                    function_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": tool_result}
                            )
                        )
                    )
                
                contents.append(content)
                contents.append(types.Content(
                    role="user",
                    parts=function_responses
                ))
                
            else:
                final_text = response.text
                break
        
        CHAT_MEMORY[user_id] = contents[-20:]
        
        return {
            "text": final_text or "處理完成！",
            "image": image_url
        }
        
    except Exception as e:
        logger.error(f"Agent 處理錯誤: {str(e)}", exc_info=True)
        return {"text": f"❌ 系統錯誤：{str(e)}"}

# =========================
# API 端點
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "Smart ERP Bot (Excel Only)"}

@app.get("/health")
def health_check():
    checks = {"database": False, "gemini": bool(client), "line": bool(LINE_CHANNEL_ACCESS_TOKEN)}
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except:
        pass
    return {"status": "healthy" if all(checks.values()) else "degraded", "checks": checks}

@app.get("/img/{img_id}")
def get_img(img_id: str):
    if img_id not in IMG_STORE: 
        raise HTTPException(status_code=404, detail="圖片不存在")
    return Response(content=IMG_STORE[img_id]["bytes"], media_type="image/png")

@app.post("/line/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    
    if LINE_CHANNEL_SECRET:
        import hmac, hashlib, base64
        hash_value = hmac.new(LINE_CHANNEL_SECRET.encode('utf-8'), body, hashlib.sha256).digest()
        expected_signature = base64.b64encode(hash_value).decode('utf-8')
        if signature != expected_signature:
            raise HTTPException(status_code=400, detail="Invalid signature")
    
    try:
        events = json.loads(body.decode("utf-8")).get("events", [])
    except:
        return {"ok": False}
    
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
        if text.lower() in ['/clear', '清除', '/reset']:
            CHAT_MEMORY.pop(user_id, None)
            await reply_line(reply_token, "記憶已清除", None)
            return
        
        # Agent 處理
        result = await agent_process(user_id, text, base_url)
        await reply_line(reply_token, result.get("text"), result.get("image"))
    except Exception as e:
        logger.error(f"Error: {e}")
        await reply_line(reply_token, "系統忙碌中，請稍後再試。", None)

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
    try:
        from data_loader import import_excel_files
        import_excel_files()
        logger.info("✅ 資料載入完成")
    except:
        pass