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
# 設定 Matplotlib 後端為 Agg (必須在 pyplot 匯入前設定)
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO

from google import genai
from google.genai import types

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart ERP Bot", version="Final_Search_Enabled")

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
# 工具函數 (Python Functions)
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
            df = pd.read_sql(text(sql), conn)
            if df.empty: 
                return "查詢成功但無資料。"
            
            # 處理日期時間欄位
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].astype(str)
            
            # 限制回傳筆數避免過大
            if len(df) > 100:
                logger.info(f"結果筆數過多 ({len(df)})，僅回傳前 100 筆")
                df = df.head(100)
                
            return df.to_json(orient="records", force_ascii=False, date_format='iso')
    except Exception as e:
        logger.error(f"SQL 執行錯誤: {str(e)}")
        return f"SQL Error: {str(e)}"

def create_chart(title: str, chart_type: str, data_json: str, x_key: str, y_key: str) -> str:
    """【工具】繪製圖表。data_json 必須是有效的 JSON 字串。"""
    logger.info(f"繪製圖表: {title} ({chart_type})")
    
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        
        if df.empty: 
            return "無資料繪圖。"
        
        if x_key not in df.columns or y_key not in df.columns:
            return f"錯誤：找不到欄位 {x_key} 或 {y_key}"
        
        # 數值轉換
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)
        
        # 繪圖
        plt.figure(figsize=(10, 6))
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
        logger.error(f"JSON 解析錯誤: {str(e)}")
        return f"JSON 格式錯誤: {str(e)}"
    except Exception as e:
        logger.error(f"圖表生成錯誤: {str(e)}")
        return f"Chart Error: {str(e)}"

def get_database_schema() -> str:
    """【工具】取得資料庫結構資訊"""
    try:
        with engine.connect() as conn:
            # 嘗試取得資料表結構，若無資料表則回傳錯誤
            sales_info = conn.execute(text("SELECT * FROM sales LIMIT 1")).keys()
            purchase_info = conn.execute(text("SELECT * FROM purchase LIMIT 1")).keys()
            
            sales_count = conn.execute(text("SELECT COUNT(*) FROM sales")).scalar()
            purchase_count = conn.execute(text("SELECT COUNT(*) FROM purchase")).scalar()
            
            return json.dumps({
                "tables": {
                    "sales": {
                        "columns": list(sales_info),
                        "count": sales_count
                    },
                    "purchase": {
                        "columns": list(purchase_info),
                        "count": purchase_count
                    }
                }
            }, ensure_ascii=False)
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# 系統提示詞
# =========================
SYSTEM_PROMPT = """你是一個智能 ERP 助理，名字是「小智」。你擁有以下能力：

## 📊 資料庫查詢能力
- 可以查詢 'sales'（銷售）和 'purchase'（採購）兩張表
- sales 欄位：date(日期), customer(客戶), product(產品), quantity(數量), amount(金額), year(年份)
- purchase 欄位：date(日期), supplier(供應商), product(產品), quantity(數量), amount(金額), year(年份)

## 🎨 資料視覺化能力
- 可以繪製折線圖(line)、長條圖(bar)、圓餅圖(pie)
- 繪圖時必須先用 execute_sql_query 取得資料，再用 create_chart 繪製

## 🌐 網路搜尋能力 (Google Search)
- **當用戶問的問題不在資料庫中（例如：最新新聞、NBA 比分、天氣、匯率、歷史事件等），請務必使用 google_search 工具查詢最新資訊。**
- 不要在沒有搜尋的情況下編造即時資訊。

## 💬 對話原則
1. **主動積極**：不要只是回答問題，要主動提供洞察和建議
2. **數據驅動**：盡可能用實際數據支持你的回答
3. **視覺化優先**：當數據適合視覺化時，主動建議或直接繪圖
4. **友善專業**：使用繁體中文，語氣友善但專業

## 🚫 限制
- 只能執行 SELECT 查詢，不能修改資料庫
- 繪圖時 data_json 必須是有效的 JSON 字串格式
"""

# =========================
# Agent 處理邏輯
# =========================
async def agent_process(user_id: str, text: str, base_url: str, max_turns: int = 5):
    """處理對話，支援 SQL、繪圖與 Google 搜尋"""
    if not client: 
        return {"text": "❌ Gemini API Key 未設定"}
    
    history = CHAT_MEMORY.get(user_id, [])
    
    try:
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=text)]
        )
        
        contents = history + [user_message]
        
        # ==========================================
        # ✅ 關鍵修復：正確定義並混合 Google Search 工具
        # ==========================================
        
        # 1. 定義搜尋工具 (正確的 SDK 寫法)
        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        # 2. 混合 Python 函式與搜尋工具
        # 我們將自定義函式與 google_search_tool 放在同一個清單中傳給 config
        my_tools = [execute_sql_query, create_chart, get_database_schema, google_search_tool]
        
        config = types.GenerateContentConfig(
            tools=my_tools, 
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7
        )
        
        final_text = ""
        image_url = None
        turn = 0
        
        while turn < max_turns:
            turn += 1
            logger.info(f"Agent 第 {turn} 輪處理")
            
            # ==========================================
            # ✅ 關鍵修復：使用 gemini-1.5-flash
            # 原因：1.5-flash 是目前最穩定支援「工具混用(SQL+Search)」的版本
            # 2.0 版本目前會報 "unsupported" 錯誤
            # ==========================================
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config=config
            )
            
            if not response.candidates:
                final_text = "抱歉，我無法處理這個請求。"
                break
            
            candidate = response.candidates[0]
            content = candidate.content
            
            # 檢查是否有工具調用
            has_function_call = any(
                part.function_call for part in content.parts if hasattr(part, 'function_call')
            )
            
            if has_function_call:
                function_responses = []
                
                for part in content.parts:
                    if not hasattr(part, 'function_call'):
                        continue
                        
                    fc = part.function_call
                    logger.info(f"調用工具: {fc.name}")
                    
                    tool_result = ""
                    
                    # 處理自定義 Python 工具
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
                        # 如果是 Google Search，模型通常會自己在伺服器端執行，
                        # 但如果跑到這裡，代表模型可能嘗試用 function call 的方式回傳。
                        # 對於 gemini-1.5-flash，通常它會自動處理 search，
                        # 我們只需回傳一個空的或提示訊息讓它繼續。
                        tool_result = f"工具 {fc.name} 已被調用"
                    
                    function_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response={"result": tool_result}
                            )
                        )
                    )
                
                # 將工具執行結果回傳給模型
                contents.append(content)
                contents.append(types.Content(
                    role="user",
                    parts=function_responses
                ))
                
            else:
                # 沒有工具調用，代表已生成最終回應
                final_text = response.text
                break
        
        CHAT_MEMORY[user_id] = contents[-20:]
        
        return {
            "text": final_text or "處理完成！",
            "image": image_url
        }
        
    except Exception as e:
        logger.error(f"Agent 處理錯誤: {str(e)}", exc_info=True)
        return {"text": f"❌ 發生錯誤：{str(e)}"}

# =========================
# API 端點
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "Smart ERP Bot (Search Enabled)"}

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
        
        # 顯示歡迎/幫助訊息
        if text.lower() in ['/help', '/說明', '說明']:
            await reply_line(reply_token, "我可以查資料庫（銷售/採購），也可以上網搜尋（NBA、天氣）。請直接問我問題！", None)
            return

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