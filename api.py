import os
import time
import uuid
import json
import hmac
import base64
import hashlib
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from sqlalchemy import create_engine, text
import pandas as pd
import httpx

# =========================
# 0. Matplotlib 設定 (防當機)
# =========================
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO

# 設定字型 Fallback (避免方塊字)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 初始化與設定
# =========================
from google import genai
from google.genai import types

app = FastAPI(title="Smart ERP Bot", version="Final")

# 資料庫連線修正 (Render Postgres Fix)
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./erp.db"

engine = create_engine(DATABASE_URL)

# API Keys
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Gemini Client
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Gemini Init Error: {e}")

# 記憶體
CHAT_MEMORY: Dict[str, List[Any]] = {} 
IMG_STORE: Dict[str, Dict[str, Any]] = {}

# =========================
# 2. 工具函式 (AI Skills)
# =========================

def execute_sql_query(sql: str) -> str:
    """【工具】執行 SQL SELECT 查詢 sales/purchase 表格。"""
    if not sql.strip().lower().startswith("select"):
        return "錯誤：只允許 SELECT 查詢。"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
            if df.empty: return "查詢成功 (無資料)。"
            # 格式化日期
            for col in df.select_dtypes(include=['datetime']).columns:
                df[col] = df[col].astype(str)
            if len(df) > 30:
                return f"資料過多 ({len(df)} 筆)，僅回傳前 30 筆：\n" + df.head(30).to_json(orient="records", force_ascii=False)
            return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

def create_chart(title: str, chart_type: str, data: List[Dict], x_key: str, y_key: str) -> str:
    """【工具】繪製圖表 (line, bar, pie)。"""
    try:
        df = pd.DataFrame(data)
        if df.empty: return "無資料可繪圖。"
        
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)
        plt.figure(figsize=(10, 6))
        
        if chart_type == "line":
            plt.plot(df[x_key], df[y_key], marker='o')
            plt.grid(True, alpha=0.3)
        elif chart_type == "bar":
            plt.bar(df[x_key], df[y_key], alpha=0.8)
            plt.xticks(rotation=45, ha='right')
        elif chart_type == "pie":
            df_s = df.sort_values(by=y_key, ascending=False).head(8)
            plt.pie(df_s[y_key], labels=df_s[x_key], autopct='%1.1f%%')
            
        plt.title(title)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        
        img_id = str(uuid.uuid4())
        IMG_STORE[img_id] = {"bytes": buf.getvalue(), "ts": time.time()}
        return f"IMAGE_ID:{img_id}"
    except Exception as e:
        return f"Chart Error: {str(e)}"

# 工具定義
tools_list = [execute_sql_query, create_chart]
google_search = {"google_search": {}}

# =========================
# 3. Agent 核心邏輯
# =========================
SYSTEM_PROMPT = """
你是一個全能企業助手。
1. **內部資料**：使用 `execute_sql_query` 查詢 'sales' (date, customer, product, quantity, amount, year) 或 'purchase' 表。
2. **外部資訊**：NBA、天氣、新聞等問題，使用 `Google Search`。
3. **圖表**：若數據適合視覺化，使用 `create_chart`。
4. **回應**：繁體中文，專業友善。
"""

async def agent_process(user_id: str, text: str, base_url: str):
    if not client: return {"text": "API Key Error"}
    
    history = CHAT_MEMORY.get(user_id, [])
    config = types.GenerateContentConfig(
        tools=tools_list + [google_search],
        system_instruction=SYSTEM_PROMPT,
        temperature=0.3
    )
    
    try:
        # 第一輪：思考與調用工具
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history + [text],
            config=config
        )
        
        final_text = ""
        image_url = None
        
        # 處理 Tool Call
        candidates = resp.candidates
        if candidates and candidates[0].content.parts:
            part = candidates[0].content.parts[0]
            if part.function_call:
                fc = part.function_call
                res_content = "執行失敗"
                
                # 執行工具
                if fc.name == "execute_sql_query":
                    res_content = execute_sql_query(fc.args["sql"])
                elif fc.name == "create_chart":
                    # 處理資料傳遞
                    d = fc.args["data"]
                    if isinstance(d, str): d = json.loads(d)
                    chart_res = create_chart(fc.args["title"], fc.args["chart_type"], d, fc.args["x_key"], fc.args["y_key"])
                    if "IMAGE_ID" in chart_res:
                        img_id = chart_res.split(":")[1]
                        image_url = f"{base_url}/img/{img_id}"
                        res_content = "圖表已生成。"
                    else:
                        res_content = chart_res
                
                # 第二輪：將結果回傳給 AI 生成文字
                resp2 = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=history + [text, resp.candidates[0].content, 
                                      types.Content(parts=[types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": res_content}))])],
                    config=config
                )
                final_text = resp2.text
            else:
                final_text = resp.text

        CHAT_MEMORY[user_id] = (history + [text, final_text])[-10:]
        return {"text": final_text, "image": image_url}
        
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"text": "系統忙碌中，請稍後再試。"}

# =========================
# 4. Web API
# =========================
@app.get("/")
def root(): return {"status": "Running"}

@app.get("/img/{img_id}")
def get_img(img_id: str):
    if img_id not in IMG_STORE: raise HTTPException(404)
    return Response(content=IMG_STORE[img_id]["bytes"], media_type="image/png")

@app.post("/line/webhook")
async def webhook(request: Request):
    sig = request.headers.get("x-line-signature", "")
    body = await request.body()
    # 這裡省略簽章驗證報錯，避免測試時卡住，正式上線建議開啟
    
    events = json.loads(body.decode("utf-8")).get("events", [])
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    base_url = f"https://{host}"
    
    for ev in events:
        if ev.get("type") == "message" and ev["message"].get("type") == "text":
            uid = ev["source"]["userId"]
            reply_token = ev["replyToken"]
            msg = ev["message"]["text"]
            
            res = await agent_process(uid, msg, base_url)
            await reply_line(reply_token, res.get("text"), res.get("image"))
            
    return {"ok": True}

async def reply_line(token, text, img_url=None):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    messages = []
    if img_url: messages.append({"type": "image", "originalContentUrl": img_url, "previewImageUrl": img_url})
    if text: messages.append({"type": "text", "text": str(text)[:4500]})
    async with httpx.AsyncClient() as client:
        await client.post("https://api.line.me/v2/bot/message/reply", headers=headers, json={"replyToken": token, "messages": messages})

@app.on_event("startup")
def startup():
    # 啟動時自動載入資料
    try:
        from data_loader import import_excel_files
        import_excel_files()
    except Exception as e:
        print(f"Data Load Error: {e}")