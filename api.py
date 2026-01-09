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

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO

from google import genai
from google.genai import types

app = FastAPI(title="Smart ERP Bot", version="Fixed_Agent")

# 連線修正
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./erp.db"

engine = create_engine(DATABASE_URL)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

CHAT_MEMORY: Dict[str, List[Any]] = {} 
IMG_STORE: Dict[str, Dict[str, Any]] = {}

# =========================
# 工具修正：將 data 改為 str 類型
# =========================
def execute_sql_query(sql: str) -> str:
    """【工具】執行 SQL SELECT 查詢 sales 或 purchase 表。"""
    if not sql.strip().lower().startswith("select"):
        return "錯誤：只允許 SELECT 查詢。"
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
            if df.empty: return "查詢成功但無資料。"
            for col in df.select_dtypes(include=['datetime']).columns:
                df[col] = df[col].astype(str)
            return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

def create_chart(title: str, chart_type: str, data_json: str, x_key: str, y_key: str) -> str:
    """【工具】繪製圖表。data_json 必須是有效的 JSON 字串。"""
    try:
        data = json.loads(data_json)
        df = pd.DataFrame(data)
        if df.empty: return "無資料繪圖。"
        
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)
        plt.figure(figsize=(10, 6))
        if chart_type == "line": plt.plot(df[x_key], df[y_key], marker='o')
        elif chart_type == "bar": 
            plt.bar(df[x_key], df[y_key])
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

tools_list = [execute_sql_query, create_chart]
google_search = {"google_search": {}}

# =========================
# Agent 處理修正
# =========================
SYSTEM_PROMPT = """
你是一個 ERP 助理。你可以查詢 'sales' 和 'purchase' 資料表。
- 繪圖時，請將資料轉換為 JSON 字串並傳遞給 `data_json` 參數。
- 欄位：date, customer, product, quantity, amount, year (sales) / date, supplier, product, quantity, amount, year (purchase)。
"""

async def agent_process(user_id: str, text: str, base_url: str):
    if not client: return {"text": "API Key Error"}
    history = CHAT_MEMORY.get(user_id, [])
    
    try:
        # 第一輪：AI 決定行動
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=history + [text],
            config=types.GenerateContentConfig(tools=tools_list + [google_search], system_instruction=SYSTEM_PROMPT)
        )
        
        final_text = ""
        image_url = None
        part = resp.candidates[0].content.parts[0]
        
        if part.function_call:
            fc = part.function_call
            tool_res = ""
            if fc.name == "execute_sql_query":
                tool_res = execute_sql_query(fc.args["sql"])
            elif fc.name == "create_chart":
                # AI 現在會傳入字串形式的 data_json
                chart_res = create_chart(fc.args["title"], fc.args["chart_type"], fc.args["data_json"], fc.args["x_key"], fc.args["y_key"])
                if "IMAGE_ID" in chart_res:
                    img_id = chart_res.split(":")[1]
                    image_url = f"{base_url}/img/{img_id}"
                    tool_res = "圖表已生成。"
                else: tool_res = chart_res
            
            # 第二輪：告訴 AI 結果
            resp2 = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=history + [text, resp.candidates[0].content, 
                                  types.Content(parts=[types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": tool_res}))])],
                config=types.GenerateContentConfig(tools=tools_list + [google_search], system_instruction=SYSTEM_PROMPT)
            )
            final_text = resp2.text
        else:
            final_text = resp.text

        CHAT_MEMORY[user_id] = (history + [text, final_text])[-10:]
        return {"text": final_text, "image": image_url}
    except Exception as e:
        return {"text": f"發生錯誤：{str(e)}"}

@app.get("/")
def root(): return {"ok": True}

@app.get("/img/{img_id}")
def get_img(img_id: str):
    if img_id not in IMG_STORE: raise HTTPException(404)
    return Response(content=IMG_STORE[img_id]["bytes"], media_type="image/png")

@app.post("/line/webhook")
async def webhook(request: Request):
    body = await request.body()
    events = json.loads(body.decode("utf-8")).get("events", [])
    base_url = f"https://{request.headers.get('host')}"
    for ev in events:
        if ev.get("type") == "message" and ev["message"]["type"] == "text":
            res = await agent_process(ev["source"]["userId"], ev["message"]["text"], base_url)
            await reply_line(ev["replyToken"], res.get("text"), res.get("image"))
    return {"ok": True}

async def reply_line(token, text, img_url=None):
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    messages = []
    if img_url: messages.append({"type": "image", "originalContentUrl": img_url, "previewImageUrl": img_url})
    if text: messages.append({"type": "text", "text": str(text)[:4500]})
    async with httpx.AsyncClient() as c: await c.post("https://api.line.me/v2/bot/message/reply", headers=headers, json={"replyToken": token, "messages": messages})

@app.on_event("startup")
def startup():
    try:
        from data_loader import import_excel_files
        import_excel_files()
    except: pass