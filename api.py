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
# 0. Matplotlib 設定 (必須在最上面)
# =========================
import matplotlib
matplotlib.use("Agg") # 設定後端，避免在無介面環境報錯
import matplotlib.pyplot as plt
from io import BytesIO

# 設定中文字型 (Render 上可能沒有中文字型，這裡設定一個 fallback 列表)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 系統初始化
# =========================
from google import genai
from google.genai import types

app = FastAPI(title="Smart ERP Bot Agent", version="3.0")

# 環境變數
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp.db")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 建立連線
engine = create_engine(DATABASE_URL)
# 初始化 Gemini Client
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# 記憶體區
CHAT_MEMORY: Dict[str, List[Any]] = {} 
IMG_STORE: Dict[str, Dict[str, Any]] = {}

# =========================
# 2. LINE 簽章驗證 (保留你原本的邏輯)
# =========================
def verify_line_signature(body_bytes: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)

# =========================
# 3. 定義工具 (Tools) - 給 AI 的技能
# =========================

def execute_sql_query(sql: str) -> str:
    """
    【工具】執行 SQL 查詢以獲取 ERP 數據。
    只允許 SELECT。AI 應根據使用者的問題生成對應的 SQL。
    """
    clean_sql = sql.strip().lower()
    if not clean_sql.startswith("select"):
        return "錯誤：基於安全考量，只允許執行 SELECT 查詢。"
    
    try:
        with engine.connect() as conn:
            # 使用 Pandas 讀取，方便處理
            df = pd.read_sql(text(sql), conn)
            
            if df.empty:
                return "查詢成功，但結果為空 (0 rows)。"
            
            # 處理日期轉字串，避免 JSON 序列化錯誤
            for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
                df[col] = df[col].astype(str)

            # 限制回傳筆數，避免 Token 爆炸
            if len(df) > 30:
                summary = f"注意：資料過多 ({len(df)} 筆)，僅回傳前 30 筆供分析。\n"
                return summary + df.head(30).to_json(orient="records", force_ascii=False)
            
            return df.to_json(orient="records", force_ascii=False)
            
    except Exception as e:
        return f"SQL 執行失敗: {str(e)}。請檢查欄位名稱 (date, year, customer, product, quantity, amount)。"

def create_chart(title: str, chart_type: str, data: List[Dict[str, Any]], x_key: str, y_key: str) -> str:
    """
    【工具】繪製圖表。
    當數據適合視覺化時使用 (例如趨勢、佔比)。
    :param title: 圖表標題
    :param chart_type: 'bar' (長條), 'line' (折線), 'pie' (圓餅)
    :param data: 數據列表 (JSON list)
    :param x_key: X軸欄位名 (類別/時間)
    :param y_key: Y軸欄位名 (數值)
    :return: 圖片 ID
    """
    try:
        df = pd.DataFrame(data)
        if df.empty: return "錯誤：資料為空，無法繪圖。"
        
        # 確保 Y 軸是數值
        df[y_key] = pd.to_numeric(df[y_key], errors='coerce').fillna(0)

        plt.figure(figsize=(10, 6))
        
        if chart_type == "line":
            plt.plot(df[x_key], df[y_key], marker='o', linewidth=2)
            plt.grid(True, linestyle='--', alpha=0.6)
        elif chart_type == "bar":
            plt.bar(df[x_key], df[y_key], alpha=0.8)
        elif chart_type == "pie":
            # 圓餅圖只取前 8 大，剩下歸類為 Other
            df_sorted = df.sort_values(by=y_key, ascending=False)
            if len(df_sorted) > 8:
                top = df_sorted.head(8)
                other = pd.DataFrame([{x_key: 'Other', y_key: df_sorted.iloc[8:][y_key].sum()}])
                df_plot = pd.concat([top, other], ignore_index=True)
            else:
                df_plot = df_sorted
            plt.pie(df_plot[y_key], labels=df_plot[x_key], autopct='%1.1f%%')

        plt.title(title)
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        
        # 針對 bar/line 的 X 軸標籤優化 (避免重疊)
        if chart_type != "pie":
            plt.xticks(rotation=45, ha='right')
            
        plt.tight_layout()

        # 存到記憶體
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()

        img_id = str(uuid.uuid4())
        IMG_STORE[img_id] = {"bytes": buf.getvalue(), "ts": time.time()}
        
        return f"IMAGE_ID:{img_id}"

    except Exception as e:
        return f"繪圖錯誤: {str(e)}"

# 定義 Gemini 可用的工具列表
my_tools = [execute_sql_query, create_chart]
# Google 內建搜尋工具
google_search_tool = {"google_search": {}}

# =========================
# 4. Agent 核心 (思考與執行)
# =========================

SYSTEM_INSTRUCTION = """
你是一個專業、聰明的企業 ERP 助理。
你的資料庫中有兩張表：
1. **sales (銷售表)**: date (日期), year (年), customer (客戶), product (產品), quantity (數量), amount (金額)
2. **purchase (進貨表)**: date, year, supplier (廠商), product, quantity, amount

**你的任務：**
1. **資料查詢**：若使用者問內部數據（如業績、銷量），請務必使用 `execute_sql_query`。
   - 技巧：加總用 SUM(amount)，計次用 COUNT(*)，排序用 ORDER BY amount DESC。
   - 技巧：文字搜尋請用 ILIKE '%關鍵字%'。
2. **外部資訊**：若使用者問天氣、NBA、股價、新聞，請使用 `Google Search`。
3. **資料視覺化**：若數據適合畫圖（如每月趨勢、前十名），先查 SQL，再呼叫 `create_chart`。
4. **回答風格**：請用繁體中文，語氣親切專業。回答要包含數據分析見解。

**重要**：
- 不要憑空捏造內部數據，一定要查資料庫。
- 如果 SQL 錯誤，請根據錯誤訊息修正後重試。
"""

async def process_chat(user_id: str, user_msg: str, base_url: str):
    if not client:
        return {"text": "錯誤：Gemini API Key 未設定。", "image": None}

    history = CHAT_MEMORY.get(user_id, [])
    
    # 建立生成設定
    config = types.GenerateContentConfig(
        tools=my_tools + [google_search_tool], # 混合使用自定義工具與 Google 搜尋
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3, # 降低隨機性，讓 SQL 更準
    )

    try:
        # 1. 發送訊息給 Gemini (自動 Agent 模式)
        # 注意：我們使用手動迴圈來處理 Tool Call，確保流程可控
        response = client.models.generate_content(
            model="gemini-2.0-flash", # 使用支援 Tool Call 穩定的模型
            contents=history + [user_msg],
            config=config
        )

        final_text = ""
        image_url = None

        # 2. 處理回應 (包含可能的 Tool Calls)
        # 這裡簡化處理：如果 AI 決定用工具，我們執行並回傳結果，最多一輪 (Query -> Answer)
        # 複雜的 Agent 可以多輪，但一輪通常夠用
        
        candidates = response.candidates
        if not candidates:
            return {"text": "抱歉，我現在無法思考。", "image": None}

        part = candidates[0].content.parts[0]
        
        # 情況 A: AI 想要呼叫工具
        if part.function_call:
            fc = part.function_call
            tool_name = fc.name
            args = fc.args
            
            print(f"🤖 AI 決定使用工具: {tool_name} | 參數: {args}")
            
            tool_result = "執行失敗"
            
            # 執行對應 Python 函數
            if tool_name == "execute_sql_query":
                tool_result = execute_sql_query(args["sql"])
            
            elif tool_name == "create_chart":
                # 處理資料格式
                data_input = args.get("data")
                if isinstance(data_input, str):
                    try:
                        data_input = json.loads(data_input)
                    except: pass
                
                chart_res = create_chart(
                    args["title"], args["chart_type"], data_input,
                    args["x_key"], args["y_key"]
                )
                
                if "IMAGE_ID:" in chart_res:
                    img_id = chart_res.split(":")[1]
                    image_url = f"{base_url}/img/{img_id}"
                    tool_result = "圖表已生成，請在回覆中告知使用者。"
                else:
                    tool_result = chart_res
            
            # 將工具執行結果回傳給 AI，讓它生成最終文字
            response_final = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=history + [
                    user_msg,
                    response.candidates[0].content, # AI 原本的 Call
                    types.Content(parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": tool_result}
                        )
                    )])
                ],
                config=config
            )
            final_text = response_final.text

        # 情況 B: AI 直接回話 (例如問 NBA，Gemini 會自己處理 Google Search Tool 並整合在 text 裡)
        else:
            final_text = response.text

        # 更新歷史紀錄
        CHAT_MEMORY[user_id] = (history + [user_msg, final_text])[-10:]
        
        return {"text": final_text, "image": image_url}

    except Exception as e:
        print(f"Agent Error: {e}")
        return {"text": "系統繁忙中，請稍後再試。", "image": None}


# =========================
# 5. API 路由
# =========================

@app.get("/")
def root():
    return {"status": "ok", "bot": "Super Smart ERP Agent"}

@app.get("/img/{img_id}")
def get_img(img_id: str):
    item = IMG_STORE.get(img_id)
    if not item: raise HTTPException(status_code=404)
    return Response(content=item["bytes"], media_type="image/png")

@app.post("/line/webhook")
async def line_webhook(request: Request):
    # 簽章驗證
    signature = request.headers.get("x-line-signature", "")
    body_bytes = await request.body()
    if not verify_line_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    body = json.loads(body_bytes.decode("utf-8"))
    
    # 取得 Base URL (for image link)
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    proto = request.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host}"

    for ev in body.get("events", []):
        if ev.get("type") == "message" and ev["message"].get("type") == "text":
            user_id = ev["source"]["userId"]
            reply_token = ev["replyToken"]
            user_text = ev["message"]["text"]

            # 呼叫 Agent
            res = await process_chat(user_id, user_text, base_url)

            # 回覆 LINE
            await reply_line(reply_token, res["text"], res["image"])

    return {"status": "ok"}

async def reply_line(token: str, text: str, image_url: Optional[str]):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    messages = []
    
    # 優先顯示圖片
    if image_url:
        messages.append({
            "type": "image",
            "originalContentUrl": image_url,
            "previewImageUrl": image_url
        })
    
    if text:
        messages.append({"type": "text", "text": str(text)[:4500]})

    async with httpx.AsyncClient() as client:
        await client.post("https://api.line.me/v2/bot/message/reply", headers=headers, json={
            "replyToken": token, "messages": messages
        })

# =========================
# 6. 啟動時自動載入資料 (重要！)
# =========================
@app.on_event("startup")
def startup_event():
    # 每次啟動都重新檢查並匯入資料，確保 Render 重啟後資料還在
    try:
        from data_loader import import_excel_files
        print("🚀 系統啟動，開始載入 Excel 資料...")
        import_excel_files()
    except Exception as e:
        print(f"⚠️ 資料載入失敗: {e}")