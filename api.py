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
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from io import BytesIO

from google import genai
from google.genai import types

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart ERP Bot", version="Enhanced_Agent")

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
# 工具函數
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
            # 檢查 sales 表
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
# 工具列表
# =========================
tools_list = [execute_sql_query, create_chart, get_database_schema]
google_search = {"google_search": {}}

# =========================
# 增強的系統提示詞
# =========================
SYSTEM_PROMPT = """你是一個智能 ERP 助理，名字是「小智」。你擁有以下能力：

## 📊 資料庫查詢能力
- 可以查詢 'sales'（銷售）和 'purchase'（採購）兩張表
- sales 欄位：date(日期), customer(客戶), product(產品), quantity(數量), amount(金額), year(年份)
- purchase 欄位：date(日期), supplier(供應商), product(產品), quantity(數量), amount(金額), year(年份)

## 🎨 資料視覺化能力
- 可以繪製折線圖(line)、長條圖(bar)、圓餅圖(pie)
- 繪圖時必須先用 execute_sql_query 取得資料，再用 create_chart 繪製

## 🌐 網路搜尋能力
- 可以搜尋最新資訊、新聞、天氣等

## 💬 對話原則
1. **主動積極**：不要只是回答問題，要主動提供洞察和建議
2. **數據驅動**：盡可能用實際數據支持你的回答
3. **視覺化優先**：當數據適合視覺化時，主動建議或直接繪圖
4. **友善專業**：使用繁體中文，語氣友善但專業
5. **舉一反三**：回答完問題後，可以主動提供相關的額外資訊或建議

## 📝 回答範例
用戶問：「2024年銷售狀況如何？」
你應該：
1. 查詢 2024 年總銷售額
2. 比較與 2023 年的差異
3. 繪製趨勢圖
4. 分析主要客戶或產品
5. 給出具體建議

## 🚫 限制
- 只能執行 SELECT 查詢，不能修改資料庫
- 繪圖時 data_json 必須是有效的 JSON 字串格式

記住：你不只是工具的執行者，更是用戶的商業顧問！
"""

# =========================
# Agent 處理邏輯（多輪對話支援）
# =========================
async def agent_process(user_id: str, text: str, base_url: str, max_turns: int = 5):
    """增強版 Agent 處理，支援多輪工具調用"""
    if not client: 
        return {"text": "❌ Gemini API Key 未設定，請檢查環境變數"}
    
    # 取得對話歷史
    history = CHAT_MEMORY.get(user_id, [])
    
    try:
        # 記錄用戶訊息
        user_message = types.Content(
            role="user",
            parts=[types.Part(text=text)]
        )
        
        # 完整對話內容
        contents = history + [user_message]
        
        # 配置
        config = types.GenerateContentConfig(
            tools=tools_list + [google_search],
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7
        )
        
        final_text = ""
        image_url = None
        turn = 0
        
        # 多輪對話循環
        while turn < max_turns:
            turn += 1
            logger.info(f"Agent 第 {turn} 輪處理")
            
            response = client.models.generate_content(
                model="gemini-1.5-pro",
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
                # 處理工具調用
                function_responses = []
                
                for part in content.parts:
                    if not hasattr(part, 'function_call'):
                        continue
                        
                    fc = part.function_call
                    logger.info(f"調用工具: {fc.name}")
                    
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
                
                # 將工具回應加入對話
                contents.append(content)
                contents.append(types.Content(
                    role="user",
                    parts=function_responses
                ))
                
            else:
                # 沒有工具調用，取得最終回應
                final_text = response.text
                break
        
        # 更新記憶（保留最近 10 輪對話）
        CHAT_MEMORY[user_id] = contents[-20:]
        
        return {
            "text": final_text or "處理完成！",
            "image": image_url
        }
        
    except Exception as e:
        logger.error(f"Agent 處理錯誤: {str(e)}", exc_info=True)
        return {"text": f"❌ 發生錯誤：{str(e)}\n\n請稍後再試或聯繫管理員。"}

# =========================
# API 端點
# =========================
@app.get("/")
def root():
    """健康檢查"""
    return {
        "status": "ok",
        "service": "Smart ERP Bot",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if engine else "disconnected",
        "gemini": "ready" if client else "not configured"
    }

@app.get("/health")
def health_check():
    """詳細健康檢查"""
    checks = {
        "database": False,
        "gemini": bool(client),
        "line": bool(LINE_CHANNEL_ACCESS_TOKEN)
    }
    
    # 測試資料庫連線
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except:
        pass
    
    return {
        "status": "healthy" if all(checks.values()) else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/img/{img_id}")
def get_img(img_id: str):
    """取得圖片"""
    if img_id not in IMG_STORE: 
        raise HTTPException(status_code=404, detail="圖片不存在")
    
    return Response(
        content=IMG_STORE[img_id]["bytes"], 
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600"
        }
    )

@app.post("/line/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    """LINE Webhook 端點"""
    
    # 取得請求內容
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    
    # 驗證簽名（重要！）
    if LINE_CHANNEL_SECRET:
        import hmac
        import hashlib
        import base64
        
        hash_value = hmac.new(
            LINE_CHANNEL_SECRET.encode('utf-8'),
            body,
            hashlib.sha256
        ).digest()
        expected_signature = base64.b64encode(hash_value).decode('utf-8')
        
        if signature != expected_signature:
            logger.warning("⚠️ LINE 簽名驗證失敗")
            raise HTTPException(status_code=400, detail="Invalid signature")
    
    # 解析事件
    try:
        events = json.loads(body.decode("utf-8")).get("events", [])
    except json.JSONDecodeError:
        logger.error("❌ JSON 解析失敗")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    base_url = f"https://{request.headers.get('host', 'localhost')}"
    
    # 處理每個事件
    for event in events:
        logger.info(f"收到事件: {event.get('type')}")
        
        # 訊息事件
        if event.get("type") == "message":
            message = event.get("message", {})
            
            # 文字訊息
            if message.get("type") == "text":
                user_id = event["source"]["userId"]
                text = message["text"]
                reply_token = event["replyToken"]
                
                logger.info(f"用戶 {user_id} 說: {text}")
                
                # 非同步處理（避免 timeout）
                background_tasks.add_task(
                    handle_message,
                    user_id,
                    text,
                    reply_token,
                    base_url
                )
        
        # 追蹤事件（用戶加入好友）
        elif event.get("type") == "follow":
            reply_token = event["replyToken"]
            welcome_msg = """👋 歡迎使用智能 ERP 助理！

我可以幫你：
📊 查詢銷售和採購數據
📈 生成視覺化圖表
🔍 搜尋最新資訊
💡 提供商業洞察

試試問我：
• 「2024年總銷售額是多少？」
• 「幫我畫出前十大客戶的銷售圖」
• 「分析一下採購趨勢」

有任何問題都可以問我！😊"""
            
            background_tasks.add_task(reply_line, reply_token, welcome_msg, None)
    
    return {"ok": True}

async def handle_message(user_id: str, text: str, reply_token: str, base_url: str):
    """處理訊息（非同步）"""
    try:
        # 處理指令
        if text.lower() in ['/清除記憶', '/clear', '/reset']:
            CHAT_MEMORY.pop(user_id, None)
            await reply_line(reply_token, "✅ 對話記憶已清除！", None)
            return
        
        if text.lower() in ['/help', '/說明', '/?']:
            help_text = """🤖 智能 ERP 助理使用說明

📊 **查詢功能**
• 直接問問題即可，例如：
  - 2024年銷售多少？
  - 哪個客戶買最多？
  - 採購金額趨勢如何？

📈 **視覺化功能**
• 要求繪圖，例如：
  - 畫出月銷售趨勢圖
  - 顯示產品銷售比例
  - 比較各年度業績

🔍 **搜尋功能**
• 問任何問題，我都會盡力回答！

⚙️ **指令**
/清除記憶 - 清除對話歷史
/說明 - 顯示此說明"""
            await reply_line(reply_token, help_text, None)
            return
        
        # Agent 處理
        result = await agent_process(user_id, text, base_url)
        await reply_line(reply_token, result.get("text"), result.get("image"))
        
    except Exception as e:
        logger.error(f"處理訊息時發生錯誤: {str(e)}", exc_info=True)
        await reply_line(reply_token, f"❌ 處理失敗：{str(e)}", None)

async def reply_line(token: str, text: Optional[str], img_url: Optional[str]):
    """回覆 LINE 訊息"""
    if not LINE_CHANNEL_ACCESS_TOKEN:
        logger.warning("⚠️ LINE_CHANNEL_ACCESS_TOKEN 未設定，無法回覆")
        return
    
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    messages = []
    
    # 圖片訊息
    if img_url:
        messages.append({
            "type": "image",
            "originalContentUrl": img_url,
            "previewImageUrl": img_url
        })
    
    # 文字訊息
    if text:
        # LINE 訊息長度限制
        if len(text) > 5000:
            text = text[:4997] + "..."
        messages.append({
            "type": "text",
            "text": text
        })
    
    if not messages:
        messages.append({
            "type": "text",
            "text": "處理完成！"
        })
    
    payload = {
        "replyToken": token,
        "messages": messages
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            response = await c.post(
                "https://api.line.me/v2/bot/message/reply",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"LINE API 錯誤: {response.status_code} - {response.text}")
            else:
                logger.info("✅ 訊息已送出")
    except Exception as e:
        logger.error(f"發送訊息失敗: {str(e)}")

# =========================
# 啟動事件
# =========================
@app.on_event("startup")
async def startup():
    """應用啟動時執行"""
    logger.info("🚀 應用啟動中...")
    
    # 測試資料庫連線
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✅ 資料庫連線成功")
    except Exception as e:
        logger.error(f"❌ 資料庫連線失敗: {str(e)}")
    
    # 載入 Excel 資料（如果有）
    try:
        from data_loader import import_excel_files
        import_excel_files()
        logger.info("✅ 資料載入完成")
    except ImportError:
        logger.info("ℹ️ data_loader 模組不存在，跳過資料載入")
    except Exception as e:
        logger.warning(f"⚠️ 資料載入失敗: {str(e)}")
    
    logger.info("✨ 應用啟動完成！")

@app.on_event("shutdown")
async def shutdown():
    """應用關閉時執行"""
    logger.info("👋 應用關閉中...")
    
    # 清理圖片快取
    IMG_STORE.clear()
    CHAT_MEMORY.clear()
    
    logger.info("✅ 清理完成")