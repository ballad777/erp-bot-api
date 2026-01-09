import os
import re
import io
import time
import json
import hmac
import base64
import hashlib
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from sqlalchemy import create_engine, text

import httpx
import matplotlib.pyplot as plt

# =========================
# App / DB
# =========================
app = FastAPI(title="ERP Bot API", version="2.0")

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

BASE_URL = os.getenv("BASE_URL", "https://erp-bot-api.onrender.com")  # 可不設，預設你現在的 Render 網址

# =========================
# LINE
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

def verify_line_signature(body_bytes: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)

async def line_reply(reply_token: str, messages: list[dict]):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"replyToken": reply_token, "messages": messages}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            print("LINE reply error:", r.status_code, r.text)

# =========================
# Gemini (Google GenAI SDK)
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 延遲 import：避免沒裝好套件時整個 service 起不來
def get_gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Gemini client init error:", str(e))
        return None

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # 官方範例用這個 :contentReference[oaicite:2]{index=2}

# =========================
# In-memory stores (Render 會重啟就清掉，先能用再說)
# =========================
# chat memory: user_id -> list[(role, text)]
CHAT_MEM: Dict[str, List[Tuple[str, str]]] = {}

# chart store: chart_id -> (bytes, mime, expire_ts)
CHARTS: Dict[str, Tuple[bytes, str, float]] = {}

def now_ts() -> float:
    return time.time()

def cleanup_charts():
    t = now_ts()
    expired = [k for k, (_, _, exp) in CHARTS.items() if exp <= t]
    for k in expired:
        del CHARTS[k]

def push_mem(user_id: str, role: str, text_: str, keep: int = 12):
    hist = CHAT_MEM.get(user_id, [])
    hist.append((role, text_))
    if len(hist) > keep:
        hist = hist[-keep:]
    CHAT_MEM[user_id] = hist

# =========================
# DB helpers (安全：只提供白名單查詢)
# =========================
def db_sales_summary(year: int) -> dict:
    sql = text("""
        SELECT
            :year AS year,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"year": year}).mappings().first()
    return dict(row) if row else {"year": year, "rows": 0, "total_qty": 0, "total_amount": 0}

def db_purchase_summary(year: int) -> dict:
    sql = text("""
        SELECT
            :year AS year,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM purchase
        WHERE year = :year
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"year": year}).mappings().first()
    return dict(row) if row else {"year": year, "rows": 0, "total_qty": 0, "total_amount": 0}

def db_sales_top_products(year: int, n: int) -> list[dict]:
    sql = text("""
        SELECT
            product,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY product
        ORDER BY total_qty DESC, rows DESC, product ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()
    return [dict(r) for r in rows]

def db_sales_top_customers(year: int, n: int) -> list[dict]:
    sql = text("""
        SELECT
            customer,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY customer
        ORDER BY total_qty DESC, rows DESC, customer ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()
    return [dict(r) for r in rows]

def db_sales_search(q: str, year: Optional[int], limit: int = 20) -> list[dict]:
    where_year = "AND year = :year" if year is not None else ""
    sql = text(f"""
        SELECT date, year, customer, product, quantity, amount
        FROM sales
        WHERE (customer ILIKE :pat OR product ILIKE :pat)
        {where_year}
        ORDER BY date DESC
        LIMIT :limit
    """)
    params = {"pat": f"%{q}%", "limit": limit}
    if year is not None:
        params["year"] = year
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]

def db_purchase_top_products(year: int, n: int) -> list[dict]:
    sql = text("""
        SELECT
            product,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM purchase
        WHERE year = :year
        GROUP BY product
        ORDER BY total_qty DESC, rows DESC, product ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()
    return [dict(r) for r in rows]

def db_purchase_search(q: str, year: Optional[int], limit: int = 20) -> list[dict]:
    where_year = "AND year = :year" if year is not None else ""
    sql = text(f"""
        SELECT date, year, supplier, product, quantity, amount
        FROM purchase
        WHERE (supplier ILIKE :pat OR product ILIKE :pat)
        {where_year}
        ORDER BY date DESC
        LIMIT :limit
    """)
    params = {"pat": f"%{q}%", "limit": limit}
    if year is not None:
        params["year"] = year
    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]

# =========================
# Chart generation (A 方法：回 LINE 圖)
# =========================
def make_bar_chart(title: str, labels: list[str], values: list[float]) -> bytes:
    plt.figure()
    plt.title(title)
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    return buf.getvalue()

def save_chart_bytes(png_bytes: bytes, ttl_sec: int = 300) -> str:
    cleanup_charts()
    chart_id = hashlib.sha256(png_bytes + str(now_ts()).encode("utf-8")).hexdigest()[:16]
    CHARTS[chart_id] = (png_bytes, "image/png", now_ts() + ttl_sec)
    return chart_id

@app.get("/charts/{chart_id}.png")
def get_chart(chart_id: str):
    cleanup_charts()
    if chart_id not in CHARTS:
        return JSONResponse({"detail": "Not Found"}, status_code=404)
    data, mime, exp = CHARTS[chart_id]
    return Response(content=data, media_type=mime)

# =========================
# Intent routing by Gemini
# =========================
ROUTER_SYSTEM = """
你是一個 ERP 助理，能跟使用者自然聊天，也能讀取 ERP 後端資料(銷售/進貨)並做簡單分析。
你必須輸出「純 JSON」(不要 markdown)，格式如下：

{
  "type": "chat" | "db" | "chart",
  "chat": {"text": "..."}                 // type=chat
  "db": {"action": "...", "params": {...}} // type=db
  "chart": {"action": "...", "params": {...}, "caption": "..."} // type=chart
}

允許的 db/chart action 白名單如下（只能選這些）：
- sales_summary: {year:int}
- purchase_summary: {year:int}
- sales_top_products: {year:int, n:int}
- sales_top_customers: {year:int, n:int}
- sales_search: {q:str, year:int|null}
- purchase_top_products: {year:int, n:int}
- purchase_search: {q:str, year:int|null}

規則：
1) 使用者只是聊天/閒聊/問你是誰 -> type=chat，正常回覆。
2) 使用者問銷售/進貨數據 -> type=db 或 type=chart。
3) 使用者要求「圖表/趨勢/長條圖/比較」-> type=chart（優先）。
4) 若缺少年份，合理猜：若問題有提到年份用它；沒提就回 chat 追問（但仍輸出 JSON）。
5) 回覆要簡短、像真人，不要機械式指令教學。
"""

def safe_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        x = int(v)
        return max(lo, min(hi, x))
    except Exception:
        return default

async def gemini_route(user_id: str, user_text: str) -> dict:
    client = get_gemini_client()
    if client is None:
        return {"type": "chat", "chat": {"text": "我目前 AI 模型沒設定好，但你可以先問我銷售/進貨資料或叫我做排行/搜尋。"}}

    # 組對話上下文（讓它更像聊天）
    hist = CHAT_MEM.get(user_id, [])
    # 只帶最近幾輪，避免爆 token
    context_lines = []
    for role, txt in hist[-8:]:
        context_lines.append(f"{role}: {txt}")
    context = "\n".join(context_lines)

    prompt = f"""{ROUTER_SYSTEM}

【對話上下文】
{context}

【使用者輸入】
{user_text}

請輸出純 JSON："""

    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        raw = (resp.text or "").strip()
        # 有些模型會在前後夾雜文字，這裡硬抓第一個 JSON 區塊
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return {"type": "chat", "chat": {"text": "我看得懂，但我需要你再說清楚一點：你想查銷售還是進貨？要哪一年？"}}
        obj = json.loads(m.group(0))
        return obj
    except Exception as e:
        print("Gemini route error:", str(e))
        # 不要 500，退回可用模式
        return {"type": "chat", "chat": {"text": "我剛剛 AI 有點連不上，但你可以直接說：例如「2025 銷售總覽」或「2025 銷售前10產品」或「銷售搜尋 ABC」。"}}

# =========================
# Execute actions
# =========================
def run_db_action(action: str, params: dict) -> dict:
    if action == "sales_summary":
        year = safe_int(params.get("year"), 2025, 2000, 2100)
        return db_sales_summary(year)
    if action == "purchase_summary":
        year = safe_int(params.get("year"), 2025, 2000, 2100)
        return db_purchase_summary(year)
    if action == "sales_top_products":
        year = safe_int(params.get("year"), 2025, 2000, 2100)
        n = safe_int(params.get("n"), 10, 1, 50)
        return {"year": year, "top_products": db_sales_top_products(year, n)}
    if action == "sales_top_customers":
        year = safe_int(params.get("year"), 2025, 2000, 2100)
        n = safe_int(params.get("n"), 10, 1, 50)
        return {"year": year, "top_customers": db_sales_top_customers(year, n)}
    if action == "sales_search":
        q = str(params.get("q") or "").strip()
        year = params.get("year", None)
        year_val = safe_int(year, 0, 2000, 2100) if year is not None else None
        return {"q": q, "year": year_val, "rows": db_sales_search(q, year_val, 20)}
    if action == "purchase_top_products":
        year = safe_int(params.get("year"), 2025, 2000, 2100)
        n = safe_int(params.get("n"), 10, 1, 50)
        return {"year": year, "top_products": db_purchase_top_products(year, n)}
    if action == "purchase_search":
        q = str(params.get("q") or "").strip()
        year = params.get("year", None)
        year_val = safe_int(year, 0, 2000, 2100) if year is not None else None
        return {"q": q, "year": year_val, "rows": db_purchase_search(q, year_val, 20)}

    return {"error": "unknown_action"}

def format_summary(title: str, d: dict) -> str:
    return (
        f"{title}\n"
        f"年：{d.get('year')}\n"
        f"筆數：{d.get('rows')}\n"
        f"總數量：{d.get('total_qty')}\n"
        f"總金額：{d.get('total_amount')}\n"
    )

# =========================
# Routes
# =========================
@app.get("/health")
def health():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar()
    return {"ok": True, "db": v}

@app.get("/")
def root():
    return {"ok": True, "hint": "Use /health or LINE webhook /line/webhook"}

@app.post("/line/webhook")
async def line_webhook(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("x-line-signature", "")

    if not verify_line_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    body = json.loads(body_bytes.decode("utf-8"))
    events = body.get("events", [])

    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        if msg.get("type") != "text":
            continue

        reply_token = ev.get("replyToken")
        user_text = msg.get("text", "")
        user_id = (ev.get("source", {}) or {}).get("userId", "unknown")

        # 記憶：使用者講了什麼
        push_mem(user_id, "user", user_text)

        route = await gemini_route(user_id, user_text)

        # type=chat
        if route.get("type") == "chat":
            text_out = (route.get("chat", {}) or {}).get("text", "").strip()
            if not text_out:
                text_out = "我在～你想聊什麼？也可以問我銷售/進貨、排行、搜尋、或叫我畫圖。"
            push_mem(user_id, "assistant", text_out)
            await line_reply(reply_token, [{"type": "text", "text": text_out[:4900]}])
            continue

        # type=db
        if route.get("type") == "db":
            dbinfo = route.get("db", {}) or {}
            action = str(dbinfo.get("action") or "")
            params = dbinfo.get("params", {}) or {}
            data = run_db_action(action, params)

            # 依 action 友善輸出
            if action == "sales_summary":
                text_out = format_summary("【銷售總覽】", data)
            elif action == "purchase_summary":
                text_out = format_summary("【進貨總覽】", data)
            elif action == "sales_top_products":
                rows = data.get("top_products", [])
                year = data.get("year")
                lines = [f"【銷售 前{len(rows)} 產品｜依數量】(年 {year})"]
                for i, r in enumerate(rows, 1):
                    lines.append(f"{i}. {r.get('product')}｜數量 {r.get('total_qty')}｜金額 {r.get('total_amount')}")
                text_out = "\n".join(lines)
            elif action == "sales_top_customers":
                rows = data.get("top_customers", [])
                year = data.get("year")
                lines = [f"【銷售 前{len(rows)} 客戶｜依數量】(年 {year})"]
                for i, r in enumerate(rows, 1):
                    lines.append(f"{i}. {r.get('customer')}｜數量 {r.get('total_qty')}｜金額 {r.get('total_amount')}")
                text_out = "\n".join(lines)
            elif action == "sales_search":
                rows = data.get("rows", [])
                q = data.get("q")
                year = data.get("year")
                if not rows:
                    text_out = f"【銷售搜尋】找不到：{q}"
                else:
                    lines = [f"【銷售搜尋】{q}（年：{year if year else '不限'}）"]
                    for r in rows[:20]:
                        lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
                    text_out = "\n".join(lines)
            elif action == "purchase_top_products":
                rows = data.get("top_products", [])
                year = data.get("year")
                lines = [f"【進貨 前{len(rows)} 產品｜依數量】(年 {year})"]
                for i, r in enumerate(rows, 1):
                    lines.append(f"{i}. {r.get('product')}｜數量 {r.get('total_qty')}｜金額 {r.get('total_amount')}")
                text_out = "\n".join(lines)
            elif action == "purchase_search":
                rows = data.get("rows", [])
                q = data.get("q")
                year = data.get("year")
                if not rows:
                    text_out = f"【進貨搜尋】找不到：{q}"
                else:
                    lines = [f"【進貨搜尋】{q}（年：{year if year else '不限'}）"]
                    for r in rows[:20]:
                        lines.append(f"{r['date']}｜{r['supplier']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
                    text_out = "\n".join(lines)
            else:
                text_out = "我查到資料了，但不知道怎麼整理輸出（action 不在預期內）。"

            push_mem(user_id, "assistant", text_out)
            await line_reply(reply_token, [{"type": "text", "text": text_out[:4900]}])
            continue

        # type=chart
        if route.get("type") == "chart":
            cinfo = route.get("chart", {}) or {}
            action = str(cinfo.get("action") or "")
            params = cinfo.get("params", {}) or {}
            caption = str(cinfo.get("caption") or "").strip() or "圖表如下"

            # 我們先用 db action 拿資料，再畫圖
            # 目前先支援：sales_top_products / sales_top_customers / purchase_top_products
            data = run_db_action(action, params)

            png = None
            if action == "sales_top_products":
                rows = data.get("top_products", [])
                year = data.get("year")
                labels = [r.get("product", "")[:10] for r in rows]
                values = [float(r.get("total_qty") or 0) for r in rows]
                png = make_bar_chart(f"Sales Top Products {year}", labels, values)

            elif action == "sales_top_customers":
                rows = data.get("top_customers", [])
                year = data.get("year")
                labels = [r.get("customer", "")[:10] for r in rows]
                values = [float(r.get("total_qty") or 0) for r in rows]
                png = make_bar_chart(f"Sales Top Customers {year}", labels, values)

            elif action == "purchase_top_products":
                rows = data.get("top_products", [])
                year = data.get("year")
                labels = [r.get("product", "")[:10] for r in rows]
                values = [float(r.get("total_qty") or 0) for r in rows]
                png = make_bar_chart(f"Purchase Top Products {year}", labels, values)

            if not png:
                text_out = "我懂你要圖，但我目前只會畫：銷售/進貨的前N排行長條圖。你可以說「畫 2025 銷售前10產品圖」。"
                push_mem(user_id, "assistant", text_out)
                await line_reply(reply_token, [{"type": "text", "text": text_out[:4900]}])
                continue

            chart_id = save_chart_bytes(png, ttl_sec=300)
            img_url = f"{BASE_URL}/charts/{chart_id}.png"

            # LINE 圖片訊息
            messages = [
                {"type": "text", "text": caption[:4900]},
                {
                    "type": "image",
                    "originalContentUrl": img_url,
                    "previewImageUrl": img_url,
                },
            ]
            push_mem(user_id, "assistant", caption)
            await line_reply(reply_token, messages)
            continue

        # unknown
        await line_reply(reply_token, [{"type": "text", "text": "我有點看不懂你的意思，你可以再講一次～"}])

    return {"status": "ok"}

# 你之前拿到 columns 的 endpoint（保留給你 debug）
@app.get("/debug/columns")
def debug_columns():
    sql = text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name IN ('sales', 'purchase')
        ORDER BY table_name, ordinal_position
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return {"ok": True, "columns": [dict(r) for r in rows]}
