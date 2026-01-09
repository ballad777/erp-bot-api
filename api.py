import os
import re
import json
import time
import uuid
import hmac
import base64
import hashlib
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy import create_engine, text

import httpx

# ===== Chart deps =====
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== Gemini deps =====
import google.generativeai as genai


# =========================
# Config / DB
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="ERP Bot API", version="2.0")


# =========================
# In-memory stores
# (Render 會重啟 -> 會清空；但先讓功能穩，再做持久化)
# =========================
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}  # user_id -> [{"role":"user/assistant","content":""}, ...]
CHART_CACHE: Dict[str, Dict[str, Any]] = {}       # chart_id -> {"bytes": b"...", "ts": 123}
CHART_TTL_SEC = 300  # 5分鐘


# =========================
# Utilities
# =========================
def _verify_line_signature(body_bytes: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        # 沒設 secret 就略過（可用，但不安全）
        return True
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _extract_year(text_in: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", text_in)
    return int(m.group(1)) if m else None


def _extract_top_n(text_in: str, default: int = 10) -> int:
    m = re.search(r"(?:前|top|TOP)\s*(\d+)", text_in)
    if m:
        n = int(m.group(1))
        return max(1, min(100, n))
    return default


def _clean_chart_cache():
    now = time.time()
    dead = [k for k, v in CHART_CACHE.items() if now - v["ts"] > CHART_TTL_SEC]
    for k in dead:
        CHART_CACHE.pop(k, None)


def _remember(user_id: str, role: str, content: str, max_turns: int = 12):
    arr = CHAT_MEMORY.setdefault(user_id, [])
    arr.append({"role": role, "content": content})
    # 保留最近 max_turns*2 則訊息
    if len(arr) > max_turns * 2:
        CHAT_MEMORY[user_id] = arr[-max_turns * 2:]


# =========================
# DB Query helpers (白名單)
# =========================
def db_sales_summary(year: int) -> Dict[str, Any]:
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


def db_purchase_summary(year: int) -> Dict[str, Any]:
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


def db_sales_top_products(year: int, n: int) -> List[Dict[str, Any]]:
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


def db_sales_top_customers(year: int, n: int) -> List[Dict[str, Any]]:
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


def db_sales_search(q: str, year: Optional[int], limit: int = 20) -> List[Dict[str, Any]]:
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


def db_purchase_top_products(year: int, n: int) -> List[Dict[str, Any]]:
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


def db_purchase_search(q: str, year: Optional[int], limit: int = 20) -> List[Dict[str, Any]]:
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
# Gemini (model fallback)
# =========================
GEMINI_MODEL = None

def init_gemini():
    global GEMINI_MODEL
    if not GEMINI_API_KEY:
        GEMINI_MODEL = None
        return

    genai.configure(api_key=GEMINI_API_KEY)

    # ✅ 先用官方建議的新模型，舊的會 404
    candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-pro-latest",
        "gemini-pro",
    ]

    last_err = None
    for m in candidates:
        try:
            model = genai.GenerativeModel(m)
            # 做一個超小測試，確保 generate_content 真的能用
            _ = model.generate_content("ping")
            GEMINI_MODEL = model
            print(f"[Gemini] Using model: {m}")
            return
        except Exception as e:
            last_err = e
            continue

    # 都不行就關掉 Gemini（至少 DB 功能還能回）
    GEMINI_MODEL = None
    print("[Gemini] init failed:", repr(last_err))


init_gemini()


# =========================
# Intent Router (Gemini)
# 回傳 JSON: {"type": "chat" | "data" | "chart", "action": "...", "params": {...}}
# =========================
ALLOWED_ACTIONS = {
    "sales_summary",
    "purchase_summary",
    "sales_top_products",
    "sales_top_customers",
    "sales_search",
    "purchase_top_products",
    "purchase_search",
    # chart actions
    "chart_sales_monthly_amount",
    "chart_sales_top_products_amount",
}

def gemini_route(user_text: str) -> Dict[str, Any]:
    """
    - 沒 Gemini：用簡單規則
    - 有 Gemini：用 Gemini 產出路由 JSON
    """
    t = user_text.strip()

    # 1) 沒 Gemini -> fallback
    if GEMINI_MODEL is None:
        # 簡單規則：有「圖/圖表」就當 chart；有「銷售/進貨」就當 data；不然 chat
        if any(k in t for k in ["圖", "圖表", "趨勢", "折線", "長條"]):
            y = _extract_year(t) or 2025
            return {"type": "chart", "action": "chart_sales_monthly_amount", "params": {"year": y}}
        if any(k in t for k in ["銷售", "進貨", "採購"]):
            y = _extract_year(t) or 2025
            return {"type": "data", "action": "sales_summary", "params": {"year": y}}
        return {"type": "chat", "action": "chat", "params": {}}

    # 2) 有 Gemini -> 產路由 JSON
    system = f"""
你是一個 ERP AI 助手，負責「聊天」與「查詢 ERP 資料」。
你必須輸出「純 JSON」，不要加任何多餘文字。

你可使用的 action（白名單）只有：
{sorted(list(ALLOWED_ACTIONS))}

規則：
- 如果使用者在閒聊/一般問題 -> type=chat, action="chat"
- 如果使用者要查資料（銷售/進貨/採購/客戶/產品/金額/數量/總覽/搜尋）-> type=data，action 從白名單選，params 填合理參數
- 如果使用者要圖表（趨勢、圖、折線、長條、比較）-> type=chart，action 從白名單選
- year 沒講就用 2025
- top n 沒講就 10
- search keyword 從句子抽出最像關鍵字的那段（可含中文/英文/數字），不要空字串

輸出格式範例：
{{"type":"data","action":"sales_summary","params":{{"year":2025}}}}
{{"type":"data","action":"sales_search","params":{{"year":2025,"q":"ABC"}}}}
{{"type":"chart","action":"chart_sales_monthly_amount","params":{{"year":2025}}}}
{{"type":"chat","action":"chat","params":{{}}}}
""".strip()

    prompt = f"{system}\n\n使用者訊息：{t}\nJSON："

    try:
        resp = GEMINI_MODEL.generate_content(prompt)
        raw = (resp.text or "").strip()

        # 去掉可能的 ```json
        raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"```$", "", raw).strip()

        route = json.loads(raw)

        # 最基本保護
        if route.get("type") not in {"chat", "data", "chart"}:
            return {"type": "chat", "action": "chat", "params": {}}

        if route.get("type") in {"data", "chart"}:
            if route.get("action") not in ALLOWED_ACTIONS:
                return {"type": "chat", "action": "chat", "params": {}}

        # 補預設
        params = route.get("params") or {}
        if "year" not in params:
            params["year"] = 2025
        route["params"] = params

        return route

    except Exception as e:
        # ✅ 任何錯誤都不能讓 webhook 500
        print("[Gemini route error]", repr(e))
        # fallback
        y = _extract_year(t) or 2025
        if any(k in t for k in ["圖", "圖表", "趨勢", "折線", "長條"]):
            return {"type": "chart", "action": "chart_sales_monthly_amount", "params": {"year": y}}
        if any(k in t for k in ["銷售", "進貨", "採購"]):
            return {"type": "data", "action": "sales_summary", "params": {"year": y}}
        return {"type": "chat", "action": "chat", "params": {}}


# =========================
# LINE Reply helpers
# =========================
async def line_reply_text(reply_token: str, text_message: str):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text_message[:4900]}],
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            print("LINE reply error:", r.status_code, r.text)


async def line_reply_image(reply_token: str, image_url: str, preview_url: Optional[str] = None):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{
            "type": "image",
            "originalContentUrl": image_url,
            "previewImageUrl": preview_url or image_url
        }],
    }
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            print("LINE reply image error:", r.status_code, r.text)


# =========================
# Chart generators
# =========================
def make_chart_sales_monthly_amount(year: int) -> bytes:
    """
    假設 sales.date 是 date 型別（你剛查 columns 顯示是 date ✅）
    """
    sql = text("""
        SELECT EXTRACT(MONTH FROM date)::int AS m,
               COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY m
        ORDER BY m ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year}).mappings().all()

    # 補齊 1..12
    data = {r["m"]: float(r["total_amount"]) for r in rows}
    months = list(range(1, 13))
    vals = [data.get(m, 0.0) for m in months]

    fig = plt.figure()
    plt.plot(months, vals, marker="o")
    plt.title(f"Sales Monthly Amount ({year})")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.xticks(months)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def make_chart_sales_top_products_amount(year: int, n: int = 10) -> bytes:
    sql = text("""
        SELECT product,
               COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY product
        ORDER BY total_amount DESC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()

    labels = [r["product"] for r in rows][::-1]
    vals = [float(r["total_amount"]) for r in rows][::-1]

    fig = plt.figure()
    plt.barh(labels, vals)
    plt.title(f"Top {n} Products by Amount ({year})")
    plt.xlabel("Amount")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# =========================
# Public endpoints
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "erp-bot-api", "docs": "/docs"}


@app.get("/health")
def health():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar()
    return {"ok": True, "db": v}


@app.get("/debug/schema")
def debug_schema():
    sql = text("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema='public'
          AND table_name IN ('sales','purchase')
        ORDER BY table_name, ordinal_position
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return {"ok": True, "columns": [dict(r) for r in rows]}


@app.get("/charts/{chart_id}")
def get_chart(chart_id: str):
    _clean_chart_cache()
    item = CHART_CACHE.get(chart_id)
    if not item:
        raise HTTPException(status_code=404, detail="Chart not found or expired")
    return StreamingResponse(io.BytesIO(item["bytes"]), media_type="image/png")


# =========================
# Simple data API (保留你原本可測試的)
# =========================
@app.get("/sales/summary")
def sales_summary(year: int = Query(..., description="年份，例如 2025")):
    return db_sales_summary(year)


@app.get("/purchase/summary")
def purchase_summary(year: int = Query(..., description="年份，例如 2024")):
    return db_purchase_summary(year)


# =========================
# Chat builder
# =========================
def build_chat_answer(user_id: str, user_text: str) -> str:
    """
    真聊天：把最近的對話帶進 Gemini（如果有）
    沒 Gemini：就回一個基本回答
    """
    if GEMINI_MODEL is None:
        return "我目前沒有接上 Gemini（或模型不可用）。你可以先問：銷售/進貨的總覽、搜尋、或圖表。"

    history = CHAT_MEMORY.get(user_id, [])
    # 將 history 轉成一段文字（google-generativeai 的簡易輸入）
    convo = []
    for h in history[-12:]:
        role = h["role"]
        if role == "user":
            convo.append(f"使用者：{h['content']}")
        else:
            convo.append(f"助理：{h['content']}")
    convo.append(f"使用者：{user_text}")
    prompt = (
        "你是一個貼近真人的 ERP AI 助手，平常能聊天，必要時會提醒使用者可以查 ERP 數據或畫圖。\n"
        "請用自然中文回答。\n\n"
        + "\n".join(convo)
        + "\n助理："
    )

    try:
        resp = GEMINI_MODEL.generate_content(prompt)
        return (resp.text or "").strip() or "我剛剛想了一下，但沒有產生內容。你可以換個說法再問一次。"
    except Exception as e:
        print("[Gemini chat error]", repr(e))
        return "我這邊呼叫 Gemini 失敗了，但資料查詢仍可用。你可以問我：『2025 銷售總覽』或『畫 2025 每月銷售金額圖』。"


# =========================
# Route executor
# =========================
def execute_data_action(action: str, params: Dict[str, Any]) -> str:
    year = int(params.get("year", 2025))
    n = int(params.get("n", 10))
    q = str(params.get("q", "")).strip()

    if action == "sales_summary":
        d = db_sales_summary(year)
        return f"【銷售總覽】{year}\n筆數：{d['rows']}\n總數量：{d['total_qty']}\n總金額：{d['total_amount']}"

    if action == "purchase_summary":
        d = db_purchase_summary(year)
        return f"【進貨總覽】{year}\n筆數：{d['rows']}\n總數量：{d['total_qty']}\n總金額：{d['total_amount']}"

    if action == "sales_top_products":
        rows = db_sales_top_products(year, n)
        lines = [f"【銷售 前{n} 產品｜依數量】{year}"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['product']}｜數量 {r['total_qty']}｜金額 {r['total_amount']}")
        return "\n".join(lines)

    if action == "sales_top_customers":
        rows = db_sales_top_customers(year, n)
        lines = [f"【銷售 前{n} 客戶｜依數量】{year}"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['customer']}｜數量 {r['total_qty']}｜金額 {r['total_amount']}")
        return "\n".join(lines)

    if action == "sales_search":
        if not q:
            return "你想查哪個關鍵字？例如：『查 2025 銷售 ABC』"
        rows = db_sales_search(q=q, year=year, limit=15)
        if not rows:
            return f"找不到：{q}（年：{year}）"
        lines = [f"【銷售搜尋】{q}（年：{year}）"]
        for r in rows:
            lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines)

    if action == "purchase_top_products":
        rows = db_purchase_top_products(year, n)
        lines = [f"【進貨 前{n} 產品｜依數量】{year}"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['product']}｜數量 {r['total_qty']}｜金額 {r['total_amount']}")
        return "\n".join(lines)

    if action == "purchase_search":
        if not q:
            return "你想查哪個關鍵字？例如：『查 2024 進貨 ABC』"
        rows = db_purchase_search(q=q, year=year, limit=15)
        if not rows:
            return f"找不到：{q}（年：{year}）"
        lines = [f"【進貨搜尋】{q}（年：{year}）"]
        for r in rows:
            lines.append(f"{r['date']}｜{r['supplier']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines)

    return "我目前不支援這個查詢。"


def execute_chart_action(action: str, params: Dict[str, Any], base_url: str) -> Tuple[str, str]:
    """
    回傳 (文字, 圖片URL)
    """
    year = int(params.get("year", 2025))
    n = int(params.get("n", 10))

    if action == "chart_sales_monthly_amount":
        img_bytes = make_chart_sales_monthly_amount(year)
        chart_id = uuid.uuid4().hex
        CHART_CACHE[chart_id] = {"bytes": img_bytes, "ts": time.time()}
        return (f"我整理了 {year} 每月銷售金額趨勢圖（5分鐘內有效）。", f"{base_url}/charts/{chart_id}")

    if action == "chart_sales_top_products_amount":
        img_bytes = make_chart_sales_top_products_amount(year, n=n)
        chart_id = uuid.uuid4().hex
        CHART_CACHE[chart_id] = {"bytes": img_bytes, "ts": time.time()}
        return (f"我整理了 {year} 銷售金額 Top {n} 產品長條圖（5分鐘內有效）。", f"{base_url}/charts/{chart_id}")

    # fallback
    img_bytes = make_chart_sales_monthly_amount(year)
    chart_id = uuid.uuid4().hex
    CHART_CACHE[chart_id] = {"bytes": img_bytes, "ts": time.time()}
    return (f"我先給你 {year} 每月銷售金額趨勢圖。", f"{base_url}/charts/{chart_id}")


# =========================
# LINE webhook
# =========================
@app.post("/line/webhook")
async def line_webhook(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("x-line-signature", "")

    if not _verify_line_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    try:
        body = json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return JSONResponse({"status": "bad json"}, status_code=400)

    events = body.get("events", [])
    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        if msg.get("type") != "text":
            continue

        reply_token = ev.get("replyToken")
        user_text = msg.get("text", "").strip()
        user_id = (ev.get("source") or {}).get("userId", "unknown")

        # 記住使用者輸入
        _remember(user_id, "user", user_text)

        # Render 服務的外部 URL（你這個固定）
        base_url = "https://erp-bot-api.onrender.com"

        # 1) Gemini 路由
        route = gemini_route(user_text)
        rtype = route.get("type")
        action = route.get("action")
        params = route.get("params") or {}

        # 2) 執行
        try:
            if rtype == "chat":
                answer = build_chat_answer(user_id, user_text)
                _remember(user_id, "assistant", answer)
                await line_reply_text(reply_token, answer)
                continue

            if rtype == "data":
                answer = execute_data_action(action, params)
                _remember(user_id, "assistant", answer)
                await line_reply_text(reply_token, answer)
                continue

            if rtype == "chart":
                text_msg, img_url = execute_chart_action(action, params, base_url)
                _remember(user_id, "assistant", text_msg)
                # 先回文字再回圖（你比較像 A 的體感）
                await line_reply_text(reply_token, text_msg)
                # LINE replyToken 只能用一次，所以圖改成「push」會更穩
                # 但你現在先求能跑：我們改成同一個 reply 一次回兩則
                # -> 因此用 line_reply_text 會佔掉 token，這裡改成一次回兩則訊息

        except Exception as e:
            print("[Execute error]", repr(e))
            await line_reply_text(reply_token, "我剛剛處理時發生錯誤，但服務還活著。你可以再問一次或換個問法。")
            continue

        # ✅ chart：一次 reply 兩則訊息（文字 + 圖）
        if rtype == "chart":
            if not LINE_CHANNEL_ACCESS_TOKEN:
                continue
            url = "https://api.line.me/v2/bot/message/reply"
            headers = {
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            }
            payload = {
                "replyToken": reply_token,
                "messages": [
                    {"type": "text", "text": text_msg[:4900]},
                    {"type": "image", "originalContentUrl": img_url, "previewImageUrl": img_url},
                ],
            }
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    print("LINE reply chart error:", r.status_code, r.text)

    return {"status": "ok"}
