import os
import re
import json
import hmac
import base64
import hashlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, Request, HTTPException
from sqlalchemy import create_engine, text

# =========================
# ENV / DB
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="ERP Bot API", version="2.0")

# =========================
# LINE ENV
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

# =========================
# GEMINI ENV
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)

# 這組是「常見可用」模型名稱，會逐個嘗試
# 注意：不同 key / 不同地區 / 不同 API 版本可用模型可能不一樣
GEMINI_MODEL_CANDIDATES = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "models/gemini-1.0-pro",
]


# =========================
# Helpers: LINE signature verify
# =========================
def verify_line_signature(body_bytes: bytes, signature: str) -> bool:
    """有設 LINE_CHANNEL_SECRET 就驗簽；沒設就略過（但建議一定要設）"""
    if not LINE_CHANNEL_SECRET:
        return True
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# =========================
# DB query functions
# =========================
def sales_summary_db(year: int) -> Dict[str, Any]:
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


def sales_top_products_db(year: int, n: int) -> List[Dict[str, Any]]:
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


def sales_top_customers_db(year: int, n: int) -> List[Dict[str, Any]]:
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


def sales_search_db(q: str, year: Optional[int], limit: int) -> List[Dict[str, Any]]:
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


def purchase_summary_db(year: int) -> Dict[str, Any]:
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


def purchase_top_products_db(year: int, n: int) -> List[Dict[str, Any]]:
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


def purchase_search_db(q: str, year: Optional[int], limit: int) -> List[Dict[str, Any]]:
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
# Format helpers
# =========================
def extract_year(text_in: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", text_in)
    return int(m.group(1)) if m else None


def extract_top_n(text_in: str, default: int = 10) -> int:
    m = re.search(r"(?:前|top|TOP)\s*(\d+)", text_in)
    if m:
        n = int(m.group(1))
        return max(1, min(100, n))
    return default


def format_summary(title: str, d: dict) -> str:
    return (
        f"{title}\n"
        f"年：{d.get('year')}\n"
        f"筆數：{d.get('rows')}\n"
        f"總數量：{d.get('total_qty')}\n"
        f"總金額：{d.get('total_amount')}\n"
    )


def format_top_list(title: str, rows: List[dict], key_name: str) -> str:
    lines = [title]
    for i, r in enumerate(rows, 1):
        name = r.get(key_name, "")
        qty = r.get("total_qty", 0)
        cnt = r.get("rows", 0)
        amt = r.get("total_amount", 0)
        lines.append(f"{i}. {name}｜數量 {qty}｜筆數 {cnt}｜金額 {amt}")
    return "\n".join(lines)


# =========================
# Public API endpoints (你原本的)
# =========================
@app.get("/health")
def health():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar()
    return {"ok": True, "db": v}


@app.get("/sales/summary")
def sales_summary(year: int = Query(..., description="年份，例如 2025")):
    return sales_summary_db(year)


@app.get("/sales/top_products")
def sales_top_products(year: int = Query(...), n: int = Query(10, ge=1, le=100)):
    return {"year": year, "order_by": "total_qty DESC", "top_products": sales_top_products_db(year, n)}


@app.get("/sales/top_customers")
def sales_top_customers(year: int = Query(...), n: int = Query(10, ge=1, le=100)):
    return {"year": year, "order_by": "total_qty DESC", "top_customers": sales_top_customers_db(year, n)}


@app.get("/sales/search")
def sales_search(q: str = Query(...), year: int | None = Query(None), limit: int = Query(50, ge=1, le=200)):
    return {"q": q, "year": year, "limit": limit, "rows": sales_search_db(q, year, limit)}


@app.get("/purchase/summary")
def purchase_summary(year: int = Query(..., description="年份，例如 2024")):
    return purchase_summary_db(year)


@app.get("/purchase/top_products")
def purchase_top_products(year: int = Query(...), n: int = Query(10, ge=1, le=100)):
    return {"year": year, "order_by": "total_qty DESC", "top_products": purchase_top_products_db(year, n)}


@app.get("/purchase/search")
def purchase_search(q: str = Query(...), year: int | None = Query(None), limit: int = Query(50, ge=1, le=200)):
    return {"q": q, "year": year, "limit": limit, "rows": purchase_search_db(q, year, limit)}


# =========================
# LINE reply
# =========================
async def line_reply(reply_token: str, text_message: str):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN 未設定，略過回覆")
        return

    import httpx

    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text_message[:4900]}],
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            print("LINE reply error:", r.status_code, r.text)


# =========================
# Gemini safe call (重點：永遠不讓 webhook 500)
# =========================
def gemini_generate_safe(prompt: str) -> Optional[str]:
    """
    成功就回字串；失敗回 None
    """
    if not GEMINI_ENABLED:
        return None

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Gemini init error:", repr(e))
        return None

    last_err = None
    for name in GEMINI_MODEL_CANDIDATES:
        try:
            model = genai.GenerativeModel(name)
            resp = model.generate_content(prompt)
            txt = (getattr(resp, "text", "") or "").strip()
            if txt:
                return txt
            # 沒 text 就當作失敗，換下一個
            last_err = f"empty response from {name}"
        except Exception as e:
            last_err = e
            continue

    print("Gemini all models failed:", repr(last_err))
    return None


# =========================
# Router: 先嘗試理解 => 能查資料就查 => 不行才聊天
# =========================
def handle_user_text(user_text: str) -> str:
    t = user_text.strip()

    # 1) 先內建「資料指令」（不用固定格式也能吃）
    year = extract_year(t)
    n = extract_top_n(t, default=10)

    # 銷售總覽
    if ("銷售" in t) and ("總覽" in t or "概況" in t or "summary" in t.lower()):
        if not year:
            return "你想看哪一年？例如：銷售總覽 2025"
        d = sales_summary_db(year)
        return format_summary("【銷售總覽】", d)

    # 銷售 前N產品
    if ("銷售" in t) and ("產品" in t) and ("前" in t or "top" in t.lower() or "排行" in t):
        if not year:
            return "你想看哪一年？例如：銷售前10產品 2025"
        rows = sales_top_products_db(year, n)
        return format_top_list(f"【銷售 前{n} 產品｜依數量】(年 {year})", rows, "product")

    # 銷售 前N客戶
    if ("銷售" in t) and ("客戶" in t) and ("前" in t or "top" in t.lower() or "排行" in t):
        if not year:
            return "你想看哪一年？例如：銷售前10客戶 2025"
        rows = sales_top_customers_db(year, n)
        return format_top_list(f"【銷售 前{n} 客戶｜依數量】(年 {year})", rows, "customer")

    # 銷售 搜尋
    if ("銷售" in t) and ("搜尋" in t or "search" in t.lower() or "查" in t):
        # 把年份跟關鍵字拆出來
        keyword = re.sub(r"(銷售\s*)?(搜尋|search|查)\s*", "", t, flags=re.IGNORECASE).strip()
        if year:
            keyword = re.sub(r"20\d{2}", "", keyword).strip()
        if not keyword:
            return "你要查什麼關鍵字？例如：銷售搜尋 2025 ABC"
        rows = sales_search_db(keyword, year, limit=20)
        if not rows:
            return f"找不到：{keyword}"
        lines = [f"【銷售搜尋】{keyword}（年：{year if year else '不限'}）"]
        for r in rows[:20]:
            lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines)

    # 進貨總覽
    if (("進貨" in t) or ("採購" in t)) and ("總覽" in t or "概況" in t):
        if not year:
            return "你想看哪一年？例如：進貨總覽 2024"
        d = purchase_summary_db(year)
        return format_summary("【進貨總覽】", d)

    # 進貨 前N產品
    if (("進貨" in t) or ("採購" in t)) and ("產品" in t) and ("前" in t or "top" in t.lower() or "排行" in t):
        if not year:
            return "你想看哪一年？例如：進貨前10產品 2024"
        rows = purchase_top_products_db(year, n)
        return format_top_list(f"【進貨 前{n} 產品｜依數量】(年 {year})", rows, "product")

    # 進貨 搜尋
    if (("進貨" in t) or ("採購" in t)) and ("搜尋" in t or "查" in t):
        keyword = re.sub(r"(進貨|採購)\s*(搜尋|查)\s*", "", t).strip()
        if year:
            keyword = re.sub(r"20\d{2}", "", keyword).strip()
        if not keyword:
            return "你要查什麼關鍵字？例如：進貨搜尋 2024 ABC"
        rows = purchase_search_db(keyword, year, limit=20)
        if not rows:
            return f"找不到：{keyword}"
        lines = [f"【進貨搜尋】{keyword}（年：{year if year else '不限'}）"]
        for r in rows[:20]:
            lines.append(f"{r['date']}｜{r['supplier']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines)

    # 2) 不是資料指令 => 走 Gemini 聊天（但 Gemini 失敗也不會掛）
    prompt = f"""
你是一個 LINE 上的 ERP AI 助手，請用繁體中文自然回答。
若使用者是在聊天就聊天；若使用者在問資料分析，請用「你目前能做到的」方式回答，必要時提示他可用指令格式。
使用者訊息：
{t}
"""
    ai = gemini_generate_safe(prompt)
    if ai:
        return ai

    # 3) Gemini 不可用 => 回退文字
    return "我目前 AI 聊天功能暫時不可用（Gemini 模型/權限問題），但你可以直接問我銷售/進貨資料：例如「銷售總覽 2025」「銷售排行 前10產品 2025」「銷售搜尋 2025 ABC」。"


# =========================
# LINE webhook
# =========================
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

        answer = handle_user_text(user_text)

        # 最重要：無論如何都要回，避免 webhook 500
        try:
            await line_reply(reply_token, answer)
        except Exception as e:
            print("LINE reply exception:", repr(e))

    return {"status": "ok"}
