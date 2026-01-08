import os
import re
import json
import hmac
import base64
import hashlib
from typing import Optional

from fastapi import FastAPI, Query, Request, HTTPException
from sqlalchemy import create_engine, text

# =========================
# DB
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="ERP Bot API", version="1.0")

# =========================
# LINE
# =========================
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")  # 可先不設；設了會驗簽更安全


def _verify_line_signature(body_bytes: bytes, signature: str) -> bool:
    """如果你有設 LINE_CHANNEL_SECRET，就驗簽；沒設就略過"""
    if not LINE_CHANNEL_SECRET:
        return True
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


def _extract_year(text_in: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", text_in)
    return int(m.group(1)) if m else None


def _extract_top_n(text_in: str, default: int = 10) -> int:
    # 例：前5、TOP 5、top5
    m = re.search(r"(?:前|top|TOP)\s*(\d+)", text_in)
    if m:
        n = int(m.group(1))
        return max(1, min(100, n))
    return default


def _format_kv(title: str, d: dict) -> str:
    return (
        f"{title}\n"
        f"年：{d.get('year')}\n"
        f"筆數：{d.get('rows')}\n"
        f"總數量：{d.get('total_qty')}\n"
        f"總金額：{d.get('total_amount')}\n"
    )


def _format_top_list(title: str, rows: list[dict], key_name: str) -> str:
    lines = [title]
    for i, r in enumerate(rows, 1):
        name = r.get(key_name, "")
        qty = r.get("total_qty", 0)
        cnt = r.get("rows", 0)
        amt = r.get("total_amount", 0)
        lines.append(f"{i}. {name}｜數量 {qty}｜筆數 {cnt}｜金額 {amt}")
    return "\n".join(lines)


def _sales_summary(year: int) -> dict:
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


def _sales_top_products(year: int, n: int) -> list[dict]:
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


def _sales_top_customers(year: int, n: int) -> list[dict]:
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


def _sales_search(q: str, year: Optional[int], limit: int) -> list[dict]:
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


def _purchase_summary(year: int) -> dict:
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


def _purchase_top_products(year: int, n: int) -> list[dict]:
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


def _purchase_search(q: str, year: Optional[int], limit: int) -> list[dict]:
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
# Public API (你原本的)
# =========================
@app.get("/health")
def health():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar()
    return {"ok": True, "db": v}


@app.get("/sales/summary")
def sales_summary(year: int = Query(..., description="年份，例如 2025")):
    return _sales_summary(year)


@app.get("/sales/top_products")
def sales_top_products(
    year: int = Query(..., description="年份，例如 2025"),
    n: int = Query(10, ge=1, le=100, description="回傳前 N 名")
):
    return {"year": year, "order_by": "total_qty DESC", "top_products": _sales_top_products(year, n)}


@app.get("/sales/top_customers")
def sales_top_customers(
    year: int = Query(..., description="年份，例如 2025"),
    n: int = Query(10, ge=1, le=100, description="回傳前 N 名")
):
    return {"year": year, "order_by": "total_qty DESC", "top_customers": _sales_top_customers(year, n)}


@app.get("/sales/search")
def sales_search(
    q: str = Query(..., description="模糊關鍵字（客戶/品號）"),
    year: int | None = Query(None, description="不填代表不限年份"),
    limit: int = Query(50, ge=1, le=200, description="最多回傳筆數")
):
    return {"q": q, "year": year, "limit": limit, "rows": _sales_search(q, year, limit)}


@app.get("/purchase/summary")
def purchase_summary(year: int = Query(..., description="年份，例如 2024")):
    return _purchase_summary(year)


@app.get("/purchase/top_products")
def purchase_top_products(
    year: int = Query(..., description="年份，例如 2024"),
    n: int = Query(10, ge=1, le=100, description="回傳前 N 名")
):
    return {"year": year, "order_by": "total_qty DESC", "top_products": _purchase_top_products(year, n)}


@app.get("/purchase/search")
def purchase_search(
    q: str = Query(..., description="模糊關鍵字（供應商/品號）"),
    year: int | None = Query(None, description="不填代表不限年份"),
    limit: int = Query(50, ge=1, le=200, description="最多回傳筆數")
):
    return {"q": q, "year": year, "limit": limit, "rows": _purchase_search(q, year, limit)}


# =========================
# LINE Webhook (回你要的資料)
# =========================
async def _line_reply(reply_token: str, text_message: str):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        # 沒設 token 也別讓 webhook 500
        return

    import httpx  # ← 需要 requirements.txt 加一行 httpx

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
        # 不要讓 webhook 因為回覆失敗而整個炸掉
        if r.status_code >= 400:
            print("LINE reply error:", r.status_code, r.text)


def _handle_query(user_text: str) -> str:
    t = user_text.strip()

    # 指令提示
    if t.lower() in {"help", "h", "指令", "幫助"}:
        return (
            "可用指令：\n"
            "1) 銷售總覽 2025\n"
            "2) 銷售前10產品 2025（可改前5/前20）\n"
            "3) 銷售前10客戶 2025\n"
            "4) 銷售搜尋 2025 關鍵字（或：銷售搜尋 關鍵字）\n"
            "5) 進貨總覽 2024\n"
            "6) 進貨前10產品 2024\n"
            "7) 進貨搜尋 2024 關鍵字\n"
        )

    year = _extract_year(t)
    n = _extract_top_n(t, default=10)

    # 銷售
    if "銷售" in t:
        if "總覽" in t or "summary" in t.lower():
            if not year:
                return "請帶年份，例如：銷售總覽 2025"
            d = _sales_summary(year)
            return _format_kv("【銷售總覽】", d)

        if ("前" in t or "top" in t.lower()) and "產品" in t:
            if not year:
                return "請帶年份，例如：銷售前10產品 2025"
            rows = _sales_top_products(year, n)
            return _format_top_list(f"【銷售 前{n} 產品｜依數量】(年 {year})", rows, "product")

        if ("前" in t or "top" in t.lower()) and "客戶" in t:
            if not year:
                return "請帶年份，例如：銷售前10客戶 2025"
            rows = _sales_top_customers(year, n)
            return _format_top_list(f"【銷售 前{n} 客戶｜依數量】(年 {year})", rows, "customer")

        if "搜尋" in t or "search" in t.lower():
            # 允許：銷售搜尋 2025 xxx / 銷售搜尋 xxx
            keyword = re.sub(r"(銷售\s*)?(搜尋|search)\s*", "", t, flags=re.IGNORECASE).strip()
            if year:
                keyword = re.sub(r"20\d{2}", "", keyword).strip()
            if not keyword:
                return "請給關鍵字，例如：銷售搜尋 2025 ABC"
            rows = _sales_search(keyword, year, limit=20)
            if not rows:
                return f"找不到：{keyword}"
            lines = [f"【銷售搜尋】{keyword}（年：{year if year else '不限'}）"]
            for r in rows[:20]:
                lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
            return "\n".join(lines)

        return "我看得懂銷售：總覽/前N產品/前N客戶/搜尋。打「指令」看範例。"

    # 進貨
    if "進貨" in t or "採購" in t:
        if "總覽" in t:
            if not year:
                return "請帶年份，例如：進貨總覽 2024"
            d = _purchase_summary(year)
            return _format_kv("【進貨總覽】", d)

        if ("前" in t or "top" in t.lower()) and "產品" in t:
            if not year:
                return "請帶年份，例如：進貨前10產品 2024"
            rows = _purchase_top_products(year, n)
            return _format_top_list(f"【進貨 前{n} 產品｜依數量】(年 {year})", rows, "product")

        if "搜尋" in t:
            keyword = re.sub(r"(進貨|採購)\s*搜尋\s*", "", t).strip()
            if year:
                keyword = re.sub(r"20\d{2}", "", keyword).strip()
            if not keyword:
                return "請給關鍵字，例如：進貨搜尋 2024 ABC"
            rows = _purchase_search(keyword, year, limit=20)
            if not rows:
                return f"找不到：{keyword}"
            lines = [f"【進貨搜尋】{keyword}（年：{year if year else '不限'}）"]
            for r in rows[:20]:
                lines.append(f"{r['date']}｜{r['supplier']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
            return "\n".join(lines)

        return "我看得懂進貨：總覽/前N產品/搜尋。打「指令」看範例。"

    return "我不知道你要查什麼。打「指令」我會給你可用格式。"


@app.post("/line/webhook")
async def line_webhook(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("x-line-signature", "")

    if not _verify_line_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    body = json.loads(body_bytes.decode("utf-8"))
    print(json.dumps(body, indent=2, ensure_ascii=False))

    events = body.get("events", [])
    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        if msg.get("type") != "text":
            continue

        reply_token = ev.get("replyToken")
        user_text = msg.get("text", "")

        answer = _handle_query(user_text)
        await _line_reply(reply_token, answer)

    return {"status": "ok"}
