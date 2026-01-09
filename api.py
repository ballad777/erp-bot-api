import os
import re
import json
import time
import hmac
import base64
import hashlib
from typing import Any, Dict, Optional, List, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, text
import httpx

import google.generativeai as genai

import matplotlib
matplotlib.use("Agg")  # Render 上沒有 GUI，要用 Agg
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# Env
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")  # e.g. https://erp-bot-api.onrender.com

if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")
if not LINE_CHANNEL_ACCESS_TOKEN:
    raise Exception("找不到 LINE_CHANNEL_ACCESS_TOKEN 環境變數")
if not LINE_CHANNEL_SECRET:
    raise Exception("找不到 LINE_CHANNEL_SECRET 環境變數")
if not GEMINI_API_KEY:
    raise Exception("找不到 GEMINI_API_KEY 環境變數")
if not PUBLIC_BASE_URL:
    raise Exception("找不到 PUBLIC_BASE_URL 環境變數（例如 https://erp-bot-api.onrender.com）")


# ============================================================
# App + Static
# ============================================================
app = FastAPI(title="ERP Bot API", version="2.0")

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================
# DB
# ============================================================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# ============================================================
# Gemini
# ============================================================
genai.configure(api_key=GEMINI_API_KEY)

# 你可以改成 gemini-1.5-pro（更準更貴）
GEMINI_MODEL = genai.GenerativeModel("models/gemini-1.5-flash")


# ============================================================
# Simple in-memory chat memory (Render 免費版重啟會清空，正常)
# key: LINE userId
# value: list of (role, text) role in {"user","assistant"}
# ============================================================
CHAT_MEM: Dict[str, List[Tuple[str, str]]] = {}
MAX_TURNS = 10  # 保留最近 10 輪對話


def remember(user_id: str, role: str, content: str):
    hist = CHAT_MEM.get(user_id, [])
    hist.append((role, content))
    hist = hist[-(MAX_TURNS * 2):]
    CHAT_MEM[user_id] = hist


def get_history_text(user_id: str) -> str:
    hist = CHAT_MEM.get(user_id, [])
    lines = []
    for role, content in hist:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


# ============================================================
# LINE helpers
# ============================================================
def verify_line_signature(body: bytes, signature: str) -> bool:
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


async def line_reply_text(reply_token: str, text_message: str):
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


async def line_reply_image(reply_token: str, image_url: str, preview_url: Optional[str] = None, caption: Optional[str] = None):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    msg_list = []
    if caption:
        msg_list.append({"type": "text", "text": caption[:4900]})

    msg_list.append({
        "type": "image",
        "originalContentUrl": image_url,
        "previewImageUrl": preview_url or image_url,
    })

    payload = {"replyToken": reply_token, "messages": msg_list}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            print("LINE reply image error:", r.status_code, r.text)


# ============================================================
# DB query tools (給 Gemini 呼叫)
# 你可以依你的資料表欄位調整
# sales: date, year, customer, product, quantity, amount
# purchase: date, year, supplier, product, quantity, amount
# ============================================================
def db_sales_latest(limit: int = 10) -> List[Dict[str, Any]]:
    sql = text("""
        SELECT date, year, customer, product, quantity, amount
        FROM sales
        ORDER BY date DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"limit": int(limit)}).mappings().all()
    return [dict(r) for r in rows]


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
        row = conn.execute(sql, {"year": int(year)}).mappings().first()
    return dict(row) if row else {"year": year, "rows": 0, "total_qty": 0, "total_amount": 0}


def db_sales_search(keyword: str, year: Optional[int] = None, limit: int = 20) -> List[Dict[str, Any]]:
    where_year = "AND year = :year" if year is not None else ""
    sql = text(f"""
        SELECT date, year, customer, product, quantity, amount
        FROM sales
        WHERE (customer ILIKE :pat OR product ILIKE :pat)
        {where_year}
        ORDER BY date DESC
        LIMIT :limit
    """)
    params = {"pat": f"%{keyword}%", "limit": int(limit)}
    if year is not None:
        params["year"] = int(year)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]


def db_sales_top_products(year: int, n: int = 10) -> List[Dict[str, Any]]:
    sql = text("""
        SELECT
            product,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY product
        ORDER BY total_amount DESC, total_qty DESC, rows DESC, product ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": int(year), "n": int(n)}).mappings().all()
    return [dict(r) for r in rows]


def db_sales_monthly(year: int) -> List[Dict[str, Any]]:
    # 兼容 date 是 timestamp / date
    sql = text("""
        SELECT
            DATE_TRUNC('month', date::timestamp) AS month,
            COALESCE(SUM(amount), 0) AS total_amount,
            COALESCE(SUM(quantity), 0) AS total_qty
        FROM sales
        WHERE year = :year
        GROUP BY 1
        ORDER BY 1 ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": int(year)}).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        # month 可能是 datetime
        d["month"] = str(d["month"])[:10]
        out.append(d)
    return out


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
        row = conn.execute(sql, {"year": int(year)}).mappings().first()
    return dict(row) if row else {"year": year, "rows": 0, "total_qty": 0, "total_amount": 0}


def db_purchase_search(keyword: str, year: Optional[int] = None, limit: int = 20) -> List[Dict[str, Any]]:
    where_year = "AND year = :year" if year is not None else ""
    sql = text(f"""
        SELECT date, year, supplier, product, quantity, amount
        FROM purchase
        WHERE (supplier ILIKE :pat OR product ILIKE :pat)
        {where_year}
        ORDER BY date DESC
        LIMIT :limit
    """)
    params = {"pat": f"%{keyword}%", "limit": int(limit)}
    if year is not None:
        params["year"] = int(year)

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]


# ============================================================
# Chart tools
# ============================================================
def save_bar_chart(title: str, labels: List[str], values: List[float], filename_prefix: str) -> str:
    ts = int(time.time())
    fname = f"{filename_prefix}_{ts}.png"
    path = os.path.join(STATIC_DIR, fname)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    # LINE 預覽要求圖檔可公開存取
    return f"{PUBLIC_BASE_URL}/static/{fname}"


def save_line_chart(title: str, x: List[str], y: List[float], filename_prefix: str) -> str:
    ts = int(time.time())
    fname = f"{filename_prefix}_{ts}.png"
    path = os.path.join(STATIC_DIR, fname)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(x, y, marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return f"{PUBLIC_BASE_URL}/static/{fname}"


# ============================================================
# Gemini Router (核心：自然語言 -> 決策)
# Gemini 必須回傳 JSON，指定要「聊天」或「查資料/畫圖」
# ============================================================
TOOL_SPEC = """
你是一個 ERP 資料助理，能和使用者正常聊天，也能查詢公司資料庫。
你必須輸出【純 JSON】且只能是 JSON（不要 Markdown、不要多餘文字）。

你可以選擇：
1) mode="chat": 正常聊天回覆
2) mode="tool": 要求系統呼叫工具取得資料，再由你產出回覆

可用工具 tools（name 與 args）：
- sales_latest: { "limit": int }
- sales_summary: { "year": int }
- sales_search: { "keyword": str, "year": int|null, "limit": int }
- sales_top_products: { "year": int, "n": int }
- sales_monthly_chart: { "year": int }   # 會回傳每月銷售金額折線圖
- purchase_summary: { "year": int }
- purchase_search: { "keyword": str, "year": int|null, "limit": int }

規則：
- 使用者問「最近幾筆銷售」=> sales_latest
- 問「某年銷售總覽/總金額/總數量」=> sales_summary
- 問「幫我找某客戶/某品號」=> sales_search 或 purchase_search
- 問「每月趨勢/折線圖/圖表」=> sales_monthly_chart（先做銷售）
- 問「前幾名產品」=> sales_top_products
- 如果使用者只是聊天、閒聊、問你是誰、問建議，不需要查資料 => mode=chat
- 如果資訊不足（例如沒給年份你真的需要年份）你可以 mode=chat 詢問，但要自然，不要像指令機器人。

輸出 JSON 格式固定：
{
  "mode": "chat" | "tool",
  "response": "你要回給使用者的文字（若 mode=tool，這裡先寫你打算怎麼回，工具結果出來後我會再組合一次）",
  "tool": { "name": "...", "args": { ... } } | null
}
"""


def parse_year(text_in: str) -> Optional[int]:
    m = re.search(r"(20\d{2})", text_in)
    return int(m.group(1)) if m else None


async def gemini_route(user_id: str, user_text: str) -> Dict[str, Any]:
    history = get_history_text(user_id)
    prompt = f"""
{TOOL_SPEC}

【對話歷史】
{history}

【使用者訊息】
{user_text}
"""
    resp = GEMINI_MODEL.generate_content(prompt)
    raw = (resp.text or "").strip()

    # 盡量把 JSON 取出（避免 Gemini 偶爾多打字）
    try:
        data = json.loads(raw)
        if "mode" not in data:
            raise ValueError("no mode")
        return data
    except Exception:
        # fallback：當作 chat
        return {"mode": "chat", "response": raw if raw else "我在喔～你想聊什麼？", "tool": None}


# ============================================================
# Tool executor
# ============================================================
def tool_execute(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "sales_latest":
        limit = int(args.get("limit", 10))
        return {"rows": db_sales_latest(limit=limit)}

    if tool_name == "sales_summary":
        year = int(args["year"])
        return db_sales_summary(year=year)

    if tool_name == "sales_search":
        keyword = str(args["keyword"])
        year = args.get("year", None)
        year = int(year) if year is not None else None
        limit = int(args.get("limit", 20))
        return {"rows": db_sales_search(keyword=keyword, year=year, limit=limit)}

    if tool_name == "sales_top_products":
        year = int(args["year"])
        n = int(args.get("n", 10))
        return {"rows": db_sales_top_products(year=year, n=n)}

    if tool_name == "sales_monthly_chart":
        year = int(args["year"])
        monthly = db_sales_monthly(year=year)
        x = [m["month"] for m in monthly]
        y = [float(m["total_amount"]) for m in monthly]
        img_url = save_line_chart(
            title=f"{year} 每月銷售金額趨勢",
            x=x,
            y=y,
            filename_prefix=f"sales_monthly_{year}"
        )
        return {"image_url": img_url, "monthly": monthly}

    if tool_name == "purchase_summary":
        year = int(args["year"])
        return db_purchase_summary(year=year)

    if tool_name == "purchase_search":
        keyword = str(args["keyword"])
        year = args.get("year", None)
        year = int(year) if year is not None else None
        limit = int(args.get("limit", 20))
        return {"rows": db_purchase_search(keyword=keyword, year=year, limit=limit)}

    return {"error": f"Unknown tool: {tool_name}"}


# ============================================================
# Final answer composer (把工具結果變成好讀的回覆)
# ============================================================
def format_tool_result(tool_name: str, result: Dict[str, Any], user_text: str) -> Tuple[str, Optional[str]]:
    """
    return (text, image_url or None)
    """
    if tool_name == "sales_latest":
        rows = result.get("rows", [])
        if not rows:
            return "我查不到最近的銷售資料耶（sales 表可能是空的）。", None
        lines = ["【最近銷售】"]
        for r in rows[:10]:
            lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines), None

    if tool_name == "sales_summary":
        return (
            f"【{result.get('year')} 銷售總覽】\n"
            f"筆數：{result.get('rows')}\n"
            f"總數量：{result.get('total_qty')}\n"
            f"總金額：{result.get('total_amount')}",
            None
        )

    if tool_name == "sales_search":
        rows = result.get("rows", [])
        if not rows:
            return "我查不到符合的銷售紀錄（你可以換個關鍵字或不指定年份）。", None
        lines = ["【銷售搜尋結果】（我先回前 20 筆）"]
        for r in rows[:20]:
            lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines), None

    if tool_name == "sales_top_products":
        rows = result.get("rows", [])
        if not rows:
            return "我找不到該年度的產品排行資料。", None
        lines = ["【產品 Top 排行】（依金額/數量綜合）"]
        for i, r in enumerate(rows, 1):
            lines.append(f"{i}. {r['product']}｜金額 {r['total_amount']}｜數量 {r['total_qty']}｜筆數 {r['rows']}")
        return "\n".join(lines), None

    if tool_name == "sales_monthly_chart":
        img = result.get("image_url")
        return "我把每月銷售趨勢圖畫好了（折線圖）。", img

    if tool_name == "purchase_summary":
        return (
            f"【{result.get('year')} 進貨總覽】\n"
            f"筆數：{result.get('rows')}\n"
            f"總數量：{result.get('total_qty')}\n"
            f"總金額：{result.get('total_amount')}",
            None
        )

    if tool_name == "purchase_search":
        rows = result.get("rows", [])
        if not rows:
            return "我查不到符合的進貨紀錄（你可以換個關鍵字或不指定年份）。", None
        lines = ["【進貨搜尋結果】（我先回前 20 筆）"]
        for r in rows[:20]:
            lines.append(f"{r['date']}｜{r['supplier']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
        return "\n".join(lines), None

    err = result.get("error")
    if err:
        return f"工具執行出錯：{err}", None

    return "我有查到結果，但不知道怎麼整理回覆（你再描述一次你要看什麼）。", None


# ============================================================
# Health
# ============================================================
@app.get("/health")
def health():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar()
    return {"ok": True, "db": v}


# ============================================================
# LINE Webhook
# ============================================================
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
        user_id = (ev.get("source") or {}).get("userId", "unknown")
        user_text = msg.get("text", "").strip()

        # 記錄使用者訊息
        remember(user_id, "user", user_text)

        # 先讓 Gemini 決策
        route = await gemini_route(user_id, user_text)

        mode = route.get("mode", "chat")
        tool = route.get("tool")

        # mode=chat：直接聊天
        if mode == "chat" or not tool:
            resp_text = route.get("response") or "我在喔～你想聊什麼？"
            remember(user_id, "assistant", resp_text)
            await line_reply_text(reply_token, resp_text)
            continue

        # mode=tool：執行工具
        tool_name = tool.get("name")
        args = tool.get("args", {}) or {}

        # 如果 Gemini 忘了年份，但你訊息有年份，補一下（容錯）
        if tool_name in {"sales_summary", "sales_top_products", "sales_monthly_chart", "purchase_summary"}:
            if "year" not in args or args["year"] in (None, ""):
                y = parse_year(user_text)
                if y:
                    args["year"] = y

        result = tool_execute(tool_name, args)
        text_reply, image_url = format_tool_result(tool_name, result, user_text)

        remember(user_id, "assistant", text_reply)

        # 回圖片 or 回文字
        if image_url:
            await line_reply_image(reply_token, image_url=image_url, caption=text_reply)
        else:
            await line_reply_text(reply_token, text_reply)

    return JSONResponse({"status": "ok"})
