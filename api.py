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
from fastapi.responses import Response
from sqlalchemy import create_engine, text

import httpx

# =========================
# Config
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # 你也可以改 gemini-3-flash-preview 等

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
app = FastAPI(title="ERP Bot API", version="2.0")

# =========================
# In-memory stores (Render 無狀態，重啟會清空；但足夠用於 demo/測試)
# =========================
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}  # user_id -> [{"role":"user/assistant","content": "..."}]
IMG_STORE: Dict[str, Dict[str, Any]] = {}          # img_id -> {"bytes": b"...", "ts": time.time()}
IMG_TTL_SEC = 60 * 10  # 圖片保留 10 分鐘

# =========================
# Gemini client (new SDK)
# =========================
GEMINI_CLIENT = None
if GEMINI_API_KEY:
    try:
        from google import genai  # google-genai
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print("Gemini init error:", repr(e))
        GEMINI_CLIENT = None


# =========================
# Helpers: LINE signature
# =========================
def verify_line_signature(body_bytes: bytes, signature: str) -> bool:
    if not LINE_CHANNEL_SECRET:
        return True  # 沒設 secret 就不驗簽（建議你最後一定要設）
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body_bytes, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# =========================
# Helpers: DB
# =========================
def db_one(sql: str, params: dict | None = None) -> Dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(text(sql), params or {}).mappings().first()
    return dict(row) if row else {}

def db_all(sql: str, params: dict | None = None) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params or {}).mappings().all()
    return [dict(r) for r in rows]

def safe_select_only(sql: str) -> bool:
    # 只允許 SELECT（避免被模型生成 DROP/UPDATE）
    s = sql.strip().lower()
    if not s.startswith("select"):
        return False
    bad = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"]
    return not any(b in s for b in bad)


# =========================
# Helpers: parse year / topN
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


# =========================
# Public endpoints
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "erp-bot-api", "version": "2.0"}

@app.get("/health")
def health():
    v = db_one("SELECT 1 AS v").get("v", None)
    return {"ok": True, "db": v}

@app.get("/schema")
def schema():
    # 讓你快速確認 date 是 date / year 是 int 等
    sql = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema='public'
      AND table_name IN ('sales', 'purchase')
    ORDER BY table_name, ordinal_position;
    """
    return {"ok": True, "columns": db_all(sql)}

@app.get("/img/{img_id}")
def get_img(img_id: str):
    item = IMG_STORE.get(img_id)
    if not item:
        raise HTTPException(status_code=404, detail="Not Found")
    # TTL
    if time.time() - item["ts"] > IMG_TTL_SEC:
        IMG_STORE.pop(img_id, None)
        raise HTTPException(status_code=404, detail="Expired")
    return Response(content=item["bytes"], media_type="image/png")


# =========================
# LINE reply (text / image)
# =========================
async def line_reply_text(reply_token: str, text_message: str):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text_message[:4900]}]}
    async with httpx.AsyncClient(timeout=15) as client:
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
        "messages": [
            {
                "type": "image",
                "originalContentUrl": image_url,
                "previewImageUrl": preview_url or image_url,
            }
        ],
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            print("LINE reply image error:", r.status_code, r.text)


# =========================
# Chart generator (Matplotlib -> PNG bytes)
# =========================
def make_line_chart_png(title: str, x: List[Any], y: List[float], x_label: str, y_label: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    return buf.getvalue()

def make_bar_chart_png(title: str, x: List[Any], y: List[float], x_label: str, y_label: str) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    plt.close()
    return buf.getvalue()


# =========================
# Weather (免費 Open-Meteo，免 key)
# =========================
async def weather_by_city(city: str) -> str:
    # 先用 geocoding 找經緯度
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    async with httpx.AsyncClient(timeout=15) as client:
        g = await client.get(geo_url, params={"name": city, "count": 1, "language": "zh", "format": "json"})
        data = g.json()
        if not data.get("results"):
            return f"我找不到「{city}」的位置。你可以換成：台北 / 新北 / 台中 / 高雄 之類的。"
        r0 = data["results"][0]
        lat, lon = r0["latitude"], r0["longitude"]
        name = r0.get("name", city)

        w_url = "https://api.open-meteo.com/v1/forecast"
        w = await client.get(
            w_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "timezone": "Asia/Taipei",
            },
        )
        wd = w.json()
        cur = wd.get("current", {})
        t = cur.get("temperature_2m")
        wind = cur.get("wind_speed_10m")
        code = cur.get("weather_code")
        return f"【{name} 即時天氣】\n溫度：{t}°C\n風速：{wind} m/s\n代碼：{code}（需要我把代碼轉成文字描述也可以）"


# =========================
# Core: deterministic data functions
# =========================
def sales_summary(year: int) -> Dict[str, Any]:
    sql = """
    SELECT :year AS year,
           COUNT(*) AS rows,
           COALESCE(SUM(quantity), 0) AS total_qty,
           COALESCE(SUM(amount), 0) AS total_amount
    FROM sales
    WHERE year = :year
    """
    row = db_one(sql, {"year": year})
    if not row:
        return {"year": year, "rows": 0, "total_qty": 0, "total_amount": 0}
    return row

def sales_top_products(year: int, n: int) -> List[Dict[str, Any]]:
    sql = """
    SELECT product,
           COUNT(*) AS rows,
           COALESCE(SUM(quantity), 0) AS total_qty,
           COALESCE(SUM(amount), 0) AS total_amount
    FROM sales
    WHERE year = :year
    GROUP BY product
    ORDER BY total_qty DESC, rows DESC, product ASC
    LIMIT :n
    """
    return db_all(sql, {"year": year, "n": n})

def sales_top_customers(year: int, n: int) -> List[Dict[str, Any]]:
    sql = """
    SELECT customer,
           COUNT(*) AS rows,
           COALESCE(SUM(quantity), 0) AS total_qty,
           COALESCE(SUM(amount), 0) AS total_amount
    FROM sales
    WHERE year = :year
    GROUP BY customer
    ORDER BY total_qty DESC, rows DESC, customer ASC
    LIMIT :n
    """
    return db_all(sql, {"year": year, "n": n})

def sales_search(keyword: str, year: Optional[int], limit: int = 30) -> List[Dict[str, Any]]:
    where_year = "AND year = :year" if year is not None else ""
    sql = f"""
    SELECT date, year, customer, product, quantity, amount
    FROM sales
    WHERE (customer ILIKE :pat OR product ILIKE :pat)
      {where_year}
    ORDER BY date DESC
    LIMIT :limit
    """
    params = {"pat": f"%{keyword}%", "limit": limit}
    if year is not None:
        params["year"] = year
    return db_all(sql, params)

def purchase_summary(year: int) -> Dict[str, Any]:
    sql = """
    SELECT :year AS year,
           COUNT(*) AS rows,
           COALESCE(SUM(quantity), 0) AS total_qty,
           COALESCE(SUM(amount), 0) AS total_amount
    FROM purchase
    WHERE year = :year
    """
    row = db_one(sql, {"year": year})
    if not row:
        return {"year": year, "rows": 0, "total_qty": 0, "total_amount": 0}
    return row

def purchase_top_products(year: int, n: int) -> List[Dict[str, Any]]:
    sql = """
    SELECT product,
           COUNT(*) AS rows,
           COALESCE(SUM(quantity), 0) AS total_qty,
           COALESCE(SUM(amount), 0) AS total_amount
    FROM purchase
    WHERE year = :year
    GROUP BY product
    ORDER BY total_qty DESC, rows DESC, product ASC
    LIMIT :n
    """
    return db_all(sql, {"year": year, "n": n})

def purchase_search(keyword: str, year: Optional[int], limit: int = 30) -> List[Dict[str, Any]]:
    where_year = "AND year = :year" if year is not None else ""
    sql = f"""
    SELECT date, year, supplier, product, quantity, amount
    FROM purchase
    WHERE (supplier ILIKE :pat OR product ILIKE :pat)
      {where_year}
    ORDER BY date DESC
    LIMIT :limit
    """
    params = {"pat": f"%{keyword}%", "limit": limit}
    if year is not None:
        params["year"] = year
    return db_all(sql, params)

def sales_monthly_amount_chart(year: int) -> Tuple[str, bytes]:
    sql = """
    SELECT to_char(date, 'YYYY-MM') AS ym, COALESCE(SUM(amount),0) AS total_amount
    FROM sales
    WHERE year = :year
    GROUP BY ym
    ORDER BY ym
    """
    rows = db_all(sql, {"year": year})
    x = [r["ym"] for r in rows]
    y = [float(r["total_amount"]) for r in rows]
    png = make_line_chart_png(f"Sales Monthly Amount {year}", x, y, "Month", "Amount")
    return ("line", png)

def sales_monthly_qty_chart(year: int) -> Tuple[str, bytes]:
    sql = """
    SELECT to_char(date, 'YYYY-MM') AS ym, COALESCE(SUM(quantity),0) AS total_qty
    FROM sales
    WHERE year = :year
    GROUP BY ym
    ORDER BY ym
    """
    rows = db_all(sql, {"year": year})
    x = [r["ym"] for r in rows]
    y = [float(r["total_qty"]) for r in rows]
    png = make_line_chart_png(f"Sales Monthly Qty {year}", x, y, "Month", "Quantity")
    return ("line", png)


# =========================
# Gemini: natural language router -> JSON intent
# =========================
ROUTER_SYSTEM = """
你是一個 ERP LINE AI 助手。你要做兩件事：
1) 能像一般聊天機器人一樣自然對話。
2) 只要使用者的問題涉及「銷售 sales」或「進貨 purchase」資料查詢/分析/圖表，你要輸出一個 JSON 指令讓後端去查資料並回覆。

【你只能輸出 JSON，不能輸出其他文字】
JSON 格式：
{
  "type": "chat" | "data",
  "intent": "...",
  "params": { ... }
}

可用 intent（type=data 時）：
- "sales_summary" params: {"year": 2025}
- "sales_top_products" params: {"year": 2025, "n": 10}
- "sales_top_customers" params: {"year": 2025, "n": 10}
- "sales_search" params: {"keyword": "ABC", "year": 2025|null}
- "purchase_summary" params: {"year": 2024}
- "purchase_top_products" params: {"year": 2024, "n": 10}
- "purchase_search" params: {"keyword": "ABC", "year": 2024|null}
- "sales_chart_monthly_amount" params: {"year": 2025}
- "sales_chart_monthly_qty" params: {"year": 2025}
- "weather" params: {"city": "台北"}

規則：
- 使用者如果只是聊天/閒聊/問你是誰/要你幫忙寫文案/工作建議 → type=chat, intent="general"
- 只要牽涉到資料（例如：某客戶金額、某產品數量、排行、趨勢、圖表、查詢）→ type=data
- year 沒講：優先推測為今年（如果看起來像要今年），不確定就填 null 並改用 search 或請對方補年份（但仍輸出 JSON）
- keyword 允許使用者打錯字（維持原字串），後端會用 ILIKE 模糊查
- 想要圖表（折線/趨勢/每月）就用 chart intents
"""

def keep_history(user_id: str, role: str, content: str, limit: int = 12):
    h = CHAT_MEMORY.get(user_id, [])
    h.append({"role": role, "content": content})
    CHAT_MEMORY[user_id] = h[-limit:]

async def gemini_route(user_id: str, user_text: str) -> Dict[str, Any]:
    if not GEMINI_CLIENT:
        raise RuntimeError("Gemini client not ready")

    # 給模型一些上下文（短記憶）
    history = CHAT_MEMORY.get(user_id, [])
    contents = []
    contents.append({"role": "user", "parts": [{"text": ROUTER_SYSTEM}]})
    for m in history:
        contents.append({"role": m["role"], "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    resp = GEMINI_CLIENT.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
    )
    txt = (resp.text or "").strip()

    # 只接受 JSON
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and "type" in data:
            return data
    except Exception:
        pass

    # JSON 失敗就當 chat
    return {"type": "chat", "intent": "general", "params": {"fallback": True, "text": txt[:1500]}}


# =========================
# Execute intent
# =========================
def format_summary(title: str, d: Dict[str, Any]) -> str:
    return (
        f"{title}\n"
        f"年：{d.get('year')}\n"
        f"筆數：{d.get('rows')}\n"
        f"總數量：{d.get('total_qty')}\n"
        f"總金額：{d.get('total_amount')}\n"
    )

def format_top(title: str, rows: List[Dict[str, Any]], name_key: str) -> str:
    lines = [title]
    for i, r in enumerate(rows, 1):
        nm = r.get(name_key, "")
        qty = r.get("total_qty", 0)
        cnt = r.get("rows", 0)
        amt = r.get("total_amount", 0)
        lines.append(f"{i}. {nm}｜數量 {qty}｜筆數 {cnt}｜金額 {amt}")
    return "\n".join(lines)

def format_sales_rows(rows: List[Dict[str, Any]], keyword: str, year: Optional[int]) -> str:
    if not rows:
        return f"找不到：{keyword}"
    head = f"【銷售搜尋】{keyword}（年：{year if year else '不限'}）"
    lines = [head]
    for r in rows[:30]:
        lines.append(f"{r['date']}｜{r['customer']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
    return "\n".join(lines)

def format_purchase_rows(rows: List[Dict[str, Any]], keyword: str, year: Optional[int]) -> str:
    if not rows:
        return f"找不到：{keyword}"
    head = f"【進貨搜尋】{keyword}（年：{year if year else '不限'}）"
    lines = [head]
    for r in rows[:30]:
        lines.append(f"{r['date']}｜{r['supplier']}｜{r['product']}｜數量 {r['quantity']}｜金額 {r['amount']}")
    return "\n".join(lines)

async def execute_intent(route: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """
    return:
      {"kind":"text","text":"..."}
      or {"kind":"image","url":"..."}
    """
    t = route.get("type")
    intent = route.get("intent", "")
    params = route.get("params", {}) or {}

    # chat
    if t == "chat":
        # 如果 Gemini 有回傳 fallback text，就用它；不然用簡單回覆
        txt = params.get("text")
        if txt:
            return {"kind": "text", "text": txt}
        return {"kind": "text", "text": "我在～你想聊什麼？也可以直接問我銷售/進貨資料或要我做趨勢圖。"}

    # data intents
    try:
        if intent == "sales_summary":
            year = int(params.get("year"))
            return {"kind": "text", "text": format_summary("【銷售總覽】", sales_summary(year))}

        if intent == "sales_top_products":
            year = int(params.get("year"))
            n = int(params.get("n", 10))
            return {"kind": "text", "text": format_top(f"【銷售 前{n} 產品｜依數量】(年 {year})", sales_top_products(year, n), "product")}

        if intent == "sales_top_customers":
            year = int(params.get("year"))
            n = int(params.get("n", 10))
            return {"kind": "text", "text": format_top(f"【銷售 前{n} 客戶｜依數量】(年 {year})", sales_top_customers(year, n), "customer")}

        if intent == "sales_search":
            keyword = str(params.get("keyword", "")).strip()
            year = params.get("year", None)
            year = int(year) if year else None
            rows = sales_search(keyword, year, limit=30)
            return {"kind": "text", "text": format_sales_rows(rows, keyword, year)}

        if intent == "purchase_summary":
            year = int(params.get("year"))
            return {"kind": "text", "text": format_summary("【進貨總覽】", purchase_summary(year))}

        if intent == "purchase_top_products":
            year = int(params.get("year"))
            n = int(params.get("n", 10))
            return {"kind": "text", "text": format_top(f"【進貨 前{n} 產品｜依數量】(年 {year})", purchase_top_products(year, n), "product")}

        if intent == "purchase_search":
            keyword = str(params.get("keyword", "")).strip()
            year = params.get("year", None)
            year = int(year) if year else None
            rows = purchase_search(keyword, year, limit=30)
            return {"kind": "text", "text": format_purchase_rows(rows, keyword, year)}

        if intent == "sales_chart_monthly_amount":
            year = int(params.get("year"))
            _, png = sales_monthly_amount_chart(year)
            img_id = str(uuid.uuid4())
            IMG_STORE[img_id] = {"bytes": png, "ts": time.time()}
            return {"kind": "image", "url": f"{base_url}/img/{img_id}"}

        if intent == "sales_chart_monthly_qty":
            year = int(params.get("year"))
            _, png = sales_monthly_qty_chart(year)
            img_id = str(uuid.uuid4())
            IMG_STORE[img_id] = {"bytes": png, "ts": time.time()}
            return {"kind": "image", "url": f"{base_url}/img/{img_id}"}

        if intent == "weather":
            city = str(params.get("city", "")).strip() or "台北"
            txt = await weather_by_city(city)
            return {"kind": "text", "text": txt}

        # Unknown intent
        return {"kind": "text", "text": "我理解你想查資料/分析，但我還不確定要怎麼做。你可以換個說法或補：年份、客戶、產品。"}

    except Exception as e:
        return {"kind": "text", "text": f"資料處理出錯：{type(e).__name__}: {e}"}


# =========================
# Fallback parser (當 Gemini 掛了/被限流時)
# =========================
def fallback_rule_route(user_text: str) -> Dict[str, Any]:
    t = user_text.strip()
    y = extract_year(t)
    n = extract_top_n(t, 10)

    # 天氣
    if "天氣" in t:
        # 盡量抓城市
        m = re.search(r"(台北|新北|桃園|台中|台南|高雄|基隆|新竹|嘉義|宜蘭|花蓮|台東)", t)
        city = m.group(1) if m else "台北"
        return {"type": "data", "intent": "weather", "params": {"city": city}}

    # 銷售
    if "銷售" in t:
        if "總覽" in t or "總結" in t:
            return {"type": "data", "intent": "sales_summary", "params": {"year": y or time.localtime().tm_year}}
        if ("趨勢" in t or "折線" in t or "每月" in t) and ("金額" in t or "業績" in t):
            return {"type": "data", "intent": "sales_chart_monthly_amount", "params": {"year": y or time.localtime().tm_year}}
        if ("趨勢" in t or "折線" in t or "每月" in t) and ("數量" in t):
            return {"type": "data", "intent": "sales_chart_monthly_qty", "params": {"year": y or time.localtime().tm_year}}
        if ("前" in t or "top" in t.lower()) and ("產品" in t or "品項" in t):
            return {"type": "data", "intent": "sales_top_products", "params": {"year": y or time.localtime().tm_year, "n": n}}
        if ("前" in t or "top" in t.lower()) and ("客戶" in t):
            return {"type": "data", "intent": "sales_top_customers", "params": {"year": y or time.localtime().tm_year, "n": n}}
        if "查" in t or "搜尋" in t or "search" in t.lower():
            kw = re.sub(r".*(搜尋|search|查)\s*", "", t, flags=re.IGNORECASE).strip()
            kw = re.sub(r"20\d{2}", "", kw).strip()
            if not kw:
                kw = t.replace("銷售", "").strip()
            return {"type": "data", "intent": "sales_search", "params": {"keyword": kw, "year": y}}

        # 沒抓到就當 chat
        return {"type": "chat", "intent": "general", "params": {"text": "你是想查銷售資料嗎？你可以說：\n- 2025 銷售總覽\n- 2025 銷售前10客戶\n- 2025 銷售每月金額趨勢\n- 查 銷售 ABC"}}

    # 進貨
    if "進貨" in t or "採購" in t:
        if "總覽" in t or "總結" in t:
            return {"type": "data", "intent": "purchase_summary", "params": {"year": y or time.localtime().tm_year}}
        if ("前" in t or "top" in t.lower()) and ("產品" in t or "品項" in t):
            return {"type": "data", "intent": "purchase_top_products", "params": {"year": y or time.localtime().tm_year, "n": n}}
        if "查" in t or "搜尋" in t or "search" in t.lower():
            kw = re.sub(r".*(搜尋|search|查)\s*", "", t, flags=re.IGNORECASE).strip()
            kw = re.sub(r"20\d{2}", "", kw).strip()
            if not kw:
                kw = t.replace("進貨", "").replace("採購", "").strip()
            return {"type": "data", "intent": "purchase_search", "params": {"keyword": kw, "year": y}}

        return {"type": "chat", "intent": "general", "params": {"text": "你是想查進貨資料嗎？你可以說：\n- 2024 進貨總覽\n- 2024 進貨前10產品\n- 2024 進貨搜尋 XYZ"}}

    # 其他就聊天
    return {"type": "chat", "intent": "general", "params": {"text": "我在～你想聊什麼？也可以問：某客戶今年買了多少、某產品銷售趨勢、要我畫折線圖。"}}


# =========================
# LINE Webhook
# =========================
@app.post("/line/webhook")
async def line_webhook(request: Request):
    body_bytes = await request.body()
    signature = request.headers.get("x-line-signature", "")

    if not verify_line_signature(body_bytes, signature):
        raise HTTPException(status_code=400, detail="Invalid signature")

    body = json.loads(body_bytes.decode("utf-8"))
    events = body.get("events", [])

    # 用 request 的 host 組出圖片 URL（Render 會是 https）
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    proto = request.headers.get("x-forwarded-proto") or "https"
    base_url = f"{proto}://{host}"

    for ev in events:
        if ev.get("type") != "message":
            continue
        msg = ev.get("message", {})
        if msg.get("type") != "text":
            continue

        reply_token = ev.get("replyToken")
        user_id = (ev.get("source") or {}).get("userId", "unknown")
        user_text = (msg.get("text") or "").strip()
        if not user_text:
            continue

        # 先記 user message
        keep_history(user_id, "user", user_text)

        # 1) 先走 Gemini（若可用）
        route = None
        if GEMINI_CLIENT:
            try:
                route = await gemini_route(user_id, user_text)
            except Exception as e:
                # Gemini 掛了就走 fallback
                print("Gemini route error:", repr(e))
                route = None

        # 2) Gemini 不可用 / 解析失敗 -> fallback
        if not route:
            route = fallback_rule_route(user_text)

        # 3) 執行 intent
        result = await execute_intent(route, base_url=base_url)

        # 4) 回 LINE
        if result["kind"] == "image":
            await line_reply_image(reply_token, result["url"])
            keep_history(user_id, "assistant", f"[image]{result['url']}")
        else:
            await line_reply_text(reply_token, result["text"])
            keep_history(user_id, "assistant", result["text"])

    return {"status": "ok"}
