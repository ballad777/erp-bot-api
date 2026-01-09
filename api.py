import os
import re
import json
import time
import hmac
import base64
import hashlib
import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import requests
import pandas as pd
import httpx

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from google import genai
from google.genai import types

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("erp_ultra_pro")

app = FastAPI(title="ERP Bot Ultra PRO", version="3.0_Commercial")

# =========================
# Environment
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

SALES_SHEET_URL = os.getenv("SALES_EXCEL_URL", "")
PURCHASE_SHEET_URL = os.getenv("PURCHASE_EXCEL_URL", "")

# Admin protect for reload
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # REQUIRED for /admin/*
AUTO_IMPORT_ON_STARTUP = os.getenv("AUTO_IMPORT_ON_STARTUP", "0") == "1"

# basic anti-spam
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "40"))
RATE_STORE: Dict[str, List[float]] = {}

engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# =========================
# Time utils
# =========================
def now_taipei() -> datetime:
    return datetime.utcnow() + timedelta(hours=8)

# =========================
# Security utils
# =========================
def verify_line_signature(body: bytes, signature: str):
    if not LINE_CHANNEL_SECRET:
        return
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    if not hmac.compare_digest(signature.strip(), expected):
        raise HTTPException(400, "Invalid Signature")

def require_admin(request: Request):
    if not ADMIN_TOKEN:
        raise HTTPException(500, "ADMIN_TOKEN not set")
    token = request.headers.get("X-Admin-Token", "")
    if not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(401, "Unauthorized")

def rate_limit_ok(user_id: str) -> bool:
    now = time.time()
    window_start = now - 60
    ts = RATE_STORE.get(user_id, [])
    ts = [t for t in ts if t >= window_start]
    if len(ts) >= RATE_LIMIT_PER_MIN:
        RATE_STORE[user_id] = ts
        return False
    ts.append(now)
    RATE_STORE[user_id] = ts
    return True

# =========================
# DB init (tables + unique index for de-dup)
# =========================
def ensure_tables():
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sales (
            date TEXT,
            customer TEXT,
            product TEXT,
            quantity REAL,
            amount REAL,
            year INTEGER
        );
        """))
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS purchase (
            date TEXT,
            supplier TEXT,
            product TEXT,
            quantity REAL,
            amount REAL,
            year INTEGER
        );
        """))
        conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_sales_row
        ON sales(date, customer, product, amount, quantity);
        """))
        conn.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_purchase_row
        ON purchase(date, supplier, product, amount, quantity);
        """))

def table_counts() -> Dict[str, int]:
    insp = inspect(engine)
    names = set(insp.get_table_names())
    out = {"sales": 0, "purchase": 0}
    with engine.connect() as conn:
        for t in out.keys():
            if t in names:
                out[t] = int(conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0)
    return out

# =========================
# Google Sheet download (stable): export xlsx
# =========================
def get_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    # also accept id= form
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else ""

def download_google_sheet_xlsx(sheet_url: str, dest_path: str, max_retries: int = 4) -> bool:
    sheet_id = get_sheet_id(sheet_url)
    if not sheet_id:
        return False

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
    params = {"format": "xlsx"}

    for attempt in range(max_retries):
        try:
            r = requests.get(export_url, params=params, stream=True, timeout=60)
            if r.status_code != 200:
                logger.error(f"Sheet export failed: {r.status_code}")
                time.sleep(min(10, 2 ** attempt))
                continue

            # xlsx is zip -> starts with PK
            first2 = r.raw.read(2)
            if first2 != b"PK":
                logger.error("Not xlsx content (permission page/HTML?)")
                time.sleep(min(10, 2 ** attempt))
                continue

            with open(dest_path, "wb") as f:
                f.write(first2)
                for chunk in r.iter_content(32768):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logger.error(f"download error attempt {attempt+1}: {e}")
            time.sleep(min(10, 2 ** attempt))
    return False

# =========================
# ETL: normalize sheets
# =========================
def normalize_sales_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = df.columns.astype(str).str.strip()
    if "日期(轉換)" not in df.columns or "進銷明細未稅金額" not in df.columns:
        return None

    clean = pd.DataFrame({
        "date": pd.to_datetime(df["日期(轉換)"], errors="coerce"),
        "customer": df.get("客戶供應商簡稱", "").astype(str).str.strip(),
        "product": df.get("品名", "").astype(str).str.strip(),
        "quantity": pd.to_numeric(df.get("數量", 0), errors="coerce").fillna(0),
        "amount": pd.to_numeric(df["進銷明細未稅金額"], errors="coerce").fillna(0),
    }).dropna(subset=["date"])

    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean

def normalize_purchase_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = df.columns.astype(str).str.strip()
    if "日期(轉換)" not in df.columns or "進銷明細未稅金額" not in df.columns:
        return None

    prod_col = "對方品名/品名備註" if "對方品名/品名備註" in df.columns else "品名"
    clean = pd.DataFrame({
        "date": pd.to_datetime(df["日期(轉換)"], errors="coerce"),
        "supplier": df.get("客戶供應商簡稱", "").astype(str).str.strip(),
        "product": df.get(prod_col, "").astype(str).str.strip(),
        "quantity": pd.to_numeric(df.get("數量", 0), errors="coerce").fillna(0),
        "amount": pd.to_numeric(df["進銷明細未稅金額"], errors="coerce").fillna(0),
    }).dropna(subset=["date"])

    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean

def upsert_rows(kind: str, rows: List[Dict[str, Any]]) -> int:
    dialect = engine.url.get_backend_name()
    inserted = 0

    with engine.begin() as conn:
        if kind == "sales":
            if dialect == "postgresql":
                stmt = text("""
                INSERT INTO sales(date, customer, product, quantity, amount, year)
                VALUES (:date, :customer, :product, :quantity, :amount, :year)
                ON CONFLICT (date, customer, product, amount, quantity) DO NOTHING;
                """)
            else:
                stmt = text("""
                INSERT OR IGNORE INTO sales(date, customer, product, quantity, amount, year)
                VALUES (:date, :customer, :product, :quantity, :amount, :year);
                """)
        else:
            if dialect == "postgresql":
                stmt = text("""
                INSERT INTO purchase(date, supplier, product, quantity, amount, year)
                VALUES (:date, :supplier, :product, :quantity, :amount, :year)
                ON CONFLICT (date, supplier, product, amount, quantity) DO NOTHING;
                """)
            else:
                stmt = text("""
                INSERT OR IGNORE INTO purchase(date, supplier, product, quantity, amount, year)
                VALUES (:date, :supplier, :product, :quantity, :amount, :year);
                """)

        for r in rows:
            conn.execute(stmt, r)
            inserted += 1

    return inserted

def import_data_to_db() -> Dict[str, Any]:
    ensure_tables()
    before = table_counts()
    msgs = []

    tmp_sales = f"./_sales_{int(time.time())}.xlsx"
    tmp_purchase = f"./_purchase_{int(time.time())}.xlsx"

    # sales
    if SALES_SHEET_URL:
        if download_google_sheet_xlsx(SALES_SHEET_URL, tmp_sales):
            xls = pd.read_excel(tmp_sales, sheet_name=None)
            dfs = []
            for _, df in xls.items():
                n = normalize_sales_df(df)
                if n is not None and len(n) > 0:
                    dfs.append(n)
            if dfs:
                final = pd.concat(dfs, ignore_index=True)
                upsert_rows("sales", final.to_dict(orient="records"))
                msgs.append(f"sales: 讀到 {len(final)} 筆，已嘗試增量匯入")
            else:
                msgs.append("sales: 沒找到符合欄位的分頁")
        else:
            msgs.append("sales: 下載失敗（請確認連結權限/格式）")
    else:
        msgs.append("sales: 未設定 SALES_EXCEL_URL")

    # purchase
    if PURCHASE_SHEET_URL:
        if download_google_sheet_xlsx(PURCHASE_SHEET_URL, tmp_purchase):
            xls = pd.read_excel(tmp_purchase, sheet_name=None)
            dfs = []
            for _, df in xls.items():
                n = normalize_purchase_df(df)
                if n is not None and len(n) > 0:
                    dfs.append(n)
            if dfs:
                final = pd.concat(dfs, ignore_index=True)
                upsert_rows("purchase", final.to_dict(orient="records"))
                msgs.append(f"purchase: 讀到 {len(final)} 筆，已嘗試增量匯入")
            else:
                msgs.append("purchase: 沒找到符合欄位的分頁")
        else:
            msgs.append("purchase: 下載失敗（請確認連結權限/格式）")
    else:
        msgs.append("purchase: 未設定 PURCHASE_EXCEL_URL")

    # cleanup
    for p in [tmp_sales, tmp_purchase]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except:
            pass

    after = table_counts()
    return {"ok": True, "before": before, "after": after, "messages": msgs}

# =========================
# LLM: ONLY output a JSON plan (no SQL)
# =========================
PLAN_SYSTEM = """你是 ERP 問題解析器。你只能輸出 JSON（不要任何多餘文字）。
把使用者問題轉成查詢計畫。

輸出 schema：
{
  "table": "sales" | "purchase",
  "metric": "amount" | "quantity",
  "agg": "detail" | "sum" | "top_party" | "top_products" | "trend_month",
  "keyword": "string",
  "year": 2020..2035 | null,
  "limit": 5..30
}

規則：
- 銷售/客戶/業績/出貨 → table=sales
- 採購/供應商/進貨 → table=purchase
- 沒指定金額/數量 → metric=amount
- 問總額/總和 → agg=sum
- 問排名/前幾/最大 → agg=top_party 或 top_products（依語意）
- 問趨勢/每月/走勢 → agg=trend_month
- 其他 → agg=detail
- keyword：抓客戶/供應商/品名關鍵字，沒有則空字串
- 若使用者說今年/去年，year 用台灣時間換算
"""

@dataclass
class Plan:
    table: str
    metric: str
    agg: str
    keyword: str
    year: Optional[int]
    limit: int

def parse_plan(user_text: str) -> Plan:
    if not client:
        return Plan("sales", "amount", "detail", "", None, 10)

    today = now_taipei().date().isoformat()
    prompt = f"今天台灣日期是 {today}。使用者問題：{user_text}"

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(system_instruction=PLAN_SYSTEM, temperature=0.1)
    )

    raw = (resp.text or "").strip()
    try:
        obj = json.loads(raw)
    except:
        return Plan("sales", "amount", "detail", "", None, 10)

    def clamp_limit(x):
        try:
            x = int(x)
        except:
            x = 10
        return max(5, min(30, x))

    table = obj.get("table", "sales")
    if table not in ["sales", "purchase"]:
        table = "sales"

    metric = obj.get("metric", "amount")
    if metric not in ["amount", "quantity"]:
        metric = "amount"

    agg = obj.get("agg", "detail")
    if agg not in ["detail", "sum", "top_party", "top_products", "trend_month"]:
        agg = "detail"

    year = obj.get("year", None)
    if isinstance(year, int):
        if year < 2000 or year > 2100:
            year = None
    else:
        year = None

    keyword = (obj.get("keyword") or "").strip()
    limit = clamp_limit(obj.get("limit", 10))

    return Plan(table, metric, agg, keyword, year, limit)

# =========================
# SAFE SQL templates
# =========================
def build_where(plan: Plan) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {}
    where = []

    if plan.year:
        where.append("year = :year")
        params["year"] = plan.year

    if plan.keyword:
        params["kw"] = f"%{plan.keyword}%"
        if plan.table == "sales":
            where.append("(customer LIKE :kw OR product LIKE :kw)")
        else:
            where.append("(supplier LIKE :kw OR product LIKE :kw)")

    clause = " WHERE " + " AND ".join(where) if where else ""
    return clause, params

def run_df(sql: str, params: Dict[str, Any]) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)

def format_text(df: pd.DataFrame) -> str:
    # LINE-friendly simple text
    lines = []
    for _, r in df.iterrows():
        lines.append(" - " + " | ".join(str(x) for x in r.tolist()))
    return "\n".join(lines)

def answer_from_plan(user_text: str) -> str:
    ensure_tables()
    counts = table_counts()
    if counts["sales"] == 0 and counts["purchase"] == 0:
        return "目前資料庫沒有資料，請先匯入 Google Sheet。"

    plan = parse_plan(user_text)
    where, params = build_where(plan)
    metric = "amount" if plan.metric == "amount" else "quantity"
    value_label = "金額" if plan.metric == "amount" else "數量"

    if plan.table == "sales":
        party_col = "customer"
        party_label = "客戶"
    else:
        party_col = "supplier"
        party_label = "供應商"

    # SUM
    if plan.agg == "sum":
        df = run_df(f"SELECT SUM({metric}) AS total FROM {plan.table}{where};", params)
        total = float(df.iloc[0]["total"] or 0)
        return f"{value_label}總和：{total:,.2f}"

    # TOP party
    if plan.agg == "top_party":
        p = dict(params); p["limit"] = plan.limit
        df = run_df(f"""
            SELECT {party_col} AS name, SUM({metric}) AS value
            FROM {plan.table}
            {where}
            GROUP BY {party_col}
            ORDER BY value DESC
            LIMIT :limit;
        """, p)
        if df.empty:
            # relax: remove year once
            relaxed = Plan(plan.table, plan.metric, plan.agg, plan.keyword, None, plan.limit)
            where2, params2 = build_where(relaxed)
            p2 = dict(params2); p2["limit"] = plan.limit
            df = run_df(f"""
                SELECT {party_col} AS name, SUM({metric}) AS value
                FROM {plan.table}
                {where2}
                GROUP BY {party_col}
                ORDER BY value DESC
                LIMIT :limit;
            """, p2)
            if df.empty:
                return "查不到資料。建議：縮短關鍵字或先不要指定年份。"

        lines = [f"Top {len(df)} {party_label}（依{value_label}）"]
        lines += [f" - {r['name']}: {float(r['value']):,.2f}" for _, r in df.iterrows()]
        return "\n".join(lines)

    # TOP products
    if plan.agg == "top_products":
        p = dict(params); p["limit"] = plan.limit
        df = run_df(f"""
            SELECT product AS name, SUM({metric}) AS value
            FROM {plan.table}
            {where}
            GROUP BY product
            ORDER BY value DESC
            LIMIT :limit;
        """, p)
        if df.empty:
            return "查不到資料。建議：縮短關鍵字或先不要指定年份。"
        lines = [f"Top {len(df)} 品名（依{value_label}）"]
        lines += [f" - {r['name']}: {float(r['value']):,.2f}" for _, r in df.iterrows()]
        return "\n".join(lines)

    # Trend by month
    if plan.agg == "trend_month":
        df = run_df(f"""
            SELECT SUBSTR(date, 1, 7) AS month, SUM({metric}) AS value
            FROM {plan.table}
            {where}
            GROUP BY SUBSTR(date, 1, 7)
            ORDER BY month ASC
            LIMIT 60;
        """, params)
        if df.empty:
            return "查不到趨勢資料。建議：先不要指定年份/關鍵字。"
        lines = [f"每月{value_label}趨勢（{plan.table}）"]
        lines += [f" - {r['month']}: {float(r['value']):,.2f}" for _, r in df.iterrows()]
        return "\n".join(lines)

    # Detail (latest)
    p = dict(params); p["limit"] = plan.limit
    if plan.table == "sales":
        df = run_df(f"""
            SELECT date, customer, product, quantity, amount
            FROM sales
            {where}
            ORDER BY date DESC
            LIMIT :limit;
        """, p)
        if df.empty:
            return "查不到明細。建議：縮短關鍵字或先不要指定年份。"
        lines = ["最新銷售明細（最多顯示前幾筆）"]
        for _, r in df.iterrows():
            lines.append(f" - {r['date']} | {r['customer']} | {r['product']} | 數量 {float(r['quantity']):,.2f} | 金額 {float(r['amount']):,.2f}")
        return "\n".join(lines)

    else:
        df = run_df(f"""
            SELECT date, supplier, product, quantity, amount
            FROM purchase
            {where}
            ORDER BY date DESC
            LIMIT :limit;
        """, p)
        if df.empty:
            return "查不到明細。建議：縮短關鍵字或先不要指定年份。"
        lines = ["最新採購明細（最多顯示前幾筆）"]
        for _, r in df.iterrows():
            lines.append(f" - {r['date']} | {r['supplier']} | {r['product']} | 數量 {float(r['quantity']):,.2f} | 金額 {float(r['amount']):,.2f}")
        return "\n".join(lines)

# =========================
# LINE reply
# =========================
async def reply_line(reply_token: str, text_out: str):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"replyToken": reply_token, "messages": [{"type": "text", "text": text_out[:4999]}]}
    async with httpx.AsyncClient(timeout=20) as c:
        await c.post("https://api.line.me/v2/bot/message/reply", headers=headers, json=payload)

# =========================
# API routes
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "ERP Bot Ultra PRO"}

@app.get("/health")
def health():
    ensure_tables()
    return {"status": "ok", "counts": table_counts()}

@app.post("/admin/reload_sync")
def admin_reload_sync(request: Request):
    require_admin(request)
    return import_data_to_db()

@app.post("/line/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    verify_line_signature(body, signature)

    try:
        events = json.loads(body.decode("utf-8")).get("events", [])
    except:
        return {"ok": False}

    for event in events:
        if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
            user_id = event["source"]["userId"]
            user_text = event["message"]["text"]
            reply_token = event["replyToken"]

            if user_text.strip().lower() in ["/reset", "清除"]:
                # no conversation memory needed for this pro design
                background_tasks.add_task(reply_line, reply_token, "已清除（本版本不依賴長對話記憶）")
                continue

            if not rate_limit_ok(user_id):
                background_tasks.add_task(reply_line, reply_token, "請稍後再試（請求過於頻繁）")
                continue

            background_tasks.add_task(handle_message, user_text, reply_token)

    return {"ok": True}

async def handle_message(user_text: str, reply_token: str):
    try:
        ans = answer_from_plan(user_text)
        await reply_line(reply_token, ans)
    except Exception as e:
        logger.error(f"handle_message error: {e}")
        await reply_line(reply_token, "系統忙碌中，請稍後再試。")

@app.on_event("startup")
async def startup():
    ensure_tables()
    if AUTO_IMPORT_ON_STARTUP:
        try:
            import_data_to_db()
        except Exception as e:
            logger.error(f"startup import error: {e}")
