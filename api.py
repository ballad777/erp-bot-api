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
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import httpx

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

# Gemini SDK
from google import genai
from google.genai import types

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("erpbot_pro")

app = FastAPI(title="ERP Bot Pro", version="Commercial_Pro")

# =========================
# Settings
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

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # required to call /admin/*
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "40"))

# Render free: avoid startup import. Use GitHub Actions daily trigger instead.
AUTO_IMPORT_ON_STARTUP = os.getenv("AUTO_IMPORT_ON_STARTUP", "0") == "1"

engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# In-memory rate store (good enough; for multi-instance use Redis)
RATE_STORE: Dict[str, List[float]] = {}

# =========================
# Utilities
# =========================
def now_taipei() -> datetime:
    # Simple fixed offset; good enough for this use-case
    return datetime.utcnow() + timedelta(hours=8)

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

def require_admin(request: Request):
    if not ADMIN_TOKEN:
        raise HTTPException(500, "ADMIN_TOKEN not set")
    token = request.headers.get("X-Admin-Token", "")
    if not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(401, "Unauthorized")

def get_drive_id(url: str) -> str:
    patterns = [
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return ""

def download_google_sheet_xlsx(sheet_url: str, dest_path: str, max_retries: int = 4) -> bool:
    sheet_id = get_drive_id(sheet_url)
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

            # xlsx is a zip -> "PK"
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
# DB schema & migrations
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

        # de-dup unique index
        # SQLite supports it; Postgres supports it
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
    names = insp.get_table_names()
    out = {"sales": 0, "purchase": 0}
    with engine.connect() as conn:
        for t in ["sales", "purchase"]:
            if t in names:
                out[t] = int(conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0)
    return out

# =========================
# Import pipeline (Google Sheet -> DB)
# =========================
def normalize_sales_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = df.columns.astype(str).str.strip()
    need = {"日期(轉換)", "進銷明細未稅金額"}
    if not need.issubset(set(df.columns)):
        return None

    clean = pd.DataFrame({
        "date": pd.to_datetime(df["日期(轉換)"], errors="coerce"),
        "customer": df.get("客戶供應商簡稱", "").astype(str),
        "product": df.get("品名", "").astype(str),
        "quantity": pd.to_numeric(df.get("數量", 0), errors="coerce").fillna(0),
        "amount": pd.to_numeric(df["進銷明細未稅金額"], errors="coerce").fillna(0),
    }).dropna(subset=["date"])

    clean["customer"] = clean["customer"].str.strip()
    clean["product"] = clean["product"].str.strip()
    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean

def normalize_purchase_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = df.columns.astype(str).str.strip()
    need = {"日期(轉換)", "進銷明細未稅金額"}
    if not need.issubset(set(df.columns)):
        return None

    prod_col = "對方品名/品名備註" if "對方品名/品名備註" in df.columns else "品名"
    clean = pd.DataFrame({
        "date": pd.to_datetime(df["日期(轉換)"], errors="coerce"),
        "supplier": df.get("客戶供應商簡稱", "").astype(str),
        "product": df.get(prod_col, "").astype(str),
        "quantity": pd.to_numeric(df.get("數量", 0), errors="coerce").fillna(0),
        "amount": pd.to_numeric(df["進銷明細未稅金額"], errors="coerce").fillna(0),
    }).dropna(subset=["date"])

    clean["supplier"] = clean["supplier"].str.strip()
    clean["product"] = clean["product"].str.strip()
    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    return clean

def upsert_rows(kind: str, rows: List[Dict[str, Any]]) -> int:
    dialect = engine.url.get_backend_name()
    inserted_attempted = 0

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
            inserted_attempted += 1

    return inserted_attempted

def import_from_sheets() -> Dict[str, Any]:
    ensure_tables()

    report = {"ok": True, "counts_before": table_counts(), "messages": [], "counts_after": None}

    tmp_sales = f"./_sales_{int(time.time())}.xlsx"
    tmp_purchase = f"./_purchase_{int(time.time())}.xlsx"

    # Sales
    if SALES_SHEET_URL:
        ok = download_google_sheet_xlsx(SALES_SHEET_URL, tmp_sales)
        if not ok:
            report["messages"].append("sales: 下載失敗（請確認連結權限/格式）")
        else:
            xls = pd.read_excel(tmp_sales, sheet_name=None)
            dfs = []
            for _, df in xls.items():
                n = normalize_sales_df(df)
                if n is not None and len(n) > 0:
                    dfs.append(n)
            if dfs:
                final = pd.concat(dfs, ignore_index=True)
                upsert_rows("sales", final.to_dict(orient="records"))
                report["messages"].append(f"sales: 讀到 {len(final)} 筆，已嘗試匯入")
            else:
                report["messages"].append("sales: 沒找到符合欄位的分頁")
    else:
        report["messages"].append("sales: 未設定 SALES_EXCEL_URL")

    # Purchase
    if PURCHASE_SHEET_URL:
        ok = download_google_sheet_xlsx(PURCHASE_SHEET_URL, tmp_purchase)
        if not ok:
            report["messages"].append("purchase: 下載失敗（請確認連結權限/格式）")
        else:
            xls = pd.read_excel(tmp_purchase, sheet_name=None)
            dfs = []
            for _, df in xls.items():
                n = normalize_purchase_df(df)
                if n is not None and len(n) > 0:
                    dfs.append(n)
            if dfs:
                final = pd.concat(dfs, ignore_index=True)
                upsert_rows("purchase", final.to_dict(orient="records"))
                report["messages"].append(f"purchase: 讀到 {len(final)} 筆，已嘗試匯入")
            else:
                report["messages"].append("purchase: 沒找到符合欄位的分頁")
    else:
        report["messages"].append("purchase: 未設定 PURCHASE_EXCEL_URL")

    # cleanup
    for p in [tmp_sales, tmp_purchase]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except:
            pass

    report["counts_after"] = table_counts()
    report["ok"] = True
    return report

# =========================
# Chart URL (QuickChart) - no file storage
# =========================
def quickchart_url(chart_type: str, title: str, labels: List[str], values: List[float]) -> str:
    if chart_type not in ["line", "bar", "pie"]:
        chart_type = "line"

    if chart_type == "line":
        cfg = {
            "type": "line",
            "data": {"labels": labels, "datasets": [{"label": title, "data": values}]},
            "options": {"plugins": {"title": {"display": True, "text": title}},
                        "scales": {"y": {"beginAtZero": True}}}
        }
    elif chart_type == "bar":
        cfg = {
            "type": "bar",
            "data": {"labels": labels, "datasets": [{"label": title, "data": values}]},
            "options": {"plugins": {"title": {"display": True, "text": title}}}
        }
    else:
        cfg = {
            "type": "pie",
            "data": {"labels": labels, "datasets": [{"data": values}]},
            "options": {"plugins": {"title": {"display": True, "text": title}}}
        }

    c = json.dumps(cfg, ensure_ascii=False)
    return "https://quickchart.io/chart?c=" + urllib.parse.quote(c)

# =========================
# “頂級商用”核心：LLM 只做意圖解析 (JSON)，SQL 由後端模板生成
# =========================
INTENT_SYSTEM = """你是一個 ERP 問題解析器。你只能輸出 JSON（不要任何多餘文字）。
目標：把使用者問題轉成後端可執行的查詢計畫。

輸出 JSON schema（必須符合）：
{
  "table": "sales" | "purchase",
  "metric": "amount" | "quantity",
  "agg": "sum" | "top_customers" | "top_products" | "trend_month" | "trend_day" | "detail",
  "keyword": "string",          // 可能是客戶/供應商/品名關鍵字，沒有就空字串
  "year": 2025 | null,          // 有提到年份就填，沒提就 null
  "date_from": "YYYY-MM-DD" | null,
  "date_to": "YYYY-MM-DD" | null,
  "limit": 5-20,                // top/detail 的筆數，預設 10
  "want_chart": true|false,
  "chart_type": "line"|"bar"|"pie"
}

規則：
- 使用者問「銷售/客戶/出貨/業績」→ table="sales"
- 使用者問「採購/供應商/進貨」→ table="purchase"
- 沒指定金額/數量時：預設 metric="amount"
- 問「趨勢/每月/走勢」→ agg="trend_month"
- 問「Top/排名/前幾」→ agg="top_customers" 或 "top_products"（依問題語意）
- 沒講趨勢也沒講Top：agg="detail"
- 想要圖表：只有使用者提「圖表/折線/長條/圓餅/畫出來/趨勢圖」才 want_chart=true
- chart_type 預設：趨勢→line、Top→bar、占比→pie
- keyword：抓最像的客戶/品名詞（可能有錯字，保留原樣）
- 若使用者說「今年/去年/本月」請換算到 date_from/date_to（以台灣時間為準）
"""

@dataclass
class Plan:
    table: str
    metric: str
    agg: str
    keyword: str
    year: Optional[int]
    date_from: Optional[str]
    date_to: Optional[str]
    limit: int
    want_chart: bool
    chart_type: str

def parse_user_plan(user_text: str) -> Plan:
    if not client:
        # fallback: default safe plan
        return Plan("sales", "amount", "detail", "", None, None, None, 10, False, "line")

    taipei = now_taipei().date()
    # give model today's date context
    user_with_context = f"今天台灣日期是 {taipei.isoformat()}。使用者問題：{user_text}"

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Content(role="user", parts=[types.Part(text=user_with_context)])],
        config=types.GenerateContentConfig(
            system_instruction=INTENT_SYSTEM,
            temperature=0.1
        )
    )

    raw = (resp.text or "").strip()
    # model must output json; still guard
    try:
        obj = json.loads(raw)
    except:
        # fallback safe
        return Plan("sales", "amount", "detail", "", None, None, None, 10, False, "line")

    def clamp_limit(x):
        try:
            x = int(x)
        except:
            return 10
        return max(5, min(20, x))

    return Plan(
        table=obj.get("table", "sales") if obj.get("table") in ["sales", "purchase"] else "sales",
        metric=obj.get("metric", "amount") if obj.get("metric") in ["amount", "quantity"] else "amount",
        agg=obj.get("agg", "detail"),
        keyword=(obj.get("keyword") or "").strip(),
        year=obj.get("year", None),
        date_from=obj.get("date_from", None),
        date_to=obj.get("date_to", None),
        limit=clamp_limit(obj.get("limit", 10)),
        want_chart=bool(obj.get("want_chart", False)),
        chart_type=obj.get("chart_type", "line") if obj.get("chart_type") in ["line","bar","pie"] else "line"
    )

# =========================
# SQL template generator (SAFE)
# =========================
def build_filters(plan: Plan) -> Tuple[str, Dict[str, Any]]:
    params: Dict[str, Any] = {}
    where = []

    # date range
    if plan.date_from and plan.date_to:
        where.append("date BETWEEN :date_from AND :date_to")
        params["date_from"] = plan.date_from
        params["date_to"] = plan.date_to
    elif plan.year:
        where.append("year = :year")
        params["year"] = plan.year

    # keyword fuzzy match
    if plan.keyword:
        like = f"%{plan.keyword}%"
        if plan.table == "sales":
            where.append("(customer LIKE :kw OR product LIKE :kw)")
        else:
            where.append("(supplier LIKE :kw OR product LIKE :kw)")
        params["kw"] = like

    clause = " WHERE " + " AND ".join(where) if where else ""
    return clause, params

def run_query(sql: str, params: Dict[str, Any]) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

def execute_plan(plan: Plan) -> Dict[str, Any]:
    """
    Returns:
    {
      "summary": "...",
      "table_text": "...",
      "chart_url": Optional[str]
    }
    """
    ensure_tables()
    counts = table_counts()
    if counts["sales"] == 0 and counts["purchase"] == 0:
        return {
            "summary": "系統目前沒有資料（sales/purchase 都是 0 筆）。請先匯入。",
            "table_text": "",
            "chart_url": None
        }

    where_clause, params = build_filters(plan)

    metric_col = "amount" if plan.metric == "amount" else "quantity"
    value_label = "金額" if plan.metric == "amount" else "數量"

    # Determine group columns
    if plan.table == "sales":
        party_col = "customer"
        party_label = "客戶"
    else:
        party_col = "supplier"
        party_label = "供應商"

    # Build SQL by agg type
    if plan.agg == "sum":
        sql = f"SELECT SUM({metric_col}) AS total FROM {plan.table}{where_clause};"
        df = run_query(sql, params)
        total = float(df.iloc[0]["total"] or 0)
        return {
            "summary": f"{plan.table} {value_label}總和：{total:,.2f}",
            "table_text": "",
            "chart_url": None
        }

    if plan.agg == "top_customers":
        sql = f"""
        SELECT {party_col} AS name, SUM({metric_col}) AS value
        FROM {plan.table}
        {where_clause}
        GROUP BY {party_col}
        ORDER BY value DESC
        LIMIT :limit;
        """
        params2 = dict(params)
        params2["limit"] = plan.limit
        df = run_query(sql, params2)

        if df.empty:
            return {"summary": "查不到資料。建議：換關鍵字或移除年份/日期限制。", "table_text": "", "chart_url": None}

        table_text = format_table(df, headers=[party_label, value_label], cols=["name", "value"])
        summary = f"Top {len(df)} {party_label}（依{value_label}）"
        chart_url = None
        if plan.want_chart:
            labels = df["name"].astype(str).tolist()
            values = df["value"].astype(float).tolist()
            chart_url = quickchart_url(plan.chart_type or "bar", summary, labels, values)

        return {"summary": summary, "table_text": table_text, "chart_url": chart_url}

    if plan.agg == "top_products":
        sql = f"""
        SELECT product AS name, SUM({metric_col}) AS value
        FROM {plan.table}
        {where_clause}
        GROUP BY product
        ORDER BY value DESC
        LIMIT :limit;
        """
        params2 = dict(params)
        params2["limit"] = plan.limit
        df = run_query(sql, params2)

        if df.empty:
            return {"summary": "查不到資料。建議：換關鍵字或移除年份/日期限制。", "table_text": "", "chart_url": None}

        table_text = format_table(df, headers=["品名", value_label], cols=["name", "value"])
        summary = f"Top {len(df)} 品名（依{value_label}）"
        chart_url = None
        if plan.want_chart:
            labels = df["name"].astype(str).tolist()
            values = df["value"].astype(float).tolist()
            chart_url = quickchart_url(plan.chart_type or "bar", summary, labels, values)

        return {"summary": summary, "table_text": table_text, "chart_url": chart_url}

    if plan.agg in ["trend_month", "trend_day"]:
        # For SQLite/Postgres, do simple substring grouping:
        # date stored as YYYY-MM-DD
        if plan.agg == "trend_month":
            group_expr = "SUBSTR(date, 1, 7)"  # YYYY-MM
            x_label = "月份"
        else:
            group_expr = "date"
            x_label = "日期"

        sql = f"""
        SELECT {group_expr} AS x, SUM({metric_col}) AS value
        FROM {plan.table}
        {where_clause}
        GROUP BY {group_expr}
        ORDER BY x ASC
        LIMIT 60;
        """
        df = run_query(sql, params)

        if df.empty:
            return {"summary": "查不到趨勢資料。建議：放寬日期/年份或換關鍵字。", "table_text": "", "chart_url": None}

        table_text = format_table(df, headers=[x_label, value_label], cols=["x", "value"])
        summary = f"{x_label}{value_label}趨勢（{plan.table}）"
        chart_url = None
        if plan.want_chart:
            labels = df["x"].astype(str).tolist()
            values = df["value"].astype(float).tolist()
            chart_url = quickchart_url(plan.chart_type or "line", summary, labels, values)

        return {"summary": summary, "table_text": table_text, "chart_url": chart_url}

    # Default: detail
    if plan.table == "sales":
        sql = f"""
        SELECT date, customer, product, quantity, amount
        FROM sales
        {where_clause}
        ORDER BY date DESC
        LIMIT :limit;
        """
        params2 = dict(params)
        params2["limit"] = plan.limit
        df = run_query(sql, params2)
        if df.empty:
            # auto relax: drop year/date filters once
            relaxed = Plan(**{**plan.__dict__, "year": None, "date_from": None, "date_to": None})
            where2, p2 = build_filters(relaxed)
            params3 = dict(p2); params3["limit"] = plan.limit
            df2 = run_query(f"""
                SELECT date, customer, product, quantity, amount
                FROM sales
                {where2}
                ORDER BY date DESC
                LIMIT :limit;
            """, params3)
            if df2.empty:
                return {"summary": "查不到資料。建議：關鍵字縮短、改客戶/品名其中一個試試。", "table_text": "", "chart_url": None}
            df = df2
            summary = "查無資料（已自動放寬年份/日期後找到結果）"
        else:
            summary = "明細查詢結果"

        table_text = format_table(df, headers=["日期","客戶","品名","數量","金額"], cols=["date","customer","product","quantity","amount"])
        return {"summary": summary, "table_text": table_text, "chart_url": None}

    else:
        sql = f"""
        SELECT date, supplier, product, quantity, amount
        FROM purchase
        {where_clause}
        ORDER BY date DESC
        LIMIT :limit;
        """
        params2 = dict(params)
        params2["limit"] = plan.limit
        df = run_query(sql, params2)
        if df.empty:
            relaxed = Plan(**{**plan.__dict__, "year": None, "date_from": None, "date_to": None})
            where2, p2 = build_filters(relaxed)
            params3 = dict(p2); params3["limit"] = plan.limit
            df2 = run_query(f"""
                SELECT date, supplier, product, quantity, amount
                FROM purchase
                {where2}
                ORDER BY date DESC
                LIMIT :limit;
            """, params3)
            if df2.empty:
                return {"summary": "查不到資料。建議：關鍵字縮短、改供應商/品名其中一個試試。", "table_text": "", "chart_url": None}
            df = df2
            summary = "查無資料（已自動放寬年份/日期後找到結果）"
        else:
            summary = "明細查詢結果"

        table_text = format_table(df, headers=["日期","供應商","品名","數量","金額"], cols=["date","supplier","product","quantity","amount"])
        return {"summary": summary, "table_text": table_text, "chart_url": None}

# =========================
# Text table formatting (LINE friendly)
# =========================
def format_table(df: pd.DataFrame, headers: List[str], cols: List[str], max_width: int = 24) -> str:
    show = df.copy()

    # Convert numbers
    for c in cols:
        if c in show.columns:
            if show[c].dtype.kind in "fi":
                show[c] = show[c].apply(lambda x: 0 if x is None else x)

    # Truncate
    def trunc(s: str) -> str:
        s = str(s)
        return s if len(s) <= max_width else s[: max_width - 1] + "…"

    for c in cols:
        show[c] = show[c].astype(str).apply(trunc)

    # column widths
    widths = []
    for h, c in zip(headers, cols):
        w = max(len(h), show[c].astype(str).map(len).max())
        widths.append(min(max_width, w))

    # build lines
    def pad(s: str, w: int) -> str:
        s = str(s)
        if len(s) > w:
            s = s[: w - 1] + "…"
        return s + " " * max(0, w - len(s))

    header_line = " | ".join(pad(h, w) for h, w in zip(headers, widths))
    sep_line = "-+-".join("-" * w for w in widths)

    rows = []
    for _, r in show.iterrows():
        row = " | ".join(pad(r[c], w) for c, w in zip(cols, widths))
        rows.append(row)

    return "\n".join([header_line, sep_line] + rows)

# =========================
# LINE webhook handlers
# =========================
async def reply_line(reply_token: str, text_out: str, image_url: Optional[str] = None):
    if not LINE_CHANNEL_ACCESS_TOKEN:
        return
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    messages = []
    if image_url:
        messages.append({"type": "image", "originalContentUrl": image_url, "previewImageUrl": image_url})
    messages.append({"type": "text", "text": text_out[:4999]})

    async with httpx.AsyncClient(timeout=20) as c:
        await c.post(
            "https://api.line.me/v2/bot/message/reply",
            headers=headers,
            json={"replyToken": reply_token, "messages": messages}
        )

def verify_line_signature(body: bytes, signature: str):
    if not LINE_CHANNEL_SECRET:
        return
    mac = hmac.new(LINE_CHANNEL_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    if not hmac.compare_digest(signature.strip(), expected):
        raise HTTPException(400, "Invalid Signature")

# =========================
# API
# =========================
@app.get("/")
def root():
    return {"status": "ok", "service": "ERP Bot Pro"}

@app.get("/health")
def health():
    ensure_tables()
    return {"status": "ok", "counts": table_counts()}

@app.post("/admin/reload_sync")
def admin_reload_sync(request: Request):
    require_admin(request)
    report = import_from_sheets()
    return report

@app.post("/line/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    verify_line_signature(body, signature)

    try:
        payload = json.loads(body.decode("utf-8"))
        events = payload.get("events", [])
    except:
        return {"ok": False}

    for event in events:
        if event.get("type") == "message" and event.get("message", {}).get("type") == "text":
            user_id = event["source"]["userId"]
            user_text = event["message"]["text"]
            reply_token = event["replyToken"]

            if not rate_limit_ok(user_id):
                background_tasks.add_task(reply_line, reply_token, "請稍後再試（請求過於頻繁）")
                continue

            background_tasks.add_task(handle_message, user_text, reply_token)

    return {"ok": True}

async def handle_message(user_text: str, reply_token: str):
    try:
        if user_text.strip().lower() in ["/help", "help", "指令"]:
            msg = (
                "可用問法例：\n"
                "1) 2025 華碩銷售總額\n"
                "2) 2025 Top 10 客戶銷售額\n"
                "3) 今年每月銷售趨勢（畫圖）\n"
                "4) 供應商XX採購明細\n"
                "小技巧：加上「畫圖/折線/長條」可附圖。"
            )
            await reply_line(reply_token, msg)
            return

        plan = parse_user_plan(user_text)
        result = execute_plan(plan)

        summary = result["summary"]
        table_text = result["table_text"]
        chart_url = result.get("chart_url")

        # 商用輸出格式：結論 + 明細表（如有）
        final_text = summary
        if table_text:
            final_text += "\n\n" + table_text

        await reply_line(reply_token, final_text, chart_url)

    except Exception as e:
        logger.error(f"handle_message error: {e}")
        await reply_line(reply_token, "系統忙碌中，請稍後再試。")

@app.on_event("startup")
async def startup():
    ensure_tables()
    if AUTO_IMPORT_ON_STARTUP:
        try:
            import_from_sheets()
        except Exception as e:
            logger.error(f"startup import error: {e}")
