import os
import re
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import requests
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_loader")

# =========================
# Env
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Google Sheet URL (not local xlsx)
SALES_SHEET_URL = os.getenv("SALES_EXCEL_URL", "").strip()
PURCHASE_SHEET_URL = os.getenv("PURCHASE_EXCEL_URL", "").strip()

engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


# =========================
# DB init + dedup safe indexes
# =========================
def ensure_tables_and_indexes() -> None:
    """
    1) Ensure tables exist
    2) Try create unique indexes
    3) If duplicates break index creation, auto-dedup then retry
    4) Never crash the app for index creation failure
    """
    dialect = engine.url.get_backend_name()

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

        def dedup_postgres(table: str, cols: List[str]) -> None:
            cond = " AND ".join([f"a.{c} = b.{c}" for c in cols])
            sql = f"""
            DELETE FROM {table} a
            USING {table} b
            WHERE {cond}
              AND a.ctid > b.ctid;
            """
            conn.execute(text(sql))

        def dedup_sqlite(table: str, cols: List[str]) -> None:
            group_by = ", ".join(cols)
            sql = f"""
            DELETE FROM {table}
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM {table}
                GROUP BY {group_by}
            );
            """
            conn.execute(text(sql))

        def create_unique_index(table: str, index_name: str, cols: List[str]) -> None:
            cols_join = ", ".join(cols)
            conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
            ON {table}({cols_join});
            """))

        # sales
        try:
            create_unique_index("sales", "ux_sales_row", ["date", "customer", "product", "amount", "quantity"])
        except Exception as e:
            logger.warning(f"sales unique index create failed, try dedup: {e}")
            try:
                if dialect == "postgresql":
                    dedup_postgres("sales", ["date", "customer", "product", "amount", "quantity"])
                else:
                    dedup_sqlite("sales", ["date", "customer", "product", "amount", "quantity"])
                create_unique_index("sales", "ux_sales_row", ["date", "customer", "product", "amount", "quantity"])
                logger.info("✅ sales dedup done, unique index created")
            except Exception as e2:
                logger.error(f"❌ sales index still failed, continue without it: {e2}")

        # purchase
        try:
            create_unique_index("purchase", "ux_purchase_row", ["date", "supplier", "product", "amount", "quantity"])
        except Exception as e:
            logger.warning(f"purchase unique index create failed, try dedup: {e}")
            try:
                if dialect == "postgresql":
                    dedup_postgres("purchase", ["date", "supplier", "product", "amount", "quantity"])
                else:
                    dedup_sqlite("purchase", ["date", "supplier", "product", "amount", "quantity"])
                create_unique_index("purchase", "ux_purchase_row", ["date", "supplier", "product", "amount", "quantity"])
                logger.info("✅ purchase dedup done, unique index created")
            except Exception as e2:
                logger.error(f"❌ purchase index still failed, continue without it: {e2}")


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
# Google Sheet download (export xlsx)
# =========================
def get_sheet_id(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else ""


def download_google_sheet_xlsx(sheet_url: str, dest_path: str, max_retries: int = 4) -> bool:
    sheet_id = get_sheet_id(sheet_url)
    if not sheet_id:
        logger.error("Invalid sheet url (no sheet id found).")
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

            # xlsx is a zip file -> starts with PK
            first2 = r.raw.read(2)
            if first2 != b"PK":
                logger.error("Not xlsx content (permission page/HTML?). Check sharing settings.")
                time.sleep(min(10, 2 ** attempt))
                continue

            with open(dest_path, "wb") as f:
                f.write(first2)
                for chunk in r.iter_content(32768):
                    if chunk:
                        f.write(chunk)

            return True
        except Exception as e:
            logger.error(f"download error attempt {attempt + 1}: {e}")
            time.sleep(min(10, 2 ** attempt))

    return False


# =========================
# Normalize
# =========================
def normalize_sales_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    df.columns = df.columns.astype(str).str.strip()

    # Your ERP exports usually contain these columns
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


# =========================
# Insert (idempotent)
# =========================
def insert_ignore(kind: str, df: pd.DataFrame) -> int:
    """
    Insert rows but ignore duplicates via unique index.
    Note: For Postgres ON CONFLICT needs a UNIQUE constraint/index on the conflict target columns.
    """
    if df.empty:
        return 0

    dialect = engine.url.get_backend_name()
    rows = df.to_dict(orient="records")

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

    # “attempted inserts” count is len(rows). Actual inserted may be less due to ignore.
    return len(rows)


# =========================
# Public: import from sheets
# =========================
def import_from_sheets() -> Dict[str, Any]:
    ensure_tables_and_indexes()

    before = table_counts()
    messages: List[str] = []
    ts = int(time.time())

    tmp_sales = f"./_sales_{ts}.xlsx"
    tmp_purchase = f"./_purchase_{ts}.xlsx"

    # Sales
    if SALES_SHEET_URL:
        ok = download_google_sheet_xlsx(SALES_SHEET_URL, tmp_sales)
        if ok:
            xls = pd.read_excel(tmp_sales, sheet_name=None)
            parts = []
            for _, sdf in xls.items():
                n = normalize_sales_df(sdf)
                if n is not None and not n.empty:
                    parts.append(n)
            if parts:
                final = pd.concat(parts, ignore_index=True)
                insert_ignore("sales", final)
                messages.append(f"sales: 讀到 {len(final)} 筆（增量匯入、重複會自動略過）")
            else:
                messages.append("sales: 沒找到符合欄位的分頁（請確認欄位是否包含：日期(轉換)、進銷明細未稅金額）")
        else:
            messages.append("sales: 下載失敗（請確認 Google Sheet 已設為「知道連結的人可檢視」）")
    else:
        messages.append("sales: 未設定 SALES_EXCEL_URL")

    # Purchase
    if PURCHASE_SHEET_URL:
        ok = download_google_sheet_xlsx(PURCHASE_SHEET_URL, tmp_purchase)
        if ok:
            xls = pd.read_excel(tmp_purchase, sheet_name=None)
            parts = []
            for _, pdf in xls.items():
                n = normalize_purchase_df(pdf)
                if n is not None and not n.empty:
                    parts.append(n)
            if parts:
                final = pd.concat(parts, ignore_index=True)
                insert_ignore("purchase", final)
                messages.append(f"purchase: 讀到 {len(final)} 筆（增量匯入、重複會自動略過）")
            else:
                messages.append("purchase: 沒找到符合欄位的分頁（請確認欄位是否包含：日期(轉換)、進銷明細未稅金額）")
        else:
            messages.append("purchase: 下載失敗（請確認 Google Sheet 已設為「知道連結的人可檢視」）")
    else:
        messages.append("purchase: 未設定 PURCHASE_EXCEL_URL")

    # cleanup
    for p in [tmp_sales, tmp_purchase]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    after = table_counts()
    return {
        "ok": True,
        "before": before,
        "after": after,
        "messages": messages,
        "db": engine.url.get_backend_name(),
        "time": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    result = import_from_sheets()
    logger.info(json.dumps(result, ensure_ascii=False))
