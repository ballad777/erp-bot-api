import os
import re
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from io import BytesIO

import requests
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_loader")

# =========================
# Env
# =========================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Google Sheet URL
SALES_SHEET_URL = os.getenv("SALES_EXCEL_URL", "").strip()
PURCHASE_SHEET_URL = os.getenv("PURCHASE_EXCEL_URL", "").strip()

engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)


# =========================
# DB init
# =========================
def ensure_tables_and_indexes() -> None:
    """
    1) 確保資料表存在
    2) 嘗試建立索引
    3) 如果有重複資料導致索引失敗，自動去重後重試
    """
    logger.info("🔧 檢查資料表和索引...")
    dialect = engine.url.get_backend_name()

    with engine.begin() as conn:
        # 建立 sales 表
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
        
        # 建立 purchase 表
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
            """PostgreSQL 去重"""
            cond = " AND ".join([f"a.{c} = b.{c}" for c in cols])
            sql = f"""
            DELETE FROM {table} a
            USING {table} b
            WHERE {cond}
              AND a.ctid > b.ctid;
            """
            conn.execute(text(sql))

        def dedup_sqlite(table: str, cols: List[str]) -> None:
            """SQLite 去重"""
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
            """建立唯一索引"""
            cols_join = ", ".join(cols)
            conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS {index_name}
            ON {table}({cols_join});
            """))

        # 處理 sales 索引
        try:
            create_unique_index("sales", "ux_sales_row", 
                              ["date", "customer", "product", "amount", "quantity"])
            logger.info("✅ sales 索引已建立")
        except Exception as e:
            logger.warning(f"⚠️ sales 索引建立失敗，嘗試去重: {e}")
            try:
                if dialect == "postgresql":
                    dedup_postgres("sales", ["date", "customer", "product", "amount", "quantity"])
                else:
                    dedup_sqlite("sales", ["date", "customer", "product", "amount", "quantity"])
                create_unique_index("sales", "ux_sales_row", 
                                  ["date", "customer", "product", "amount", "quantity"])
                logger.info("✅ sales 去重完成，索引已建立")
            except Exception as e2:
                logger.error(f"❌ sales 索引仍然失敗，繼續執行: {e2}")

        # 處理 purchase 索引
        try:
            create_unique_index("purchase", "ux_purchase_row", 
                              ["date", "supplier", "product", "amount", "quantity"])
            logger.info("✅ purchase 索引已建立")
        except Exception as e:
            logger.warning(f"⚠️ purchase 索引建立失敗，嘗試去重: {e}")
            try:
                if dialect == "postgresql":
                    dedup_postgres("purchase", ["date", "supplier", "product", "amount", "quantity"])
                else:
                    dedup_sqlite("purchase", ["date", "supplier", "product", "amount", "quantity"])
                create_unique_index("purchase", "ux_purchase_row", 
                                  ["date", "supplier", "product", "amount", "quantity"])
                logger.info("✅ purchase 去重完成，索引已建立")
            except Exception as e2:
                logger.error(f"❌ purchase 索引仍然失敗，繼續執行: {e2}")


def table_counts() -> Dict[str, int]:
    """取得各表筆數"""
    insp = inspect(engine)
    names = set(insp.get_table_names())
    out = {"sales": 0, "purchase": 0}
    
    with engine.connect() as conn:
        for t in out.keys():
            if t in names:
                try:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                    out[t] = int(result.scalar() or 0)
                except:
                    out[t] = 0
    
    return out


# =========================
# Google Sheet 下載
# =========================
def extract_sheet_id(url: str) -> Optional[str]:
    """從 URL 提取 Sheet ID"""
    patterns = [
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"id=([a-zA-Z0-9_-]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def download_google_sheet_xlsx(sheet_url: str, max_retries: int = 3) -> Optional[BytesIO]:
    """
    下載 Google Sheet 為 Excel 格式
    回傳 BytesIO 或 None
    """
    sheet_id = extract_sheet_id(sheet_url)
    if not sheet_id:
        logger.error(f"❌ 無法提取 Sheet ID: {sheet_url}")
        return None

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export"
    params = {"format": "xlsx"}
    
    logger.info(f"📥 下載 Google Sheet: {sheet_id}")

    for attempt in range(max_retries):
        try:
            response = requests.get(
                export_url,
                params=params,
                timeout=60,
                allow_redirects=True
            )
            
            if response.status_code != 200:
                logger.error(f"❌ HTTP {response.status_code}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

            content = response.content
            
            # 檢查是否為 Excel (ZIP 格式，以 PK 開頭)
            if not content.startswith(b'PK'):
                logger.error("❌ 回應不是 Excel 格式")
                
                # 檢查是否為 HTML (權限問題)
                if b'<html' in content[:500].lower():
                    logger.error("❌ 收到 HTML，可能是權限問題")
                    logger.error("請確認 Google Sheet 已設為「知道連結的人可以檢視」")
                    logger.error(f"前 200 字元: {content[:200]}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            logger.info(f"✅ 下載成功: {len(content)} bytes")
            return BytesIO(content)

        except requests.exceptions.Timeout:
            logger.error(f"⏱️ 下載超時 (嘗試 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"❌ 下載錯誤: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# =========================
# 欄位匹配輔助函數
# =========================
def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """從候選欄位名稱中找到第一個存在的"""
    df.columns = df.columns.astype(str).str.strip()
    for candidate in candidates:
        for col in df.columns:
            if candidate in col:
                return col
    return None


# =========================
# 標準化函數（彈性版本）
# =========================
def normalize_sales_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """標準化銷售資料（支援多種欄位名稱）"""
    logger.info(f"處理銷售資料，欄位: {list(df.columns)}")
    
    # 彈性尋找欄位
    date_col = find_column(df, ["日期(轉換)", "日期", "Date", "date", "交易日期"])
    customer_col = find_column(df, ["客戶供應商簡稱", "客戶簡稱", "客戶", "Customer", "客戶名稱"])
    product_col = find_column(df, ["品名", "產品", "Product", "品號", "產品代號"])
    quantity_col = find_column(df, ["數量", "Quantity", "銷售數量"])
    amount_col = find_column(df, ["進銷明細未稅金額", "未稅金額", "金額", "Amount"])

    # 檢查必要欄位
    if not all([date_col, customer_col, product_col]):
        logger.warning(f"⚠️ 缺少必要欄位: date={date_col}, customer={customer_col}, product={product_col}")
        return None

    # 建立標準化 DataFrame
    clean = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "customer": df[customer_col].astype(str).str.strip(),
        "product": df[product_col].astype(str).str.strip(),
        "quantity": pd.to_numeric(df[quantity_col], errors="coerce").fillna(0) if quantity_col else 0,
        "amount": pd.to_numeric(df[amount_col], errors="coerce").fillna(0) if amount_col else 0,
    }).dropna(subset=["date"])

    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    
    logger.info(f"✅ 標準化完成: {len(clean)} 筆")
    return clean


def normalize_purchase_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """標準化採購資料（支援多種欄位名稱）"""
    logger.info(f"處理採購資料，欄位: {list(df.columns)}")
    
    # 彈性尋找欄位
    date_col = find_column(df, ["日期(轉換)", "日期", "Date", "date", "交易日期"])
    supplier_col = find_column(df, ["客戶供應商簡稱", "供應商", "Supplier", "廠商"])
    product_col = find_column(df, ["對方品名/品名備註", "品名", "產品", "Product", "品號"])
    quantity_col = find_column(df, ["數量", "Quantity", "採購數量"])
    amount_col = find_column(df, ["進銷明細未稅金額", "未稅金額", "金額", "Amount"])

    # 檢查必要欄位
    if not all([date_col, supplier_col, product_col]):
        logger.warning(f"⚠️ 缺少必要欄位: date={date_col}, supplier={supplier_col}, product={product_col}")
        return None

    # 建立標準化 DataFrame
    clean = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "supplier": df[supplier_col].astype(str).str.strip(),
        "product": df[product_col].astype(str).str.strip(),
        "quantity": pd.to_numeric(df[quantity_col], errors="coerce").fillna(0) if quantity_col else 0,
        "amount": pd.to_numeric(df[amount_col], errors="coerce").fillna(0) if amount_col else 0,
    }).dropna(subset=["date"])

    clean["year"] = clean["date"].dt.year
    clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")
    
    logger.info(f"✅ 標準化完成: {len(clean)} 筆")
    return clean


# =========================
# 插入資料（忽略重複）
# =========================
def insert_ignore(kind: str, df: pd.DataFrame) -> int:
    """
    插入資料，重複的會自動忽略
    回傳嘗試插入的筆數
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
        else:  # purchase
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

    return len(rows)


# =========================
# 主要匯入函數
# =========================
def import_from_sheets() -> Dict[str, Any]:
    """從 Google Sheets 匯入資料"""
    logger.info("🔄 開始資料匯入...")
    
    ensure_tables_and_indexes()

    before = table_counts()
    messages: List[str] = []

    # 匯入銷售資料
    if SALES_SHEET_URL:
        logger.info("📊 處理銷售資料...")
        excel_bytes = download_google_sheet_xlsx(SALES_SHEET_URL)
        
        if excel_bytes:
            try:
                xls = pd.read_excel(excel_bytes, sheet_name=None)
                logger.info(f"找到 {len(xls)} 個工作表")
                
                parts = []
                for sheet_name, df in xls.items():
                    logger.info(f"處理工作表: {sheet_name}")
                    normalized = normalize_sales_df(df)
                    if normalized is not None and not normalized.empty:
                        parts.append(normalized)
                
                if parts:
                    final = pd.concat(parts, ignore_index=True)
                    insert_ignore("sales", final)
                    messages.append(f"✅ sales: 讀到 {len(final)} 筆（增量匯入，重複會自動略過）")
                else:
                    messages.append("⚠️ sales: 沒找到符合欄位的分頁")
            except Exception as e:
                logger.error(f"❌ sales 處理錯誤: {str(e)}", exc_info=True)
                messages.append(f"❌ sales: 處理失敗 - {str(e)}")
        else:
            messages.append("❌ sales: 下載失敗（請確認 Google Sheet 已設為「知道連結的人可檢視」）")
    else:
        messages.append("ℹ️ sales: 未設定 SALES_EXCEL_URL")

    # 匯入採購資料
    if PURCHASE_SHEET_URL:
        logger.info("📦 處理採購資料...")
        excel_bytes = download_google_sheet_xlsx(PURCHASE_SHEET_URL)
        
        if excel_bytes:
            try:
                xls = pd.read_excel(excel_bytes, sheet_name=None)
                logger.info(f"找到 {len(xls)} 個工作表")
                
                parts = []
                for sheet_name, df in xls.items():
                    logger.info(f"處理工作表: {sheet_name}")
                    normalized = normalize_purchase_df(df)
                    if normalized is not None and not normalized.empty:
                        parts.append(normalized)
                
                if parts:
                    final = pd.concat(parts, ignore_index=True)
                    insert_ignore("purchase", final)
                    messages.append(f"✅ purchase: 讀到 {len(final)} 筆（增量匯入，重複會自動略過）")
                else:
                    messages.append("⚠️ purchase: 沒找到符合欄位的分頁")
            except Exception as e:
                logger.error(f"❌ purchase 處理錯誤: {str(e)}", exc_info=True)
                messages.append(f"❌ purchase: 處理失敗 - {str(e)}")
        else:
            messages.append("❌ purchase: 下載失敗（請確認 Google Sheet 已設為「知道連結的人可檢視」）")
    else:
        messages.append("ℹ️ purchase: 未設定 PURCHASE_EXCEL_URL")

    after = table_counts()
    
    result = {
        "ok": True,
        "before": before,
        "after": after,
        "messages": messages,
        "db": engine.url.get_backend_name(),
        "time": datetime.utcnow().isoformat() + "Z",
    }
    
    logger.info(f"✨ 資料匯入完成: {result}")
    return result


if __name__ == "__main__":
    result = import_from_sheets()
    print(json.dumps(result, ensure_ascii=False, indent=2))