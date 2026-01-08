import os
import pandas as pd
from sqlalchemy import create_engine, text

# ====== 設定：檔名（放在同一個資料夾即可）======
PURCHASE_FILES = [
    "purchase_2023.xlsx",
    "purchase_2024.xlsx",
]
SALES_FILES = [
    "sales_2023_2025.xlsx",
]

# ====== DB 連線 ======
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數，請先 setx DATABASE_URL ... 再重開 PowerShell")

engine = create_engine(DATABASE_URL)


def pick_col(df, candidates):
    """從多個候選欄位中找第一個存在的欄位名稱（容錯）"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fuzzy_find_col(df, include_keywords, exclude_keywords=None):
    """
    從 df.columns 用「包含關鍵字」找欄位（模糊匹配）
    include_keywords: list[str] 任一命中即可
    exclude_keywords: list[str] 命中就排除
    """
    exclude_keywords = exclude_keywords or []
    cols = [str(c) for c in df.columns]

    for c in cols:
        hit_incl = any(k in c for k in include_keywords)
        hit_excl = any(k in c for k in exclude_keywords)
        if hit_incl and not hit_excl:
            return c
    return None


def ensure_tables():
    """建表並 commit（SQLAlchemy 最穩方式）"""
    with engine.begin() as conn:  # ✅ commit
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS purchase (
            id SERIAL PRIMARY KEY,
            date DATE,
            year INT,
            supplier TEXT,
            product TEXT,
            quantity NUMERIC,
            amount NUMERIC
        );
        """))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sales (
            id SERIAL PRIMARY KEY,
            date DATE,
            year INT,
            customer TEXT,
            product TEXT,
            quantity NUMERIC,
            amount NUMERIC
        );
        """))

    print("✅ Table 確認完成：purchase / sales（已 commit）")


def read_all_sheets(excel_path: str) -> list[pd.DataFrame]:
    """
    讀取 Excel 的所有工作表，每張表回傳一個 DataFrame。
    會自動跳過完全空的 sheet。
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"找不到檔案：{excel_path}（請放在同一個資料夾）")

    xls = pd.ExcelFile(excel_path)
    dfs = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        if df is None or df.empty:
            print(f"⚠️ 跳過空白工作表：{excel_path} / {sheet}")
            continue
        df["__source_file__"] = excel_path
        df["__source_sheet__"] = sheet
        dfs.append(df)

    print(f"📄 讀取完成：{excel_path} 共 {len(dfs)} 張工作表")
    return dfs


def normalize_purchase(df: pd.DataFrame) -> pd.DataFrame | None:
    date_col = pick_col(df, ["日期(轉換)", "日期", "Date", "date"])
    supplier_col = pick_col(df, ["客戶供應商簡稱", "供應商", "supplier", "Supplier", "廠商", "廠商名稱"])
    product_col = pick_col(df, ["產品代號", "品號", "product", "Product", "料號"])

    qty_col = pick_col(df, ["數量", "進貨數量", "quantity", "Qty"])
    amt_col = pick_col(df, ["金額", "未稅金額", "含稅金額", "amount", "Amount", "總金額"])

    if not all([date_col, supplier_col, product_col]):
        print("⚠️ purchase 工作表缺必要欄位，已跳過：",
              {"date": date_col, "supplier": supplier_col, "product": product_col,
               "sheet": df.get('__source_sheet__', 'unknown')})
        print("   columns =", list(df.columns))
        return None

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["year"] = out["date"].dt.year
    out["supplier"] = df[supplier_col].astype(str).str.strip()
    out["product"] = df[product_col].astype(str).str.strip()

    out["quantity"] = pd.to_numeric(df[qty_col], errors="coerce") if qty_col else None
    out["amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else None

    out = out.dropna(subset=["date", "supplier", "product"])
    return out


def normalize_sales(df: pd.DataFrame) -> pd.DataFrame | None:
    date_col = pick_col(df, ["日期(轉換)", "日期", "Date", "date"])
    product_col = pick_col(df, ["產品代號", "品號", "product", "Product", "料號"])

    # 先用明確候選找 customer
    customer_col = pick_col(df, ["客戶簡稱", "客戶", "customer", "Customer", "客戶名稱", "客戶代號", "客戶全名"])

    # 如果還找不到，改用模糊規則：欄位只要含「客戶」或 customer 就算
    if customer_col is None:
        customer_col = fuzzy_find_col(
            df,
            include_keywords=["客戶", "customer", "Customer"],
            exclude_keywords=["供應商", "廠商"]
        )

    qty_col = pick_col(df, ["數量", "銷售數量", "quantity", "Qty"])

    # ✅✅✅ 金額欄：優先抓「進銷明細未稅金額(含正負號)」
    amt_col = pick_col(df, [
        "進銷明細未稅金額(含正負號)",
        "進銷明細未稅金額",
        "明細金額(含正負號)",
        "明細金額",
        "銷貨小計",
        "含稅金額(主檔)",
        "金額",
        "未稅金額",
        "含稅金額"
    ])

    # 如果還是找不到，再用模糊匹配（欄名只要含 金額/未稅/含稅）
    if amt_col is None:
        amt_col = fuzzy_find_col(
            df,
            include_keywords=["金額", "未稅", "含稅", "amount", "amt"],
            exclude_keywords=["單價", "價格", "登打單價"]
        )

    # 分析頁通常連 date/customer/product 都沒有，正常跳過
    if not all([date_col, customer_col, product_col]):
        print("⚠️ sales 工作表缺必要欄位，已跳過：",
              {"date": date_col, "customer": customer_col, "product": product_col,
               "sheet": df.get('__source_sheet__', 'unknown')})
        print("   columns =", list(df.columns))
        return None

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["year"] = out["date"].dt.year
    out["customer"] = df[customer_col].astype(str).str.strip()
    out["product"] = df[product_col].astype(str).str.strip()

    out["quantity"] = pd.to_numeric(df[qty_col], errors="coerce") if qty_col else None
    out["amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else None

    out = out.dropna(subset=["date", "customer", "product"])
    return out


def replace_table(table_name: str, df: pd.DataFrame):
    """先 TRUNCATE 再寫入（穩定）"""
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name};"))
    df.to_sql(table_name, engine, if_exists="append", index=False)
    print(f"✅ 已寫入 {table_name}: {len(df)} rows")


def load_purchase_all() -> pd.DataFrame:
    all_norm = []
    for path in PURCHASE_FILES:
        sheet_dfs = read_all_sheets(path)
        for raw_df in sheet_dfs:
            norm = normalize_purchase(raw_df)
            if norm is not None and not norm.empty:
                all_norm.append(norm)

    if not all_norm:
        raise Exception("purchase 沒有任何工作表可用（都空或欄位不符）")

    return pd.concat(all_norm, ignore_index=True)


def load_sales_all() -> pd.DataFrame:
    all_norm = []
    for path in SALES_FILES:
        sheet_dfs = read_all_sheets(path)
        for raw_df in sheet_dfs:
            norm = normalize_sales(raw_df)
            if norm is not None and not norm.empty:
                all_norm.append(norm)

    if not all_norm:
        raise Exception("sales 沒有任何工作表可用（都空或欄位不符）")

    return pd.concat(all_norm, ignore_index=True)


def main():
    ensure_tables()

    purchase_all = load_purchase_all()
    print("📌 purchase 合併後筆數：", len(purchase_all))
    replace_table("purchase", purchase_all)

    sales_all = load_sales_all()
    print("📌 sales 合併後筆數：", len(sales_all))
    replace_table("sales", sales_all)

    print("🎉 ETL 完成：所有工作表資料已匯入雲端 PostgreSQL")


if __name__ == "__main__":
    main()
