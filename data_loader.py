import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(DATABASE_URL or "sqlite:///./erp.db")

COLUMN_MAPPING = {
    "日期": "date", "交易日期": "date", "客戶": "customer", "客戶名稱": "customer",
    "產品": "product", "品名": "product", "數量": "quantity", "金額": "amount", "廠商": "supplier"
}

def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = [str(c).strip() for c in df.columns]
    
    new_cols = {}
    for col in df.columns:
        for k, v in COLUMN_MAPPING.items():
            if k in col:
                new_cols[col] = v
                break
    df = df.rename(columns=new_cols)
    
    for col in ['amount', 'quantity']:
        if col in df.columns:
            # 修正：確保處理的是 Series 而不是整個 DataFrame
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce').fillna(0)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
    return df

def import_excel_files():
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("DROP TABLE IF EXISTS purchase"))
        if "sqlite" not in engine.url.drivername: conn.commit()

    all_sales, all_purchase = [], []
    for f in glob.glob("*.xlsx"):
        try:
            xls = pd.read_excel(f, sheet_name=None)
            for sheet_name, df in xls.items():
                if len(df) < 1: continue
                cleaned = clean_and_rename(df)
                if 'supplier' in cleaned.columns: all_purchase.append(cleaned)
                elif 'customer' in cleaned.columns: all_sales.append(cleaned)
        except: continue

    if all_sales:
        pd.concat(all_sales, ignore_index=True).to_sql("sales", engine, if_exists='replace', index=False)
    if all_purchase:
        pd.concat(all_purchase, ignore_index=True).to_sql("purchase", engine, if_exists='replace', index=False)
    print("✅ 資料匯入完成")

if __name__ == "__main__":
    import_excel_files()