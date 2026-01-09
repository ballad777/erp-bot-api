import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob

# 資料庫連線設定
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./erp.db"

engine = create_engine(DATABASE_URL)

COLUMN_MAPPING = {
    "日期": "date", "交易日期": "date", "客戶": "customer", "客戶名稱": "customer",
    "產品": "product", "品名": "product", "數量": "quantity", "金額": "amount",
    "總金額": "amount", "年": "year", "廠商": "supplier", "供應商": "supplier"
}

def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """強化版清理：處理重複標頭與型別問題"""
    # 移除全空的行或列
    df = df.dropna(how='all').dropna(axis=1, how='all')
    if df.empty: return pd.DataFrame()

    df.columns = [str(c).strip() for c in df.columns]
    
    # 重新命名欄位
    new_cols = {}
    for col in df.columns:
        for k, v in COLUMN_MAPPING.items():
            if k in col:
                new_cols[col] = v
                break
    df = df.rename(columns=new_cols)
    
    # 強制轉換數值欄位，避免 "arg must be a list" 錯誤
    for col in ['amount', 'quantity', 'year']:
        if col in df.columns:
            # 先轉成字串處理掉可能存在的非數字字元，再轉數值
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'year' not in df.columns or (df['year'] == 0).all():
            df['year'] = df['date'].dt.year

    return df

def import_excel_files():
    print(f"🚀 開始匯入資料... (目標: {DATABASE_URL.split(':')[0]})")
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("DROP TABLE IF EXISTS purchase"))
        if "sqlite" not in DATABASE_URL: conn.commit()

    xlsx_files = glob.glob("*.xlsx")
    all_sales = []
    all_purchase = []
    
    for f in xlsx_files:
        print(f"📄 讀取檔案: {f}")
        try:
            # 讀取 Excel (不指定分頁，讀取全部)
            xls = pd.read_excel(f, sheet_name=None, engine='openpyxl')
            for sheet_name, df in xls.items():
                if len(df) < 1: continue
                
                df_clean = clean_and_rename(df)
                if df_clean.empty: continue

                # 判定邏輯優化
                fname_lower = f.lower()
                is_p = "purchase" in fname_lower or "進貨" in fname_lower or "supplier" in df_clean.columns
                
                if is_p:
                    all_purchase.append(df_clean)
                    print(f"   -> [進貨] {sheet_name} 已就緒")
                else:
                    all_sales.append(df_clean)
                    print(f"   -> [銷售] {sheet_name} 已就緒")
                    
        except Exception as e:
            print(f"❌ 讀取失敗 {f}: {e}")

    # 使用 ignore_index=True 解決 "duplicate keys" 錯誤
    try:
        if all_sales:
            final_sales = pd.concat(all_sales, ignore_index=True, sort=False)
            final_sales.to_sql("sales", engine, if_exists='replace', index=False)
            print(f"✅ Sales 匯入完成：共 {len(final_sales)} 筆")
            
        if all_purchase:
            final_purchase = pd.concat(all_purchase, ignore_index=True, sort=False)
            final_purchase.to_sql("purchase", engine, if_exists='replace', index=False)
            print(f"✅ Purchase 匯入完成：共 {len(final_purchase)} 筆")
    except Exception as e:
        print(f"❌ 資料合併寫入失敗: {e}")

if __name__ == "__main__":
    import_excel_files()