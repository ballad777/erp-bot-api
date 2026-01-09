import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob

# =========================
# 資料庫連線設定 (含 Render 修正)
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 如果沒有設定或設定為空，預設使用 SQLite
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./erp.db"

engine = create_engine(DATABASE_URL)

# =========================
# 欄位對照表 (AI 認得英文，Excel 是中文)
# =========================
COLUMN_MAPPING = {
    # 共同欄位
    "日期": "date", "交易日期": "date", "訂單日期": "date",
    "年": "year", "年份": "year",
    # 銷售
    "客戶": "customer", "客戶名稱": "customer", "客戶代號": "customer_id",
    "產品": "product", "品名": "product", "產品名稱": "product",
    "數量": "quantity", "銷售數量": "quantity",
    "金額": "amount", "銷售金額": "amount", "總金額": "amount", "未稅金額": "amount",
    # 進貨
    "廠商": "supplier", "供應商": "supplier", "廠商名稱": "supplier",
}

def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """清理 Excel 資料並轉成資料庫欄位"""
    # 1. 清除欄位空白
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 中翻英重命名
    new_cols = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            new_cols[col] = COLUMN_MAPPING[col]
        else:
            # 模糊比對
            for k, v in COLUMN_MAPPING.items():
                if k in col:
                    new_cols[col] = v
                    break
            if col not in new_cols:
                new_cols[col] = col.lower()
    df = df.rename(columns=new_cols)
    
    # 3. 補充年份 (Year)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
            
    # 4. 數值補零
    for col in ['amount', 'quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def import_excel_files():
    """讀取 Excel 並寫入資料庫"""
    print(f"🚀 開始匯入資料... (目標資料庫: {DATABASE_URL.split(':')[0]})")
    
    # 重置資料表
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("DROP TABLE IF EXISTS purchase"))
        if "sqlite" not in DATABASE_URL:
            conn.commit() # PostgreSQL 需要明確 commit

    xlsx_files = glob.glob("*.xlsx")
    if not xlsx_files:
        print("⚠️ 警告：找不到任何 .xlsx 檔案！請確認檔案已上傳。")
        return

    all_sales = []
    all_purchase = []
    
    for f in xlsx_files:
        print(f"📄 讀取檔案: {f}")
        try:
            xls = pd.read_excel(f, sheet_name=None, engine='openpyxl')
            for sheet_name, df in xls.items():
                if df.empty or len(df) < 2: continue
                
                df_clean = clean_and_rename(df)
                
                # 簡單判斷類型
                fname_lower = f.lower()
                sheet_lower = str(sheet_name).lower()
                is_purchase = "purchase" in fname_lower or "進貨" in fname_lower or "purchase" in sheet_lower
                
                if is_purchase and 'supplier' in df_clean.columns:
                    all_purchase.append(df_clean)
                    print(f"   -> [進貨] {sheet_name}: {len(df_clean)} 筆")
                elif 'customer' in df_clean.columns:
                    all_sales.append(df_clean)
                    print(f"   -> [銷售] {sheet_name}: {len(df_clean)} 筆")
                    
        except Exception as e:
            print(f"❌ 讀取錯誤 {f}: {e}")

    # 寫入資料庫
    if all_sales:
        final_sales = pd.concat(all_sales, ignore_index=True)
        final_sales.to_sql("sales", engine, if_exists='replace', index=False)
        print(f"✅ Sales 表匯入成功：共 {len(final_sales)} 筆")
        
    if all_purchase:
        final_purchase = pd.concat(all_purchase, ignore_index=True)
        final_purchase.to_sql("purchase", engine, if_exists='replace', index=False)
        print(f"✅ Purchase 表匯入成功：共 {len(final_purchase)} 筆")

if __name__ == "__main__":
    import_excel_files()