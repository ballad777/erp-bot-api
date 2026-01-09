import pandas as pd
from sqlalchemy import create_engine, text
import os
import glob

# 設定資料庫連線 (Render 重啟後會自動透過這個腳本重建資料)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./erp.db")
engine = create_engine(DATABASE_URL)

# ==========================================
# 欄位中翻英對照表 (AI 只認得英文欄位)
# 這裡涵蓋了你 Excel 可能出現的各種標頭寫法
# ==========================================
COLUMN_MAPPING = {
    # 時間相關
    "日期": "date", "交易日期": "date", "訂單日期": "date",
    "年": "year", "年份": "year",
    
    # 銷售相關
    "客戶": "customer", "客戶名稱": "customer", "客戶代號": "customer_id",
    "產品": "product", "品名": "product", "產品名稱": "product",
    "數量": "quantity", "銷售數量": "quantity",
    "金額": "amount", "銷售金額": "amount", "總金額": "amount", "未稅金額": "amount",
    
    # 進貨相關
    "廠商": "supplier", "供應商": "supplier", "廠商名稱": "supplier",
}

def clean_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """清理資料並重新命名欄位"""
    # 1. 移除欄位名稱的前後空白
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 重新命名 (中文 -> 英文)
    new_cols = {}
    for col in df.columns:
        # 直接比對
        if col in COLUMN_MAPPING:
            new_cols[col] = COLUMN_MAPPING[col]
        # 模糊比對 (例如 "金額(含稅)" -> "amount")
        else:
            for k, v in COLUMN_MAPPING.items():
                if k in col:
                    new_cols[col] = v
                    break
            # 如果都對不到，轉小寫英文
            if col not in new_cols:
                new_cols[col] = col.lower()
                
    df = df.rename(columns=new_cols)
    
    # 3. 確保有 year 欄位
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'year' not in df.columns:
            df['year'] = df['date'].dt.year
            
    # 4. 數值補零 (避免計算錯誤)
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)

    # 5. 確保這張表至少有日期或金額，否則可能是無效的 Sheet
    if 'amount' not in df.columns and 'date' not in df.columns:
        return pd.DataFrame() # 回傳空表

    return df

def import_excel_files():
    """讀取當前目錄下的 Excel 並匯入資料庫"""
    print("🔄 開始資料匯入程序...")
    
    # 1. 重置資料庫
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("DROP TABLE IF EXISTS purchase"))
    
    # 2. 讀取所有 .xlsx 檔案
    xlsx_files = glob.glob("*.xlsx")
    
    all_sales = []
    all_purchase = []
    
    for f in xlsx_files:
        print(f"📄 正在讀取: {f} ...")
        try:
            # 讀取所有分頁
            xls = pd.read_excel(f, sheet_name=None, engine='openpyxl')
            
            for sheet_name, df in xls.items():
                # 跳過空的分頁
                if df.empty or len(df) < 2: continue
                
                # 簡單判斷這張表是 Sales 還是 Purchase
                # 判斷邏輯：檔名或分頁名稱包含關鍵字
                fname_lower = f.lower()
                sheet_lower = str(sheet_name).lower()
                
                cleaned_df = clean_and_rename(df)
                if cleaned_df.empty: continue

                # 判定類型
                is_purchase = "purchase" in fname_lower or "進貨" in fname_lower or "purchase" in sheet_lower or "進貨" in sheet_lower
                is_sales = "sales" in fname_lower or "sale" in fname_lower or "銷" in fname_lower or "sales" in sheet_lower
                
                if is_purchase and 'supplier' in cleaned_df.columns:
                    all_purchase.append(cleaned_df)
                    print(f"   -> 識別為 [進貨] 資料: {sheet_name} ({len(cleaned_df)}筆)")
                elif 'customer' in cleaned_df.columns: # 預設如果有 customer 就當 sales
                    all_sales.append(cleaned_df)
                    print(f"   -> 識別為 [銷售] 資料: {sheet_name} ({len(cleaned_df)}筆)")
                    
        except Exception as e:
            print(f"❌ 讀取 {f} 失敗: {e}")

    # 3. 寫入資料庫
    if all_sales:
        final_sales = pd.concat(all_sales, ignore_index=True)
        final_sales.to_sql("sales", engine, if_exists='replace', index=False)
        print(f"✅ Sales 表匯入完成，共 {len(final_sales)} 筆。")
    
    if all_purchase:
        final_purchase = pd.concat(all_purchase, ignore_index=True)
        final_purchase.to_sql("purchase", engine, if_exists='replace', index=False)
        print(f"✅ Purchase 表匯入完成，共 {len(final_purchase)} 筆。")

if __name__ == "__main__":
    import_excel_files()