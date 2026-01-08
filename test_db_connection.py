import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")

print("DATABASE_URL 是否存在：", bool(DATABASE_URL))

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("✅ 成功連線！資料庫回傳：", result.scalar())
