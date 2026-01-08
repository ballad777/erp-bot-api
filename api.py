import os
from fastapi import FastAPI, Query
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise Exception("找不到 DATABASE_URL 環境變數")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="ERP Bot API", version="1.0")


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    with engine.connect() as conn:
        v = conn.execute(text("SELECT 1")).scalar()
    return {"ok": True, "db": v}


# -------------------------
# Sales
# -------------------------
@app.get("/sales/summary")
def sales_summary(year: int = Query(..., description="年份，例如 2025")):
    sql = text("""
        SELECT
            :year AS year,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"year": year}).mappings().first()
    return dict(row)


@app.get("/sales/summary_all_years")
def sales_summary_all_years():
    sql = text("""
        SELECT
            year,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        GROUP BY year
        ORDER BY year ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return {"years": [dict(r) for r in rows]}


@app.get("/sales/top_products")
def sales_top_products(
    year: int = Query(..., description="年份，例如 2025"),
    n: int = Query(10, ge=1, le=100, description="回傳前 N 名")
):
    # ✅ 依數量排序（你要求的）
    sql = text("""
        SELECT
            product,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY product
        ORDER BY total_qty DESC, rows DESC, product ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()
    return {"year": year, "order_by": "total_qty DESC", "top_products": [dict(r) for r in rows]}


@app.get("/sales/top_customers")
def sales_top_customers(
    year: int = Query(..., description="年份，例如 2025"),
    n: int = Query(10, ge=1, le=100, description="回傳前 N 名")
):
    # ✅ 依數量排序（你要求的）
    sql = text("""
        SELECT
            customer,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM sales
        WHERE year = :year
        GROUP BY customer
        ORDER BY total_qty DESC, rows DESC, customer ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()
    return {"year": year, "order_by": "total_qty DESC", "top_customers": [dict(r) for r in rows]}


@app.get("/sales/search")
def sales_search(
    q: str = Query(..., description="模糊關鍵字（客戶/品號）"),
    year: int | None = Query(None, description="不填代表不限年份"),
    limit: int = Query(50, ge=1, le=200, description="最多回傳筆數")
):
    where_year = "AND year = :year" if year is not None else ""
    sql = text(f"""
        SELECT date, year, customer, product, quantity, amount
        FROM sales
        WHERE (customer ILIKE :pat OR product ILIKE :pat)
        {where_year}
        ORDER BY date DESC
        LIMIT :limit
    """)
    params = {"pat": f"%{q}%", "limit": limit}
    if year is not None:
        params["year"] = year

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return {"q": q, "year": year, "limit": limit, "rows": [dict(r) for r in rows]}


# -------------------------
# Purchase
# -------------------------
@app.get("/purchase/summary")
def purchase_summary(year: int = Query(..., description="年份，例如 2024")):
    sql = text("""
        SELECT
            :year AS year,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM purchase
        WHERE year = :year
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"year": year}).mappings().first()
    return dict(row)


@app.get("/purchase/top_products")
def purchase_top_products(
    year: int = Query(..., description="年份，例如 2024"),
    n: int = Query(10, ge=1, le=100, description="回傳前 N 名")
):
    sql = text("""
        SELECT
            product,
            COUNT(*) AS rows,
            COALESCE(SUM(quantity), 0) AS total_qty,
            COALESCE(SUM(amount), 0) AS total_amount
        FROM purchase
        WHERE year = :year
        GROUP BY product
        ORDER BY total_qty DESC, rows DESC, product ASC
        LIMIT :n
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"year": year, "n": n}).mappings().all()
    return {"year": year, "order_by": "total_qty DESC", "top_products": [dict(r) for r in rows]}


@app.get("/purchase/search")
def purchase_search(
    q: str = Query(..., description="模糊關鍵字（供應商/品號）"),
    year: int | None = Query(None, description="不填代表不限年份"),
    limit: int = Query(50, ge=1, le=200, description="最多回傳筆數")
):
    where_year = "AND year = :year" if year is not None else ""
    sql = text(f"""
        SELECT date, year, supplier, product, quantity, amount
        FROM purchase
        WHERE (supplier ILIKE :pat OR product ILIKE :pat)
        {where_year}
        ORDER BY date DESC
        LIMIT :limit
    """)
    params = {"pat": f"%{q}%", "limit": limit}
    if year is not None:
        params["year"] = year

    with engine.connect() as conn:
        rows = conn.execute(sql, params).mappings().all()

    return {"q": q, "year": year, "limit": limit, "rows": [dict(r) for r in rows]}
