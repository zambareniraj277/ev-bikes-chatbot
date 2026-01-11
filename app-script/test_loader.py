from db.db_utils import run_query

query = "SELECT * FROM sales_data LIMIT 5;"  # adjust table name if needed
df = run_query(query)
print(df)
