# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest H&M data into Delta
# MAGIC **STUB.** For now it only creates the schemas, which also proves the service principal
# MAGIC has the Unity Catalog permissions it needs. Real ingest comes in build step 2.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("raw_path", "/Volumes/dev_ml/hm/raw")
dbutils.widgets.text("sample_customers", "200000")

catalog = dbutils.widgets.get("catalog")
raw_path = dbutils.widgets.get("raw_path")
sample_customers = int(dbutils.widgets.get("sample_customers"))

# COMMAND ----------

for schema in ("hm", "repurchase", "ts_demand"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"Schemas ready in catalog {catalog}. raw_path={raw_path}, sample={sample_customers}")
