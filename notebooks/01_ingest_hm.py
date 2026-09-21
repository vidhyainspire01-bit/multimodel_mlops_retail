# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest H&M data into Delta
# MAGIC Reads the Kaggle H&M CSVs from a Unity Catalog Volume, samples customers
# MAGIC (deterministically), and writes `transactions`, `customers`, `articles` to `<catalog>.hm`.
# MAGIC
# MAGIC * Idempotent: if all three tables already exist it skips the load (set `force_reload=true` to rebuild).
# MAGIC * Validation always runs and fails the task if the data looks wrong.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("raw_path", "/Volumes/dev_ml/hm/raw")
dbutils.widgets.text("sample_customers", "200000")   # 0 or less = keep all customers
dbutils.widgets.text("force_reload", "false")

catalog = dbutils.widgets.get("catalog")
raw_path = dbutils.widgets.get("raw_path").rstrip("/")
sample_customers = int(dbutils.widgets.get("sample_customers"))
force_reload = dbutils.widgets.get("force_reload").lower() == "true"

schema_fqn = f"{catalog}.hm"
TABLES = ("transactions", "customers", "articles")
RAW_FILES = ("transactions_train.csv", "customers.csv", "articles.csv")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

for s in ("hm", "repurchase", "ts_demand"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{s}")


def write_table(df, name):
    (
        df.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{schema_fqn}.{name}")
    )


def load():
    present = {f.name for f in dbutils.fs.ls(raw_path)}
    missing = [f for f in RAW_FILES if f not in present]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {raw_path}. Upload the extracted Kaggle H&M CSVs to this volume."
        )

    # customers: deterministic sample (smallest hashes) so every run and environment agrees
    customers = spark.read.csv(f"{raw_path}/customers.csv", header=True, inferSchema=True)
    if sample_customers > 0:
        customers = (
            customers.withColumn("_h", F.xxhash64("customer_id"))
            .orderBy("_h")
            .limit(sample_customers)
            .drop("_h")
        )
    write_table(customers, "customers")

    # transactions: explicit schema avoids an extra pass over the 3.5 GB file
    tx_schema = StructType(
        [
            StructField("t_dat", DateType()),
            StructField("customer_id", StringType()),
            StructField("article_id", LongType()),
            StructField("price", DoubleType()),
            StructField("sales_channel_id", IntegerType()),
        ]
    )
    tx = spark.read.csv(f"{raw_path}/transactions_train.csv", header=True, schema=tx_schema)
    ids = spark.table(f"{schema_fqn}.customers").select("customer_id")
    write_table(tx.join(F.broadcast(ids), "customer_id", "inner"), "transactions")

    # articles: small, keep all of them
    articles = spark.read.csv(f"{raw_path}/articles.csv", header=True, inferSchema=True)
    write_table(articles, "articles")


already_loaded = all(spark.catalog.tableExists(f"{schema_fqn}.{t}") for t in TABLES)
if already_loaded and not force_reload:
    print(f"{schema_fqn} already populated; skipping load (force_reload=false).")
else:
    load()
    print(f"Loaded H&M tables into {schema_fqn}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation (fails the task if any check breaks)

# COMMAND ----------

tx = spark.table(f"{schema_fqn}.transactions")
customers = spark.table(f"{schema_fqn}.customers")
articles = spark.table(f"{schema_fqn}.articles")

n_tx, n_cust, n_art = tx.count(), customers.count(), articles.count()
d_min, d_max = tx.agg(F.min("t_dat"), F.max("t_dat")).first()
null_keys = tx.filter(
    F.col("customer_id").isNull() | F.col("article_id").isNull() | F.col("t_dat").isNull()
).count()
orphan_customers = tx.join(customers, "customer_id", "left_anti").count()
unknown_articles = tx.join(articles, "article_id", "left_anti").count()

print(f"transactions={n_tx:,} customers={n_cust:,} articles={n_art:,}")
print(f"date range {d_min} -> {d_max}")
print(f"null keys={null_keys}, orphan customers={orphan_customers}, unknown articles={unknown_articles}")

assert n_tx > 0, "transactions table is empty"
assert n_cust > 0, "customers table is empty"
assert n_art > 0, "articles table is empty"
assert null_keys == 0, f"{null_keys} transactions have null keys"
assert orphan_customers == 0, f"{orphan_customers} transactions reference unknown customers"
assert d_max > d_min, "date range looks wrong"

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from dev_ml.hm.transactions