# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / batch_score  (STUB)

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
catalog = dbutils.widgets.get("catalog")
print(f"batch_score stub running against {catalog}")
