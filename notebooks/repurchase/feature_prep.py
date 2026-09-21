# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / feature_prep  (STUB)

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
catalog = dbutils.widgets.get("catalog")
print(f"feature_prep stub running against {catalog}")
