# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / assert_outputs  (STUB - always passes)
# MAGIC Real version asserts on the scored table: row count, no null scores, 0 <= p <= 1,
# MAGIC champion alias exists. This is the integration test CI relies on.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
print("assert_outputs stub: nothing to assert yet")
