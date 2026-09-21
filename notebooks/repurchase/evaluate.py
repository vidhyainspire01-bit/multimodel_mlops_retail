# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / evaluate  (STUB)
# MAGIC Real version calls `promote_if_better` from `common.promotion`.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("challenger_version", "")
dbutils.widgets.text("fail_on_no_promote", "false")

# COMMAND ----------

# Make the bundle root importable (notebooks run from notebooks/repurchase/).
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from common.promotion import decide  # proves the shared-code import works in every environment

print("import ok; decide(0.8, 0.9, True) =", decide(0.8, 0.9, True))
print("challenger_version param =", dbutils.widgets.get("challenger_version"))
