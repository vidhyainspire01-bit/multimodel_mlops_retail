# Databricks notebook source
# MAGIC %md
# MAGIC # repurchase / feature_prep
# MAGIC Builds snapshot-based training, holdout and scoring tables in `<catalog>.repurchase`.
# MAGIC
# MAGIC * Features use purchases **on or before** the snapshot date.
# MAGIC * Label = customer buys again in the next `horizon_days` (default 28).
# MAGIC * Holdout snapshot is the latest one with a fully observed label window; training snapshots
# MAGIC   are earlier and their label windows never overlap it (time-based split, no leakage).

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_ml")
dbutils.widgets.text("horizon_days", "28")
dbutils.widgets.text("n_train_snapshots", "4")

catalog = dbutils.widgets.get("catalog")
horizon_days = int(dbutils.widgets.get("horizon_days"))
n_train = int(dbutils.widgets.get("n_train_snapshots"))

# COMMAND ----------

# Make the bundle root importable (notebooks run from notebooks/repurchase/).
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))

from functools import reduce

from pyspark.sql import functions as F

from common.feature_builders import build_snapshot_features
from common.features import check_no_leakage, make_snapshots

# COMMAND ----------

tx = spark.table(f"{catalog}.hm.transactions")
customers = spark.table(f"{catalog}.hm.customers")
articles = spark.table(f"{catalog}.hm.articles")

min_date, max_date = tx.agg(F.min("t_dat"), F.max("t_dat")).first()
snaps = make_snapshots(max_date, horizon_days, n_train)
check_no_leakage(snaps, max_date, horizon_days)

# every snapshot needs some history before it
if min(snaps["train"]) <= min_date:
    raise ValueError(f"Earliest snapshot {min(snaps['train'])} has no history (data starts {min_date})")
print(f"data {min_date} -> {max_date}")
print(f"holdout snapshot {snaps['holdout']}, train snapshots {snaps['train']}, score {snaps['score']}")

# COMMAND ----------


def build(snapshot_list, with_label):
    parts = [
        build_snapshot_features(tx, customers, articles, s, horizon_days, with_label)
        for s in snapshot_list
    ]
    return reduce(lambda a, b: a.unionByName(b), parts)


def write(df, name):
    (
        df.write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{catalog}.repurchase.{name}")
    )


write(build(snaps["train"], True), "train_features")
write(build([snaps["holdout"]], True), "holdout")
write(build([snaps["score"]], False), "score_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity checks (fail the task if something is off)

# COMMAND ----------

train = spark.table(f"{catalog}.repurchase.train_features")
holdout = spark.table(f"{catalog}.repurchase.holdout")
score = spark.table(f"{catalog}.repurchase.score_features")

n_train_rows, n_hold, n_score = train.count(), holdout.count(), score.count()
train_rate = train.agg(F.avg("label")).first()[0]
hold_rate = holdout.agg(F.avg("label")).first()[0]
dupes = n_train_rows - train.select("customer_id", "snapshot_date").distinct().count()
footwear_signal = train.agg(F.sum("share_footwear")).first()[0]

print(f"train rows={n_train_rows:,} positive rate={train_rate:.3f}")
print(f"holdout rows={n_hold:,} positive rate={hold_rate:.3f}")
print(f"score rows={n_score:,}")

assert min(n_train_rows, n_hold, n_score) > 0, "an output table is empty"
assert dupes == 0, "duplicate (customer, snapshot) rows in train_features"
assert 0.005 < train_rate < 0.7, f"implausible train positive rate {train_rate}"
assert 0.005 < hold_rate < 0.7, f"implausible holdout positive rate {hold_rate}"
assert footwear_signal and footwear_signal > 0, (
    "no footwear purchases found - check product_group_name == 'Shoes' in the articles table"
)
