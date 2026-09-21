"""Spark feature + label builder for one snapshot date (imported by the notebooks)."""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_snapshot_features(
    tx: DataFrame,
    customers: DataFrame,
    articles: DataFrame,
    snapshot,                 # datetime.date
    horizon_days: int = 28,
    with_label: bool = True,
) -> DataFrame:
    """One row per customer with purchase history up to `snapshot`.

    Features use only transactions with t_dat <= snapshot.
    Label (if requested) = 1 if the customer buys in (snapshot, snapshot + horizon_days].
    """
    s = F.lit(snapshot).cast("date")

    tx_a = tx.join(articles.select("article_id", "product_group_name"), "article_id", "left")
    hist = tx_a.filter(F.col("t_dat") <= s).withColumn("age_days", F.datediff(s, F.col("t_dat")))

    is_shoe = F.col("product_group_name") == "Shoes"
    feats = hist.groupBy("customer_id").agg(
        F.min("age_days").alias("recency_days"),
        F.max("age_days").alias("tenure_days"),
        F.countDistinct("t_dat").alias("purchase_days_total"),
        F.count("*").alias("items_total"),
        F.sum("price").alias("spend_total"),
        F.avg("price").alias("avg_item_price"),
        F.countDistinct(F.when(F.col("age_days") < 28, F.col("t_dat"))).alias("purchase_days_28d"),
        F.countDistinct(F.when(F.col("age_days") < 90, F.col("t_dat"))).alias("purchase_days_90d"),
        F.sum(F.when(F.col("age_days") < 90, F.col("price")).otherwise(0.0)).alias("spend_90d"),
        F.avg((F.col("sales_channel_id") == 2).cast("double")).alias("online_share"),
        F.avg(is_shoe.cast("double")).alias("share_footwear"),
        F.avg(F.col("product_group_name").rlike("^Garment").cast("double")).alias("share_garment"),
        F.avg((F.col("product_group_name") == "Accessories").cast("double")).alias(
            "share_accessories"
        ),
        F.countDistinct("product_group_name").alias("n_product_groups"),
        F.min(F.when(is_shoe, F.col("age_days"))).alias("days_since_last_footwear"),
    )

    cust = customers.select("customer_id", F.col("age").cast("double").alias("age"))
    out = feats.join(cust, "customer_id", "left").withColumn("snapshot_date", s)

    if with_label:
        future = (
            tx.filter((F.col("t_dat") > s) & (F.col("t_dat") <= F.date_add(s, horizon_days)))
            .select("customer_id")
            .distinct()
            .withColumn("label", F.lit(1))
        )
        out = out.join(future, "customer_id", "left").withColumn(
            "label", F.coalesce(F.col("label"), F.lit(0))
        )
    return out
