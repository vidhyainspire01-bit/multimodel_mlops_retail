"""Spark tests on a toy dataset. Skipped automatically when pyspark is not installed."""

from datetime import date

import pytest

# importorskip skips this whole module when pyspark is missing (e.g. the light CI job)
SparkSession = pytest.importorskip("pyspark.sql").SparkSession
build_snapshot_features = pytest.importorskip("common.feature_builders").build_snapshot_features

S = date(2020, 8, 25)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[1]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def frames(spark):
    d = date.fromisoformat
    tx = spark.createDataFrame(
        [
            # A: history, buys again 10 days after snapshot -> label 1
            (d("2020-08-01"), "A", 1, 10.0, 2),
            (d("2020-08-20"), "A", 2, 20.0, 1),
            (d("2020-09-04"), "A", 1, 5.0, 2),
            # B: history, next purchase is 29 days after snapshot -> label 0
            (d("2020-08-10"), "B", 2, 30.0, 2),
            (d("2020-09-23"), "B", 2, 99.0, 2),
            # C: first ever purchase is after the snapshot -> must be excluded
            (d("2020-09-01"), "C", 1, 15.0, 1),
        ],
        "t_dat date, customer_id string, article_id long, price double, sales_channel_id int",
    )
    customers = spark.createDataFrame(
        [("A", 30), ("B", 41), ("C", 25)], "customer_id string, age int"
    )
    articles = spark.createDataFrame(
        [(1, "Shoes"), (2, "Garment Upper body")], "article_id long, product_group_name string"
    )
    return tx, customers, articles


def test_labels_and_population(frames):
    tx, customers, articles = frames
    rows = {
        r.customer_id: r
        for r in build_snapshot_features(tx, customers, articles, S, 28, True).collect()
    }
    assert set(rows) == {"A", "B"}          # C has no history at the snapshot
    assert rows["A"].label == 1
    assert rows["B"].label == 0             # purchase on day 29 is outside the window


def test_features_use_only_past_data(frames):
    tx, customers, articles = frames
    rows = {
        r.customer_id: r
        for r in build_snapshot_features(tx, customers, articles, S, 28, True).collect()
    }
    a = rows["A"]
    assert a.items_total == 2               # the 2020-09-04 purchase must not be counted
    assert a.spend_total == 30.0
    assert a.recency_days == 5              # 2020-08-20 -> 2020-08-25
    assert a.tenure_days == 24
    assert a.days_since_last_footwear == 24  # only shoe purchase was 2020-08-01
    assert a.share_footwear == 0.5
    assert rows["B"].days_since_last_footwear is None


def test_scoring_mode_has_no_label(frames):
    tx, customers, articles = frames
    df = build_snapshot_features(tx, customers, articles, S, 28, with_label=False)
    assert "label" not in df.columns
