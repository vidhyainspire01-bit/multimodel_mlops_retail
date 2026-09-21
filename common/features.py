"""Pure-Python helpers for the repurchase model: snapshot logic and the feature list.

No pyspark import here, so these run in the lightweight CI unit tests.
"""

from datetime import date, timedelta
from itertools import pairwise

# Columns the model is trained on. train / evaluate / batch_score must all use this list.
# Note: static customer fields (club status, Active, news frequency) are deliberately excluded.
# customers.csv is a single end-of-period snapshot, so they can leak future behaviour.
FEATURES = [
    "recency_days",
    "tenure_days",
    "purchase_days_total",
    "items_total",
    "spend_total",
    "avg_item_price",
    "purchase_days_28d",
    "purchase_days_90d",
    "spend_90d",
    "online_share",
    "share_footwear",
    "share_garment",
    "share_accessories",
    "n_product_groups",
    "days_since_last_footwear",
    "age",
]
LABEL = "label"
KEYS = ["customer_id", "snapshot_date"]


def make_snapshots(max_date: date, horizon_days: int = 28, n_train: int = 4) -> dict:
    """Snapshot dates for training, holdout and scoring.

    Features for a snapshot S use purchases on or before S.
    The label for S is 'any purchase in (S, S + horizon_days]'.
    Holdout: the latest snapshot whose label window is fully observed.
    Train: earlier snapshots spaced one horizon apart, so no label window overlaps the holdout's.
    Score: max_date (label unknown - this is what batch scoring uses).
    """
    if horizon_days < 1 or n_train < 1:
        raise ValueError("horizon_days and n_train must be >= 1")
    h = timedelta(days=horizon_days)
    holdout = max_date - h
    train = [holdout - h * k for k in range(1, n_train + 1)]
    return {"train": train, "holdout": holdout, "score": max_date}


def check_no_leakage(snapshots: dict, max_date: date, horizon_days: int = 28) -> None:
    """Raise ValueError if any training label window reaches into the holdout's period."""
    h = timedelta(days=horizon_days)
    holdout, train = snapshots["holdout"], snapshots["train"]
    if holdout + h > max_date:
        raise ValueError("holdout label window extends beyond the available data")
    if any(s + h > holdout for s in train):
        raise ValueError("a training label window overlaps the holdout snapshot")
    ordered = sorted(train)
    if any(b - a < h for a, b in pairwise(ordered)):
        raise ValueError("training label windows overlap each other")
