"""Business rules applied on top of the raw model probability at serving time.

Pure Python (no mlflow / pyspark), so these are unit-tested without any infra and can be
edited by a business stakeholder's PR without touching model code. `apply_rules` is called
once per customer row by the serving wrapper in common/serving.py.

These thresholds and actions are illustrative for the demo - tune them with the business
team before using them for anything real.
"""

RISK_TIERS = ("High", "Medium", "Low")
HIGH_THRESHOLD = 0.5
MEDIUM_THRESHOLD = 0.2

ACTION_BY_TIER = {
    "High": "send_personalized_offer",
    "Medium": "include_in_nurture_campaign",
    "Low": "no_action",
}

# Below this many days of purchase history, the model has too little signal to trust -
# it was trained on customers who already had at least one prior purchase.
MIN_TENURE_DAYS_FOR_CONFIDENCE = 14

# A customer who bought within this many days is in a natural post-purchase lull; offering
# them another discount immediately tends to erode margin rather than drive incremental sales.
COOLDOWN_RECENCY_DAYS = 3


def risk_tier(probability: float) -> str:
    if probability >= HIGH_THRESHOLD:
        return "High"
    if probability >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def apply_rules(probability: float, recency_days: float, tenure_days: float) -> dict:
    """Turn a raw probability + a couple of raw features into a decision the business can act on.

    Returns a dict with repurchase_probability (unchanged - the model's number is never
    silently altered), risk_tier, recommended_action, low_confidence and cooldown flags.
    """
    tier = risk_tier(probability)
    low_confidence = tenure_days < MIN_TENURE_DAYS_FOR_CONFIDENCE
    cooldown = recency_days < COOLDOWN_RECENCY_DAYS

    action = ACTION_BY_TIER[tier]
    if cooldown and action == "send_personalized_offer":
        # Don't override the tier (that's the model's honest read); override the ACTION,
        # since acting on it immediately would be the wrong business move regardless of score.
        action = "defer_offer_cooldown"

    return {
        "repurchase_probability": float(probability),
        "risk_tier": tier,
        "recommended_action": action,
        "low_confidence": bool(low_confidence),
    }
