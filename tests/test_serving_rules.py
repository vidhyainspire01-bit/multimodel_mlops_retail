
from common.serving_rules import (
    COOLDOWN_RECENCY_DAYS,
    MIN_TENURE_DAYS_FOR_CONFIDENCE,
    apply_rules,
    risk_tier,
)


def test_risk_tier_boundaries():
    assert risk_tier(0.5) == "High"
    assert risk_tier(0.499) == "Medium"
    assert risk_tier(0.2) == "Medium"
    assert risk_tier(0.199) == "Low"
    assert risk_tier(0.0) == "Low"
    assert risk_tier(1.0) == "High"


def test_apply_rules_probability_is_never_modified():
    # the model's number is always returned untouched, even when business rules kick in
    out = apply_rules(probability=0.73, recency_days=1, tenure_days=1)
    assert out["repurchase_probability"] == 0.73


def test_high_probability_normal_customer_gets_offer():
    out = apply_rules(probability=0.8, recency_days=30, tenure_days=200)
    assert out["risk_tier"] == "High"
    assert out["recommended_action"] == "send_personalized_offer"
    assert out["low_confidence"] is False


def test_high_probability_recent_purchase_gets_deferred_not_suppressed():
    out = apply_rules(probability=0.8, recency_days=1, tenure_days=200)
    assert out["risk_tier"] == "High"                      # the model's read is untouched
    assert out["recommended_action"] == "defer_offer_cooldown"  # only the ACTION changes


def test_cooldown_does_not_affect_non_high_tiers():
    out = apply_rules(probability=0.3, recency_days=1, tenure_days=200)
    assert out["risk_tier"] == "Medium"
    assert out["recommended_action"] == "include_in_nurture_campaign"


def test_new_customer_flagged_low_confidence():
    out = apply_rules(probability=0.6, recency_days=30, tenure_days=1)
    assert out["low_confidence"] is True
    assert out["risk_tier"] == "High"   # confidence flag informs the caller, doesn't hide the score


def test_boundary_values_are_inclusive_thresholds():
    just_below = apply_rules(0.5, recency_days=100, tenure_days=MIN_TENURE_DAYS_FOR_CONFIDENCE)
    assert just_below["low_confidence"] is False
    on_cooldown_edge = apply_rules(0.9, recency_days=COOLDOWN_RECENCY_DAYS, tenure_days=100)
    assert on_cooldown_edge["recommended_action"] == "send_personalized_offer"  # not < threshold
