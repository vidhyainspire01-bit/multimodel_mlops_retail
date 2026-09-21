"""Champion-vs-challenger gate for models registered in Unity Catalog.

Outcomes
  promoted : challenger beat the champion by more than min_delta (or no champion existed)
  kept     : no meaningful difference; champion stays, nothing fails
  rejected : challenger regressed beyond tolerance, or missed the absolute threshold;
             champion stays, and the task fails if fail_on_regression=True

The decision logic is pure (no mlflow) so it can be unit-tested without Databricks.
"""

PROMOTED = "promoted"
KEPT = "kept"
REJECTED = "rejected"


def meets_threshold(score, higher_is_better, absolute_threshold=None):
    """Absolute quality floor (or ceiling for error metrics). None disables the check."""
    if absolute_threshold is None:
        return True
    return score >= absolute_threshold if higher_is_better else score <= absolute_threshold


def decide(champion_score, challenger_score, higher_is_better, min_delta=0.0):
    """True only if the challenger beats the champion by more than min_delta."""
    if higher_is_better:
        return challenger_score > champion_score + min_delta
    return challenger_score < champion_score - min_delta


def is_regression(champion_score, challenger_score, higher_is_better, tolerance=0.0):
    """True if the challenger is worse than the champion by more than `tolerance`."""
    if higher_is_better:
        return challenger_score < champion_score - tolerance
    return challenger_score > champion_score + tolerance


def _reject(message, fail_on_regression):
    print(f"REJECTED. {message}")
    if fail_on_regression:
        raise RuntimeError(f"Promotion gate failed: {message}")
    return REJECTED


def promote_if_better(
    model_name,             # "<catalog>.<schema>.<model>"
    challenger_version,
    score_fn,               # callable(model_uri) -> float, evaluated on the SAME holdout
    higher_is_better,       # True for AUC / PR-AUC, False for WAPE / SMAPE
    min_delta=0.0,          # improvement required to take the champion alias
    absolute_threshold=None,  # e.g. the baseline score the model must beat
    fail_on_regression=False,
    regression_tolerance=0.005,
):
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    version = str(challenger_version)
    client.set_registered_model_alias(model_name, "challenger", version)

    chall_score = score_fn(f"models:/{model_name}/{version}")
    if not meets_threshold(chall_score, higher_is_better, absolute_threshold):
        return _reject(
            f"challenger v{version} score {chall_score:.4f} misses the required threshold "
            f"{absolute_threshold:.4f}.",
            fail_on_regression,
        )

    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        champ = None

    if champ is None:
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"No champion existed. Promoted v{version} (score {chall_score:.4f}).")
        return PROMOTED

    champ_score = score_fn(f"models:/{model_name}/{champ.version}")
    if decide(champ_score, chall_score, higher_is_better, min_delta):
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"PROMOTED v{version}: {chall_score:.4f} vs champion {champ_score:.4f}")
        return PROMOTED

    if is_regression(champ_score, chall_score, higher_is_better, regression_tolerance):
        return _reject(
            f"challenger v{version} {chall_score:.4f} is worse than champion v{champ.version} "
            f"{champ_score:.4f}. Champion stays.",
            fail_on_regression,
        )

    print(
        f"KEPT champion v{champ.version}: challenger v{version} {chall_score:.4f} vs "
        f"champion {champ_score:.4f} (no meaningful difference)."
    )
    return KEPT
