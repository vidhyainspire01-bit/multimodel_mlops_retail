"""Champion-vs-challenger gate for models registered in Unity Catalog.

The decision logic is pure (no mlflow) so it can be unit-tested in CI without Databricks.
"""


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


def _no_promote(message, fail_on_no_promote):
    print(f"NOT promoted. {message}")
    if fail_on_no_promote:
        raise RuntimeError(f"Promotion gate failed: {message}")
    return False


def promote_if_better(
    model_name,             # "<catalog>.<schema>.<model>"
    challenger_version,
    score_fn,               # callable(model_uri) -> float, evaluated on the SAME holdout
    higher_is_better,       # True for AUC / PR-AUC, False for WAPE / SMAPE
    min_delta=0.0,
    absolute_threshold=None,
    fail_on_no_promote=False,
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
        return _no_promote(
            f"challenger v{version} score {chall_score:.4f} fails the absolute threshold "
            f"{absolute_threshold}.",
            fail_on_no_promote,
        )

    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        champ = None

    if champ is None:
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"No champion existed. Promoted v{version} (score {chall_score:.4f}).")
        return True

    champ_score = score_fn(f"models:/{model_name}/{champ.version}")
    if decide(champ_score, chall_score, higher_is_better, min_delta):
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"Promoted v{version}: {chall_score:.4f} vs champion {champ_score:.4f}")
        return True

    return _no_promote(
        f"challenger v{version} {chall_score:.4f} vs champion v{champ.version} "
        f"{champ_score:.4f}. Champion stays.",
        fail_on_no_promote,
    )
