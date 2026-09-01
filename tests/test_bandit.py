from a2r.learning.bandit import BanditRouter


def test_bandit_persists_and_updates_weight(tmp_path):
    db_path = tmp_path / "weights.sqlite"
    router = BanditRouter(db_path, ["billing", "product", "hr"], learning_rate=0.1, min_weight=0.1)
    chosen = router.route("billing", explore=False)
    assert chosen == "billing"

    old_weight = router.weights["billing"]
    router.update("billing", 1.0)
    assert router.weights["billing"] > old_weight
