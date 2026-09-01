from a2r.learning.bandit import BanditRouter


def test_bandit_persists_and_updates_weight(tmp_path):
    router = BanditRouter(tmp_path / "weights.sqlite", ["billing", "product"], exploration_rate=0)
    assert router.rank_pipelines("billing") == ["billing", "product"]
    new_weight = router.update_weight("billing", "billing", 1)
    assert new_weight > 0.5
    restarted = BanditRouter(tmp_path / "weights.sqlite", ["billing", "product"], exploration_rate=0)
    assert restarted.matrix()["billing"]["billing"]["weight"] == new_weight
