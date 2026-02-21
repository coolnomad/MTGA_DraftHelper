from scripts.run_deckbuild_ui import (
    BASIC_LANDS,
    DeckBuildService,
    DeckBumpScorer,
    card_to_feature_token,
    parse_csv_text,
    parse_pasted_list,
)


class DummyAssets:
    def find_by_name(self, name):
        return None

    def art_uri_for_name(self, name):
        return None

    def scryfall_image_url(self, name, version="art_crop"):
        return f"https://example.test/{version}/{name}"


class DummyScorer:
    def predict_many(self, locked_counts, base_p_values):
        return [{"base_p": float(v), "deck_bump": float(sum(locked_counts.values())) * 0.001} for v in base_p_values]

    def predict_batch(self, decks, base_p):
        return [float(sum(d.values())) * 0.001 for d in decks]


class MockBeamScorer:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, decks, base_p):
        self.calls += 1
        out = []
        for d in decks:
            s = (
                3.0 * d.get("Upgrade2", 0)
                + 2.0 * d.get("Upgrade1", 0)
                - 1.5 * d.get("Bad1", 0)
                - 1.0 * d.get("Bad2", 0)
            )
            out.append(s * float(base_p))
        return out


def test_parse_pasted_list():
    text = "2 Lightning Bolt\nLightning Bolt\n\n1 Shock\n"
    out = parse_pasted_list(text)
    assert out["Lightning Bolt"] == 3
    assert out["Shock"] == 1


def test_parse_csv_text_header_and_noheader():
    csv1 = "name,count\nLightning Bolt,2\nShock,1\n"
    out1 = parse_csv_text(csv1)
    assert out1["Lightning Bolt"] == 2
    assert out1["Shock"] == 1

    csv2 = "Lightning Bolt,2\nShock,1\n"
    out2 = parse_csv_text(csv2)
    assert out2["Lightning Bolt"] == 2
    assert out2["Shock"] == 1


def test_three_zone_move_rules_and_basics():
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=DummyScorer())
    ses = svc.create_session(0.55)
    ses.locked_counts = {"Lightning Bolt": 1}
    ses.wobble_counts = {"Lightning Bolt": 1}
    svc._recompute_pool_counts(ses)

    svc.move(ses.session_id, "Lightning Bolt", "wobble", "locked")
    assert ses.locked_counts["Lightning Bolt"] == 2

    svc.move(ses.session_id, "Island", "pool", "locked")
    assert ses.locked_counts["Island"] == 1

    svc.move(ses.session_id, "Island", "locked", "ignored")
    assert ses.ignored_counts.get("Island", 0) == 1

    try:
        svc.move(ses.session_id, "Island", "ignored", "locked")
        assert False, "ignored->locked should be blocked"
    except Exception:
        pass


def test_load_pool_with_three_zones_and_pool_invariant():
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=DummyScorer())
    ses = svc.create_session(0.55)
    svc.load_pool(
        ses.session_id,
        locked_list_text="2 Lightning Bolt\n",
        locked_csv_text=None,
        wobble_list_text="1 Negate\n",
        wobble_csv_text=None,
        ignored_list_text="1 Cancel\n",
        ignored_csv_text=None,
    )
    assert ses.locked_counts.get("Lightning Bolt", 0) == 2
    assert ses.wobble_counts.get("Negate", 0) == 1
    assert ses.ignored_counts.get("Cancel", 0) == 1
    assert ses.pool_counts.get("Cancel", 0) == 0
    assert ses.pool_counts.get("Lightning Bolt", 0) == 2
    assert ses.pool_counts.get("Negate", 0) == 1


def test_suggest_swaps_and_auto_iterate():
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=DummyScorer())
    ses = svc.create_session(0.55)
    ses.locked_counts = {f"Card{i}": 1 for i in range(35)}
    ses.wobble_counts = {"Negate": 1, "Disdainful Stroke": 1}
    svc._recompute_pool_counts(ses)
    out = svc.suggest_swaps(ses.session_id, top_k=5, rank_mode="user")
    assert "current" in out
    assert "suggestions" in out
    it = svc.auto_iterate(ses.session_id, max_steps=2)
    assert "applied_swaps" in it
    assert "final" in it


def test_min_lock_prevents_removal():
    scorer = MockBeamScorer()
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=scorer)
    ses = svc.create_session(0.55)
    ses.locked_counts = {f"Filler{i}": 1 for i in range(38)}
    ses.locked_counts["Bad1"] = 2
    ses.wobble_counts = {"Upgrade1": 1}
    svc._recompute_pool_counts(ses)
    svc.set_min_lock(ses.session_id, "Bad1", 2)
    out = svc.suggest_swaps(ses.session_id, top_k=10)
    assert not any(s["remove"] == "Bad1" for s in out["suggestions"])


def test_partial_min_lock_and_manual_removable_filtering():
    scorer = MockBeamScorer()
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=scorer)
    ses = svc.create_session(0.55)
    ses.locked_counts = {f"Filler{i}": 1 for i in range(37)}
    ses.locked_counts["Bad1"] = 2
    ses.locked_counts["Bad2"] = 1
    ses.wobble_counts = {"Upgrade1": 1, "Upgrade2": 1}
    svc._recompute_pool_counts(ses)
    svc.set_min_lock(ses.session_id, "Bad1", 1)
    out = svc.suggest_swaps(ses.session_id, top_k=20, removable_cards=["Bad1", "Bad2"])
    assert all(s["remove"] in {"Bad1", "Bad2"} for s in out["suggestions"])


def test_all_locked_returns_warning():
    scorer = MockBeamScorer()
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=scorer)
    ses = svc.create_session(0.55)
    ses.locked_counts = {"Bad1": 1}
    ses.wobble_counts = {"Upgrade1": 1}
    svc._recompute_pool_counts(ses)
    svc.set_min_lock(ses.session_id, "Bad1", 1)
    out = svc.suggest_swaps(ses.session_id, top_k=5)
    assert len(out["suggestions"]) == 0
    assert any("min locks" in w for w in out["current"]["warnings"])


def test_land_swap_only_enforced():
    scorer = MockBeamScorer()
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=scorer)
    ses = svc.create_session(0.55)
    ses.locked_counts = {f"Filler{i}": 1 for i in range(38)}
    ses.locked_counts["Bad1"] = 1
    ses.locked_counts["Island"] = 1
    ses.wobble_counts = {"Upgrade1": 1}
    svc._recompute_pool_counts(ses)
    svc.set_land_swap_only(ses.session_id, "Upgrade1", True)
    out = svc.suggest_swaps(ses.session_id, top_k=20)
    assert all(s["remove"] in BASIC_LANDS for s in out["suggestions"])


def test_beam_search_and_constraints():
    scorer = MockBeamScorer()
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=scorer)
    ses = svc.create_session(0.55)
    ses.locked_counts = {f"Filler{i}": 1 for i in range(38)}
    ses.locked_counts["Bad1"] = 1
    ses.locked_counts["Bad2"] = 1
    ses.wobble_counts = {"Upgrade1": 1, "Upgrade2": 1}
    svc._recompute_pool_counts(ses)
    out = svc.optimize_beam(
        session_id=ses.session_id,
        steps=3,
        beam_width=4,
        top_children_per_parent=20,
        R=5,
        rank_mode="user",
    )
    assert out["best"]["rank_score"] >= out["start"]["rank_score"]

    svc.set_min_lock(ses.session_id, "Bad1", 1)
    out2 = svc.optimize_beam(session_id=ses.session_id, steps=3, beam_width=4, top_children_per_parent=20, R=5)
    assert all(step["remove"] != "Bad1" for step in out2["path"])


def test_beam_scoring_cache_reuse():
    scorer = MockBeamScorer()
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=scorer)
    ses = svc.create_session(0.55)
    ses.locked_counts = {f"Filler{i}": 1 for i in range(39)}
    ses.locked_counts["Bad1"] = 1
    ses.wobble_counts = {"Upgrade1": 1}
    svc._recompute_pool_counts(ses)
    _ = svc.optimize_beam(session_id=ses.session_id, steps=2, beam_width=3, R=4)
    calls_after_first = scorer.calls
    _ = svc.optimize_beam(session_id=ses.session_id, steps=2, beam_width=3, R=4)
    assert scorer.calls == calls_after_first


def test_feature_encoding_correctness():
    scorer = DeckBumpScorer.__new__(DeckBumpScorer)
    scorer.feature_cols = ["base_p", "deck_lightning_bolt", "deck_shock"]
    scorer.col_to_idx = {c: i for i, c in enumerate(scorer.feature_cols)}

    x = scorer.vectorize_deck({"Lightning Bolt": 2, "Shock": 1, "Missing": 4}, base_p=0.6)
    assert x.shape[0] == 3
    assert x[0] == 0.6
    assert x[1] == 2
    assert x[2] == 1


def test_model_smoke_predict():
    scorer = DeckBumpScorer()
    preds = scorer.predict_many({}, [0.5])
    assert len(preds) == 1
    assert "deck_bump" in preds[0]


def test_token_format():
    assert card_to_feature_token("Lightning Bolt") == "lightning_bolt"
    assert card_to_feature_token("Oko, Thief of Crowns") == "oko,_thief_of_crowns"
    assert "Island" in BASIC_LANDS
