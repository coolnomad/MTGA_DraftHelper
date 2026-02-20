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


def test_move_logic_and_basics():
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=DummyScorer())
    ses = svc.create_session(0.55)
    ses.pool_counts = {"Lightning Bolt": 2}

    svc.move(ses.session_id, "Lightning Bolt", "pool", "locked")
    assert ses.pool_counts["Lightning Bolt"] == 1
    assert ses.locked_counts["Lightning Bolt"] == 1

    # Unlimited basic from pool -> locked even when absent in pool
    svc.move(ses.session_id, "Island", "pool", "locked")
    assert ses.locked_counts["Island"] == 1
    assert "Island" not in ses.pool_counts

    # Basic removed to pool should just decrement locked
    svc.move(ses.session_id, "Island", "locked", "pool")
    assert ses.locked_counts.get("Island", 0) == 0


def test_load_pool_with_explicit_locked_and_wobble():
    svc = DeckBuildService(asset_loader=DummyAssets(), scorer=DummyScorer())
    ses = svc.create_session(0.55)
    svc.load_pool(
        ses.session_id,
        list_text="Card In Pool\n",
        csv_text=None,
        locked_list_text="2 Lightning Bolt\n",
        locked_csv_text=None,
        wobble_list_text="1 Negate\n",
        wobble_csv_text=None,
    )
    assert ses.pool_counts.get("Card In Pool", 0) == 1
    assert ses.locked_counts.get("Lightning Bolt", 0) == 2
    assert ses.wobble_counts.get("Negate", 0) == 1


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
