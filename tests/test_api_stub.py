import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    TestClient = None
from src.api.server import app, recommend


@pytest.mark.skipif(TestClient is None, reason="httpx dependency missing for TestClient")
def test_recommend_endpoint_stub():
    client = TestClient(app)
    payload = {
        "pack_cards": ["card_a", "card_b"],
        "pool_counts": {"card_a": 2},
        "pick_number": 1,
        "pack_number": 1,
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2


def test_recommend_function_direct_call():
    payload = {
        "pack_cards": ["card_a", "card_b"],
        "pool_counts": {"card_a": 2},
        "pick_number": 1,
        "pack_number": 1,
    }
    data = recommend(payload)
    assert "recommendations" in data
    assert len(data["recommendations"]) == 2

