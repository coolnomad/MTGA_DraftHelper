import os
from pathlib import Path

import pytest
import numpy as np

from draft_env.pack_sampler import load_card_pool, sample_pack, CARDS_PARQUET


@pytest.mark.skipif(not Path(CARDS_PARQUET).exists(), reason="cards.parquet missing")
def test_fin_and_fca_pools_present():
    fin_pool = load_card_pool("FIN")
    fca_pool = load_card_pool("FCA")
    # ensure we have cards to sample from and basic land slot is defined
    assert len(fin_pool["common"]) > 0
    assert "basic_land" in fin_pool
    assert len(fca_pool["common"]) > 0
    assert "basic_land" in fca_pool


@pytest.mark.skipif(not Path(CARDS_PARQUET).exists(), reason="cards.parquet missing")
def test_sample_pack_includes_basic_land_slot():
    rng = np.random.default_rng(0)
    pack = sample_pack(rng, pack_size=15, set_code="FIN")
    assert len(pack) == 15
    pool = load_card_pool("FIN")
    basics = set(pool.get("basic_land", []))
    basics_in_pack = [c for c in pack if c in basics]
    if basics:
        assert len(basics_in_pack) <= 1
        assert len(basics_in_pack) == 1  # expects basic slot filled when available
