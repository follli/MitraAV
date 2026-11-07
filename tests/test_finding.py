from mitraav.core.finding import Finding


def test_undetected_scores_zero():
    assert Finding(probe="x", detected=False, strength=1, narrows=1).score == 0


def test_cheaper_followup_ranks_higher():
    cheap = Finding(probe="a", detected=True, strength=0.8, narrows=0.8, effort=1)
    costly = Finding(probe="b", detected=True, strength=0.8, narrows=0.8, effort=3)
    assert cheap.score > costly.score
