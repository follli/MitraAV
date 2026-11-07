import numpy as np

from mitraav.core.birdgeo import locate, weeks_to_text


def _fake(regions):
    """regions: species index -> (lat0, lat1, lon0, lon1, week0, week1)."""
    def predict(lat, lon, week):
        p = np.full(10, 0.001)
        for i, (a, b, c, d, w0, w1) in regions.items():
            if a <= lat <= b and c <= lon <= d and (week is None or w0 <= week <= w1):
                p[i] = 0.9
        return p
    return predict


def test_locates_region_and_season():
    pred = _fake({1: (30, 45, -85, -65, 10, 30), 2: (35, 50, -90, -60, 8, 32),
                  3: (20, 80, -180, 0, 1, 48)})
    r = locate(pred, [1, 2, 3], [1.0, 1.0, 1.0])
    c = r["clusters"][0]
    assert 33 <= c["lat"] <= 47 and -88 <= c["lon"] <= -62
    assert "to" in c["season"]


def test_one_wrong_species_cannot_veto():
    pred = _fake({1: (30, 45, -85, -65, 1, 48), 2: (30, 45, -85, -65, 1, 48),
                  3: (-40, -20, 100, 150, 1, 48)})          # third species is elsewhere
    c = locate(pred, [1, 2, 3], [1.0, 1.0, 0.3])["clusters"][0]
    assert 25 <= c["lat"] <= 50 and -90 <= c["lon"] <= -60


def test_no_species_gives_none():
    assert locate(_fake({}), [], []) is None


def test_weeks_to_text():
    assert weeks_to_text([17, 19, 21, 23]) == "May to Jun"
    assert weeks_to_text([]) == "year-round"
    assert weeks_to_text([47, 1, 3]) == "Dec to Jan"


def test_region_names_and_merge():
    from mitraav.core.birdgeo import merge_regions, region_name
    assert region_name(40.9, -73.8) == "eastern North America"
    assert region_name(51.5, -0.1) == "Europe"
    assert region_name(50.0, 45.0) == "Europe"            # the Volga is Europe, not central Asia
    assert region_name(41.0, 29.0) == "the Middle East and central Asia"
    assert region_name(-33.9, 151.2) == "Australia and New Zealand"
    m = merge_regions([dict(region="Europe", share=0.3, season="Apr", best_score=10.0),
                       dict(region="Europe", share=0.4, season="Jul", best_score=9.0),
                       dict(region="Africa", share=0.1, season="Apr", best_score=7.0)], 10.0)
    assert m[0]["region"] == "Europe" and abs(m[0]["share"] - 0.7) < 1e-9
    assert m[0]["seasons"] == ["Apr", "Jul"] and m[0]["fit"] == 1.0
    assert m[1]["region"] == "Africa" and 0.04 < m[1]["fit"] < 0.06      # exp(-3)


def test_one_mistaken_species_does_not_drag_the_region_away():
    # four species agree on a western range, one (likely a misidentification) only exists far east
    pred = _fake({1: (45, 60, -10, 10, 1, 48), 2: (45, 60, -10, 10, 1, 48), 3: (45, 60, -10, 10, 1, 48),
                  4: (45, 60, -10, 30, 1, 48), 5: (45, 60, 60, 120, 1, 48)})
    r = locate(pred, [1, 2, 3, 4, 5], [2.0, 2.0, 2.0, 2.0, 1.0])
    assert 44 <= r["clusters"][0]["lat"] <= 61 and r["clusters"][0]["lon"] < 35
