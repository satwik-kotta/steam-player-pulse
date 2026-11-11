import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spp import anomaly, config as C, cycle, ingest, sales  # noqa: E402


def synthetic_samples(appid, ratio, days=80, mean=1000, seed=0, start="2026-08-10"):
    """One sample a day at a rotating hour, from a cycle whose peak / trough ratio is known."""
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.arange(days), "D") \
        + pd.to_timedelta((np.arange(days) * 7) % 24, "h")
    amp = np.log(ratio) / 2
    hour = ts.hour.values
    y = mean * np.exp(amp * np.cos(2 * np.pi * (hour - 18) / 24) + rng.normal(0, 0.05, days))
    return pd.DataFrame({"appid": appid, "ts": ts, "players": np.round(y)})


def test_cycle_recovers_known_ratio_and_flat_game():
    s = pd.concat([synthetic_samples(1, 2.5), synthetic_samples(2, 1.0, seed=1)])
    cyc, adj = cycle.fit_all(s, boot=20)
    r = cyc.set_index("appid")
    assert r.loc[1, "peak_trough"] == pytest.approx(2.5, rel=0.12)
    assert r.loc[1, "peak_hour_utc"] == pytest.approx(18, abs=1.5)
    assert r.loc[2, "peak_trough"] < 1.25
    assert r.loc[2, "trough_share"] > r.loc[1, "trough_share"]
    # hour adjustment removes the cycle: adjusted values no longer depend on the hour
    a = adj[adj.appid == 1].assign(h=s[s.appid == 1].ts.dt.hour.values)
    assert abs(np.corrcoef(np.cos(2 * np.pi * (a.h - 18) / 24), a.adjusted)[0, 1]) < 0.3


def test_quiet_and_short_games_are_not_fit():
    s = pd.concat([synthetic_samples(1, 2.0, mean=50), synthetic_samples(2, 2.0, days=10)])
    cyc, _ = cycle.fit_all(s, boot=0)
    assert cyc.empty


def test_validation_counts_idle_names():
    cyc = pd.DataFrame({"appid": [1, 2, 3, 4], "peak_trough": [1.1, 1.2, 2.0, 2.5]})
    names = pd.Series({1: "Idle Miner", 2: "Space Shooter", 3: "Cookie Clicker Deluxe", 4: "Racer"})
    v = cycle.validate(cyc, names)
    assert v["idle_named"] == 2 and v["base_rate"] == 0.5
    assert v["auc_flatness_vs_idle_name"] == pytest.approx(0.75)


def test_ingest_keeps_empty_answers_and_reads_both_price_shapes(tmp_path):
    ccu = tmp_path / "ccu" / "2026-10"
    ccu.mkdir(parents=True)
    (ccu / "2026-10-01_0500_s0.csv").write_text(
        "ts,appid,concurrent,peak_in_game,src\n1790000000,10,5,,single\n1790000001,20,,,single\n1790000002,30,0,,single\n")
    pr = tmp_path / "prices" / "2026-07"
    pr.mkdir(parents=True)
    (pr / "2026-07-10_0617.csv").write_text("appid,currency,initial,final,discount_pct,is_free\n10,USD,999,999,0,0\n")
    (pr / "2026-07-10_1817.csv").write_text(
        "appid,cc,currency,initial,final,discount_pct,is_free\n10,us,USD,999,499,50,0\n10,de,EUR,999,499,50,0\n")
    names = tmp_path / "games.json"
    names.write_text(json.dumps([{"appid": 10, "name": "Ten"}, {"appid": 20, "name": "Twenty"}]))
    con = ingest.connect(":memory:")
    ingest.load(con, str(tmp_path / "ccu" / "*" / "*.csv"), str(tmp_path / "prices" / "*" / "*.csv"), str(names))
    res = ingest.checks(con)
    assert res["ccu"]["declined_empty"] == 1 and res["ccu"]["zero_players"] == 1
    assert res["ccu"]["games_without_a_name"] == 1
    day = con.execute("select discount_pct, samples from price_day where appid = 10").fetchone()
    assert day == (50, 2)          # the last US price of the day wins; the German row is not US
    assert res["prices"]["discount_pct_disagrees_with_prices"] == 0


def _panel_with_lift(lift=0.3, n_t=60, n_c=40, seed=0):
    rng = np.random.default_rng(seed)
    days = [sales.PRE_START + timedelta(days=i) for i in range((sales.POST_END - sales.PRE_START).days + 1)]
    rows, prices = [], []
    for g in range(n_t + n_c):
        treated = g < n_t
        level = rng.normal(5, 1)
        for d in days:
            in_sale = C.SALE_START <= d < C.SALE_END
            common = 0.1 * np.sin(d.toordinal())                       # store-wide day effect
            y = level + common + (np.log1p(lift) if treated and in_sale else 0) + rng.normal(0, 0.05)
            rows.append({"appid": g, "date": d, "adjusted": y})
            prices.append({"appid": g, "day": d, "discount_pct": 50 if treated and in_sale else 0,
                           "is_free": 0, "initial": 1999})
    return pd.DataFrame(rows), pd.DataFrame(prices)


def test_autumn_did_recovers_injected_lift():
    adj, price_day = _panel_with_lift(0.3)
    groups = sales.autumn_groups(price_day)
    assert groups.treated.sum() == 60 and (groups.treated == 0).sum() == 40
    res = sales.autumn_did(sales.daily_panel(adj, price_day), groups)
    assert res["periods"]["sale"]["lift"] == pytest.approx(0.3, abs=0.03)
    assert abs(res["periods"]["week_after"]["lift"]) < 0.03
    assert res["mean_abs_pretrend"] < 0.03


def test_spike_found_and_store_wide_day_ignored():
    rng = np.random.default_rng(1)
    days = [date(2026, 9, 1) + timedelta(days=i) for i in range(40)]
    rows = []
    for g in range(30):
        for i, d in enumerate(days):
            y = 6 + rng.normal(0, 0.05) + (0.8 if i == 25 else 0)        # every game jumps on day 25
            if g == 0 and i == 32:
                y += np.log(5)                                            # one game jumps 5x on day 32
            rows.append({"appid": g, "date": d, "adjusted": y})
    sp = anomaly.spikes(pd.DataFrame(rows))
    assert list(zip(sp.appid, sp.date)) == [(0, days[32])]


def test_spike_explained_by_discount_start():
    sp = pd.DataFrame({"appid": [1, 2], "date": [date(2026, 9, 10)] * 2, "jump": [1.0, 1.0]})
    price_day = pd.DataFrame({"appid": [1, 1, 2, 2], "day": [date(2026, 9, 8), date(2026, 9, 9)] * 2,
                              "discount_pct": [0, 40, 0, 0]})
    out = anomaly.explain(sp, price_day).set_index("appid").cause
    assert out[1] == "discount started" and out[2] == "no price change"
