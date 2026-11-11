"""Run every analysis after ingest and write docs/data/results.json and outputs/*.csv.

    python -m spp.ingest
    python -m spp.run
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from . import anomaly, config as C, cycle, ingest, sales

warnings.filterwarnings("ignore", category=FutureWarning)

# Titles shown on the public dashboard skip adult games; they stay in every calculation and in outputs/*.csv.
ADULT = r"(?i)hentai|nsfw|🔞|boob|butt|waifu|sex|porn|nude|lewd|carnal|erotic"

SHOWCASE = {730: "Counter-Strike 2", 570: "Dota 2", 578080: "PUBG: BATTLEGROUNDS", 1172470: "Apex Legends",
            3419430: "Bongo Cat", 2923300: "Banana", 1623660: "MIR4", 304930: "Unturned"}


def main():
    C.DOCS.mkdir(parents=True, exist_ok=True)
    C.OUT.mkdir(exist_ok=True)
    con = ingest.connect()
    checks = json.loads((C.OUT / "ingest.json").read_text())
    samples = con.execute("select appid, ts, players from ccu").df()
    samples["ts"] = pd.to_datetime(samples.ts, utc=True)
    names = con.execute("select appid, name from games").df().drop_duplicates("appid").set_index("appid").name
    price_day = con.execute("select * from price_day").df()
    price_day["day"] = pd.to_datetime(price_day.day).dt.date

    # 1. daily cycles of established games (200+ players on average)
    cyc, _ = cycle.fit_all(samples, boot=200)
    cyc["name"] = cyc.appid.map(names)
    val = cycle.validate(cyc, names)
    cyc["idle_name"] = cyc.name.fillna("").str.contains(cycle.IDLE_WORDS, case=False, regex=True)
    cyc.drop(columns="hourly").sort_values("peak_trough").to_csv(C.OUT / "game_cycles.csv", index=False)
    flat = cyc[cyc.peak_trough < 1.3]
    weights = {"games": int(len(cyc)), "flat_games": int(len(flat)),
               "flat_share_of_games": float(len(flat) / len(cyc)),
               "flat_share_of_players": float(flat.mean_players.sum() / cyc.mean_players.sum())}
    bins = np.arange(1.0, 6.01, 0.25)
    hist = np.histogram(cyc.peak_trough.clip(upper=6.0), bins)[0]
    peak_hours = np.histogram(cyc.peak_hour_utc, np.arange(0, 25, 1))[0]
    col = ["appid", "name", "mean_players", "peak_trough", "peak_trough_low", "peak_trough_high", "trough_share",
           "peak_hour_utc", "idle_name"]
    showcase = cyc[cyc.appid.isin(SHOWCASE)].assign(name=lambda d: d.appid.map(SHOWCASE))[col + ["hourly"]]

    # 2. hour adjusted daily series for every game with 20+ players, for the sale and spike analyses
    _, adj = cycle.fit_all(samples, boot=0, min_mean=20)
    daily = adj.groupby(["appid", "date"], as_index=False)["adjusted"].mean()
    panel = sales.daily_panel(adj, price_day)
    eligible = set(adj.appid.unique())
    groups = sales.autumn_groups(price_day[price_day.appid.isin(eligible)])
    autumn = sales.autumn_did(panel, groups)
    solo = sales.stacked_event_study(panel, price_day, eligible)
    pd.DataFrame(autumn["by_day"]).to_csv(C.OUT / "autumn_sale_by_day.csv", index=False)
    pd.DataFrame(solo["by_day"]).to_csv(C.OUT / "discount_event_study.csv", index=False)

    # 3. spikes in established games, and whether a price change explains them
    ep = anomaly.explain(anomaly.first_spike_per_game(anomaly.spikes(daily)), price_day)
    paid = price_day.groupby("appid").agg(free=("is_free", "max"), initial=("initial", "min"))
    paid = set(paid[(paid.free == 0) & (paid.initial > 0)].index)
    ep["name"] = ep.appid.map(names)
    ep["times"] = np.exp(ep.jump)
    ep["baseline_players"] = np.expm1(ep.baseline)
    ep["paid"] = ep.appid.isin(paid)
    ep.sort_values("jump", ascending=False).to_csv(C.OUT / "spikes.csv", index=False)
    ep_paid = ep[ep.paid]
    top = ep_paid[~ep_paid.name.fillna("").str.contains(ADULT)].sort_values("jump", ascending=False).head(15)

    payload = {
        "checks": checks,
        "cycles": {"summary": weights, "validation": val, "hist_bins": bins.tolist(), "hist": hist.tolist(),
                   "peak_hours": peak_hours.tolist(),
                   "showcase": showcase.to_dict(orient="records"),
                   "flattest": cyc[~cyc.name.fillna("").str.contains(ADULT)].sort_values("peak_trough")
                   .head(15)[col].to_dict(orient="records"),
                   "median_ratio": float(cyc.peak_trough.median()),
                   "scatter": cyc.assign(name=cyc.name.fillna("").where(~cyc.name.fillna("").str.contains(ADULT), ""))
                   [["mean_players", "peak_trough", "idle_name", "name"]].to_dict(orient="records")},
        "autumn_sale": autumn,
        "discounts_outside_sale": solo,
        "spikes": {"games_screened": int(daily.appid.nunique()), "episodes": int(len(ep)),
                   "paid_episodes": int(len(ep_paid)),
                   "paid_by_cause": ep_paid.cause.value_counts().to_dict(),
                   "top": [{"name": r.name, "date": str(r.date), "baseline": float(r.baseline_players),
                            "times": float(r.times), "cause": r.cause} for r in top.itertuples()]},
    }
    (C.DOCS / "results.json").write_text(json.dumps(payload, indent=1, default=float))
    print(json.dumps({"cycles": weights, "validation": val}, indent=1, default=float))
    print(json.dumps({k: autumn[k] for k in ["treated_games", "control_games", "periods", "mean_abs_pretrend"]},
                     indent=1, default=float))
    print(json.dumps({k: solo[k] for k in ["events", "games", "median_discount", "first_week", "second_week"]},
                     indent=1, default=float))
    print(json.dumps(payload["spikes"]["paid_by_cause"], indent=1))


if __name__ == "__main__":
    main()
