"""Find days when a game's players jumped far above its own recent level, and check which a discount explains.

For each game, on its hour adjusted daily series (cycle.py):
  baseline   median of the previous 14 observed days
  jump       today minus baseline, on the log scale, minus the median jump across all games that day
             (so store-wide days, such as the start of the Autumn Sale, do not count as spikes)
  spike      players at least doubled against the baseline (jump >= log 2),
             the jump is at least 5 robust standard deviations of the game's own day to day noise,
             and the baseline had at least 50 players (so launches and playtests from zero are left out)
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C

WINDOW = 14
MIN_HISTORY = 7
Z = 5.0
MIN_BASELINE_PLAYERS = 50   # below this a jump is usually a launch or a playtest, not a change in an existing game


def jumps(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["appid", "date"]).copy()
    g = d.groupby("appid")["adjusted"]
    d["baseline"] = g.transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=MIN_HISTORY).median())
    d["raw_jump"] = d.adjusted - d.baseline
    d["jump"] = d.raw_jump - d.groupby("date")["raw_jump"].transform("median")
    diff = d.groupby("appid")["adjusted"].diff()
    mad = diff.abs().groupby(d.appid).transform("median") * 1.4826
    d["z"] = d.jump / mad.replace(0, np.nan)
    return d


def spikes(daily: pd.DataFrame, z: float = Z) -> pd.DataFrame:
    d = jumps(daily)
    base_players = np.expm1(d.baseline)
    return d[(d.jump >= np.log(2)) & (d.z >= z) & (base_players >= MIN_BASELINE_PLAYERS)].copy()


def explain(sp: pd.DataFrame, price_day: pd.DataFrame) -> pd.DataFrame:
    """Label each spike: a discount that started in the 3 days before it, the Autumn Sale, or neither."""
    p = price_day[["appid", "day", "discount_pct"]].sort_values(["appid", "day"]).copy()
    p["started"] = (p.discount_pct > 0) & (p.groupby("appid").discount_pct.shift(1).fillna(0) == 0)
    starts = p[p.started][["appid", "day"]]
    sp = sp.copy()
    sp["cause"] = "no price change"
    m = sp[["appid", "date"]].merge(starts, on="appid")
    near = m[(m.day <= m.date) & (m.day >= m.date - timedelta(days=3))]
    key = set(zip(near.appid, near.date))
    in_sale = sp.date.between(C.SALE_START, C.SALE_END).values
    disc = sp.merge(price_day[["appid", "day", "discount_pct"]].rename(columns={"day": "date"}),
                    on=["appid", "date"], how="left")["discount_pct"].fillna(0).values > 0
    started = np.array([k in key for k in zip(sp.appid, sp.date)])
    sp.loc[in_sale & disc, "cause"] = "Autumn Sale discount"
    sp.loc[started & ~(in_sale & disc), "cause"] = "discount started"
    return sp


def first_spike_per_game(sp: pd.DataFrame) -> pd.DataFrame:
    """A run of spike days in one game counts once, at its biggest day."""
    sp = sp.sort_values(["appid", "date"]).copy()
    gap = pd.to_datetime(sp.date).groupby(sp.appid).diff().dt.days
    sp["episode"] = (gap.isna() | (gap > 3)).cumsum()
    return sp.loc[sp.groupby("episode").jump.idxmax()]
