"""Do discounts bring players in, and do they stay?

Outcome: each game's daily log player count after removing its own hour of day effect (see cycle.py),
so a sample taken at 4 am is comparable with one taken at 8 pm.

1. Steam Autumn Sale (Oct 1 to 8, 2026): difference in differences. Games discounted in the sale against
   paid games that stayed at full price, with game and calendar day fixed effects.
2. Discounts outside the sale: a stacked event study. Each discount start is compared with games that had
   no discount in the surrounding weeks, on the same calendar days.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyfixest as pf

from . import config as C

PRE_START = date(2026, 9, 17)
POST_END = date(2026, 10, 22)


def daily_panel(adj: pd.DataFrame, price_day: pd.DataFrame) -> pd.DataFrame:
    d = adj.groupby(["appid", "date"], as_index=False)["adjusted"].mean()
    p = price_day.rename(columns={"day": "date"})
    return d.merge(p[["appid", "date", "discount_pct", "is_free", "initial"]], on=["appid", "date"], how="left")


def _period(dt: date) -> str | None:
    if PRE_START <= dt <= C.SALE_START - timedelta(days=1):
        return "pre"
    if C.SALE_START + timedelta(days=1) <= dt <= C.SALE_END - timedelta(days=1):
        return "sale"            # Oct 1 and Oct 8 are part day: the sale starts and ends at 17:00 UTC
    if C.SALE_END + timedelta(days=1) <= dt <= C.SALE_END + timedelta(days=7):
        return "week_after"
    if C.SALE_END + timedelta(days=8) <= dt <= POST_END:
        return "two_weeks_after"
    return None


def autumn_groups(price_day: pd.DataFrame) -> pd.DataFrame:
    """Treated: full price for the two weeks before, discounted during the sale, full price again by Oct 9.
    Control: paid games at full price on every day of the window."""
    p = price_day[(price_day.day >= PRE_START) & (price_day.day <= POST_END)].copy()
    p["seg"] = p.day.map(_period)
    days_needed = (POST_END - PRE_START).days - 2   # allow a missed collection run or two
    g = p.groupby("appid").agg(days=("day", "nunique"), free=("is_free", "max"), initial=("initial", "min"),
                               first=("day", "min"))
    seg = p.pivot_table(index="appid", columns="seg", values="discount_pct", aggfunc="max")
    sale_disc = p[p.day == C.SALE_START + timedelta(days=2)].set_index("appid")["discount_pct"]
    after = p[(p.day >= C.SALE_END + timedelta(days=1)) & (p.day <= C.SALE_END + timedelta(days=1))] \
        .set_index("appid")["discount_pct"]
    ok = g[(g.days >= days_needed) & (g.free == 0) & (g.initial > 0) & (g["first"] <= PRE_START)].index
    seg = seg.reindex(ok)
    treated = seg[(seg["pre"] == 0) & (sale_disc.reindex(ok) > 0) & (after.reindex(ok) == 0)].index
    control = seg[(seg[["pre", "sale", "week_after", "two_weeks_after"]].max(axis=1) == 0)].index
    out = pd.DataFrame({"appid": list(treated) + list(control),
                        "treated": [1] * len(treated) + [0] * len(control)})
    out["discount"] = out.appid.map(sale_disc).fillna(0)
    return out


def _lift(b: float) -> float:
    return float(np.exp(b) - 1)


def autumn_did(panel: pd.DataFrame, groups: pd.DataFrame) -> dict:
    d = panel.merge(groups, on="appid")
    d = d[(d.date >= PRE_START) & (d.date <= POST_END)].copy()
    d["period"] = d.date.map(_period)
    d = d.dropna(subset=["period"])
    d["date_s"] = d.date.astype(str)
    fit = pf.feols("adjusted ~ i(period, treated, ref='pre') | appid + date_s", data=d, vcov={"CRV1": "appid"})
    co, se = fit.coef(), fit.se()
    res = {}
    for k in ["sale", "week_after", "two_weeks_after"]:
        name = [c for c in co.index if k in c and (k != "sale" or "after" not in c)][0]
        b, s = co[name], se[name]
        res[k] = {"lift": _lift(b), "low": _lift(b - 1.96 * s), "high": _lift(b + 1.96 * s),
                  "p": float(fit.pvalue()[name])}

    # day by day event study, relative to the last full day before the sale
    d["rel"] = (pd.to_datetime(d.date) - pd.Timestamp(C.SALE_START)).dt.days
    ev = pf.feols("adjusted ~ i(rel, treated, ref=-1) | appid + date_s", data=d, vcov={"CRV1": "appid"})
    ec, es = ev.coef(), ev.se()
    days = []
    for name in ec.index:
        r = int(name.split("::")[1].split(":")[0])
        days.append({"day": r, "lift": _lift(ec[name]), "low": _lift(ec[name] - 1.96 * es[name]),
                     "high": _lift(ec[name] + 1.96 * es[name])})
    days.sort(key=lambda x: x["day"])
    pre = [x for x in days if x["day"] < -1]
    pretrend = float(np.mean([abs(x["lift"]) for x in pre])) if pre else None

    # depth of discount during the sale
    t = d[d.period.isin(["pre", "sale"])].copy()
    t["band"] = pd.cut(t.discount, [-1, 0, 33, 66, 100], labels=["none", "up to 33%", "34 to 66%", "67% or more"])
    t["sale_x_band"] = np.where(t.period == "sale", t.band.astype(str), "none")
    dose_fit = pf.feols("adjusted ~ i(sale_x_band, ref='none') | appid + date_s", data=t, vcov={"CRV1": "appid"})
    dc, ds = dose_fit.coef(), dose_fit.se()
    dose = []
    band_of = pd.cut(groups.discount, [-1, 0, 33, 66, 100], labels=["none", "up to 33%", "34 to 66%", "67% or more"])
    for band in ["up to 33%", "34 to 66%", "67% or more"]:
        found = [c for c in dc.index if band in c]
        if not found:
            continue
        name = found[0]
        dose.append({"band": band, "games": int(((groups.treated == 1) & (band_of == band)).sum()),
                     "lift": _lift(dc[name]), "low": _lift(dc[name] - 1.96 * ds[name]),
                     "high": _lift(dc[name] + 1.96 * ds[name])})
    return {"treated_games": int(groups.treated.sum()), "control_games": int((groups.treated == 0).sum()),
            "rows": int(len(d)), "periods": res, "by_day": days, "mean_abs_pretrend": pretrend,
            "by_depth": dose, "median_discount": float(groups[groups.treated == 1].discount.median())}


def discount_starts(price_day: pd.DataFrame, eligible: set) -> pd.DataFrame:
    """First day of a discount after at least 14 days at full price, away from the Autumn Sale window."""
    p = price_day[price_day.appid.isin(eligible)].sort_values(["appid", "day"]).copy()
    p["on"] = p.discount_pct > 0
    p["prev_on_14"] = p.groupby("appid")["on"].transform(lambda s: s.shift(1).rolling(14, min_periods=10).max())
    starts = p[p.on & (p.prev_on_14 == 0)][["appid", "day", "discount_pct"]]
    last = price_day.day.max()
    safe = starts.day.map(lambda d: (d + timedelta(days=14) < C.SALE_START) or
                          (d - timedelta(days=14) > C.SALE_END and d + timedelta(days=14) <= last))
    return starts[safe]


def stacked_event_study(panel: pd.DataFrame, price_day: pd.DataFrame, eligible: set,
                        max_controls: int = 400, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    starts = discount_starts(price_day, eligible)
    on = price_day[price_day.discount_pct > 0][["appid", "day"]]
    paid = price_day.groupby("appid").agg(free=("is_free", "max"), initial=("initial", "min"))
    paid = set(paid[(paid.free == 0) & (paid.initial > 0)].index) & eligible
    stacks = []
    for i, (d0, ev) in enumerate(starts.groupby("day")):
        lo, hi = d0 - timedelta(days=14), d0 + timedelta(days=21)
        busy = set(on[(on.day >= lo) & (on.day <= hi)].appid)
        pool = np.array(sorted(paid - busy))
        if len(pool) == 0:
            continue
        ctrl = rng.choice(pool, min(max_controls, len(pool)), replace=False)
        ids = pd.DataFrame({"appid": list(ev.appid) + list(ctrl), "treat": [1] * len(ev) + [0] * len(ctrl)})
        w = panel[(panel.date >= d0 - timedelta(days=7)) & (panel.date <= d0 + timedelta(days=14))]
        w = w.merge(ids, on="appid")
        w["rel"] = (pd.to_datetime(w.date) - pd.Timestamp(d0)).dt.days
        w["stack"] = i
        stacks.append(w)
    s = pd.concat(stacks, ignore_index=True)
    s["sg"] = s["stack"].astype(str) + "_" + s["appid"].astype(str)
    s["sd"] = s["stack"].astype(str) + "_" + s["date"].astype(str)
    fit = pf.feols("adjusted ~ i(rel, treat, ref=-1) | sg + sd", data=s, vcov={"CRV1": "appid"})
    co, se = fit.coef(), fit.se()
    days = []
    for name in co.index:
        r = int(name.split("::")[1].split(":")[0])
        days.append({"day": r, "lift": _lift(co[name]), "low": _lift(co[name] - 1.96 * se[name]),
                     "high": _lift(co[name] + 1.96 * se[name])})
    days.sort(key=lambda x: x["day"])
    s["w1"] = ((s.rel >= 0) & (s.rel <= 6) & (s.treat == 1)).astype(int)
    s["w2"] = ((s.rel >= 7) & (s.rel <= 14) & (s.treat == 1)).astype(int)
    avg = pf.feols("adjusted ~ w1 + w2 | sg + sd", data=s[s.rel != -1], vcov={"CRV1": "appid"})
    out = {}
    for k in ["w1", "w2"]:
        b, sd_ = avg.coef()[k], avg.se()[k]
        out[k] = {"lift": _lift(b), "low": _lift(b - 1.96 * sd_), "high": _lift(b + 1.96 * sd_)}
    return {"events": int(len(starts)), "games": int(starts.appid.nunique()), "start_days": int(starts.day.nunique()),
            "median_discount": float(starts.discount_pct.median()), "by_day": days,
            "first_week": out["w1"], "second_week": out["w2"]}
