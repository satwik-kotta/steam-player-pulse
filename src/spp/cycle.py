"""The daily player cycle of each game, and how flat it is.

Each game is sampled about once a day, and the collector rotates the hour it samples at, so over three
months every game is seen at most hours of the day. For each game we fit

    log(players + 1) = smooth trend over the days  +  two harmonics of the UTC hour

The harmonic part is the game's average day. Real players sleep, so a normal game's busiest hour has about
twice the players of its quietest. Games that keep running while nobody is at the keyboard (idle games,
item-drop farming, bots) have a flat day. Flatness is the fitted peak / trough ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

HARMONICS = 2
TREND_KNOTS = 6
IDLE_WORDS = r"\b(?:idle|incremental|clicker|afk|desktop|desk)\b"


def _design(day: np.ndarray, hour: np.ndarray, day_range=(0.0, 100.0)):
    """Columns: piecewise linear trend over the days (hinge basis), then cos/sin of the hour."""
    lo, hi = day_range
    knots = np.linspace(lo, hi, TREND_KNOTS + 2)[1:-1]
    cols = [np.ones_like(day), (day - lo) / (hi - lo)]
    cols += [np.clip(day - k, 0, None) / (hi - lo) for k in knots]
    for k in range(1, HARMONICS + 1):
        w = 2 * np.pi * k * hour / 24
        cols += [np.cos(w), np.sin(w)]
    return np.column_stack(cols), 2 + len(knots)


def cycle_curve(coef_h: np.ndarray, hours=np.arange(0, 24, 0.25)) -> np.ndarray:
    """Hour effect (log scale) at the given hours, from the harmonic coefficients."""
    out = np.zeros_like(hours, dtype=float)
    for k in range(1, HARMONICS + 1):
        a, b = coef_h[2 * (k - 1)], coef_h[2 * (k - 1) + 1]
        w = 2 * np.pi * k * hours / 24
        out += a * np.cos(w) + b * np.sin(w)
    return out


def fit_one(day, hour, y, day_range=(0.0, 100.0), ridge=1e-3):
    """Fit one game. Returns (coefficients, number of trend columns, residual sd)."""
    X, nt = _design(np.asarray(day, float), np.asarray(hour, float), day_range)
    z = np.log1p(np.asarray(y, float))
    pen = ridge * np.eye(X.shape[1])
    pen[0, 0] = 0
    coef = np.linalg.solve(X.T @ X + pen, X.T @ z)
    resid = z - X @ coef
    return coef, nt, float(resid.std(ddof=X.shape[1]) if len(z) > X.shape[1] else np.nan)


def summarise(coef_h: np.ndarray) -> dict:
    hours = np.arange(0, 24, 0.25)
    c = cycle_curve(coef_h, hours)
    level = np.exp(c)
    return {"peak_trough": float(np.exp(c.max() - c.min())),
            "trough_share": float(level.min() / level.mean()),
            "peak_hour_utc": float(hours[c.argmax()]),
            "trough_hour_utc": float(hours[c.argmin()])}


def fit_all(samples: pd.DataFrame, boot: int = 200, seed: int = 7,
            min_mean: float = C.MIN_MEAN_PLAYERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """samples: appid, ts (UTC), players. Returns per game summary, and per sample hour adjusted values."""
    s = samples.dropna(subset=["players"]).copy()
    t0 = s["ts"].min()
    s["day"] = (s["ts"] - t0).dt.total_seconds() / 86400
    s["hour"] = s["ts"].dt.hour + s["ts"].dt.minute / 60
    span = (0.0, float(s["day"].max()) + 1e-6)
    stats = s.groupby("appid")["players"].agg(n="size", mean_players="mean")
    keep = stats[(stats.n >= C.MIN_SAMPLES) & (stats.mean_players >= min_mean)].index
    rng = np.random.default_rng(seed)
    rows, adj = [], []
    for appid, g in s[s.appid.isin(keep)].groupby("appid"):
        day, hour, y = g["day"].values, g["hour"].values, g["players"].values
        coef, nt, sd = fit_one(day, hour, y, span)
        summ = summarise(coef[nt:])
        hours_seen = len(np.unique(np.floor(hour)))
        ratios = []
        for _ in range(boot):
            i = rng.integers(0, len(g), len(g))
            cb, ntb, _ = fit_one(day[i], hour[i], y[i], span)
            ratios.append(summarise(cb[ntb:])["peak_trough"])
        hourly = np.exp(cycle_curve(coef[nt:], np.arange(24.0)))
        rows.append({"appid": appid, "samples": len(g), "hours_seen": hours_seen,
                     "hourly": (hourly / hourly.mean()).round(4).tolist(),
                     "mean_players": float(y.mean()), "resid_sd": sd, **summ,
                     "peak_trough_low": float(np.quantile(ratios, 0.025)) if ratios else np.nan,
                     "peak_trough_high": float(np.quantile(ratios, 0.975)) if ratios else np.nan})
        h_eff = cycle_curve(coef[nt:], hour)
        adj.append(pd.DataFrame({"appid": appid, "date": g["ts"].dt.date.values,
                                 "log_players": np.log1p(y), "adjusted": np.log1p(y) - h_eff}))
    if not rows:
        return pd.DataFrame(), pd.DataFrame(columns=["appid", "date", "log_players", "adjusted"])
    return pd.DataFrame(rows), pd.concat(adj, ignore_index=True)


def validate(cycles: pd.DataFrame, names: pd.Series) -> dict:
    """Weak labels: games whose name says they run in the background (idle, clicker, AFK, desktop)."""
    from sklearn.metrics import roc_auc_score

    c = cycles.assign(name=cycles.appid.map(names)).dropna(subset=["name"])
    c["idle_name"] = c.name.str.contains(IDLE_WORDS, case=False, regex=True)
    c = c.sort_values("peak_trough")
    base = float(c.idle_name.mean())
    at_k = {k: float(c.head(k).idle_name.mean()) for k in (25, 50, 100, 200)}
    return {"games": int(len(c)), "idle_named": int(c.idle_name.sum()), "base_rate": base,
            "precision_at_k": at_k, "lift_at_25": at_k[25] / base,
            "auc_flatness_vs_idle_name": float(roc_auc_score(c.idle_name, -c.peak_trough)),
            "median_ratio_idle": float(c[c.idle_name].peak_trough.median()),
            "median_ratio_other": float(c[~c.idle_name].peak_trough.median())}
