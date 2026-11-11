"""Load the three git-scraped sources into one DuckDB file and check them.

    python -m spp.ingest

Tables written to data/panel.duckdb:
  ccu          one row per player-count sample (ts, appid, players)   empty answers kept as NULL
  price_runs   one row per US price sample (run, appid, initial, final, discount_pct, is_free)
  price_day    one row per game per day: the last US price of the day
  games        app id to name
Checks go to outputs/ingest.json.
"""
from __future__ import annotations

import json

import duckdb

from . import config as C


def connect(path=None):
    con = duckdb.connect(str(path or C.DB))
    con.execute("set TimeZone = 'UTC'")
    return con


def load(con, ccu_glob=C.CCU_GLOB, price_glob=C.PRICE_GLOB, names=C.NAMES):
    con.execute(f"""
        create or replace table ccu as
        select to_timestamp(ts) as ts, appid, concurrent as players,
               regexp_extract(filename, '([^/]+)\\.csv$', 1) as run
        from read_csv('{ccu_glob}', filename = true, header = true,
                      columns = {{'ts': 'BIGINT', 'appid': 'BIGINT', 'concurrent': 'BIGINT',
                                  'peak_in_game': 'BIGINT', 'src': 'VARCHAR'}})
    """)
    # The price files changed shape once: before 2026-07-11 12:00 UTC they were US only, without a cc column.
    con.execute(f"""
        create or replace table price_raw as
        select regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}}_\\d{{4}})', 1) as run,
               appid, coalesce(cc, 'us') as cc, currency, initial, final, discount_pct, is_free
        from read_csv('{price_glob}', filename = true, header = true, union_by_name = true)
    """)
    con.execute("""
        create or replace table price_runs as
        select strptime(run, '%Y-%m-%d_%H%M') as run_ts, appid, initial, final, discount_pct, is_free
        from price_raw where cc = 'us'
    """)
    con.execute("""
        create or replace table price_day as
        select appid, cast(run_ts as date) as day,
               arg_max(discount_pct, run_ts) as discount_pct,
               arg_max(final, run_ts) as final, arg_max(initial, run_ts) as initial,
               arg_max(is_free, run_ts) as is_free, count(*) as samples
        from price_runs group by 1, 2
    """)
    con.execute(f"""
        create or replace table games as
        select appid, name from read_json('{names}', format = 'array')
    """)


def checks(con) -> dict:
    one = lambda q: con.execute(q).fetchone()
    rows, games, declined, zeros, t0, t1 = one("""
        select count(*), count(distinct appid), count(*) filter (where players is null),
               count(*) filter (where players = 0), min(ts), max(ts) from ccu""")
    dup = one("select count(*) - count(distinct (ts, appid)) from ccu")[0]
    hours = con.execute("select hour(ts) h, count(*) from ccu group by 1 order by 1").fetchall()
    per_game = one("""select median(n), max(n) from (select appid, count(*) n from ccu group by 1)""")
    runs = one("select count(distinct run) from ccu")[0]
    unnamed = one("select count(distinct c.appid) from ccu c anti join games g using (appid)")[0]

    p_rows, p_games, p_runs = one("select count(*), count(distinct appid), count(distinct run_ts) from price_runs")
    regions = [r[0] for r in con.execute("select distinct cc from price_raw order by 1").fetchall()]
    mixed_ccy = one("""select count(*) from (select cc from price_raw where currency is not null
                       group by cc having count(distinct currency) > 1)""")[0]
    bad_pct = one("""
        select count(*) from price_runs
        where initial > 0 and abs(discount_pct - round(100.0 * (1 - final / initial))) > 1""")[0]
    dup_p = one("select count(*) - count(distinct (run_ts, appid)) from price_runs")[0]
    size_by_day = con.execute("""
        select cast(run_ts as date) d, max(n) from (select run_ts, count(*) n from price_runs group by 1)
        group by 1 order by 1""").fetchall()
    # the price watchlist grew in steps; record each change so a reader knows which games have short histories
    growth = [(str(d), int(n)) for i, (d, n) in enumerate(size_by_day) if i == 0 or n != size_by_day[i - 1][1]]
    disc_share = con.execute("""
        select day, avg(case when discount_pct > 0 then 1.0 else 0.0 end), count(*)
        from price_day where day between date '2026-09-24' and date '2026-10-15' group by 1 order by 1""").fetchall()
    return {
        "sources": C.SOURCES,
        "ccu": {"rows": rows, "games": games, "collection_runs": runs, "first": str(t0), "last": str(t1),
                "declined_empty": declined, "zero_players": zeros, "duplicate_samples": dup,
                "samples_per_game_median": float(per_game[0]), "samples_per_game_max": per_game[1],
                "samples_by_utc_hour": {int(h): n for h, n in hours},
                "games_without_a_name": unnamed},
        "prices": {"rows_us": p_rows, "games": p_games, "runs": p_runs, "regions": regions,
                   "regions_with_mixed_currency": mixed_ccy, "discount_pct_disagrees_with_prices": bad_pct,
                   "duplicate_samples": dup_p, "watchlist_size_changes": growth,
                   "share_discounted_by_day": {str(d): round(s, 4) for d, s, _ in disc_share}},
    }


def main():
    C.OUT.mkdir(exist_ok=True)
    C.DB.parent.mkdir(parents=True, exist_ok=True)
    con = connect()
    load(con)
    res = checks(con)
    (C.OUT / "ingest.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res["ccu"].items() if k != "samples_by_utc_hour"}, indent=1, default=str))
    print(json.dumps({k: v for k, v in res["prices"].items() if k != "share_discounted_by_day"}, indent=1))


if __name__ == "__main__":
    main()
