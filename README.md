# Steam Player Pulse

Bots and idle play, sale lift, and sudden spikes in **1.9 million Steam player count samples** for 35,244 games, joined to store prices in 8 regions, August to November 2026.

**[Dashboard](https://satwik-kotta.github.io/steam-player-pulse/)** · **[One page memo](https://satwik-kotta.github.io/steam-player-pulse/memo.html)**

## Findings

**1. A flat day is a fingerprint of play without a person.** Each game is sampled about once a day at a rotating hour, so over three months every hour of its day is seen. Fitting each game's daily cycle shows real players sleep: the typical game has 2.2x more players at its busiest hour than its quietest.

| Game | Busiest / quietest hour | Players at the quietest hour |
|---|---|---|
| PUBG: BATTLEGROUNDS | 5.8x | 33% of average |
| Counter-Strike 2 | 2.4x | 61% |
| Bongo Cat | 1.36x | 85% |
| Banana (known for item farming) | 1.37x | 85% |
| MIR4 | 1.13x | 94% |

Of the 25 flattest of 1,807 established games, **48% are idle games by name, against 2.7% overall (18x)**, AUC 0.85. Player counts drive store rankings, so this is a cheap first screen for inflated numbers. It flags play without a person, not cheating.

**2. The Steam Autumn Sale lifted players 28%, and most of it stayed.** Difference in differences with game and calendar day fixed effects, 1,559 discounted games against 250 paid games at full price:

| Period | Lift in players (95% CI) |
|---|---|
| During the sale, Oct 2 to 7 | +28% (25 to 32%) |
| Week after | +21% (16 to 25%) |
| Two weeks after | +17% (12 to 22%) |

Flat before the sale (mean absolute pre trend 1.5%). Deeper discounts lifted more: +25% for up to 33% off, +31% for 67% or more.

**3. A discount on its own did about twice as much.** A stacked event study of 3,111 discounts outside the sale (median 65% off), each against games with no discount in the surrounding five weeks on the same dates: **+56% (52 to 60%) in the first week**. During the sale 81% of tracked games were discounted at once.

**4. 55% of player spikes in paid games came with no price change.** 275 spikes (players at least doubled against the last two weeks, beyond five robust standard deviations, store wide days removed). The rest followed a discount start (36%) or the Autumn Sale (9%). Updates, seasons and free weekends move players more often than price.

## Data checks

- Every UTC hour is sampled, from 32,077 samples at 4:00 to 117,310 at 20:00.
- 5 samples where the API declined to answer stay missing, separate from 564,816 real zeros.
- No duplicate samples in either source; one currency per region.
- 79 of 4.39M US price rows have a discount that disagrees with the prices by more than a point.
- The price watchlist grew from 6,386 to 31,550 games, so late additions are left out of comparisons that need earlier prices.
- 1,116 games have no name in the app list.

## Method

| Step | File |
|---|---|
| Load the two git scraped sources and the app list into DuckDB, run the checks | `src/spp/ingest.py` |
| Daily cycle per game: piecewise linear trend plus two harmonics of the UTC hour, bootstrap intervals, hour adjusted series | `src/spp/cycle.py` |
| Autumn Sale difference in differences and day by day event study, discount depth, stacked event study for other discounts (pyfixest) | `src/spp/sales.py` |
| Spike detection with a rolling baseline and robust z score, matched to price changes | `src/spp/anomaly.py` |
| Tests on synthetic data with known cycles, lifts and spikes | `tests/` |

## Run it

```bash
pip install -r requirements.txt
bash scripts/get_data.sh       # clones the sources at pinned commits into data/src (about 1 GB)
cd src
python -m spp.ingest           # data/panel.duckdb and outputs/ingest.json
python -m spp.run              # docs/data/results.json and outputs/*.csv
cd .. && pytest -q
```

## Data

- [scos-lab/steamcozy-ccu](https://github.com/scos-lab/steamcozy-ccu): concurrent players from Steam's public API, collected by GitHub Actions, commit `328642e`.
- [scos-lab/steamcozy-prices](https://github.com/scos-lab/steamcozy-prices): Steam store prices in 8 regions, commit `ba29457`.
- [jsnli/steamappidlist](https://github.com/jsnli/steamappidlist): Steam app id to name, commit `cd44cea`.

Not affiliated with Valve. Adult titles are left off the public lists but stay in every calculation.

## Limits

Three months of data. Games choose whether to join a sale, so the comparisons control for each game's level and each calendar day, not for why a game was discounted. Lift is in concurrent players, not revenue. Idle names are a weak label.
