[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MYusuf0703/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/MYusuf0703/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of
Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z, as
five quantiles per trophy. Entry for
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on make.fun.

## Run

```sh
python3 main.py   # reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Python 3.13, standard library only, no network, deterministic (two runs on
the same case write byte-identical output), well under the 8 s per-case cap.

## Model (`main.py`)

Rates are counts: rate = earners / players tracked. Two regimes, chosen by
whether the case history moves:

- **Flat history** (the practice cases: every read identical on launch
  night). No trend can be fitted, and Exophase refreshes its counts every few
  hours, so the target probably falls after a refresh: the tracked population
  jumps while the new players have barely earned anything beyond the one
  trophy everybody holds, so most rates fall. The forecast is a two-mode
  mixture on the logit scale: with probability (1 - p) no refresh lands in
  the horizon (rates stay), with probability p one does (rates are diluted
  to (E + dE) / (N * G)). The low quantiles sit at the diluted level, the
  median and above near the last rate.
- **Trending history** (the final case). Per trophy, a concave quadratic of
  logit(rate) on ln(1 + hours since release) over the last 24 h of reads,
  frozen at its vertex so it saturates instead of turning down; a convex fit
  or too few points falls back to a shrunk linear extrapolation. The spread
  is calibrated by backtesting the same one-step-ahead forecast inside the
  history, floored below the house baseline's random-walk spread and capped.

## Local verification (no local grade in the kit; the server grades)

- Practice cases p00/p01, scored against answers reconstructed from the
  aggregates the brief publishes (04:30Z read: 3,868 players, 4,276 trophies
  earned; universal trophy stayed 100%): **loss ~0.60** (mean over both
  cases, three allocation variants 0.59–0.63) vs the house persistence
  baseline **~1.41** (published ~1.51) — about **57% better**. The
  reconstruction reproduces the baseline's published typical miss (0.68 pp)
  and 90% coverage (0.00) exactly, so it is consistent with the house's
  numbers. The true server practice score is only known after submission.
- Synthetic trending final-like case (2.5 days of reads, 4.5-day horizon):
  **loss 0.161** vs persistence baseline **0.243** (34% better),
  typical miss 0.99 pp vs 3.09 pp.
- Two runs on the same case are byte-identical (sha256-verified); output is
  valid (55 rows, ids 1–55, quantiles non-decreasing, 0–100, 3,418 bytes);
  runtime ~0.5 s per case.
