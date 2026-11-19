#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case per *.in file (Exophase earn rates for every GTA VI PS5 trophy
as the house recorded them before the case checkpoint) and writes, for each
trophy, five quantiles of its earn rate at the case target, as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.

Stdlib only, no network, deterministic: two runs on the same case write
byte-identical output.

Model
-----
Rates are counts: rate_i = earners_i / players_tracked. Two regimes:

* Flat history (the practice cases: every read identical, 638 players, rates
  frozen on launch night). No trend can be fitted, and Exophase refreshes its
  counts every few hours, so the target probably falls after a refresh: the
  tracked population jumps while the new players have barely earned anything
  beyond the one trophy everybody holds, so most rates fall. The forecast is a
  two-mode mixture on the logit scale: with probability (1 - p) no refresh
  lands in the horizon (rates stay), with probability p one does (rates are
  diluted to (E + dE) / (N * G), where G is the population jump at a refresh
  and new players earn each trophy at a fraction kappa of the launch cohort's
  rate, except the trophy everybody holds, which stays). Quantiles are the
  mixture's quantiles, so the low quantiles sit at the diluted level and the
  median and above stay near the last rate.

* Trending history (the final case: two days of reads with real movement).
  Per trophy, a least-squares line of logit(rate) on ln(1 + hours since
  release) over the last 24 hours of reads, extrapolated to the target with
  the slope shrunk (growth in log-time saturates). The spread is calibrated
  by backtesting inside the history: the same one-step-ahead forecast is
  scored against later reads, and the spread is the backtest residual scale,
  floored at the house baseline's random-walk spread and capped.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Exophase refresh cadence on launch night: the page sat frozen ~4-5 h, then
# refreshed once. Probability a refresh lands in a horizon of h hours.
REFRESH_INTERVAL_H = 4.5
# Population jump at a refresh, early after launch (first refresh: 638 -> 3868
# players tracked), decaying as the launch surge passes.
REFRESH_JUMP_0 = 5.0
REFRESH_JUMP_FLOOR = 1.5
# New players' earn rate for a trophy, as a fraction of the launch cohort's
# rate (first refresh: new players earned non-universal trophies at ~7% the
# launch cohort's per-player rate), rising as the game ages.
KAPPA_0 = 0.067
# Slope shrinkage when extrapolating a fitted trend past the observed window
# (linear fallback only; the quadratic fit saturates on its own).
SLOPE_SHRINK = 0.6
# Jitter (logit sd) around each mode of the flat-history mixture.
MIXTURE_JITTER = 0.35
# Spread floor/cap (logit sd) on the trending path. The floor is below the
# house baseline's 0.5/sqrt(day) random walk: a rate that follows a smooth
# saturating curve is more predictable than a random walk, and the backtest
# residual scale lifts the spread when the history is actually noisy.
SPREAD_FLOOR_SIGMA = 0.35   # per sqrt(day)
SPREAD_CAP = 2.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    l = max(-20.0, min(20.0, l))
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    v = round(x, 4)
    return 0.0 if v == 0 else v


def parse_time(s):
    """ISO 8601 (with Z, or a plain date) -> aware datetime (UTC)."""
    if s is None:
        return None
    s = str(s).strip()
    if len(s) == 10:
        s += "T00:00:00Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(lo_a, lo_b, p_b, tau, sd):
    """Quantile tau of the two-mode mixture (1-p_b) * N(lo_a, sd) + p_b * N(lo_b, sd).

    lo_a is the no-refresh mode (higher), lo_b the diluted mode (lower).
    Inverted by bisection on the mixture CDF; deterministic.
    """
    lo, hi = min(lo_a, lo_b) - 8.0 * sd - 1.0, max(lo_a, lo_b) + 8.0 * sd + 1.0

    def cdf(x):
        return ((1.0 - p_b) * norm_cdf((x - lo_a) / sd)
                + p_b * norm_cdf((x - lo_b) / sd))

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def least_squares(xs, ys):
    """Least-squares line y = a + b*x. Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def fit_quadratic(xs, ys):
    """Least-squares y = a + b*x + c*x^2 via the normal equations (3x3 solve).

    Returns (a, b, c), or None when the system is degenerate.
    """
    n = len(xs)
    s0 = float(n)
    s1 = sum(xs)
    s2 = sum(x * x for x in xs)
    s3 = sum(x ** 3 for x in xs)
    s4 = sum(x ** 4 for x in xs)
    t0 = sum(ys)
    t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum(x * x * y for x, y in zip(xs, ys))
    m = [[s0, s1, s2, t0], [s1, s2, s3, t1], [s2, s3, s4, t2]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for cc in range(col, 4):
                m[r][cc] -= f * m[col][cc]
    out = [0.0, 0.0, 0.0]
    for r in (2, 1, 0):
        out[r] = (m[r][3] - sum(m[r][cc] * out[cc] for cc in range(r + 1, 3))) / m[r][r]
    return out[0], out[1], out[2]


def extrapolate(pts, x_target):
    """Predict logit(rate) at x_target from (x, logit rate) points.

    A concave quadratic in log-time (rates saturate), frozen at its vertex so
    it never turns down; a convex fit or too few points falls back to a
    shrunk linear extrapolation; a degenerate fit falls back to the last
    level. Returns the predicted logit.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_max = max(xs)
    if len(pts) >= 4 and x_max - min(xs) > 1e-9:
        q = fit_quadratic(xs, ys)
        if q is not None:
            a, b, c = q
            if c <= 0.0:
                if c < -1e-12:
                    vertex = -b / (2.0 * c)
                    x_hat = min(x_target, vertex)
                else:
                    x_hat = x_target
                return a + b * x_hat + c * x_hat * x_hat
    if len(pts) >= 2 and x_max - min(xs) > 1e-9:
        a, b = least_squares(xs, ys)
        return a + b * x_max + SLOPE_SHRINK * b * (x_target - x_max)
    return ys[-1]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0

    release = parse_time(case.get("game", {}).get("release_date"))
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    if release is None:
        release = parse_time(history[0]["at"]) if history else at
    if at is None:
        at = parse_time(history[-1]["at"]) if history else target
    if target is None:
        target = at

    def hours_since_release(dt):
        return max(0.0, (dt - release).total_seconds() / 3600.0)

    # Per-read series: (hours since release, players tracked, rates list).
    reads = []
    for h in history:
        t = parse_time(h.get("at")) or parse_time(h.get("slot"))
        if t is None:
            continue
        reads.append((hours_since_release(t), h.get("players_tracked"),
                      h.get("rates") or []))
    reads.sort(key=lambda r: r[0])

    # Last non-null rate per trophy, and the last read overall.
    n_tr = len(trophies)
    last_rate = [None] * n_tr
    last_n = None
    for _, n, rates in reads:
        if n is not None:
            last_n = n
        for i in range(min(n_tr, len(rates))):
            if rates[i] is not None:
                last_rate[i] = rates[i]
    if last_n is None:
        last_n = 638

    # --- Flat-history test: neither the player count nor (almost) any rate
    # moved across the reads. Only the practice cases look like this.
    flat = False
    if len(reads) >= 2:
        ns = [n for _, n, _ in reads if n]
        log_n_range = (math.log(max(ns)) - math.log(min(ns))) if len(ns) >= 2 else 0.0
        moved = 0
        for i in range(n_tr):
            vals = [r[i] for _, _, r in reads
                    if i < len(r) and r[i] is not None]
            if len(vals) >= 2:
                lo_vals = [logit(v) for v in vals]
                if max(lo_vals) - min(lo_vals) > 0.01:
                    moved += 1
        flat = log_n_range < 0.01 and moved <= max(1, n_tr // 10)

    out_rows = []
    if flat:
        out_rows = forecast_flat(trophies, last_rate, last_n,
                                 hours_since_release(at), horizon_h)
    else:
        out_rows = forecast_trend(trophies, last_rate, reads,
                                  hours_since_release(at),
                                  hours_since_release(target), horizon_d)
    return {"trophies": out_rows}


def forecast_flat(trophies, last_rate, last_n, hours_launch, horizon_h):
    """Refresh-mixture forecast for a flat history (practice cases)."""
    p_refresh = min(0.9, horizon_h / REFRESH_INTERVAL_H)
    jump = max(REFRESH_JUMP_FLOOR,
               1.0 + REFRESH_JUMP_0 * math.exp(-hours_launch / 24.0))
    kappa = KAPPA_0 + (1.0 - KAPPA_0) * (1.0 - math.exp(-hours_launch / 48.0))

    rows = []
    for t, r in zip(trophies, last_rate):
        r = 0.01 if r is None else r
        lo_last = logit(r)
        if r >= 99.9:
            # The trophy everybody holds: new players hold it too.
            lo_dil = lo_last
        else:
            diluted = r * (1.0 + (jump - 1.0) * kappa) / jump
            lo_dil = logit(diluted)
        qs = []
        for tau in LEVELS:
            q = inv_logit(mixture_quantile(lo_last, lo_dil, p_refresh,
                                           tau, MIXTURE_JITTER))
            qs.append(round4(clamp_pct(q)))
        for j in range(1, len(qs)):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return rows


def forecast_trend(trophies, last_rate, reads, hours_at, hours_target,
                   horizon_d):
    """Trend extrapolation with backtest-calibrated spread."""
    n_tr = len(trophies)
    x_target = math.log1p(hours_target)

    # --- Backtest: one-step-ahead forecasts inside the history, pooled over
    # trophies, to calibrate the spread. Splits use reads up to j to predict
    # read j+1 (same trend method, no shrinkage, no refresh prior).
    errors = []
    if len(reads) >= 4:
        for j in range(2, len(reads) - 1):
            sub = reads[:j + 1]
            nxt = reads[j + 1]
            x_nxt = math.log1p(nxt[0])
            for i in range(n_tr):
                pts = [(math.log1p(t), logit(r[i])) for t, _, r in sub
                       if i < len(r) and r[i] is not None]
                if len(pts) < 3 or nxt[2] is None or i >= len(nxt[2]) \
                        or nxt[2][i] is None:
                    continue
                xs = [p[0] for p in pts]
                if max(xs) - min(xs) < 1e-9:
                    continue
                # Same extrapolation as the median uses, so the residuals
                # calibrate the spread of this exact method.
                pred = extrapolate(pts, x_nxt)
                errors.append(logit(nxt[2][i]) - pred)

    if errors:
        med = sorted(errors)[len(errors) // 2]
        mad = sorted(abs(e - med) for e in errors)[len(errors) // 2]
        sd = 1.4826 * mad
    else:
        sd = 0.8
    floor = SPREAD_FLOOR_SIGMA * math.sqrt(max(horizon_d, 1e-9))
    sd = min(SPREAD_CAP, max(floor, sd))

    # --- Fit each trophy on the last 24 h of reads (all reads if fewer).
    cutoff = hours_at - 24.0
    recent = [r for r in reads if r[0] >= cutoff]
    if len(recent) < 3:
        recent = reads

    rows = []
    for i, t in enumerate(trophies):
        pts = [(math.log1p(h), logit(r[i])) for h, _, r in recent
               if i < len(r) and r[i] is not None]
        r_last = last_rate[i]
        if r_last is None:
            r_last = 0.01
        if pts:
            pred = extrapolate(pts, x_target)
        else:
            pred = logit(r_last)
        median = clamp_pct(inv_logit(pred))
        lo_med = logit(median)
        qs = []
        for z in Z:
            qs.append(round4(clamp_pct(inv_logit(lo_med + z * sd))))
        for j in range(1, len(qs)):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in inputs:
        cid = name[:-3]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        out_path = os.path.join(OUTPUT_DIR, cid + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
