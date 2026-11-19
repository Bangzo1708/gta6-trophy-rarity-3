#!/usr/bin/env python3
"""GTA VI trophy-rarity forecaster, round 3.

Model: a holders-over-players (H/N) cohort decomposition.

A trophy's earn rate is r = H / N, holders over players tracked.  Holders
only grow (a player who earned a trophy keeps it) and players tracked only
grow (more profiles sync).  The two move for different reasons, so the
program projects them separately and divides, instead of extrapolating the
rate itself:

* N(t): players tracked.  Take the recent hourly growth rate of log N
  (last day of refresh states) and project it with an exponentially
  decaying rate (72 h time constant): a launch's inflow slows over days,
  so the recent rate is not charged over the whole horizon.
* H_i(t): holders of trophy i.  Same local-slope projection per trophy,
  anchored at the last observed holders, with each trophy's slope shrunk
  toward the cross-trophy median slope (a trophy with few reads borrows
  the typical progression speed).  Holders are capped at N(target) and
  never projected below their last value.

When the history is a single flat state (launch night: no refresh has been
seen yet), there is nothing to fit, so the program uses the refresh prior
the competition itself documents (RULES.txt / competition.yaml): Exophase
refreshes a game's counts every few hours, and its first refresh took the
page from 638 players tracked / 943 trophies earned to 3,868 / 4,276,
every rate but the 100% one falling.  One such refresh is expected before
the target: N grows by 3868/638 and non-intro holders by
(4276-3868)/(943-638), spread evenly across trophies (a constant
per-player-hour hazard makes every trophy's holders grow by the same
factor; the intro trophy, held by everyone, stays at 100%).

Spread: quantiles are the median shifted on the logit scale by the
standard-normal quantiles times a per-trophy sigma.  Sigma combines the
fit residuals, a term growing with the horizon, and a floor for the
uncertainty no fit can see (a refresh landing, an allocation surprise).
A 90-minute horizon with a documented refresh gets a tight band; a
five-day extrapolation gets a wide one.

Standard library only, no network, no randomness: the same input always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# The first refresh the house recorded, as documented in the kit
# (competition.yaml / RULES.txt): 638 -> 3,868 players tracked,
# 943 -> 4,276 trophies earned, the 100% trophy unmoved.
DOC_N_MULT = 3868.0 / 638.0
DOC_H_MULT = (4276.0 - 3868.0) / (943.0 - 638.0)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l > 40.0:
        return 100.0
    if l < -40.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (slope, intercept, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return b, a, resid


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantiles(med, sigma):
    """Five quantiles around a median rate, non-decreasing, in [0, 100]."""
    if med >= 99.99:
        return [99.9, 99.97, 99.99, 100.0, 100.0]
    base = logit(med)
    out = []
    for z in Z:
        out.append(clamp(inv_logit(base + z * sigma), 0.01, 100.0))
    # Enforce non-decreasing, then round, then enforce again (rounding
    # can only break order by a hair, the second pass fixes it).
    for _ in range(2):
        run = -1.0
        for i, v in enumerate(out):
            if v < run:
                out[i] = run
            else:
                run = v
        out = [round(v, 4) for v in out]
    return out


def refresh_states(history):
    """Collapse consecutive identical reads into one state each.

    Exophase refreshes every few hours, so most reads repeat the previous
    page exactly.  A state is (time, players_tracked, rates); its time is
    the first read that showed it.
    """
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(read["at"])
        except Exception:
            continue
        states.append((t, read.get("players_tracked"), rates))
    return states


def flat_branch(case, states, last_rates, n_last, horizon_days, horizon_hours):
    """No refresh seen yet: expect the documented refresh before target."""
    n_target = n_last * DOC_N_MULT
    # A long flat horizon would hold more than one refresh; after the first
    # (which absorbs all pending accumulation) let counts drift mildly.
    if horizon_hours > 6.0:
        extra_days = (horizon_hours - 6.0) / 24.0
        n_target *= 1.0 + 0.10 * extra_days
    sigma = math.sqrt(0.12 ** 2 + (0.09 * math.sqrt(max(0.0, horizon_days))) ** 2)
    rows = []
    for i, rate in enumerate(last_rates):
        if rate is None:
            rows.append((i, 1.0))
            continue
        if rate >= 99.99:
            rows.append((i, 100.0))
            continue
        holders = rate / 100.0 * n_last
        h_target = holders * DOC_H_MULT
        if horizon_hours > 6.0:
            extra_days = (horizon_hours - 6.0) / 24.0
            h_target *= 1.0 + 0.05 * extra_days
        med = clamp(100.0 * h_target / n_target, 0.01, 100.0)
        rows.append((i, med))
    return rows, sigma


DECAY_HOURS = 72.0


def _local_slope(ts, ys, window_hours=24.0):
    """Slope of y against t over the most recent window (or all points).

    Returns (slope, resid_std, n_points).  Launch growth is local: what
    matters for the next days is how fast counts have been moving lately,
    not the average speed since release.
    """
    if len(ts) < 2:
        return None, 0.0, len(ts)
    t_last = ts[-1]
    idx = [k for k, t in enumerate(ts) if t >= t_last - window_hours]
    if len(idx) < 2:
        idx = list(range(len(ts)))
    xs = [ts[k] for k in idx]
    ys2 = [ys[k] for k in idx]
    b, _a, r = ols(xs, ys2)
    return b, r, len(idx)


def trend_branch(case, states, last_rates, n_last, horizon_days, target):
    """Project N(t) and H_i(t) from recent local growth, then divide.

    Growth is extrapolated with an exponentially decaying rate (time
    constant 72 h): a launch's inflow and progression slow over days, so
    the recent hourly rate is not charged over the whole horizon.  The
    effective horizon is DECAY * (1 - exp(-horizon / DECAY)).
    """
    horizon_hours = horizon_days * 24.0
    tau_eff = DECAY_HOURS * (1.0 - math.exp(-horizon_hours / DECAY_HOURS)) \
        if horizon_hours > 0.0 else 0.0

    ts = []
    log_ns = []
    for t, n, _rates in states:
        if not n or n <= 0:
            continue
        ts.append(hours_since_release(t))
        log_ns.append(math.log(float(n)))
    if len(ts) < 2:
        raise ValueError("not enough states for trend branch")

    g_n, s_n, _ = _local_slope(ts, log_ns)
    g_n = 0.0 if g_n is None else clamp(g_n, 0.0, 0.08)
    s_n_eff = s_n if s_n > 0.0 else 0.10
    n_target = float(n_last) * math.exp(g_n * tau_eff)
    n_target = clamp(n_target, float(n_last), float(n_last) * 40.0)

    # Per-trophy local holder growth, shrunk toward the typical trophy.
    n_troph = len(last_rates)
    g_by_trophy = {}
    s_by_trophy = {}
    for i in range(n_troph):
        tpts = []
        ypts = []
        for (t, n, rates) in states:
            if i >= len(rates) or rates[i] is None or not n:
                continue
            tpts.append(hours_since_release(t))
            ypts.append(math.log(max(rates[i] / 100.0 * float(n), 0.5)))
        if len(tpts) >= 3:
            g, s, _ = _local_slope(tpts, ypts)
            if g is not None:
                g_by_trophy[i] = g
                s_by_trophy[i] = s
    g_med = median(list(g_by_trophy.values())) if g_by_trophy else 0.0
    g_med = clamp(g_med, 0.0, 0.08)
    s_med = median(list(s_by_trophy.values())) if s_by_trophy else 0.15
    horizon_term = 0.09 * math.sqrt(max(0.0, horizon_days))

    rows = []
    sigmas = []
    for i, rate in enumerate(last_rates):
        if rate is None:
            rows.append((i, 1.0))
            sigmas.append(0.5)
            continue
        if rate >= 99.99:
            rows.append((i, 100.0))
            sigmas.append(0.12)
            continue
        h_last = max(rate / 100.0 * float(n_last), 0.5)
        if i in g_by_trophy:
            # Count how many points the local fit used, for shrinkage.
            tpts = []
            for (t, n, rates) in states:
                if i < len(rates) and rates[i] is not None and n:
                    tpts.append(hours_since_release(t))
            npts = sum(1 for t in tpts if t >= tpts[-1] - 24.0)
            npts = max(npts, 2)
            w = npts / (npts + 3.0)
            g_use = w * g_by_trophy[i] + (1.0 - w) * g_med
            s_i = s_by_trophy.get(i, s_med)
        else:
            g_use = g_med
            s_i = s_med
        g_use = clamp(g_use, 0.0, 0.10)
        h_target = h_last * math.exp(g_use * tau_eff)
        h_target = clamp(h_target, h_last, n_target)
        h_target = min(h_target, h_last * 100.0)
        med = clamp(100.0 * h_target / n_target, 0.01, 100.0)
        rows.append((i, med))
        sig = math.sqrt(s_i ** 2 + s_n_eff ** 2 + horizon_term ** 2 + 0.10 ** 2)
        sigmas.append(clamp(sig, 0.15, 0.90))
    return rows, sigmas


def persistence_fallback(case):
    """Last rates, honest wide band. Used only if the model fails."""
    history = case.get("history") or []
    if not history:
        return [(i, 1.0) for i in range(len(case.get("trophies", [])))], 0.5
    last_rates = history[-1].get("rates") or []
    rows = []
    for i, rate in enumerate(last_rates):
        rows.append((i, 1.0 if rate is None else clamp(float(rate), 0.01, 100.0)))
    return rows, 0.30


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    try:
        at = parse_time(case["at"])
        target = parse_time(case["target"])
    except Exception:
        at = RELEASE
        target = RELEASE
    horizon_days = max(0.0, (target - at).total_seconds() / 86400.0)
    horizon_hours = horizon_days * 24.0

    if not history:
        rows = [(i, 1.0) for i in range(len(trophies))]
        return [{"id": t["id"], "q": quantiles(1.0, 0.5)} for t in trophies]

    last = history[-1]
    last_rates = last.get("rates") or []
    n_last = last.get("players_tracked") or 0
    if not n_last or not last_rates:
        rows, sig = persistence_fallback(case)
        return [{"id": t["id"], "q": quantiles(rows[i][1], sig)}
                for i, t in enumerate(trophies)]

    try:
        states = refresh_states(history)
        if len(states) <= 1:
            rows, sig = flat_branch(case, states, last_rates, n_last,
                                    horizon_days, horizon_hours)
            sigmas = [sig] * len(rows)
        else:
            rows, sigmas = trend_branch(case, states, last_rates, n_last,
                                        horizon_days, target)
    except Exception as exc:  # never crash a case: fall back cleanly
        print("forecast fallback: %r" % (exc,), file=sys.stderr)
        rows, sig = persistence_fallback(case)
        sigmas = [sig] * len(rows)

    out = []
    for idx, t in enumerate(trophies):
        med = rows[idx][1] if idx < len(rows) else 1.0
        sig = sigmas[idx] if idx < len(sigmas) else 0.3
        out.append({"id": t["id"], "q": quantiles(med, sig)})
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        rows = forecast_case(case)
        payload = {"trophies": rows}
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        out_name = os.path.splitext(name)[0] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
