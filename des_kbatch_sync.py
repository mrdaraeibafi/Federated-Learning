"""Independent microscopic DES for the grouped K-batch CTMC.

Run: python des_kbatch_sync.py
Dependencies: numpy; the original CTMC module is needed only for comparison.
Edit GROUPS and CONFIGS below. Rates determine the time unit.
No explicit job identities are modeled: J enters only through b=B*J.
"""
import heapq
from numbers import Integral
import numpy as np

GROUPS = [(3, 2.0), (4, 1.0), (3, 0.5)]  # Example only: (worker count, service rate)
CONFIGS = [{"B": 1, "J": 2, "K": k} for k in (1, 2, 4, 8, 10)]
WARMUP = 1000.0
DURATION = 10000.0
REPLICATIONS = 8
SEED = 2024
COMPARE_CTMC = True  # Dense CTMC can be expensive for large configurations.


def validate(groups, B, J, K):
    for name, value in (("B", B), ("J", J), ("K", K)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not groups:
        raise ValueError("groups cannot be empty")
    for count, mu in groups:
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError("worker counts must be positive integers")
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("service rates must be finite and positive")
    if K > sum(n for n, _ in groups) * B:
        raise ValueError("infeasible K: exceeds N*B")


def simulate(groups, B, J, K, warmup, duration, seed):
    """Simulate fixed observation time after warmup, starting from x_i=0.

    One scheduled completion per active worker. Aggregation never cancels an
    ongoing service. Workers blocked at b restart when aggregation frees space.
    All reported occupancy/activity averages are time-weighted.
    """
    validate(groups, B, J, K)
    if not np.isfinite(warmup) or warmup < 0:
        raise ValueError("warmup must be finite and nonnegative")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    rng = np.random.default_rng(seed)
    mus = np.array([mu for count, mu in groups for _ in range(count)])
    N, b = len(mus), B * J
    x = np.zeros(N, dtype=int)
    events = [(rng.exponential(1 / mu), i) for i, mu in enumerate(mus)]
    heapq.heapify(events)
    end = warmup + duration
    now = 0.0
    count = 0
    occupancy_area = active_area = 0.0
    consumed = np.zeros(N, dtype=np.int64)
    trigger_counts = np.zeros(N, dtype=np.int64)
    observed_event_times = []
    while events:
        t, worker = heapq.heappop(events)
        boundary = min(t, end)
        dt = max(0.0, boundary - max(now, warmup))
        occupancy_area += dt * int(x.sum())
        active_area += dt * int(np.count_nonzero(x < b))
        if t >= end:
            break
        now = t
        x[worker] += 1
        score = int(np.minimum(x, B).sum())
        if score == K:
            # Only workers already blocked have no pending service event.
            blocked = np.flatnonzero(x == b)
            # The triggering worker also has no pending event: reschedule it below.
            take = np.minimum(x, B)
            if t >= warmup:
                count += 1
                consumed += take
                trigger_counts[worker] += 1
                observed_event_times.append(t)
            x -= take
            for i in blocked:
                if i != worker:
                    heapq.heappush(events, (t + rng.exponential(1 / mus[i]), int(i)))
        elif score > K:
            raise AssertionError("aggregation threshold was skipped")
        if x[worker] < b:
            heapq.heappush(events, (t + rng.exponential(1 / mus[worker]), worker))
    else:
        raise RuntimeError("event queue exhausted before observation ended")
    if int(consumed.sum()) != K * count:
        raise AssertionError("consumption conservation failed")
    rate = count / duration
    return {
        "aggregation_count": count,
        "aggregation_rate": rate,
        "T_agg_rate_inverse": 1 / rate if rate else np.inf,
        "mean_complete_interval": float(np.mean(np.diff(observed_event_times)))
            if len(observed_event_times) > 1 else np.nan,
        "mean_total_pending": occupancy_area / duration,
        "active_fraction": active_area / (duration * N),
        "consumed_by_worker": consumed,
        "trigger_counts_by_worker": trigger_counts,
    }


def run_config(groups, config, warmup=WARMUP, duration=DURATION,
               replications=REPLICATIONS, seed=SEED, compare_ctmc=True):
    """Independent-replication rate estimate; normal-approximation 95% CI.

    CI quantifies replication variability, not possible warmup bias. Increase
    warmup/duration and check stability before interpreting stationary results.
    """
    if not isinstance(replications, Integral) or replications < 2:
        raise ValueError("at least two replications required")
    B, J, K = (config[key] for key in ("B", "J", "K"))
    seeds = np.random.SeedSequence(seed).spawn(replications)
    runs = [simulate(groups, B, J, K, warmup, duration, s) for s in seeds]
    rates = np.array([r["aggregation_rate"] for r in runs])
    rate = float(rates.mean())
    halfwidth = float(1.96 * rates.std(ddof=1) / np.sqrt(replications))
    result = dict(B=B, J=J, b=B*J, K=K, DES_rate=rate,
                  CI95_low=max(0.0, rate-halfwidth), CI95_high=rate+halfwidth,
                  DES_T=1/rate if rate else np.inf,
                  active_fraction=float(np.mean([r["active_fraction"] for r in runs])),
                  mean_total_pending=float(np.mean([r["mean_total_pending"] for r in runs])))
    if compare_ctmc:
        from limited_ctmc_kbatch_sync_lumped_grouped_metrics import HeterogeneousLumpedKBatchSyncCTMC
        model = HeterogeneousLumpedKBatchSyncCTMC(groups=groups, K=K, b=B*J, B=B)
        exact = model.compute_result()
        result.update(CTMC_rate=exact.aggregation_rate,
                      CTMC_T=exact.mean_aggregation_interval,
                      relative_rate_error=abs(rate/exact.aggregation_rate-1))
    return result


def main():
    import csv
    import json
    rows = []
    print("Independent DES; J enters only via b=B*J. No SGD or job scheduler.")
    for index, config in enumerate(CONFIGS):
        try:
            row = run_config(GROUPS, config, seed=SEED+index, compare_ctmc=COMPARE_CTMC)
            row["Status"] = "OK"
        except Exception as exc:
            row = {**config, "Status": f"ERROR: {type(exc).__name__}: {exc}"}
        rows.append(row)
        print(" | ".join(f"{key}={value:.7g}" if isinstance(value, float)
                         else f"{key}={value}" for key, value in row.items()))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open("des_ctmc_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open("des_run_settings.json", "w") as f:
        json.dump(dict(groups=GROUPS, configs=CONFIGS, warmup=WARMUP,
                       duration=DURATION, replications=REPLICATIONS, seed=SEED,
                       compare_ctmc=COMPARE_CTMC,
                       ci="normal approximation: mean +/- 1.96*SE across replications"), f, indent=2)
    print("Saved des_ctmc_comparison.csv and des_run_settings.json")


if __name__ == "__main__":
    main()
