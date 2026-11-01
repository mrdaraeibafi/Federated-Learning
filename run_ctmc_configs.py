"""Run an explicit list of (B, J, K) configurations using the original CTMC.

Place next to limited_ctmc_kbatch_sync_lumped_grouped_metrics.py.
Requires numpy. Run: python run_ctmc_configs.py
Edit GROUPS and CONFIGS below; no Cartesian sweep is performed.
Times use the inverse unit of the service rates (seconds if mu is in 1/s).
The original solver builds a dense generator; large models can be expensive.
"""

from numbers import Integral

from limited_ctmc_kbatch_sync_lumped_grouped_metrics import (
    HeterogeneousLumpedKBatchSyncCTMC,
)

# EXAMPLE ONLY: 10 identical workers with service rate mu=1 per time unit.
# Replace with your workers, e.g. [(4, 1.0), (3, 2.0), (3, 3.0)].
GROUPS = [(3, 2.0), (4, 1.0), (3, 0.5)]

CONFIGS = [
    {"B": 1, "J": 1, "K": 1},
    {"B": 1, "J": 1, "K": 2},
    {"B": 1, "J": 1, "K": 4},
    {"B": 1, "J": 1, "K": 8},
    {"B": 1, "J": 1, "K": 10},
]

COLUMNS = (
    "B", "J", "b", "K", "States", "Aggregation_rate", "T_agg",
    "T_job_equal_share", "Status",
)


def run_configs(configs, groups):
    """Return one result per supplied configuration, preserving input order.

    T_job_equal_share is derived assuming equal aggregation shares among jobs;
    it is not a job-level metric directly produced by the CTMC.
    Errors are recorded per row without stopping subsequent configurations.
    """
    rows = []
    for config in configs:
        row = dict.fromkeys(COLUMNS, "-")
        try:
            for key in ("B", "J", "K"):
                value = config[key]
                row[key] = value
                if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
            B, J, K = (int(row[key]) for key in ("B", "J", "K"))
            b = B * J
            row["b"] = b
            N = sum(count for count, _ in groups)
            capacity = N * min(b, B)
            if K > capacity:
                raise ValueError(f"infeasible: K={K} > N*min(b,B)={capacity}")
            model = HeterogeneousLumpedKBatchSyncCTMC(groups=groups, K=K, b=b, B=B)
            result = model.compute_result()
            row.update({
                "States": len(result.states),
                "Aggregation_rate": result.aggregation_rate,
                "T_agg": result.mean_aggregation_interval,
                "T_job_equal_share": J * result.mean_aggregation_interval,
                "Status": "OK",
            })
        except Exception as exc:
            row["Status"] = f"ERROR: {type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def print_table(rows):
    def format_value(value):
        return f"{value:.8g}" if isinstance(value, float) else str(value)

    cells = [[format_value(row[column]) for column in COLUMNS] for row in rows]
    widths = [max([len(column)] + [len(row[i]) for row in cells])
              for i, column in enumerate(COLUMNS)]

    def line(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print(line(COLUMNS))
    print("-+-".join("-" * width for width in widths))
    for row in cells:
        print(line(row))


def main():
    print(f"Worker groups (count, service rate): {GROUPS}")
    print("b = B*J; time unit = inverse service-rate unit.\n")
    print_table(run_configs(CONFIGS, GROUPS))
    print("\nT_job_equal_share = J*T_agg assumes equal aggregation shares among jobs.")
    print("These are stationary mean intervals, not simulated wall-clock trajectories.")


if __name__ == "__main__":
    main()
