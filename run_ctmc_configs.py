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
# GROUPS = [(3, 2.0), (4, 1.0), (3, 0.5)]
GROUPS = [(10, 1.0)]

CONFIGS = [
    # {"B": 1, "J": 1, "K": 1},
    # {"B": 1, "J": 1, "K": 2},
    # {"B": 1, "J": 1, "K": 4},
    # {"B": 1, "J": 1, "K": 8},
    # {"B": 1, "J": 1, "K": 10},
    # {"B": 1, "J": 2, "K": 1},
    # {"B": 1, "J": 2, "K": 2},
    # {"B": 1, "J": 2, "K": 4},
    # {"B": 1, "J": 2, "K": 8},
    # {"B": 1, "J": 2, "K": 10},
    # {"B": 1, "J": 3, "K": 1},
    # {"B": 1, "J": 3, "K": 2},
    # {"B": 1, "J": 3, "K": 4},
    # {"B": 1, "J": 3, "K": 8},
    # {"B": 1, "J": 3, "K": 10},
    # {"B": 1, "J": 4, "K": 1},
    # {"B": 1, "J": 4, "K": 2},
    # {"B": 1, "J": 4, "K": 4},
    # {"B": 1, "J": 4, "K": 8},
    # {"B": 1, "J": 4, "K": 10},
    # {"B": 1, "J": 5, "K": 1},
    # {"B": 1, "J": 5, "K": 2},
    # {"B": 1, "J": 5, "K": 4},
    # {"B": 1, "J": 5, "K": 8},
    # {"B": 1, "J": 5, "K": 10},
    {"B": 4, "J": 1, "K": 1},
    {"B": 4, "J": 1, "K": 2},
    {"B": 4, "J": 1, "K": 4},
    {"B": 4, "J": 1, "K": 8},
    {"B": 4, "J": 1, "K": 10},
    {"B": 4, "J": 2, "K": 1},
    {"B": 4, "J": 2, "K": 2},
    {"B": 4, "J": 2, "K": 4},
    {"B": 4, "J": 2, "K": 8},
    {"B": 4, "J": 2, "K": 10},
    {"B": 4, "J": 3, "K": 1},
    {"B": 4, "J": 3, "K": 2},
    {"B": 4, "J": 3, "K": 4},
    {"B": 4, "J": 3, "K": 8},
    {"B": 4, "J": 3, "K": 10},
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
        print(config)
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


def plot_job_time_vs_J(rows, output_dir="ctmc_plots"):
    from pathlib import Path
    import math
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_K = (1, 2, 4, 8, 10)
    markers = {1: "o", 2: "s", 4: "^", 8: "D", 10: "v"}

    valid_rows = [
        row for row in rows
        if row["Status"] == "OK"
        and row["K"] in selected_K
        and math.isfinite(float(row["T_job_equal_share"]))
    ]

    if not valid_rows:
        print("No valid results available for plotting.")
        return

    for B in sorted({row["B"] for row in valid_rows}):
        subset = [row for row in valid_rows if row["B"] == B]

        fig, ax = plt.subplots(figsize=(8, 5.5))

        # Draw one curve per K.
        for K in selected_K:
            points = sorted(
                [row for row in subset if row["K"] == K],
                key=lambda row: row["J"],
            )

            if not points:
                continue

            ax.plot(
                [row["J"] for row in points],
                [float(row["T_job_equal_share"]) for row in points],
                marker=markers[K],
                linewidth=1.8,
                markersize=5,
                label=fr"$K={K}$",
            )

        j_min = min(row["J"] for row in subset)
        j_max = max(row["J"] for row in subset)

        # Baseline: y = T_job_equal_share(J=1, K=1) * J.
        baseline = next(
            (
                float(row["T_job_equal_share"])
                for row in subset
                if row["J"] == 1 and row["K"] == 1
            ),
            None,
        )

        # if baseline is not None:
        #     ax.plot(
        #         [0, j_max],
        #         [0, baseline * j_max],
        #         linestyle="--",
        #         color="black",
        #         linewidth=1.8,
        #         zorder=5,
        #         label=(
        #             r"Reference: $J\,T_{\mathrm{job,eq}}(1,1)$"
        #             f" (slope={baseline:.4g})"
        #         ),
        #     )
        # else:
        #     print(
        #         f"Baseline unavailable for B={B}: "
        #         "missing valid result for J=1, K=1."
        #     )

        ax.set_xlabel(r"Number of jobs, $J$")
        ax.set_ylabel(r"$T_{\mathrm{job,eq}}$ (time units)")
        ax.set_title(
            fr"Equal-share mean per-job interval ($B={B}$)"
        )

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(max(0, j_min - 0.25), j_max + 0.25)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()

        for extension in ("png", "pdf"):
            path = (
                output_dir
                / f"T_job_equal_share_vs_J_B{B}.{extension}"
            )
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved: {path}")

        plt.close(fig)


# def main():
#     print(f"Worker groups (count, service rate): {GROUPS}")
#     print("b = B*J; time unit = inverse service-rate unit.\n")
#     print_table(run_configs(CONFIGS, GROUPS))
#     print("\nT_job_equal_share = J*T_agg assumes equal aggregation shares among jobs.")
#     print("These are stationary mean intervals, not simulated wall-clock trajectories.")

def main():
    print(f"Worker groups (count, service rate): {GROUPS}")
    print("b = B*J; time unit = inverse service-rate unit.\n")

    rows = run_configs(CONFIGS, GROUPS)

    print_table(rows)
    plot_job_time_vs_J(rows)

    print(
        "\nT_job_equal_share = J*T_agg assumes "
        "equal aggregation shares among jobs."
    )
    print(
        "These are stationary mean intervals, "
        "not simulated wall-clock trajectories."
    )


if __name__ == "__main__":
    main()
