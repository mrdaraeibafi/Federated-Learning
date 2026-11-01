#!/usr/bin/env python3
"""CTMC-driven quadratic SGD simulation with Palm-corrected learning mode.

This module integrates with `limited_ctmc_kbatch_sync_lumped_grouped_metrics.py`
to evaluate the optimization convergence across four learning modes:
1. `ideal`: True population weighting (w_g = N_g / N). Baseline MSE floor.
2. `mean`: Stationary Palm expectation (P_bar). Captures structural bias without variance.
3. `iid_palm`: Per-round i.i.d. sampling from the CTMC Palm participation distribution q(U).
4. `palm_corrected` (alias: `importance_weighted`): Importance weighting applied to sampled
   compositions to eliminate the structural bias floor:
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence
from tqdm import tqdm

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from limited_ctmc_kbatch_sync_lumped_grouped_metrics import (
    HeterogeneousLumpedKBatchSyncCTMC,
    LumpedCTMCResult,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "ctmc_driven_quadratic_sgd_palm_corrected_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUTPUT_DIR / "ctmc_sgd_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "ctmc_sgd_summary.json"


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment parameters for CTMC queue dynamics and quadratic SGD."""

    # CTMC queue parameters
    N: tuple[int, ...] = (4, 3, 3)
    mu: tuple[float, ...] = (1.2, 0.6, 0.2)
    K: int = 5
    b: int = 6
    B: int = 2

    # Quadratic objective parameters per group: f_g(theta) = 0.5 * a_g * (theta - c_g)^2
    a: tuple[float, ...] = (1.0, 1.5, 2.0)
    c: tuple[float, ...] = (1.0, -1.0, 3.0)
    sigma: tuple[float, ...] = (0.2, 0.3, 0.4)

    # SGD optimization parameters
    eta: float = 0.05
    theta0: float = -2.0
    n_aggregations: int = 200
    n_trajectories: int = 200
    seed: int = 42
    modes: tuple[str, ...] = ("ideal", "mean", "iid_palm", "palm_corrected")


def compute_corrected_weights(
    p_sample: np.ndarray,
    w_target: np.ndarray,
    p_bar: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute Palm-corrected (importance-weighted) participation weights.

    Formula:
        raw_w_g = w_g * (P_g / max(P_bar_g, eps))
        w_tilde_g = raw_w_g / sum(raw_w)
    """
    safe_pbar = np.maximum(p_bar, eps)
    raw_weights = w_target * (p_sample / safe_pbar)
    total = float(np.sum(raw_weights))
    if np.isfinite(total) and total > eps:
        return raw_weights / total
    return w_target.copy()


def noisy_aggregate_gradient(
    theta: float,
    a: np.ndarray,
    c: np.ndarray,
    sigma: np.ndarray,
    weights: np.ndarray,
    effective_counts: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute noisy aggregated gradient across groups."""
    g = len(a)
    grad = 0.0
    for i in range(g):
        u_eff = max(float(effective_counts[i]), 1e-6)
        noise_std = sigma[i] / np.sqrt(u_eff)
        noise = rng.normal(0.0, noise_std)
        group_grad = a[i] * (theta - c[i]) + noise
        grad += weights[i] * group_grad
    return float(grad)


def _json_array_or_none(arr: np.ndarray | None) -> str | None:
    if arr is None:
        return None
    return json.dumps(arr.tolist())


def _final_statistics(
    dataframe: pd.DataFrame,
    mode_name: str,
    final_step: int,
) -> dict[str, float]:
    df_mode = dataframe[
        (dataframe["mode"] == mode_name) & (dataframe["step"] == final_step)
    ]
    if len(df_mode) == 0:
        return {
            "final_mean_mse": float("nan"),
            "final_std_mse": float("nan"),
            "final_sem_mse": float("nan"),
            "final_mean_theta": float("nan"),
        }
    mse_vals = df_mode["mse"].to_numpy(dtype=float)
    theta_vals = df_mode["theta"].to_numpy(dtype=float)
    n = len(mse_vals)
    mean_mse = float(np.mean(mse_vals))
    std_mse = float(np.std(mse_vals, ddof=1)) if n > 1 else 0.0
    sem_mse = std_mse / np.sqrt(n) if n > 0 else 0.0
    mean_theta = float(np.mean(theta_vals))
    return {
        "final_mean_mse": mean_mse,
        "final_std_mse": std_mse,
        "final_sem_mse": sem_mse,
        "final_mean_theta": mean_theta,
    }


def run_experiment(
    config: ExperimentConfig,
    results_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:

    modes = list(config.modes)
    rows: list[dict[str, Any]] = []



    for mode in modes:
        mode_key = "palm_corrected" if mode == "importance_weighted" else mode
        rng = np.random.Generator(np.random.PCG64(config.seed))

        if mode_key == "ideal":
            """Execute CTMC computation and multi-mode distributed SGD."""
            groups = list(zip(config.N, config.mu))
            ctmc_model = HeterogeneousLumpedKBatchSyncCTMC(
                groups=groups,
                K=sum(config.N),
                b=1,
                B=1,
            )
            result: LumpedCTMCResult = ctmc_model.compute_result()

            if result.aggregation_rate <= 0:
                raise RuntimeError("CTMC aggregation rate is non-positive; check queue parameters.")

            a_arr = np.array(config.a, dtype=float)
            c_arr = np.array(config.c, dtype=float)
            sigma_arr = np.array(config.sigma, dtype=float)
            n_arr = np.array(config.N, dtype=float)

            w_target = n_arr / np.sum(n_arr)
            denom = float(np.sum(w_target * a_arr))
            theta_star = float(np.sum(w_target * a_arr * c_arr)) / denom

            p_bar = np.asarray(result.participation_composition_mean, dtype=float)
            denom_induced = float(np.sum(p_bar * a_arr))
            theta_mean_induced = float(np.sum(p_bar * a_arr * c_arr)) / denom_induced

            aggregation_rate = float(result.aggregation_rate*config.B/config.b)
            mean_interval = float(result.mean_aggregation_interval*config.b/config.B)

            participation_compositions = np.asarray(result.participation_compositions, dtype=float)
            comp_probs = np.asarray(result.participation_composition_probabilities, dtype=float)
            comp_probs /= np.sum(comp_probs)

        else:
            """Execute CTMC computation and multi-mode distributed SGD."""
            groups = list(zip(config.N, config.mu))
            ctmc_model = HeterogeneousLumpedKBatchSyncCTMC(
                groups=groups,
                K=config.K,
                b=config.b,
                B=config.B,
            )
            result: LumpedCTMCResult = ctmc_model.compute_result()

            if result.aggregation_rate <= 0:
                raise RuntimeError("CTMC aggregation rate is non-positive; check queue parameters.")

            a_arr = np.array(config.a, dtype=float)
            c_arr = np.array(config.c, dtype=float)
            sigma_arr = np.array(config.sigma, dtype=float)
            n_arr = np.array(config.N, dtype=float)

            w_target = n_arr / np.sum(n_arr)
            denom = float(np.sum(w_target * a_arr))
            theta_star = float(np.sum(w_target * a_arr * c_arr)) / denom

            p_bar = np.asarray(result.participation_composition_mean, dtype=float)
            denom_induced = float(np.sum(p_bar * a_arr))
            theta_mean_induced = float(np.sum(p_bar * a_arr * c_arr)) / denom_induced

            aggregation_rate = float(result.aggregation_rate*config.B/config.b)
            mean_interval = float(result.mean_aggregation_interval*config.b/config.B)

            participation_compositions = np.asarray(result.participation_compositions, dtype=float)
            comp_probs = np.asarray(result.participation_composition_probabilities, dtype=float)
            comp_probs /= np.sum(comp_probs)

        for traj in tqdm(range(config.n_trajectories), desc=f"Mode {mode}"): #traj in range(config.n_trajectories):
            theta = float(config.theta0)
            t_clock = 0.0

            rows.append(
                {
                    "mode": mode,
                    "trajectory": traj,
                    "step": 0,
                    "time": 0.0,
                    "theta": theta,
                    "mse": float((theta - theta_star) ** 2),
                    "theta_star": theta_star,
                    "participation": _json_array_or_none(None),
                    "effective_counts": _json_array_or_none(None),
                    "seed": int(config.seed),
                }
            )

            for step in range(1, config.n_aggregations + 1):
                dt = rng.exponential(mean_interval)
                t_clock += dt

                if mode_key == "ideal":
                    p_vec = w_target.copy()
                    u_eff = float(config.K) * w_target
                    weights = w_target.copy()
                elif mode_key == "mean":
                    p_vec = p_bar.copy()
                    u_eff = float(config.K) * p_bar
                    weights = p_bar.copy()
                elif mode_key == "iid_palm":
                    idx = rng.choice(len(comp_probs), p=comp_probs)
                    u_sample = participation_compositions[idx]
                    p_vec = u_sample / float(config.K)
                    u_eff = u_sample
                    weights = p_vec.copy()
                elif mode_key == "palm_corrected":
                    idx = rng.choice(len(comp_probs), p=comp_probs)
                    u_sample = participation_compositions[idx]
                    p_vec = u_sample / float(config.K)
                    u_eff = u_sample
                    weights = compute_corrected_weights(p_vec, w_target, p_bar)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                grad = noisy_aggregate_gradient(
                    theta, a_arr, c_arr, sigma_arr, weights, u_eff, rng
                )
                theta -= config.eta * grad
                mse = float((theta - theta_star) ** 2)

                rows.append(
                    {
                        "mode": mode,
                        "trajectory": traj,
                        "step": step,
                        "time": t_clock,
                        "theta": theta,
                        "mse": mse,
                        "theta_star": theta_star,
                        "participation": _json_array_or_none(p_vec),
                        "effective_counts": _json_array_or_none(u_eff),
                        "seed": int(config.seed),
                    }
                )

    dataframe = pd.DataFrame(rows)
    dataframe.sort_values(
        by=["mode", "trajectory", "step"], inplace=True, ignore_index=True
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(results_path, index=False)

    final_statistics = {
        m: _final_statistics(dataframe, m, int(config.n_aggregations))
        for m in modes
    }

    summary: dict[str, Any] = {
        "config": asdict(config),
        "aggregation_rate": aggregation_rate,
        "mean_aggregation_interval": mean_interval,
        "group_weights": w_target.tolist(),
        "participation_mean": p_bar.tolist(),
        "theta_star": theta_star,
        "theta_mean_induced": theta_mean_induced,
        "final_statistics": final_statistics,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return dataframe, summary


# def main() -> None:
#     # parser = argparse.ArgumentParser(
#     #     description="Run CTMC-driven quadratic SGD with Palm correction."
#     # )
#     # parser.add_argument(
#     #     "--trajectories", type=int, default=200, help="Number of Monte Carlo trajectories."
#     # )
#     # parser.add_argument(
#     #     "--steps", type=int, default=60, help="Number of aggregation steps per trajectory."
#     # )
#     # parser.add_argument(
#     #     "--results-csv", type=Path, default=RESULTS_PATH, help="Path for output CSV."
#     # )
#     # parser.add_argument(
#     #     "--summary-json", type=Path, default=SUMMARY_PATH, help="Path for output JSON."
#     # )
#     # args = parser.parse_args()

#     # config = ExperimentConfig(
#     #     n_trajectories=args.trajectories,
#     #     n_aggregations=args.steps,
#     # )

#     parser = argparse.ArgumentParser(
#         description="Run CTMC-driven quadratic SGD with Palm correction."
#     )
#     # پارامترهای شبیه‌سازی مونت‌کارلو و خروجی‌ها
#     parser.add_argument(
#         "--trajectories", type=int, default=200, help="Number of Monte Carlo trajectories."
#     )
#     parser.add_argument(
#         "--steps", type=int, default=60, help="Number of aggregation steps per trajectory."
#     )
#     parser.add_argument(
#         "--results-csv", type=Path, default=RESULTS_PATH, help="Path for output CSV."
#     )
#     parser.add_argument(
#         "--summary-json", type=Path, default=SUMMARY_PATH, help="Path for output JSON."
#     )

#     parser.add_argument(
#         "-N", "--N", type=int, nargs="+", default=[4, 3, 3],
#         help="Worker counts per group (e.g. --N 4 3 3)"
#     )
#     parser.add_argument(
#         "--mu", type=float, nargs="+", default=[1.2, 0.6, 0.2],
#         help="Service rates per group (e.g. --mu 1.2 0.6 0.2)"
#     )
#     parser.add_argument(
#         "-K", "--K", type=int, default=5,
#         help="Aggregation threshold K (subtasks required to trigger aggregation)"
#     )
#     parser.add_argument(
#         "-b", "--b", type=int, default=6,
#         help="Maximum per-worker backlog limit b"
#     )
#     parser.add_argument(
#         "-B", "--B", type=int, default=2,
#         help="Batch size per aggregation / subtasks per job B"
#     )

#     args = parser.parse_args()

#     N_tuple = tuple(args.N)
#     mu_tuple = tuple(args.mu)
#     if len(N_tuple) != len(mu_tuple):
#         parser.error(f"Length mismatch: N has {len(N_tuple)} groups but mu has {len(mu_tuple)} rates.")

#     config = ExperimentConfig(
#         n_trajectories=args.trajectories,
#         n_aggregations=args.steps,
#         N=N_tuple,
#         mu=mu_tuple,
#         K=args.K,
#         b=args.b,
#         B=args.B,
#     )

#     _, summary = run_experiment(
#         config,
#         results_path=args.results_csv,
#         summary_path=args.summary_json,
#     )
#     print("=== CTMC-driven Quadratic SGD Run Summary ===")
#     print(f"Aggregation Rate (A): {summary['aggregation_rate']:.4f} agg/s")
#     print(f"Global Optimum (theta*): {summary['theta_star']:.4f}")
#     print(f"Mean-Induced Stationary Theta: {summary['theta_mean_induced']:.4f}")
#     print(f"Final Mode Performance (Step {config.n_aggregations}) ---")
#     for mode_name, stats in summary["final_statistics"].items():
#         print(
#             f"  {mode_name:17s}: Mean MSE = {stats['final_mean_mse']:.6e} "
#             f"(SEM = {stats['final_sem_mse']:.4e}) | Final Theta = {stats['final_mean_theta']:.4f}"
#         )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CTMC-driven quadratic SGD with Palm correction."
    )
    
    # Simulation parameters
    parser.add_argument(
        "--trajectories", type=int, default=200,
        help="Number of Monte Carlo trajectories."
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of aggregation steps per trajectory."
    )
    parser.add_argument(
        "--results-csv", type=Path, default=None,
        help="Path for output CSV."
    )
    parser.add_argument(
        "--summary-json", type=Path, default=None,
        help="Path for output JSON."
    )
    
    # CTMC parameters
    parser.add_argument(
        "-N", "--N", type=int, nargs="+", default=[4, 3, 3],
        help="Worker counts per group (e.g. --N 4 3 3)"
    )
    parser.add_argument(
        "--mu", type=float, nargs="+", default=[1.2, 0.6, 0.2],
        help="Service rates per group (e.g. --mu 1.2 0.6 0.2)"
    )
    parser.add_argument(
        "-K", "--K", type=int, default=5,
        help="Aggregation threshold K (subtasks required to trigger aggregation)"
    )
    parser.add_argument(
        "-b", "--b", type=int, default=6,
        help="Maximum per-worker backlog limit b"
    )
    parser.add_argument(
        "-B", "--B", type=int, default=2,
        help="Batch size per aggregation / subtasks per job B"
    )
    
    # Quadratic objective parameters
    parser.add_argument(
        "--a", type=float, nargs="+", default=[1.0, 1.5, 2.0],
        help="Curvature parameters per group (e.g. --a 1.0 1.5 2.0)"
    )
    parser.add_argument(
        "--c", type=float, nargs="+", default=[1.0, -1.0, 3.0],
        help="Optima per group (e.g. --c 1.0 -1.0 3.0)"
    )
    parser.add_argument(
        "--sigma", type=float, nargs="+", default=[0.2, 0.3, 0.4],
        help="Noise standard deviation per group (e.g. --sigma 0.2 0.3 0.4)"
    )
    
    # SGD parameters
    parser.add_argument(
        "--eta", type=float, default=0.05,
        help="Learning rate"
    )
    parser.add_argument(
        "--theta0", type=float, default=-2.0,
        help="Initial theta value"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    # Modes to run
    parser.add_argument(
        "--modes", type=str, nargs="+", 
        default=["ideal", "mean", "iid_palm", "palm_corrected"],
        choices=["ideal", "mean", "iid_palm", "palm_corrected", "importance_weighted"],
        help="Modes to run (space-separated)"
    )

    args = parser.parse_args()

    # Validate lengths
    N_tuple = tuple(args.N)
    mu_tuple = tuple(args.mu)
    a_tuple = tuple(args.a)
    c_tuple = tuple(args.c)
    sigma_tuple = tuple(args.sigma)
    
    if len(N_tuple) != len(mu_tuple):
        parser.error(f"Length mismatch: N has {len(N_tuple)} groups but mu has {len(mu_tuple)} rates.")
    
    if len(a_tuple) != len(c_tuple) or len(a_tuple) != len(sigma_tuple):
        parser.error(f"Length mismatch: a ({len(a_tuple)}), c ({len(c_tuple)}), sigma ({len(sigma_tuple)}) must have same length.")
    
    if len(a_tuple) != len(N_tuple):
        parser.error(f"Length mismatch: N has {len(N_tuple)} groups but a,c,sigma have {len(a_tuple)} groups.")

    config = ExperimentConfig(
        n_trajectories=args.trajectories,
        n_aggregations=args.steps,
        N=N_tuple,
        mu=mu_tuple,
        K=args.K,
        b=args.b,
        B=args.B,
        a=a_tuple,
        c=c_tuple,
        sigma=sigma_tuple,
        eta=args.eta,
        theta0=args.theta0,
        seed=args.seed,
        modes=tuple(args.modes),
    )

    results_path = args.results_csv or OUTPUT_DIR / "ctmc_sgd_results.csv"
    summary_path = args.summary_json or OUTPUT_DIR / "ctmc_sgd_summary.json"

    _, summary = run_experiment(
        config,
        results_path=results_path,
        summary_path=summary_path,
    )
    
    print("=" * 60)
    print("CTMC-driven Quadratic SGD Run Summary")
    print("=" * 60)
    print(f"Worker topology: N={N_tuple}, mu={mu_tuple}")
    print(f"Quadratic parameters: a={a_tuple}, c={c_tuple}, sigma={sigma_tuple}")
    print(f"Parameters: K={config.K}, B={config.B}, b={config.b}")
    print(f"Aggregation Rate (A): {summary['aggregation_rate']:.4f} agg/s")
    print(f"Global Optimum (theta*): {summary['theta_star']:.4f}")
    print(f"Mean-Induced Stationary Theta: {summary['theta_mean_induced']:.4f}")
    print("-" * 60)
    print("Final Mode Performance (Step {})".format(config.n_aggregations))
    print("-" * 60)
    for mode_name, stats in summary["final_statistics"].items():
        print(
            f"  {mode_name:17s}: Mean MSE = {stats['final_mean_mse']:.6e} "
            f"(SEM = {stats['final_sem_mse']:.4e}) | Final Theta = {stats['final_mean_theta']:.4f}"
        )
    print("=" * 60)

if __name__ == "__main__":
    main()
