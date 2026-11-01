#!/usr/bin/env python3
"""
K-Sweep Analysis for CTMC-driven Quadratic SGD (API Version)
Sweeps K from 1 to 10 and plots:
1. Physical Wall-Clock Time vs K
2. Accuracy (MSE) vs K

Two subplots in a single publication-quality figure.
Style matches plot_ctmc_fl_results.py.
Modes: only 'ideal' and 'mean'

Supports command-line arguments for b and B.
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Import your CTMC and SGD modules
from ctmc_driven_quadratic_sgd_palm_corrected import (
    ExperimentConfig,
    run_experiment,
)

# ============================================================================
# PUBLICATION-QUALITY STYLE SETTINGS (from plot_ctmc_fl_results.py)
# ============================================================================

# Set up matplotlib publication style
try:
    plt.style.use('seaborn-v0-8-paper')
except OSError:
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        try:
            plt.style.use('seaborn')
        except OSError:
            plt.style.use('default')

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif'],
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.1,
    'grid.linewidth': 0.6,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'legend.frameon': True,
    'legend.framealpha': 0.92,
    'legend.edgecolor': '#cccccc',
})

# ============================================================================
# COLOR PALETTE & STYLING (only ideal and mean)
# ============================================================================

SCHEME_STYLES = {
    'ideal': {
        'label': r'FedAvg ($K=N={}$, $b=1$, $B=1$)',
        'short': 'Ideal FedAvg',
        'color': '#1f77b4',       # Classic Royal Blue
        'linestyle': '-',
        'marker': 'o',
        'markevery': 6,
        'zorder': 5
    },
    'mean': {
        'label': r'CTMC-Biased ($n_g^{(k)}/K$ weighted)',
        'short': 'CTMC-Biased',
        'color': '#d62728',       # Crimson Red
        'linestyle': '--',
        'marker': 's',
        'markevery': 6,
        'zorder': 4
    },
}

SCHEME_ORDER = ['ideal', 'mean']

# ============================================================================
# CONSTANTS
# ============================================================================

# Base parameters (everything fixed except K, b, B)
BASE_CONFIG = {
    "N": (4, 3, 3),
    "mu": (1.2, 0.6, 0.2),
    "a": (1.0, 1.5, 2.0),
    "c": (1.0, -1.0, 3.0),
    "sigma": (0.2, 0.3, 0.4),
    "n_aggregations": 60,
    "n_trajectories": 200,
    "eta": 0.05,
    "theta0": -2.0,
    "seed": 42,
    "modes": ("ideal", "mean"),  # Only ideal and mean
}

# Default b and B
DEFAULT_B = 1
DEFAULT_b = 1

# K values to sweep (1 to 10)
K_VALUES = list(range(1, 11))

# Base output directory
BASE_OUTPUT_DIR = Path("k_sweep_results_api")


# ============================================================================
# CORE SWEEP FUNCTION
# ============================================================================

def run_single_k_api(K: int, b: int, B: int, verbose: bool = True) -> dict:
    """Run a single K experiment using the API directly."""
    
    config = ExperimentConfig(
        N=BASE_CONFIG["N"],
        mu=BASE_CONFIG["mu"],
        K=K,
        b=b,
        B=B,
        a=BASE_CONFIG["a"],
        c=BASE_CONFIG["c"],
        sigma=BASE_CONFIG["sigma"],
        n_aggregations=BASE_CONFIG["n_aggregations"],
        n_trajectories=BASE_CONFIG["n_trajectories"],
        eta=BASE_CONFIG["eta"],
        theta0=BASE_CONFIG["theta0"],
        seed=BASE_CONFIG["seed"],
        modes=BASE_CONFIG["modes"],
    )
    
    if verbose:
        print(f"  Running K={K} (b={b}, B={B})...")
    
    try:
        dataframe, summary = run_experiment(config)
    except Exception as e:
        print(f"  ❌ K={K} failed: {e}")
        return {"K": K, "b": b, "B": B, "success": False, "error": str(e)}
    
    final_stats = summary.get("final_statistics", {})
    aggregation_rate = summary.get("aggregation_rate", 0)
    mean_interval = 1.0 / aggregation_rate if aggregation_rate > 0 else float('inf')
    predicted_wall_clock = BASE_CONFIG["n_aggregations"] * mean_interval
    
    # result_dict = {
    #     "K": K,
    #     "b": b,
    #     "B": B,
    #     "aggregation_rate": aggregation_rate,
    #     "mean_interval": mean_interval,
    #     "predicted_wall_clock": predicted_wall_clock,
    #     "success": True,
    #     "theta_star": summary.get("theta_star", 0),
    #     "theta_mean_induced": summary.get("theta_mean_induced", 0),
    # }
    
    # # Extract MSE values for each mode (ideal and mean)
    # for mode, stats in final_stats.items():
    #     mode_key = mode.replace(" ", "_")
    #     mse_value = stats.get("final_mean_mse", float('nan'))
    #     sem_value = stats.get("final_sem_mse", float('nan'))
    #     result_dict[f"{mode_key}_mse"] = mse_value
    #     result_dict[f"{mode_key}_sem"] = sem_value
    
#
    # ============================================================
    # Compute SEM for wall-clock time
    # ============================================================
    # If we have multiple trajectories, compute the standard deviation
    # and then SEM = std / sqrt(n_trajectories)
    n_trajectories = BASE_CONFIG["n_trajectories"]
    
    # Get the trajectory times from the dataframe
    if 'time' in dataframe.columns:
        # For each mode, get the final time across trajectories
        wall_clock_times = []
        for mode in config.modes:
            df_mode = dataframe[dataframe['mode'] == mode]
            if len(df_mode) > 0:
                # Get the final time for each trajectory
                final_times = df_mode.groupby('trajectory')['time'].last()
                wall_clock_times.extend(final_times.values)
        
        if wall_clock_times:
            std_wall_clock = np.std(wall_clock_times, ddof=1)
            sem_wall_clock = std_wall_clock / np.sqrt(len(wall_clock_times))
        else:
            sem_wall_clock = 0.0
    else:
        sem_wall_clock = 0.0
    
    result_dict = {
        "K": K,
        "b": b,
        "B": B,
        "aggregation_rate": aggregation_rate,
        "mean_interval": mean_interval,
        "predicted_wall_clock": predicted_wall_clock,
        "predicted_wall_clock_sem": sem_wall_clock,  # <-- ADD THIS
        "success": True,
        "theta_star": summary.get("theta_star", 0),
        "theta_mean_induced": summary.get("theta_mean_induced", 0),
    }
    
    # Extract MSE values for each mode
    for mode, stats in final_stats.items():
        mode_key = mode.replace(" ", "_")
        mse_value = stats.get("final_mean_mse", float('nan'))
        sem_value = stats.get("final_sem_mse", float('nan'))
        result_dict[f"{mode_key}_mse"] = mse_value
        result_dict[f"{mode_key}_sem"] = sem_value
#

    # Debug: Print extracted values
    if verbose:
        print(f"    ideal_mse = {result_dict.get('ideal_mse', 'N/A')}")
        print(f"    mean_mse = {result_dict.get('mean_mse', 'N/A')}")
    
    # Create output directory for this b, B
    output_dir = BASE_OUTPUT_DIR / f"b{b}_B{B}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"ctmc_sgd_results_K{K}.csv"
    dataframe.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"  ✅ K={K}: A={aggregation_rate:.3f}, Wall-clock={predicted_wall_clock:.1f}s")
    
    return result_dict


def run_k_sweep_api(K_values: list = None, b: int = DEFAULT_b, B: int = DEFAULT_B, 
                    verbose: bool = True) -> pd.DataFrame:
    if K_values is None:
        K_values = K_VALUES
    
    # Create output directory for this b, B
    output_dir = BASE_OUTPUT_DIR / f"b{b}_B{B}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("K-SWEEP ANALYSIS (API VERSION)")
    print("=" * 70)
    print(f"K values: {K_values}")
    print(f"Total runs: {len(K_values)}")
    print(f"b = {b}, B = {B}")
    print(f"Modes: {BASE_CONFIG['modes']}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    
    results = []
    for K in K_values:
        result = run_single_k_api(K, b, B, verbose)
        results.append(result)
    
    df = pd.DataFrame(results)
    csv_path = output_dir / "k_sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")
    print(f"\nColumns in saved CSV: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df[['K', 'b', 'B', 'predicted_wall_clock', 'aggregation_rate', 'ideal_mse', 'mean_mse']].head())
    
    return df


def load_existing_results_api(b: int = DEFAULT_b, B: int = DEFAULT_B) -> pd.DataFrame:
    output_dir = BASE_OUTPUT_DIR / f"b{b}_B{B}"
    csv_path = output_dir / "k_sweep_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\nLoaded CSV columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df[['K', 'b', 'B', 'predicted_wall_clock', 'aggregation_rate']].head())
        return df
    return None


# ============================================================================
# MAIN FIGURE: TWO SUBPLOTS IN ONE FIGURE (Publication Quality)
# ============================================================================

def plot_k_sweep_two_subplots(df: pd.DataFrame, b: int, B: int, output_dir: Path = None):
    """
    Main figure with two subplots:
    - Left: Physical Wall-Clock Time vs K
    - Right: Accuracy (MSE) vs K for ideal and mean modes
    
    Publication-quality style matching plot_ctmc_fl_results.py.
    Legend positioned in the upper right corner with adjusted y-limits.
    
    Color scheme:
    - FedAvg = Ideal = Blue (#1f77b4) - BOTH figures
    - CTMC-Based = Mean = K-Batch-Sync = Red (#d62728) - BOTH figures
    
    All curves use SOLID lines ('-').
    NO confidence bands (SEM data not available).
    
    IMPORTANT: 
    - FedAvg baseline (both wall-clock and MSE) ALWAYS uses b=1, B=1, K=N
    - The ideal curve in the right subplot is CONSTANT (read from baseline)
    - The mean curve varies with K (from current data)
    """
    
    # Filter successful runs
    df_success = df[df['success'] == True].copy()
    
    if len(df_success) == 0:
        print("❌ No successful runs to plot")
        print("Available columns:", list(df.columns))
        print("Data summary:")
        print(df.to_string())
        return
    
    # Set output directory
    if output_dir is None:
        output_dir = BASE_OUTPUT_DIR / f"b{b}_B{B}" / "figures"
    else:
        output_dir = output_dir / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Load Baseline Data (b=1, B=1) for FedAvg/Ideal
    # ========================================================================
    N_total = sum(BASE_CONFIG["N"])  # Should be 10
    base_K = N_total
    
    baseline_dir = BASE_OUTPUT_DIR / "b1_B1"
    baseline_csv = baseline_dir / "k_sweep_summary.csv"
    
    if not baseline_csv.exists():
        raise RuntimeError(
            f"FedAvg baseline data not found at {baseline_csv}!\n"
            f"Please run: python3 k_sweep_analysis_api.py --run --b 1 --B 1"
        )
    
    # Read the baseline data
    baseline_df = pd.read_csv(baseline_csv)
    baseline_success = baseline_df[baseline_df['success'] == True].copy()
    
    # Find the FedAvg baseline (K=N, b=1, B=1)
    fedavg_row = baseline_success[
        (baseline_success['K'] == base_K) & 
        (baseline_success['b'] == 1) & 
        (baseline_success['B'] == 1)
    ]
    
    if len(fedavg_row) > 0:
        base_wall_clock = fedavg_row['predicted_wall_clock'].values[0] * b/B
        # IDEAL MSE is also from the baseline (constant)
        if 'ideal_mse' in fedavg_row.columns:
            base_ideal_mse = fedavg_row['ideal_mse'].values[0]
        else:
            base_ideal_mse = None
        print(f"✅ FedAvg baseline: K={base_K}, b=1, B=1")
        print(f"   Wall-clock: {base_wall_clock:.1f}s")
        print(f"   Ideal MSE: {base_ideal_mse:.2e}")
    else:
        available_configs = baseline_success[['K', 'b', 'B']].drop_duplicates().to_string(index=False)
        raise RuntimeError(
            f"FedAvg baseline data not found in {baseline_csv}!\n"
            f"Need K={base_K}, b=1, B=1.\n"
            f"Available configurations in baseline:\n{available_configs}"
        )
    
    # ========================================================================
    # Create figure with two subplots side by side
    # ========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ========================================================================
    # SUBPLOT 1: Physical Wall-Clock Time vs K
    # ========================================================================
    ax1 = axes[0]
    
    # ---- K-Batch-Sync Wall-Clock (varies with K, from current data) ----
    ax1.plot(df_success['K'], df_success['predicted_wall_clock'], 
             's-', color='#d62728', linewidth=2.2, markersize=7,
             markerfacecolor='white', markeredgewidth=1.5,
             label=r'$K$-Batch-Sync ($b={}, B={}$)'.format(b, B))
    
    # Add value labels
    for idx, row in df_success.iterrows():
        if idx % 2 == 0:
            ax1.annotate(f'{row["predicted_wall_clock"]:.0f}s', 
                        (row['K'], row['predicted_wall_clock']),
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center', 
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # ---- FedAvg Wall-Clock (CONSTANT, from baseline) ----
    fedavg_K = df_success['K'].values
    fedavg_time = np.full_like(fedavg_K, base_wall_clock, dtype=float)
    
    ax1.plot(fedavg_K, fedavg_time, 
             'o-', color='#1f77b4', linewidth=2.0, markersize=7,
             markerfacecolor='white', markeredgewidth=1.5,
             label=r'FedAvg ($K=N={}$, $b=1$, $B=1$)'.format(N_total))
    
    # Add value labels
    for idx, (k, t) in enumerate(zip(fedavg_K, fedavg_time)):
        if idx % 2 == 0:
            ax1.annotate(f'{t:.0f}s', 
                        (k, t),
                        textcoords="offset points", 
                        xytext=(0, -15), 
                        ha='center', 
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax1.set_xlabel(r'$K$ (Aggregation Threshold)', fontsize=12)
    ax1.set_ylabel(r'Physical Wall-Clock Time [seconds]', fontsize=12)
    ax1.set_title(rf'Wall-Clock Delay', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xticks(df_success['K'])
    
    y_min = -20
    y_max = df_success['predicted_wall_clock'].max() * 1.3
    if base_wall_clock is not None:
        y_max = max(y_max, base_wall_clock * 1.3)
    ax1.set_ylim(y_min, y_max)
    
    ax1.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.92, 
               edgecolor='#cccccc', handlelength=3)
    
    # ========================================================================
    # SUBPLOT 2: Accuracy (MSE) vs K
    # ========================================================================
    ax2 = axes[1]
    
    # ---- CTMC-Biased (Mean) - VARIES with K (from current data) ----
    mode = 'mean'
    mse_col = f"{mode}_mse"
    
    if mse_col in df_success.columns:
        valid = df_success[mse_col].notna()
        if valid.any():
            df_plot = df_success.loc[valid]
            
            # Plot mean curve
            ax2.plot(df_plot['K'], 
                     df_plot[mse_col],
                     label=r'$K$-Batch-Sync ($b={}, B={}$)'.format(b, B),
                     color='#d62728',
                     marker='s',
                     linestyle='-',
                     linewidth=2.2,
                     markersize=8,
                     markerfacecolor='white',
                     markeredgewidth=1.5,
                     zorder=4)
            
            # Add value labels
            for idx, row in df_plot.iterrows():
                if idx % 2 == 0:
                    ax2.annotate(f'{row[mse_col]:.1e}', 
                                (row['K'], row[mse_col]),
                                textcoords="offset points", 
                                xytext=(0, -14), 
                                ha='center', 
                                va='top',
                                fontsize=8,
                                alpha=0.8,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # ---- Ideal FedAvg - CONSTANT (from baseline) ----
    if base_ideal_mse is not None:
        ideal_K = df_success['K'].values
        ideal_time = np.full_like(ideal_K, base_ideal_mse, dtype=float)
        
        # Plot Ideal FedAvg (CONSTANT line)
        ax2.plot(ideal_K, ideal_time, 
                 'o-', color='#1f77b4', linewidth=2.0, markersize=7,
                 markerfacecolor='white', markeredgewidth=1.5,
                 label=r'FedAvg ($K=N={}$, $b=1$, $B=1$)'.format(N_total))
        
        # Add value labels
        for idx, (k, t) in enumerate(zip(ideal_K, ideal_time)):
            if idx % 2 == 0:
                ax2.annotate(f'{t:.1e}', 
                            (k, t),
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center', 
                            va='bottom',
                            fontsize=8,
                            alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax2.set_xlabel(r'$K$ (Aggregation Threshold)', fontsize=12)
    ax2.set_ylabel(r'Mean Squared Error $\mathbb{E}[(\hat{\theta}(t) - \theta^*)^2]$', fontsize=12)
    ax2.set_title(rf'Convergence Accuracy', fontsize=13)
    ax2.set_yscale('log')
    
    # Adjust y-axis limits
    mse_values = []
    if mse_col in df_success.columns:
        valid = df_success[mse_col].notna()
        if valid.any():
            mse_values.extend(df_success.loc[valid, mse_col].values)
    if base_ideal_mse is not None:
        mse_values.append(base_ideal_mse)
    
    if mse_values:
        y_min_log = max(1e-8, min(mse_values) / 2)
        y_max_log = max(mse_values) * 5
        ax2.set_ylim(y_min_log, y_max_log)
    
    ax2.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.92, 
               edgecolor='#cccccc', handlelength=3)
    
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xticks(df_success['K'])
    
    # Add overall figure title
    fig.suptitle(rf'K-Batch-Sync Performance Analysis: Wall-Clock vs Accuracy $(b={b}, B={B})$', 
                 fontsize=14, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    # Save in both formats
    png_path = output_dir / f'k_sweep_two_subplots_b{b}_B{B}.png'
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {png_path}")
    
    pdf_path = output_dir / f'k_sweep_two_subplots_b{b}_B{B}.pdf'
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"✅ Saved: {pdf_path}")
    
    plt.close(fig)
    print(f"✅ Figure saved to {output_dir}/")
    
# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sweep K from 1 to 10 using API (ideal and mean modes only)."
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run the K sweep (otherwise just plot existing results)"
    )
    parser.add_argument(
        "--K", type=int, nargs="+", default=None,
        help="Specific K values to sweep (e.g., --K 1 3 5 7 9)"
    )
    parser.add_argument(
        "--b", type=int, default=DEFAULT_b,
        help=f"Maximum per-worker backlog limit b (default: {DEFAULT_b})"
    )
    parser.add_argument(
        "--B", type=int, default=DEFAULT_B,
        help=f"Batch size per aggregation / subtasks per job B (default: {DEFAULT_B})"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose output"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Print debug information"
    )
    args = parser.parse_args()
    
    b = args.b
    B = args.B
    K_values = args.K if args.K is not None else K_VALUES
    
    if args.run:
        df = run_k_sweep_api(K_values, b, B, args.verbose)
    else:
        df = load_existing_results_api(b, B)
        if df is None:
            print(f"No existing results found for b={b}, B={B}. Running sweep...")
            df = run_k_sweep_api(K_values, b, B, args.verbose)
        else:
            print(f"Loaded existing results from {BASE_OUTPUT_DIR}/b{b}_B{B}/k_sweep_summary.csv")
            print(f"Total records: {len(df)}")
            print(df[['K', 'b', 'B', 'predicted_wall_clock', 'aggregation_rate']].to_string(index=False))
    
    if args.debug:
        print("\n" + "=" * 70)
        print("DEBUG INFORMATION")
        print("=" * 70)
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10))
        print("\nData types:")
        print(df.dtypes)
    
    if len(df) > 0:
        print("\n" + "=" * 70)
        print("GENERATING FIGURE")
        print("=" * 70)
        
        plot_k_sweep_two_subplots(df, b, B)
        
        output_dir = BASE_OUTPUT_DIR / f"b{b}_B{B}" / "figures"
        print(f"\n✅ Figure saved to {output_dir}/")
        print(f"   - k_sweep_two_subplots_b{b}_B{B}.png")
        print(f"   - k_sweep_two_subplots_b{b}_B{B}.pdf")
    else:
        print("❌ No data to plot")


if __name__ == "__main__":
    main()