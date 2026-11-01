import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from simpy_multi_job_kbatch_sync_baseline import run_experiment_monte_carlo, compute_statistics
from limited_ctmc_kbatch_sync_lumped_grouped_metrics import HeterogeneousLumpedKBatchSyncCTMC

def get_simulation_point_metrics(N=10, J=1, B=1, K=1, R=50, n_trajectories=30, service_rates=None, base_seed=42):
    """
    Executes SimPy Monte-Carlo microscopic simulation and extracts terminal metrics at Round R.
    """
    if service_rates is None:
        service_rates = np.array([1.0 + 0.5 * (i % 3) for i in range(N)], dtype=np.float64)
    elif not isinstance(service_rates, np.ndarray):
        service_rates = np.array(service_rates, dtype=np.float64)

    # In-memory execution using exact signature
    raw_df = run_experiment_monte_carlo(
        N=N,
        J=J,
        B=B,
        K=K,
        R=R,
        n_trajectories=n_trajectories,
        service_rates=service_rates,
        base_seed=base_seed
    )
    
    stats_df = compute_statistics(raw_df)
    
    # Extract terminal round R metrics for Job 0
    terminal_row = stats_df[(stats_df['job_id'] == 0) & (stats_df['round'] == R)].iloc[0]
    sim_mse = float(terminal_row['mean_mse'])
    sim_time = float(terminal_row['mean_time'])
    
    return sim_mse, sim_time

def get_analytical_point_metrics(N=10, J=1, B=1, K=1, R=50, service_rates=None, lr=0.05, c_param=1.0, sigma=0.5, theta_0=15.0):
    """
    Computes analytical metrics at round R using Lumped CTMC renewal interval + quadratic SGD recursion.
    """
    if service_rates is None:
        service_rates = np.array([1.0 + 0.5 * (i % 3) for i in range(N)], dtype=np.float64)
    elif not isinstance(service_rates, np.ndarray):
        service_rates = np.array(service_rates, dtype=np.float64)
    
    # Buffer capacity per worker: b = B * J
    b = B * J
    
    # Heterogeneous grouping (N_g, mu_g) for lumped CTMC
    grouped_counts = Counter(np.round(service_rates, 8))
    groups = [(count, float(mu)) for mu, count in sorted(grouped_counts.items())]
    
    # Solve Lumped CTMC
    ctmc_model = HeterogeneousLumpedKBatchSyncCTMC(groups=groups, K=K, b=b, B=B)
    ctmc_res = ctmc_model.compute_result()
    
    # Analytical expected wall-clock time for R rounds
    ana_time = float(R * ctmc_res.mean_aggregation_interval)
    
    # Analytical expected MSE at round R: E[(theta_R - theta*)^2]
    theta_star = 0.0
    init_mse = (theta_0 - theta_star) ** 2
    
    contract_factor = (1.0 - lr * c_param) ** 2
    geom_sum = (1.0 - (contract_factor ** R)) / (1.0 - contract_factor) if contract_factor != 1.0 else R
    noise_variance = (lr ** 2) * (sigma ** 2) / float(K)
    ana_mse = float((contract_factor ** R) * init_mse + noise_variance * geom_sum)
    
    return ana_mse, ana_time

def plot_pointwise_benchmark(sim_metrics, ana_metrics, N=10, K=1, R=50):
    """
    Generates bar chart comparing SimPy empirical values vs CTMC analytical values at Round R.
    """
    labels = ['Terminal MSE @ R', 'Wall-Clock Time (s) @ R']
    sim_vals = [sim_metrics[0], sim_metrics[1]]
    ana_vals = [ana_metrics[0], ana_metrics[1]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, sim_vals, width, label='SimPy Microscopic', color='#1f77b4')
    ax.bar(x + width/2, ana_vals, width, label='Lumped CTMC Theory', color='#d62728')

    ax.set_yscale('log')
    ax.set_ylabel('Metric Value (Log Scale)')
    ax.set_title(f'Pointwise Validation at Round R={R} (N={N}, K={K})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_filename = f"pointwise_benchmark_N{N}_K{K}_R{R}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"[+] Output validation plot saved: {output_filename}")

if __name__ == '__main__':
    N, J, B, K, R = 10, 1, 1, 10, 200
    service_rates = np.array([1.0 + 0.5 * (i % 3) for i in range(N)], dtype=np.float64)
    base_seed = 42
    
    print(f"[*] Starting Pointwise Comparison for N={N}, K={K}, B={B}, J={J}, R={R}...")
    sim_mse, sim_time = get_simulation_point_metrics(
        N=N, J=J, B=B, K=K, R=R, n_trajectories=200, service_rates=service_rates, base_seed=base_seed
    )
    ana_mse, ana_time = get_analytical_point_metrics(
        N=N, J=J, B=B, K=K, R=R, service_rates=service_rates
    )
    
    print("\n" + "="*52)
    print(f"Validation Summary at Final Round R = {R}")
    print("="*52)
    print(f"Simulation (SimPy) -> MSE: {sim_mse:.6e} | Wall-Clock Time: {sim_time:.4f} s")
    print(f"Analytical (CTMC)  -> MSE: {ana_mse:.6e} | Wall-Clock Time: {ana_time:.4f} s")
    print("="*52)
    
    plot_pointwise_benchmark((sim_mse, sim_time), (ana_mse, ana_time), N=N, K=K, R=R)
