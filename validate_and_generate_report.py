import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from simpy_multi_job_kbatch_sync_baseline import run_experiment_monte_carlo
from limited_ctmc_kbatch_sync_lumped_grouped_metrics import HeterogeneousLumpedKBatchSyncCTMC

def run_analytical_model(N, J, B, K, R, service_rates, lr=0.05, c_param=1.0, sigma=0.5, theta_0=15.0):
    """
    Computes analytical metrics for a given configuration using Lumped CTMC + Quadratic SGD recursion.
    """
    b = B * J
    grouped_counts = Counter(np.round(service_rates, 8))
    groups = [(count, float(mu)) for mu, count in sorted(grouped_counts.items())]
    
    # Solve Lumped CTMC
    ctmc_model = HeterogeneousLumpedKBatchSyncCTMC(groups=groups, K=K, b=b, B=B)
    ctmc_res = ctmc_model.compute_result()
    
    mean_agg_interval = float(ctmc_res.mean_aggregation_interval)
    ana_time = float(R * mean_agg_interval)
    
    # SGD analytical convergence
    theta_star = 0.0
    init_mse = (theta_0 - theta_star) ** 2
    contract_factor = (1.0 - lr * c_param) ** 2
    
    if contract_factor != 1.0:
        geom_sum = (1.0 - (contract_factor ** R)) / (1.0 - contract_factor)
    else:
        geom_sum = float(R)
        
    noise_var = (lr ** 2) * (sigma ** 2) / float(K)
    ana_mse = float((contract_factor ** R) * init_mse + noise_var * geom_sum)
    
    return {
        'ana_time': ana_time,
        'ana_mse': ana_mse,
        'mean_agg_interval': mean_agg_interval
    }

def run_simulation_sweep(K_values, N=10, J=1, B=1, R=50, n_trajectories=30, base_seed=42):
    """
    Sweeps over K values and compares SimPy Monte-Carlo results with Analytical predictions.
    """
    # Heterogeneous service rates
    service_rates = np.array([1.0 + 0.5 * (i % 3) for i in range(N)], dtype=np.float64)
    
    records = []
    
    print(f"[*] Starting Validation Sweep over K={K_values} (Trajectories={n_trajectories}, R={R})...\n")
    
    for K in K_values:
        print(f"--- Processing K = {K} ---")
        # 1. Analytical Evaluation
        ana_res = run_analytical_model(N=N, J=J, B=B, K=K, R=R, service_rates=service_rates)
        
        # 2. Monte-Carlo Simulation Execution
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
        
        # Extract Job 0 terminal metrics across all trajectories
        term_df = raw_df[(raw_df['job_id'] == 0) & (raw_df['round'] == R)]
        
        # Exact column names in SimPy DataFrame: 'wallclock_time' and 'mse'
        sim_times = term_df['wallclock_time'].values
        sim_mses = term_df['mse'].values
        
        mean_sim_time = np.mean(sim_times)
        std_sim_time = np.std(sim_times, ddof=1) if len(sim_times) > 1 else 0.0
        ci_sim_time = 1.96 * (std_sim_time / np.sqrt(n_trajectories))
        
        mean_sim_mse = np.mean(sim_mses)
        std_sim_mse = np.std(sim_mses, ddof=1) if len(sim_mses) > 1 else 0.0
        ci_sim_mse = 1.96 * (std_sim_mse / np.sqrt(n_trajectories))
        
        # Error metrics
        rel_err_time = abs(mean_sim_time - ana_res['ana_time']) / ana_res['ana_time'] * 100.0
        rel_err_mse = abs(mean_sim_mse - ana_res['ana_mse']) / ana_res['ana_mse'] * 100.0
        
        records.append({
            'K': K,
            'Ana_Time': ana_res['ana_time'],
            'Sim_Time_Mean': mean_sim_time,
            'Sim_Time_CI': ci_sim_time,
            'Time_Rel_Err_%': rel_err_time,
            'Ana_MSE': ana_res['ana_mse'],
            'Sim_MSE_Mean': mean_sim_mse,
            'Sim_MSE_CI': ci_sim_mse,
            'MSE_Rel_Err_%': rel_err_mse
        })
        
    df_results = pd.DataFrame(records)
    return df_results

def print_validation_table(df):
    """
    Prints a formatted, publication-ready comparison table.
    """
    print("\n" + "="*95)
    print("                 VALIDATION SUMMARY: SIMULATION (SimPy) vs. THEORY (CTMC)")
    print("="*95)
    header = f"{'K':^4} | {'Sim Time (s) [95% CI]':^24} | {'Ana Time (s)':^13} | {'Err(%)':^7} | {'Sim MSE [95% CI]':^23} | {'Ana MSE':^11} | {'Err(%)':^7}"
    print(header)
    print("-" * 95)
    
    for _, r in df.iterrows():
        sim_t_str = f"{r['Sim_Time_Mean']:.3f} +/- {r['Sim_Time_CI']:.3f}"
        sim_m_str = f"{r['Sim_MSE_Mean']:.4e} +/- {r['Sim_MSE_CI']:.2e}"
        print(f"{int(r['K']):^4} | {sim_t_str:^24} | {r['Ana_Time']:^13.3f} | {r['Time_Rel_Err_%']:^7.2f} | {sim_m_str:^23} | {r['Ana_MSE']:^11.4e} | {r['MSE_Rel_Err_%']:^7.2f}")
    print("="*95 + "\n")

def plot_validation_panels(df, N=10, R=50):
    """
    Generates a 2-panel publication plot for Time Domain and Optimization Domain.
    """
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Subplot (a): Wall-Clock Time vs K
    ax1 = axes[0]
    ax1.plot(df['K'], df['Ana_Time'], 'r-', label='CTMC Theory', linewidth=2)
    ax1.errorbar(
        df['K'], df['Sim_Time_Mean'], yerr=df['Sim_Time_CI'],
        fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.5, capsize=4, label='SimPy (95% CI)'
    )
    ax1.set_xlabel('Sync Threshold ($K$)', fontweight='bold')
    ax1.set_ylabel(f'Wall-Clock Time @ Round {R} (s)', fontweight='bold')
    ax1.set_title('(a) Stochastic Delay & Straggler Scaling', pad=10)
    ax1.set_xticks(df['K'])
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True)
    
    # Subplot (b): Terminal MSE vs K
    ax2 = axes[1]
    ax2.plot(df['K'], df['Ana_MSE'], 'r-', label='SGD Theory', linewidth=2)
    ax2.errorbar(
        df['K'], df['Sim_MSE_Mean'], yerr=df['Sim_MSE_CI'],
        fmt='s', color='#2ca02c', ecolor='#2ca02c', elinewidth=1.5, capsize=4, label='SimPy (95% CI)'
    )
    ax2.set_yscale('log')
    ax2.set_xlabel('Sync Threshold ($K$)', fontweight='bold')
    ax2.set_ylabel(f'Terminal MSE @ Round {R} (Log Scale)', fontweight='bold')
    ax2.set_title('(b) Gradient Variance & Error Floor', pad=10)
    ax2.set_xticks(df['K'])
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True)
    
    plt.suptitle(f'Comprehensive Theoretical Validation ($N={N}$, $B=1$, $J=1$, $R={R}$)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    out_file = f"validation_panel_N{N}_R{R}.png"
    plt.savefig(out_file, dpi=300)
    print(f"[+] Multi-panel validation plot saved: {out_file}")

if __name__ == '__main__':
    N_WORKERS = 10
    K_SWEEP = [1, 2, 4, 6, 8, 10]
    ROUNDS = 200
    TRAJECTORIES = 2000
    
    results_df = run_simulation_sweep(
        K_values=K_SWEEP,
        N=N_WORKERS,
        J=1,
        B=1,
        R=ROUNDS,
        n_trajectories=TRAJECTORIES,
        base_seed=100
    )
    
    # 1. Print Text Summary Table
    print_validation_table(results_df)
    
    # 2. Render and save 2-Panel Plot
    plot_validation_panels(results_df, N=N_WORKERS, R=ROUNDS)
