#!/usr/bin/env python3
"""
Publication-Quality Plotting Script for CTMC-Driven Quadratic SGD Federated Learning.

Generates 4 figures in both PNG and PDF:
- Figure 1: Iteration Convergence (MSE vs Aggregation Round k)
- Figure 2: Wall-Clock Physical Time Convergence (MSE vs Time t)
- Figure 3: Mechanistic Bias Diagnosis & Asymptotic Fixed Points
- Figure 4: Parameter Distribution & Bias-Variance Decomposition

Supports mode selection: only selected modes are plotted.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set up matplotlib publication style
plt.rcParams.update({
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

# Color palette & styling definition
SCHEME_STYLES = {
    'ideal': {
        'label': r'Ideal FedAvg ($w_g$ weighted)',
        'short': 'Ideal FedAvg',
        'color': '#1f77b4',       # Classic Royal Blue
        'linestyle': '-',
        'marker': 'o',
        'markevery': 6,
        'zorder': 5
    },
    'palm_corrected': {
        'label': r'Palm-Corrected ($w_g / \bar{P}_g$)',
        'short': 'Palm-Corrected',
        'color': '#2ca02c',       # Forest Green
        'linestyle': '-.',
        'marker': '^',
        'markevery': 6,
        'zorder': 4
    },
    'mean': {
        'label': r'CTMC-Biased ($n_g^{(k)}/K$ weighted)',
        'short': 'CTMC-Biased',
        'color': '#d62728',       # Crimson Red
        'linestyle': '--',
        'marker': 's',
        'markevery': 6,
        'zorder': 3
    },
    'iid_palm': {
        'label': r'i.i.d. Palm Baseline',
        'short': 'i.i.d. Palm',
        'color': '#ff7f0e',       # Amber Orange
        'linestyle': ':',
        'marker': 'd',
        'markevery': 6,
        'zorder': 2
    }
}

# All available modes
ALL_MODES = ['ideal', 'palm_corrected', 'mean', 'iid_palm']
DEFAULT_MODES = ['ideal', 'mean']  # Default: only ideal and mean


def load_data(csv_path, json_path):
    """Load simulation results and theoretical summary."""
    df = pd.read_csv(csv_path)
    with open(json_path, 'r') as f:
        summary = json.load(f)
    return df, summary


def filter_modes(df, selected_modes):
    """Filter dataframe to only include selected modes."""
    if selected_modes is None:
        return df
    return df[df['mode'].isin(selected_modes)]


def get_mode_styles(selected_modes):
    """Get style dictionary for selected modes only."""
    if selected_modes is None:
        return SCHEME_STYLES
    return {mode: SCHEME_STYLES[mode] for mode in selected_modes if mode in SCHEME_STYLES}


def get_mode_order(selected_modes):
    """Get mode order for selected modes only."""
    if selected_modes is None:
        return SCHEME_ORDER
    return [mode for mode in ALL_MODES if mode in selected_modes]


def plot_figure_1(df, summary, output_dirs, selected_modes=DEFAULT_MODES):
    """
    Figure 1: Iteration Convergence (MSE vs Aggregation Round k).
    Log scale y-axis with 95% CI shaded bounds and asymptotic limit markers.
    Only selected modes are plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))

    theta_star = summary['theta_star']
    theta_mean_induced = summary['theta_mean_induced']
    asymptotic_mse_bias = (theta_mean_induced - theta_star)**2

    # Filter data for selected modes
    df_filtered = filter_modes(df, selected_modes)
    mode_styles = get_mode_styles(selected_modes)
    mode_order = get_mode_order(selected_modes)

    for mode in mode_order:
        sub = df_filtered[df_filtered['mode'] == mode]
        if sub.empty:
            continue
        
        # Group by step across trajectories
        grp = sub.groupby('step')['mse']
        mean_mse = grp.mean().values
        sem_mse = grp.sem().values
        steps = np.sort(sub['step'].unique())
        
        style = mode_styles[mode]
        ax.plot(steps, mean_mse, label=style['label'],
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markevery=style['markevery'],
                linewidth=2.2, zorder=style['zorder'])
        
        # 95% CI shaded bound: mean +- 1.96 * SEM
        ci_lower = np.maximum(mean_mse - 1.96 * sem_mse, 1e-6)
        ci_upper = mean_mse + 1.96 * sem_mse
        ax.fill_between(steps, ci_lower, ci_upper, color=style['color'], alpha=0.15, zorder=style['zorder']-1)

    # Theoretical Asymptotic Reference Line for CTMC Bias
    ax.axhline(asymptotic_mse_bias, color='#7f7f7f', linestyle=':', linewidth=1.5,
               label=r'Asymptotic CTMC Bias Limit $(\bar{\theta}_{\mathrm{mean}} - \theta^\star)^2 = ' + f'{asymptotic_mse_bias:.3f}$')

    ax.set_yscale('log')
    ax.set_xlabel(r'Aggregation Round $k$')
    ax.set_ylabel(r'Mean Squared Error $\mathbb{E}[(\theta_k - \theta^\star)^2]$ (Log Scale)')
    ax.set_title(r'Figure 1: Convergence Trajectories vs. Aggregation Round', pad=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.set_xlim(0, summary['config']['n_aggregations'])
    
    # Adjust legend based on number of modes
    if len(mode_order) <= 2:
        ax.legend(loc='upper right', framealpha=0.95)
    else:
        ax.legend(loc='lower left', framealpha=0.95)

    # # Annotate the asymptotic gap (only if mean mode is selected)
    # if 'mean' in mode_order:
    #     ax.annotate(r'Palm Sampling Bias Gap',
    #                 xy=(45, asymptotic_mse_bias), xytext=(32, 0.05),
    #                 arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', arrowstyle='->', lw=1.5),
    #                 fontsize=10, fontweight='bold', color='#d62728',
    #                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#d62728', alpha=0.8))

    plt.tight_layout()
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'fig1_iteration_convergence.png'), dpi=300)
        fig.savefig(os.path.join(out_dir, 'fig1_iteration_convergence.pdf'))
    plt.close(fig)
    print(f"Saved Figure 1. (Modes: {mode_order})")


def plot_figure_2(df, summary, output_dirs, selected_modes=DEFAULT_MODES):
    """
    Figure 2: Wall-Clock Physical Time Convergence (MSE vs Time t).
    Compares optimization progress against real system delay.
    Only selected modes are plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    
    A = summary.get('aggregation_rate', 1.43976)
    theta_star = summary['theta_star']

    # Filter data for selected modes
    df_filtered = filter_modes(df, selected_modes)
    mode_styles = get_mode_styles(selected_modes)
    mode_order = get_mode_order(selected_modes)

    for mode in mode_order:
        sub = df_filtered[df_filtered['mode'] == mode]
        if sub.empty:
            continue

        # Group by step to get average physical time and average MSE across trajectories
        grp_mse = sub.groupby('step')['mse']
        grp_time = sub.groupby('step')['time']
        
        mean_time = grp_time.mean().values
        mean_mse = grp_mse.mean().values
        sem_mse = grp_mse.sem().values
        
        style = mode_styles[mode]
        ax.plot(mean_time, mean_mse, label=style['label'],
                color=style['color'], linestyle=style['linestyle'],
                marker=style['marker'], markevery=style['markevery'],
                linewidth=2.2, zorder=style['zorder'])
        
        ci_lower = np.maximum(mean_mse - 1.96 * sem_mse, 1e-6)
        ci_upper = mean_mse + 1.96 * sem_mse
        ax.fill_between(mean_time, ci_lower, ci_upper, color=style['color'], alpha=0.15, zorder=style['zorder']-1)

    ax.set_yscale('log')
    ax.set_xlabel(r'Physical Wall-Clock Time $t = k / \mathcal{A}$ [seconds]')
    ax.set_ylabel(r'Mean Squared Error $\mathbb{E}[(\theta(t) - \theta^\star)^2]$ (Log Scale)')
    ax.set_title(r'Figure 2: Wall-Clock Optimization Delay ($\mathcal{A}\approx ' + f'{A:.3f}' + r'\,\mathrm{agg/s}$)', pad=12)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    
    # Adjust legend based on number of modes
    if len(mode_order) <= 2:
        ax.legend(loc='upper right', framealpha=0.95)
    else:
        ax.legend(loc='lower left', framealpha=0.95)

    plt.tight_layout()
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'fig2_wallclock_convergence.png'), dpi=300)
        fig.savefig(os.path.join(out_dir, 'fig2_wallclock_convergence.pdf'))
    plt.close(fig)
    print(f"Saved Figure 2. (Modes: {mode_order})")


def plot_figure_3(df, summary, output_dirs, selected_modes=DEFAULT_MODES):
    """
    Figure 3: Mechanistic Bias Diagnosis & Asymptotic Fixed Points.
    3-panel detailed layout:
      (a) Population Weights w_g vs Palm Participation Probabilities P_bar_g
      (b) Group Heterogeneous Losses f_g(theta) and Equilibrium Points
      (c) Per-round Empirical Group Participation Dynamics
    Only selected modes are plotted in panel (c).
    """
    fig = plt.figure(figsize=(13, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.2], wspace=0.32)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    groups = [r'$\mu_1=1.2$', r'$\mu_2=0.6$', r'$\mu_3=0.2$']
    w_g = np.array(summary['group_weights'])
    P_g = np.array(summary['participation_mean'])
    
    x = np.arange(len(groups))
    width = 0.35

    # Panel (a): w_g vs P_bar_g (unchanged)
    rects1 = ax1.bar(x - width/2, w_g, width, label=r'Population $w_g = N_g/N$', color='#1f77b4', alpha=0.85, edgecolor='black', lw=0.8)
    rects2 = ax1.bar(x + width/2, P_g, width, label=r'Palm-Stationary $\bar{P}_g$', color='#d62728', alpha=0.85, edgecolor='black', lw=0.8)
    
    ax1.set_ylabel('Probability / Weight Mass')
    ax1.set_title(r'(a) Straggler Sampling Distortion', fontsize=11, pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=9.5)
    ax1.set_ylim(0, 0.8)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax1.legend(loc='upper right', fontsize=8.5)

    # Add percentage labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h*100:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h*100:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, color='#b71c1c', fontweight='bold')

    # Panel (b): Heterogeneous Objective Functions & Fixed Points (unchanged)
    cfg = summary['config']
    a = np.array(cfg['a'])
    c = np.array(cfg['c'])
    theta_grid = np.linspace(-0.5, 2.5, 300)
    
    F_pop = np.zeros_like(theta_grid)
    F_palm = np.zeros_like(theta_grid)
    for i in range(len(w_g)):
        f_i = 0.5 * a[i] * (theta_grid - c[i])**2
        F_pop += w_g[i] * f_i
        F_palm += P_g[i] * f_i
        ax2.plot(theta_grid, f_i, linestyle=':', alpha=0.45, label=rf'$f_{i+1}(\theta)$ ($c_{i+1}={c[i]}$)')

    ax2.plot(theta_grid, F_pop, color='#1f77b4', linewidth=2.2, label=r'Global $F_{\mathrm{pop}}(\theta)$')
    ax2.plot(theta_grid, F_palm, color='#d62728', linewidth=2.2, linestyle='--', label=r'CTMC-Biased $F_{\mathrm{palm}}(\theta)$')
    
    theta_star = summary['theta_star']
    theta_mean_induced = summary['theta_mean_induced']
    
    ax2.axvline(theta_star, color='#1f77b4', linestyle='-', linewidth=1.2, alpha=0.7)
    ax2.axvline(theta_mean_induced, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.7)
    
    ax2.scatter([theta_star], [np.interp(theta_star, theta_grid, F_pop)], color='#1f77b4', s=60, zorder=6)
    ax2.scatter([theta_mean_induced], [np.interp(theta_mean_induced, theta_grid, F_palm)], color='#d62728', s=60, zorder=6)

    ax2.set_xlabel(r'Parameter $\theta$')
    ax2.set_ylabel(r'Loss Value $F(\theta)$')
    ax2.set_title(r'(b) Objective Landscape & Asymptotic Minima', fontsize=11, pad=8)
    ax2.legend(loc='upper right', fontsize=8.0)
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    ax2.annotate(rf'$\theta^\star = {theta_star:.3f}$', xy=(theta_star, 0.2), xytext=(theta_star+0.15, 1.2),
                 arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2), fontsize=9, color='#1f77b4', fontweight='bold')
    ax2.annotate(rf'$\bar{{\theta}}_{{\mathrm{{mean}}}} = {theta_mean_induced:.3f}$', xy=(theta_mean_induced, 0.4), xytext=(theta_mean_induced-1.0, 1.8),
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2), fontsize=9, color='#d62728', fontweight='bold')

    # Panel (c): Parameter Trajectory Convergence - only selected modes
    mode_styles = get_mode_styles(selected_modes)
    mode_order = get_mode_order(selected_modes)
    
    for mode in mode_order:
        sub = df[df['mode'] == mode]
        if sub.empty:
            continue
        grp = sub.groupby('step')['theta']
        mean_theta = grp.mean().values
        sem_theta = grp.sem().values
        steps = np.sort(sub['step'].unique())
        style = mode_styles[mode]
        ax3.plot(steps, mean_theta, label=style['short'], color=style['color'],
                 linestyle=style['linestyle'], linewidth=2.0)
        ax3.fill_between(steps, mean_theta - 1.96*sem_theta, mean_theta + 1.96*sem_theta,
                         color=style['color'], alpha=0.15)

    ax3.axhline(theta_star, color='#1f77b4', linestyle='-', linewidth=1.2, alpha=0.6, label=r'$\theta^\star$')
    ax3.axhline(theta_mean_induced, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.6, label=r'$\bar{\theta}_{\mathrm{mean}}$')
    ax3.set_xlabel(r'Aggregation Round $k$')
    ax3.set_ylabel(r'Parameter Estimate $\bar{\theta}_k$')
    ax3.set_title(r'(c) Convergence to Distinct Fixed Points', fontsize=11, pad=8)
    
    # Adjust legend based on number of modes
    if len(mode_order) <= 2:
        ax3.legend(loc='lower right', fontsize=8.5)
    else:
        ax3.legend(loc='lower right', fontsize=8.5)
    
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.set_xlim(0, 60)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'fig3_bias_diagnosis.png'), dpi=300)
        fig.savefig(os.path.join(out_dir, 'fig3_bias_diagnosis.pdf'))
    plt.close(fig)
    print(f"Saved Figure 3. (Modes in panel c: {mode_order})")


def plot_figure_4(df, summary, output_dirs, selected_modes=DEFAULT_MODES):
    """
    Figure 4: Parameter Distribution & Bias-Variance Decomposition.
    (a) Boxplot of final parameter estimates theta_{60} across trajectories.
    (b) Stacked bar chart decomposing Final MSE into Bias^2 and Variance.
    Only selected modes are shown.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1.1, 1]})

    theta_star = summary['theta_star']
    step60 = df[df['step'] == summary['config']['n_aggregations']]
    
    # Filter for selected modes
    step60_filtered = filter_modes(step60, selected_modes)
    mode_styles = get_mode_styles(selected_modes)
    mode_order = get_mode_order(selected_modes)

    # Prepare data
    data_boxplot = []
    labels = []
    biases_sq = []
    variances = []
    total_mses = []
    colors = []

    for mode in mode_order:
        sub = step60_filtered[step60_filtered['mode'] == mode]
        thetas = sub['theta'].values
        mses = sub['mse'].values
        
        mean_t = np.mean(thetas)
        var_t = np.var(thetas, ddof=0)
        b_sq = (mean_t - theta_star)**2
        
        data_boxplot.append(thetas)
        labels.append(mode_styles[mode]['short'])
        biases_sq.append(b_sq)
        variances.append(var_t)
        total_mses.append(np.mean(mses))
        colors.append(mode_styles[mode]['color'])

    # Panel (a): Boxplot / Distribution
    bp = ax1.boxplot(data_boxplot, patch_artist=True, tick_labels=labels,
                     showmeans=True, meanline=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     meanprops=dict(color='yellow', linewidth=2.0, linestyle='--'),
                     flierprops=dict(marker='o', markersize=3, alpha=0.5))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor('black')

    ax1.axhline(theta_star, color='#1f77b4', linestyle='-', linewidth=1.5, label=rf'Target $\theta^\star = {theta_star:.3f}$')
    ax1.axhline(summary['theta_mean_induced'], color='#d62728', linestyle='--', linewidth=1.5, label=rf'Biased Limit $\bar{{\theta}}_{{\mathrm{{mean}}}} = {summary["theta_mean_induced"]:.3f}$')
    
    ax1.set_ylabel(r'Final Parameter Estimate $\theta_{60}$')
    ax1.set_title(r'(a) Distribution of $\theta_{60}$ Across 200 Trajectories', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.4, axis='y')
    ax1.legend(loc='upper right', fontsize=9.0)
    ax1.tick_params(axis='x', rotation=15)

    # Panel (b): Bias-Variance Decomposition (Stacked Bar Chart)
    x_pos = np.arange(len(labels))
    bar_width = 0.55

    p1 = ax2.bar(x_pos, biases_sq, bar_width, label=r'$\mathrm{Bias}^2 = (\bar{\theta}_{60} - \theta^\star)^2$',
                 color='#d62728', alpha=0.8, edgecolor='black', hatch='//')
    p2 = ax2.bar(x_pos, variances, bar_width, bottom=biases_sq, label=r'$\mathrm{Variance} = \mathrm{Var}(\theta_{60})$',
                 color='#1f77b4', alpha=0.8, edgecolor='black', hatch='\\')

    # Add numeric annotations on top of bars
    for i in range(len(labels)):
        tot = biases_sq[i] + variances[i]
        b_pct = (biases_sq[i] / tot) * 100 if tot > 0 else 0
        v_pct = (variances[i] / tot) * 100 if tot > 0 else 0
        ax2.annotate(
            f'MSE: {tot:.4f}\n'
            f'({b_pct:.0f}% B / {v_pct:.0f}% V)',
                     xy=(x_pos[i], tot), xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel(r'Decomposed Error Metric')
    ax2.set_title(r'(b) Bias-Variance Decomposition of Final MSE', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    if total_mses:
        ax2.set_ylim(0, max(total_mses) * 1.25)
    
    ax2.legend(loc='upper left', fontsize=9.0)
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, 'fig4_bias_variance_decomposition.png'), dpi=300)
        fig.savefig(os.path.join(out_dir, 'fig4_bias_variance_decomposition.pdf'))
    plt.close(fig)
    print(f"Saved Figure 4. (Modes: {mode_order})")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready figures for CTMC FL SGD.")
    parser.add_argument('--csv', type=str, 
                        default='ctmc_driven_quadratic_sgd_palm_corrected_results/ctmc_sgd_results.csv', 
                        help='Path to results CSV')
    parser.add_argument('--json', type=str, 
                        default='ctmc_driven_quadratic_sgd_palm_corrected_results/ctmc_sgd_summary.json', 
                        help='Path to summary JSON')
    parser.add_argument('--out_dirs', nargs='+', 
                        default=['ctmc_driven_quadratic_sgd_palm_corrected_results/figures/'], 
                        help='Output directories for figures')
    parser.add_argument('--modes', nargs='+', 
                        default=DEFAULT_MODES,
                        choices=ALL_MODES,
                        help=f'Modes to plot (default: {DEFAULT_MODES})')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output')
    args = parser.parse_args()

    print(f"Loading data from {args.csv} and {args.json}...")
    print(f"Modes selected: {args.modes}")
    
    df, summary = load_data(args.csv, args.json)

    print("Generating Figure 1: Iteration Convergence...")
    plot_figure_1(df, summary, args.out_dirs, args.modes)

    print("Generating Figure 2: Wall-Clock Physical Time Convergence...")
    plot_figure_2(df, summary, args.out_dirs, args.modes)

    print("Generating Figure 3: Bias Diagnosis & Asymptotic Fixed Points...")
    plot_figure_3(df, summary, args.out_dirs, args.modes)

    print("Generating Figure 4: Parameter Distribution & Bias-Variance Decomposition...")
    plot_figure_4(df, summary, args.out_dirs, args.modes)

    print("All publication figures generated successfully in:")
    for out_dir in args.out_dirs:
        print(f" - {out_dir}")


if __name__ == '__main__':
    main()