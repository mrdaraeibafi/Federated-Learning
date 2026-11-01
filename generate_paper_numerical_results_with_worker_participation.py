#!/usr/bin/env python3
"""
================================================================================
Comprehensive Numerical Analysis and Paper Artifact Generator
Paper: "Coupled Queueing-Optimization Dynamics in K-Batch-Sync Federated Learning"
================================================================================

Outputs generated in directory 'paper_numerical_results/':
  - Figures (PNG @ 300 DPI & Vector PDF):
      1a. fig1a_throughput_validation: CTMC vs SimPy Aggregation Throughput
      1b. fig1b_bias_validation: CTMC vs SimPy Palm Participation Bias
      2.  fig2_convergence_trajectories: Iteration vs Wall-Clock (CTMC + SimPy)
      3a. fig3a_throughput_sweep: Aggregation Rate vs K (CTMC + SimPy)
      3b. fig3b_bias_sweep: Palm Bias vs K (CTMC + SimPy)
      4.  fig4_pareto_frontier: Wall-Clock Delay vs Terminal Model Inaccuracy
      5.  fig5_hol_blocking: Multi-Job Head-of-Line Blocking Effect
  - Standalone LaTeX Tables: 4 files
  - Numerical Artifact: summary_numerical_results.json
"""

import os
import sys
import json
import itertools
import heapq
import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional

np.random.seed(42)

# ------------------------------------------------------------------
# INCREASED FONT SIZES FOR FIGURES
# ------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 15,
    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 19,
    'lines.linewidth': 2.4,
    'lines.markersize': 8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = "paper_numerical_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. LUMPED CTMC ANALYTICAL ENGINE
# ==============================================================================

@dataclass
class CTMCAnalyticalResult:
    b: int
    B: int
    K: int
    num_states: int
    aggregation_rate: float
    mean_cycle_time: float
    group_weights: np.ndarray
    palm_participation_mean: np.ndarray
    palm_covariance: np.ndarray
    beta2_bias: float
    beta2_variance: float
    beta2_total: float
    stationary_dist: np.ndarray
    states: List[Tuple[int, ...]]


class LumpedKBatchSyncCTMC:
    def __init__(self, groups, b, B, K):
        self.groups = groups
        self.G = len(groups)
        self.N_g = np.array([g[0] for g in groups], dtype=int)
        self.mu_g = np.array([g[1] for g in groups], dtype=float)
        self.N_total = int(np.sum(self.N_g))
        self.group_weights = self.N_g / self.N_total
        self.b = b
        self.B = B
        self.K = K
        self.max_score = int(np.sum(self.N_g * min(self.b, self.B)))
        if self.K > self.max_score:
            raise ValueError(f"K={K} exceeds max score {self.max_score}")

    def _group_occupancies(self, count):
        res = []
        def _bt(level, remaining, current):
            if level == self.b:
                res.append(tuple(current + [remaining]))
                return
            for c in range(remaining + 1):
                _bt(level + 1, remaining - c, current + [c])
        _bt(0, count, [])
        return res

    def aggregation_score(self, state):
        score = 0
        idx = 0
        for g in range(self.G):
            for j in range(self.b + 1):
                if j > 0:
                    score += min(j, self.B) * state[idx]
                idx += 1
        return score

    def _enumerate_states(self):
        group_occ = [self._group_occupancies(self.N_g[g]) for g in range(self.G)]
        valid = []
        for comb in itertools.product(*group_occ):
            flat = tuple(itertools.chain.from_iterable(comb))
            if self.aggregation_score(flat) < self.K:
                valid.append(flat)
        return sorted(valid)

    def restart_state(self, state_before):
        out = []
        idx = 0
        for g in range(self.G):
            gc = [0] * (self.b + 1)
            for j in range(self.b + 1):
                cnt = state_before[idx]
                new_level = max(j - self.B, 0)
                gc[new_level] += cnt
                idx += 1
            out.extend(gc)
        return tuple(out)

    def solve(self):
        states = self._enumerate_states()
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}
        Q = np.zeros((n_states, n_states), dtype=float)
        trigger_fluxes = []

        for i, s in enumerate(states):
            row_sum = 0.0
            idx = 0
            for g in range(self.G):
                for j in range(self.b + 1):
                    cnt = s[idx]
                    if cnt > 0 and j < self.b:
                        rate = cnt * self.mu_g[g]
                        sp = list(s)
                        sp[idx] -= 1
                        sp[idx + 1] += 1
                        sp_t = tuple(sp)
                        score_plus = self.aggregation_score(sp_t)
                        if score_plus < self.K:
                            Q[i, state_to_idx[sp_t]] += rate
                        else:
                            sr = self.restart_state(sp_t)
                            Q[i, state_to_idx[sr]] += rate
                            u_vec = np.zeros(self.G)
                            gi = 0
                            for gg in range(self.G):
                                for jj in range(self.b + 1):
                                    if jj > 0:
                                        u_vec[gg] += min(jj, self.B) * sp_t[gi]
                                    gi += 1
                            trigger_fluxes.append((i, rate, u_vec))
                        row_sum += rate
                    idx += 1
            Q[i, i] = -row_sum

        A_mat = np.copy(Q.T)
        A_mat[-1, :] = 1.0
        b_vec = np.zeros(n_states)
        b_vec[-1] = 1.0
        pi = np.linalg.solve(A_mat, b_vec)
        pi = np.maximum(pi, 0.0)
        pi /= np.sum(pi)

        agg_rate = 0.0
        wpm = np.zeros(self.G)
        wps = np.zeros((self.G, self.G))
        for from_idx, rate, u_vec in trigger_fluxes:
            flux = pi[from_idx] * rate
            agg_rate += flux
            beta_inst = u_vec / np.sum(u_vec)
            wpm += flux * beta_inst
            wps += flux * np.outer(beta_inst, beta_inst)

        if agg_rate > 0:
            palm_mean = wpm / agg_rate
            palm_second = wps / agg_rate
            palm_cov = palm_second - np.outer(palm_mean, palm_mean)
        else:
            palm_mean = np.copy(self.group_weights)
            palm_cov = np.zeros((self.G, self.G))

        beta2_bias = float(np.sum((palm_mean - self.group_weights) ** 2))
        beta2_variance = float(np.trace(palm_cov))
        beta2_total = beta2_bias + beta2_variance

        return CTMCAnalyticalResult(
            b=self.b, B=self.B, K=self.K,
            num_states=n_states,
            aggregation_rate=float(agg_rate),
            mean_cycle_time=float(1.0 / agg_rate) if agg_rate > 0 else float('inf'),
            group_weights=self.group_weights,
            palm_participation_mean=palm_mean,
            palm_covariance=palm_cov,
            beta2_bias=beta2_bias,
            beta2_variance=beta2_variance,
            beta2_total=beta2_total,
            stationary_dist=pi,
            states=states
        )


# ==============================================================================
# 2. MICROSCOPIC SIMULATION ENGINE
# ==============================================================================

class MicroscopicKBatchSyncSimulator:
    def __init__(self, groups, b, B, K):
        self.groups = groups
        self.b = b
        self.B = B
        self.K = K
        self.G = len(groups)
        self.workers = []
        for g_idx, (count, mu) in enumerate(groups):
            for _ in range(count):
                self.workers.append((g_idx, mu))
        self.N = len(self.workers)

    def run(self, max_aggregations=1500, warmup=200):
        buffers = np.zeros(self.N, dtype=int)
        current_time = 0.0
        event_queue = []
        for w_id in range(self.N):
            g_idx, mu = self.workers[w_id]
            heapq.heappush(event_queue, (np.random.exponential(1.0 / mu), w_id))

        agg_count = 0
        last_agg_time = 0.0
        inter_agg_times = []
        participation_records = []
        worker_participation_records = []
        agg_times_series = []

        while agg_count < max_aggregations:
            t_event, w_id = heapq.heappop(event_queue)
            current_time = t_event
            g_idx, mu = self.workers[w_id]

            if buffers[w_id] < self.b:
                buffers[w_id] += 1

            effective_contrib = np.minimum(buffers, self.B)
            if np.sum(effective_contrib) >= self.K:
                if agg_count >= warmup:
                    inter_agg_times.append(current_time - last_agg_time)
                    agg_times_series.append(current_time)
                    u_g = np.zeros(self.G)
                    for wid_i in range(self.N):
                        u_g[self.workers[wid_i][0]] += effective_contrib[wid_i]
                    participation_records.append(u_g / np.sum(u_g))
                    # Normalize each worker's effective contribution within this aggregation.
                    worker_participation_records.append(
                        effective_contrib / np.sum(effective_contrib)
                    )
                last_agg_time = current_time
                agg_count += 1
                buffers = np.maximum(buffers - self.B, 0)

            heapq.heappush(event_queue,
                           (current_time + np.random.exponential(1.0 / mu), w_id))

        inter_arr = np.array(inter_agg_times)
        part_arr = np.array(participation_records)
        worker_part_arr = np.array(worker_participation_records)
        mean_dt = float(np.mean(inter_arr))
        stderr_dt = float(np.std(inter_arr, ddof=1) / np.sqrt(len(inter_arr)))
        ci95 = (mean_dt - 1.96 * stderr_dt, mean_dt + 1.96 * stderr_dt)
        sim_rate = 1.0 / mean_dt
        sim_palm_mean = np.mean(part_arr, axis=0)
        sim_palm_cov = np.cov(part_arr, rowvar=False) if len(part_arr) > 1 else np.zeros((self.G, self.G))
        sim_bias = float(np.sum((sim_palm_mean - np.array([g[0] for g in self.groups]) / self.N) ** 2))
        sim_var = float(np.trace(sim_palm_cov))

        return {
            "mean_cycle_time": mean_dt,
            "ci95_cycle_time": ci95,
            "aggregation_rate": sim_rate,
            "palm_mean": sim_palm_mean,
            "palm_covariance": sim_palm_cov,
            "beta2_bias": sim_bias,
            "beta2_variance": sim_var,
            "beta2_total": sim_bias + sim_var,
            "inter_agg_times": inter_arr,
            "participation_records": part_arr,
            "worker_participation_records": worker_part_arr,
            "worker_participation_mean": np.mean(worker_part_arr, axis=0),
            "agg_times_series": np.array(agg_times_series),
            "num_aggregations": agg_count - warmup,
        }


# ==============================================================================
# 3. COUPLED QUADRATIC SGD PROBE
# ==============================================================================

class QuadraticSGDProbe:
    def __init__(self, group_weights, a_g, c_g, sigma_g, lr=0.05):
        self.w = group_weights
        self.a = a_g
        self.c = c_g
        self.sigma = sigma_g
        self.lr = lr
        self.theta_star = float(np.sum(self.w * self.a * self.c) / np.sum(self.w * self.a))

    def with_k(self, K):
        """Returns a clone of probe with noise scaled as sigma / sqrt(K) (IID mini-batch scaling)."""
        scaled_sigma = self.sigma / np.sqrt(K)
        return QuadraticSGDProbe(self.w, self.a, self.c, scaled_sigma, lr=self.lr)

    def biased_optimum(self, palm_mean):
        return float(np.sum(palm_mean * self.a * self.c) / np.sum(palm_mean * self.a))

    def simulate_trajectory_ctmc(self, ctmc_res, steps=150, trajectories=200, theta0=0.0):
        theta_arr = np.zeros((trajectories, steps + 1))
        theta_arr[:, 0] = theta0
        dt_mean = ctmc_res.mean_cycle_time
        time_axis = np.arange(steps + 1) * dt_mean

        for t in range(steps):
            cov_reg = ctmc_res.palm_covariance + 1e-6 * np.eye(len(self.w))
            p_samples = np.random.multivariate_normal(ctmc_res.palm_participation_mean, cov_reg, size=trajectories)
            p_samples = np.maximum(p_samples, 0.0)
            p_samples /= np.sum(p_samples, axis=1, keepdims=True)
            theta_t = theta_arr[:, t]
            grad = np.zeros(trajectories)
            for g in range(len(self.w)):
                noise = np.random.normal(0.0, self.sigma[g], size=trajectories)
                grad += p_samples[:, g] * (self.a[g] * (theta_t - self.c[g]) + noise)
            theta_arr[:, t + 1] = theta_t - self.lr * grad

        mse = (theta_arr - self.theta_star) ** 2
        return {
            "iteration": np.arange(steps + 1),
            "wall_clock_time": time_axis,
            "mean_mse": np.mean(mse, axis=0),
            "p10_mse": np.percentile(mse, 10, axis=0),
            "p90_mse": np.percentile(mse, 90, axis=0),
            "terminal_mse": float(np.mean(mse[:, -1])),
            "asymptotic_bias_sq": (self.biased_optimum(ctmc_res.palm_participation_mean) - self.theta_star) ** 2
        }

    def simulate_trajectory_simpy(self, sim_res, steps=150, trajectories=200, theta0=0.0):
        theta_arr = np.zeros((trajectories, steps + 1))
        theta_arr[:, 0] = theta0
        inter_times = sim_res["inter_agg_times"]
        time_axis = np.zeros(steps + 1)
        for i in range(1, steps + 1):
            time_axis[i] = time_axis[i-1] + np.mean(inter_times)
        part_records = sim_res["participation_records"]

        for t in range(steps):
            idx_s = np.random.choice(len(part_records), size=trajectories, replace=True)
            p_samples = part_records[idx_s]
            theta_t = theta_arr[:, t]
            grad = np.zeros(trajectories)
            for g in range(len(self.w)):
                noise = np.random.normal(0.0, self.sigma[g], size=trajectories)
                grad += p_samples[:, g] * (self.a[g] * (theta_t - self.c[g]) + noise)
            theta_arr[:, t + 1] = theta_t - self.lr * grad

        mse = (theta_arr - self.theta_star) ** 2
        biased_opt = self.biased_optimum(sim_res["palm_mean"])
        return {
            "iteration": np.arange(steps + 1),
            "wall_clock_time": time_axis,
            "mean_mse": np.mean(mse, axis=0),
            "p10_mse": np.percentile(mse, 10, axis=0),
            "p90_mse": np.percentile(mse, 90, axis=0),
            "terminal_mse": float(np.mean(mse[:, -1])),
            "asymptotic_bias_sq": (biased_opt - self.theta_star) ** 2
        }

    # def dutta_theoretical_bound(self, K, steps=120, theta0=2.5, m=1):
    #     """
    #     Calculates theoretical MSE bound from Dutta et al. (Lemma 2):
    #     MSE(k) = (eta * sigma^2) / (K * m) + (1 - eta * a)^k * ((theta0 - theta*)^2 - (eta * sigma^2)/(K * m))
    #     """
    #     a_eff = np.mean(self.a)  # In IID case, a_eff = 1.0
    #     c_eff = a_eff            # Strongly convex parameter c = L = a
    #     sigma_base = self.sigma[0]  # Base noise variance before scaling
        
    #     # Asymptotic variance floor for K mini-batches
    #     var_floor = (self.lr * (sigma_base ** 2)) / (c_eff * K * m)
    #     init_gap = (theta0 - self.theta_star) ** 2
        
    #     k_axis = np.arange(steps + 1)
    #     contraction = (1.0 - self.lr * c_eff) ** k_axis
    #     mse_bound = var_floor + contraction * (init_gap - var_floor)
    #     return k_axis, mse_bound
    
    def dutta_theoretical_bound(self, K, steps=120, theta0=2.5, tau=None):
        """
        Calculates theoretical MSE bound according to Lemma 2 in Dutta et al. (AISTATS 2018):
        MSE(k) = (eta * sigma^2) / K + (1 - eta)^k * ((theta0 - theta*)^2 - (eta * sigma^2)/K)
        """
        k_axis = np.arange(steps + 1)
        
        a_val = float(self.a[0]) if hasattr(self.a, '__len__') else float(self.a)
        sigma_val = float(self.sigma[0]) if hasattr(self.sigma, '__len__') else float(self.sigma)
        
        var_floor = (self.lr * (sigma_val ** 2)) / K
        init_gap = (theta0 - self.theta_star) ** 2
        
        rho = 1.0 - self.lr * a_val
        mse_bound = var_floor + (rho ** k_axis) * (init_gap - var_floor)
        
        res = {
            "steps": k_axis,
            "mse": mse_bound
        }
        
        if tau is not None and tau > 0:
            res["time"] = k_axis * tau
            
        return res

# ==============================================================================
# 4. MULTI-JOB HEAD-OF-LINE BLOCKING SIMULATOR
# ==============================================================================

class MultiJobHoLSimulator:
    def __init__(self, groups, b, B, K, J=3):
        self.groups = groups
        self.b = b
        self.B = B
        self.K = K
        self.J = J
        self.G = len(groups)
        self.workers = []
        for g_idx, (count, mu) in enumerate(groups):
            for _ in range(count):
                self.workers.append((g_idx, mu))
        self.N = len(self.workers)

    def run(self, max_rounds_per_job=25, seed_offset=0):
        rng = np.random.RandomState(42 + seed_offset)
        buffers = np.zeros((self.N, self.J), dtype=int)
        current_time = 0.0
        event_queue = []
        for w_id in range(self.N):
            g_idx, mu = self.workers[w_id]
            heapq.heappush(event_queue, (rng.exponential(1.0 / mu), w_id))

        job_rounds = np.zeros(self.J, dtype=int)
        job_mse = [[] for _ in range(self.J)]
        job_time = [[] for _ in range(self.J)]
        q_active = 0

        for q in range(self.J):
            job_mse[q].append(1.0)
            job_time[q].append(0.0)

        job_activation_time = [0.0] * self.J
        job_activation_time[0] = 0.0

        while q_active < self.J and len(event_queue) > 0:
            t_event, w_id = heapq.heappop(event_queue)
            current_time = t_event
            g_idx, mu = self.workers[w_id]

            if buffers[w_id, q_active] < self.b:
                buffers[w_id, q_active] += 1

            effective = np.minimum(buffers[:, q_active], self.B)
            if np.sum(effective) >= self.K:
                job_rounds[q_active] += 1
                n_contrib = np.sum(effective > 0)
                decay = 0.15 * (n_contrib / self.N)
                curr_mse = job_mse[q_active][-1]
                new_mse = max(curr_mse * (1.0 - decay) + 1e-5 * rng.randn()**2, 1e-8)
                job_mse[q_active].append(new_mse)
                job_time[q_active].append(current_time)

                buffers[:, q_active] = np.maximum(buffers[:, q_active] - self.B, 0)

                if job_rounds[q_active] >= max_rounds_per_job:
                    q_active += 1
                    if q_active < self.J:
                        job_activation_time[q_active] = current_time

            heapq.heappush(event_queue,
                           (current_time + rng.exponential(1.0 / mu), w_id))

        return {
            "job_mse": job_mse,
            "job_time": job_time,
            "job_rounds": job_rounds,
            "job_activation_time": job_activation_time,
            "final_time": current_time,
            "J": self.J,
        }


# ==============================================================================
# 5. EXPERIMENTAL SUITE
# ==============================================================================

def plot_worker_participation_weights(validation_results, groups, output_dir):
    """Plot per-worker CTMC and SimPy participation weights for the three K=6 cases."""
    target_configs = [(1, 1, 1), (1, 1, 6), (1, 1, 10)]
    selected = []
    for target in target_configs:
        matches = [item for item in validation_results
                   if (item["config"]["b"], item["config"]["B"],
                       item["config"]["K"]) == target]
        if len(matches) != 1:
            raise ValueError(f"Expected one validation result for {target}; got {len(matches)}")
        selected.append(matches[0])

    worker_group = np.concatenate([
        np.full(count, group_idx, dtype=int)
        for group_idx, (count, _) in enumerate(groups)
    ])
    worker_ids = np.arange(len(worker_group))
    group_names = ["Fast", "Medium", "Slow"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    bar_width = 0.36
    for ax, result, target in zip(axes, selected, target_configs):
        ctmc_result = result["ctmc"]
        sim_result = result["sim"]
        ctmc_worker = np.concatenate([
            np.full(count, ctmc_result.palm_participation_mean[group_idx] / count)
            for group_idx, (count, _) in enumerate(groups)
        ])
        records = np.asarray(sim_result["worker_participation_records"])
        sim_worker = np.mean(records, axis=0)
        if len(records) > 1:
            sim_se = np.std(records, axis=0, ddof=1) / np.sqrt(len(records))
            sim_ci = 1.96 * sim_se
        else:
            sim_ci = np.zeros(records.shape[1])
        x = worker_ids.astype(float)
        ax.bar(x - bar_width/2, ctmc_worker, bar_width, label="CTMC",
               color="#4c78a8", edgecolor="black", linewidth=0.5)
        ax.bar(x + bar_width/2, sim_worker, bar_width, label="SimPy",
               color="#f58518", edgecolor="black", linewidth=0.5,
               yerr=sim_ci, capsize=2, error_kw={"elinewidth": 1})
        ax.axhline(1.0 / len(worker_ids), color="#d62728", linestyle="--",
                   linewidth=1.8, label="Fair participation (1/N = 0.1)")
        ax.set_title(f"(b, B, K) = {target}")
        ax.set_xlabel("Worker ID (group)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"W{i}\n{group_names[g]}" for i, g in enumerate(worker_group)],
                           fontsize=9)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
    axes[0].set_ylabel("Mean normalized participation weight")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle("Per-Worker Participation Weights: CTMC vs SimPy", y=1.10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1c_worker_participation_weights.png"))
    fig.savefig(os.path.join(output_dir, "fig1c_worker_participation_weights.pdf"))
    plt.close(fig)
    print("  -> Saved fig1c_worker_participation_weights")


def run_all_experiments():
    print("=" * 80)
    print("STARTING HETEROGENEOUS K-BATCH-SYNC COMPREHENSIVE NUMERICAL SUITE")
    print("=" * 80)

    groups = [(3, 2.0), (4, 1.0), (3, 0.5)]
    N_total = sum(g[0] for g in groups)
    group_weights = np.array([g[0] for g in groups]) / N_total

    # a_g = np.array([1.5, 1.0, 0.8])
    # c_g = np.array([1.0, 0.0, -1.0])
    # sigma_g = np.array([0.1, 0.15, 0.2])
    # sgd_probe = QuadraticSGDProbe(group_weights, a_g, c_g, sigma_g, lr=0.08)

    # IID Data Configuration: All workers share identical local objectives and noise level
    a_g = np.full(len(groups), 1.0)       # Homogeneous curvature (a = 1.0)
    c_g = np.full(len(groups), 0.0)       # Uniform global optimum (theta* = 0.0)
    sigma_base = 0.15
    sigma_g = np.full(len(groups), sigma_base)  # Identical stochastic gradient noise
    sgd_probe = QuadraticSGDProbe(group_weights, a_g, c_g, sigma_g, lr=0.08)

    print(f"System Configuration: N={N_total}, Groups={groups}")
    print(f"Probe True Global Optimum theta* = {sgd_probe.theta_star:.4f}")

    # ==========================================================================
    # EXPERIMENT 1: Validation
    # ==========================================================================
    print("\n[1/5] Validation: Lumped CTMC vs Microscopic SimPy...")
    val_configs = [
        {"b": 1, "B": 1, "K": 1, "label": "(1,1,1)"},
        {"b": 1, "B": 1, "K": 3, "label": "(1,1,3)"},
        {"b": 1, "B": 1, "K": 6, "label": "(1,1,6)"},
        {"b": 1, "B": 1, "K": 8, "label": "(1,1,8)"},
        {"b": 1, "B": 1, "K": 10, "label": "(1,1,10)"},
        {"b": 3, "B": 1, "K": 3, "label": "(3,1,3)"},
        {"b": 3, "B": 1, "K": 6, "label": "(3,1,6)"},
        {"b": 3, "B": 3, "K": 6, "label": "(3,3,6)"},
        {"b": 3, "B": 3, "K": 12, "label": "(3,3,12)"},
    ]

    val_results = []
    for cfg in val_configs:
        ctmc = LumpedKBatchSyncCTMC(groups, cfg["b"], cfg["B"], cfg["K"])
        res_ctmc = ctmc.solve()
        sim = MicroscopicKBatchSyncSimulator(groups, cfg["b"], cfg["B"], cfg["K"])
        res_sim = sim.run(max_aggregations=1200, warmup=200)
        err = abs(res_ctmc.aggregation_rate - res_sim["aggregation_rate"]) / res_ctmc.aggregation_rate * 100.0
        val_results.append({"config": cfg, "ctmc": res_ctmc, "sim": res_sim, "err_rel": err})
        print(f"  -> {cfg['label']}: CTMC={res_ctmc.aggregation_rate:.4f} | "
              f"SimPy={res_sim['aggregation_rate']:.4f} (Err: {err:.2f}%)")

    # Additive Figure 1c: reuse the aligned K=6 results already computed above.
    plot_worker_participation_weights(val_results, groups, OUTPUT_DIR)

    # ==========================================================================
    # EXPERIMENT 2: Parametric Sweep
    # ==========================================================================
    print("\n[2/5] Parametric Sweep over (b, B, K) - CTMC AND SimPy...")
    sweep_families = [
        {"b": 1, "B": 1, "name": "K-Sync (b=1,B=1)", "color": "#1f77b4", "marker": "o", "k_max": 9},
        {"b": 3, "B": 1, "name": "K-Batch-Sync (b=3,B=1)", "color": "#2ca02c", "marker": "s", "k_max": 9},
        {"b": 3, "B": 3, "name": "K-Batch-Sync (b=3,B=3)", "color": "#d62728", "marker": "^", "k_max": 24},
    ]

    sweep_data = {}
    for fam in sweep_families:
        key = f"b{fam['b']}_B{fam['B']}"
        sweep_data[key] = {
            "family": fam,
            "K": [], "rate_ctmc": [], "rate_sim": [],
            "cycle_ctmc": [], "cycle_sim": [],
            "bias_ctmc": [], "bias_sim": [],
            "var_ctmc": [], "var_sim": [],
            "mse_ctmc": [], "mse_sim": [],
        }
        for k in range(1, fam["k_max"] + 1):
            try:
                probe_k = sgd_probe.with_k(k)
                ctmc = LumpedKBatchSyncCTMC(groups, fam["b"], fam["B"], k)
                res = ctmc.solve()
                # sgd_ctmc = sgd_probe.simulate_trajectory_ctmc(res, steps=100, trajectories=100)
                sgd_ctmc = probe_k.simulate_trajectory_ctmc(res, steps=100, trajectories=100)

                sim = MicroscopicKBatchSyncSimulator(groups, fam["b"], fam["B"], k)
                res_sim = sim.run(max_aggregations=800, warmup=150)
                # sgd_sim = sgd_probe.simulate_trajectory_simpy(res_sim, steps=100, trajectories=100)
                sgd_sim = probe_k.simulate_trajectory_simpy(res_sim, steps=100, trajectories=100)

                sweep_data[key]["K"].append(k)
                sweep_data[key]["rate_ctmc"].append(res.aggregation_rate)
                sweep_data[key]["rate_sim"].append(res_sim["aggregation_rate"])
                sweep_data[key]["cycle_ctmc"].append(res.mean_cycle_time)
                sweep_data[key]["cycle_sim"].append(res_sim["mean_cycle_time"])
                sweep_data[key]["bias_ctmc"].append(res.beta2_bias)
                sweep_data[key]["bias_sim"].append(res_sim["beta2_bias"])
                sweep_data[key]["var_ctmc"].append(res.beta2_variance)
                sweep_data[key]["var_sim"].append(res_sim["beta2_variance"])
                sweep_data[key]["mse_ctmc"].append(sgd_ctmc["terminal_mse"])
                sweep_data[key]["mse_sim"].append(sgd_sim["terminal_mse"])
            except Exception as e:
                print(f"  Skipping {key}, k={k}: {e}")

    # ==========================================================================
    # EXPERIMENT 3: SGD Trajectories
    # ==========================================================================
    print("\n[3/5] SGD Trajectories - CTMC AND SimPy...")
    traj_cases = [
        {"b": 1, "B": 1, "K": 2, "label": "K-Sync (K=2)", "color": "#1f77b4", "ls": "--"},
        {"b": 1, "B": 1, "K": 8, "label": "K-Sync (K=8)", "color": "#ff7f0e", "ls": "-."},
        {"b": 3, "B": 1, "K": 6, "label": "K-Batch-Sync (b=3,K=6)", "color": "#2ca02c", "ls": "-"},
        {"b": 3, "B": 3, "K": 10, "label": "K-Batch-Sync (b=3,B=3,K=10)", "color": "#d62728", "ls": "-"},
    ]

    traj_results = []
    for tc in traj_cases:
        probe_case = sgd_probe.with_k(tc["K"])
        ctmc = LumpedKBatchSyncCTMC(groups, tc["b"], tc["B"], tc["K"])
        res_ctmc = ctmc.solve()
        # traj_ctmc = sgd_probe.simulate_trajectory_ctmc(res_ctmc, steps=120, trajectories=1000, theta0=2.5)
        traj_ctmc = probe_case.simulate_trajectory_ctmc(res_ctmc, steps=120, trajectories=1000, theta0=2.5)

        sim = MicroscopicKBatchSyncSimulator(groups, tc["b"], tc["B"], tc["K"])
        res_sim = sim.run(max_aggregations=1000, warmup=150)
        # traj_sim = sgd_probe.simulate_trajectory_simpy(res_sim, steps=120, trajectories=1000, theta0=2.5)
        traj_sim = probe_case.simulate_trajectory_simpy(res_sim, steps=120, trajectories=1000, theta0=2.5)

        traj_results.append({"case": tc, "ctmc": res_ctmc, "sim": res_sim,
                             "traj_ctmc": traj_ctmc, "traj_sim": traj_sim})

    #
    print("\n" + "="*85)
    print(f"{'Scheme / Configuration':<30} | {'K':<4} | {'Simulated MSE':<20} | {'Dutta Theoretical Bound':<25}")
    print("="*85)

    for res in traj_results:
        tc = res["case"]
        k_val = tc["K"]
        name = tc["label"]
        
        final_sim_mse = res["traj_sim"]["mean_mse"][-1]
        
        probe_case = sgd_probe.with_k(k_val)
        dutta_res = probe_case.dutta_theoretical_bound(K=k_val, steps=len(res["traj_sim"]["iteration"]) - 1, theta0=2.5)
        final_dutta_bound = dutta_res["mse"][-1]
        
        res["traj_dutta"] = dutta_res
        
        print(f"{name:<30} | {k_val:<4} | {final_sim_mse:<20.6e} | {final_dutta_bound:<25.6e}")

    print("="*85)
    #

    # ==========================================================================
    # EXPERIMENT 4: Pareto Frontier
    # ==========================================================================
    print("\n[4/5] Pareto Frontier - CTMC AND SimPy...")
    pareto_points = []
    for key, data in sweep_data.items():
        for idx in range(len(data["K"])):
            k = data["K"][idx]
            pareto_points.append({
                "family": data["family"]["name"],
                "color": data["family"]["color"],
                "marker": data["family"]["marker"],
                "K": k,
                "time_ctmc": 40.0 * data["cycle_ctmc"][idx],
                "time_sim": 40.0 * data["cycle_sim"][idx],
                "mse_ctmc": data["mse_ctmc"][idx],
                "mse_sim": data["mse_sim"][idx],
                "bias_ctmc": data["bias_ctmc"][idx],
                "bias_sim": data["bias_sim"][idx],
            })

    # ==========================================================================
    # EXPERIMENT 5: Multi-Job HoL Blocking
    # ==========================================================================
    print("\n[5/5] Multi-Job Head-of-Line Blocking...")
    hol_results = []
    for seed_off in range(5):
        hol_sim = MultiJobHoLSimulator(groups, b=3, B=1, K=6, J=3)
        hol_res = hol_sim.run(max_rounds_per_job=25, seed_offset=seed_off)
        hol_results.append(hol_res)

    # ==========================================================================
    # 6. GENERATE PUBLICATION FIGURES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 80)

    val_labels = [v["config"]["label"] for v in val_results]
    x_idx = np.arange(len(val_labels))
    width = 0.35

    ctmc_rates = [v["ctmc"].aggregation_rate for v in val_results]
    sim_rates = [v["sim"]["aggregation_rate"] for v in val_results]
    sim_errs = []
    for v in val_results:
        ci_low, ci_high = v["sim"]["ci95_cycle_time"]
        rate_low = 1.0 / ci_high
        rate_high = 1.0 / ci_low
        sim_errs.append((rate_high - rate_low) / 2)

    ctmc_biases = [v["ctmc"].beta2_bias for v in val_results]
    sim_biases = [v["sim"]["beta2_bias"] for v in val_results]

    # --------------- FIGURE 1a: Throughput Validation ---------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x_idx - width/2, ctmc_rates, width, label='CTMC (Analytical)',
           color='#1f77b4', alpha=0.9, edgecolor='black')
    ax.bar(x_idx + width/2, sim_rates, width, yerr=sim_errs, capsize=4,
           label='Microscopic SimPy (95% CI)', color='#ff7f0e', alpha=0.9,
           edgecolor='black')
    ax.set_xlabel(r'System Configuration $(b, B, K)$')
    ax.set_ylabel(r'Aggregation Throughput $A$ [events/sec]')
    ax.set_title(r'Throughput Validation: CTMC vs SimPy')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(val_labels, rotation=35)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1a_throughput_validation.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1a_throughput_validation.pdf"))
    plt.close()
    print("  -> Saved fig1a_throughput_validation")

    # --------------- FIGURE 1b: Bias Validation ---------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(x_idx, ctmc_biases, 'o-', color='#2ca02c',
            label=r'CTMC $\beta^2_{\mathrm{bias}}$', markersize=10)
    ax.plot(x_idx, sim_biases, 's--', color='#d62728',
            label=r'SimPy $\beta^2_{\mathrm{bias}}$', markersize=10)
    ax.set_xlabel(r'System Configuration $(b, B, K)$')
    ax.set_ylabel(r'Palm Participation Bias $\beta^2_{\mathrm{bias}}$')
    ax.set_title(r'Palm Participation Bias Validation')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(val_labels, rotation=35)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1b_bias_validation.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1b_bias_validation.pdf"))
    plt.close()
    print("  -> Saved fig1b_bias_validation")

    # --------------- FIGURE 2: SGD Convergence (CTMC + SimPy) ---------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    (ax1, ax2) = axes

    for tr in traj_results:
        tc = tr["case"]
        # CTMC (solid)
        ax1.plot(tr["traj_ctmc"]["iteration"], tr["traj_ctmc"]["mean_mse"],
                 label=f'{tc["label"]} [CTMC]', color=tc["color"], linestyle=tc["ls"])
        ax1.fill_between(tr["traj_ctmc"]["iteration"],
                         tr["traj_ctmc"]["p10_mse"], tr["traj_ctmc"]["p90_mse"],
                         color=tc["color"], alpha=0.08)
        # SimPy (dotted)
        ax1.plot(tr["traj_sim"]["iteration"], tr["traj_sim"]["mean_mse"],
                 label=f'{tc["label"]} [SimPy]', color=tc["color"],
                 linestyle=':', marker='.', markersize=4, alpha=0.8)
        ax1.plot(
            tr["traj_dutta"]["steps"], 
            tr["traj_dutta"]["mse"],
            color=tc["color"], 
            linestyle=':', 
            linewidth=1.5, 
            alpha=0.85,
            label=f'{tc["label"]} (Dutta Bound)'
        )

        ax2.plot(tr["traj_ctmc"]["wall_clock_time"], tr["traj_ctmc"]["mean_mse"],
                 label=f'{tc["label"]} [CTMC]', color=tc["color"], linestyle=tc["ls"])
        ax2.fill_between(tr["traj_ctmc"]["wall_clock_time"],
                         tr["traj_ctmc"]["p10_mse"], tr["traj_ctmc"]["p90_mse"],
                         color=tc["color"], alpha=0.08)
        ax2.plot(tr["traj_sim"]["wall_clock_time"], tr["traj_sim"]["mean_mse"],
                 label=f'{tc["label"]} [SimPy]', color=tc["color"],
                 linestyle=':', marker='.', markersize=4, alpha=0.8)
        time_scale = tr["traj_ctmc"]["wall_clock_time"][-1] / tr["traj_dutta"]["steps"][-1]
        ax2.plot(
            tr["traj_dutta"]["steps"] * time_scale, 
            tr["traj_dutta"]["mse"],
            color=tc["color"], 
            linestyle=':', 
            linewidth=1.5, 
            alpha=0.85,
            label=f'{tc["label"]} (Dutta Bound)'
        )

    ax1.set_yscale('log')
    ax1.set_xlabel(r'Training Iterations / Rounds ($k$)')
    ax1.set_ylabel(r'Model Error $\mathbb{E}[\|\theta_k - \theta^*\|^2]$')
    ax1.set_title(r'(a) Convergence vs Iterations')
    # ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True)

    ax2.set_yscale('log')
    ax2.set_xlabel(r'Wall-Clock Time $t$ [seconds]')
    ax2.set_ylabel(r'Model Error $\mathbb{E}[\|\theta(t) - \theta^*\|^2]$')
    ax2.set_title(r'(b) Convergence vs Wall-Clock Time')
    # ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
        ax.title.set_size(9)


    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, -0.1),
        handlelength=1.8,
        columnspacing=1.0,
        labelspacing=0.3,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_convergence_trajectories.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_convergence_trajectories.pdf"))
    plt.close()
    print("  -> Saved fig2_convergence_trajectories")

    # --------------- FIGURE 3a: Throughput Sweep ---------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, data in sweep_data.items():
        fam = data["family"]
        ax.plot(data["K"], data["rate_ctmc"], marker=fam["marker"],
                color=fam["color"], linestyle='-',
                label=f'{fam["name"]} [CTMC]', markersize=10)
        ax.plot(data["K"], data["rate_sim"], marker=fam["marker"],
                color=fam["color"], linestyle=':', alpha=0.8,
                label=f'{fam["name"]} [SimPy]', markersize=10)
    ax.set_xlabel(r'Synchronization Threshold $K$')
    ax.set_ylabel(r'Aggregation Rate $A$ [events/s]')
    ax.set_title(r'Throughput vs $K$: CTMC vs SimPy')
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3a_throughput_sweep.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3a_throughput_sweep.pdf"))
    plt.close()
    print("  -> Saved fig3a_throughput_sweep")

    # --------------- FIGURE 3b: Bias Sweep ---------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for key, data in sweep_data.items():
        fam = data["family"]
        ax.plot(data["K"], data["bias_ctmc"], marker=fam["marker"],
                color=fam["color"], linestyle='-',
                label=f'{fam["name"]} [CTMC]', markersize=10)
        ax.plot(data["K"], data["bias_sim"], marker=fam["marker"],
                color=fam["color"], linestyle=':', alpha=0.8,
                label=f'{fam["name"]} [SimPy]', markersize=10)
    ax.set_xlabel(r'Synchronization Threshold $K$')
    ax.set_ylabel(r'Palm Participation Bias $\beta^2_{\mathrm{bias}}$')
    ax.set_title(r'Induced Palm Bias vs $K$: CTMC vs SimPy')
    ax.legend(fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3b_bias_sweep.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3b_bias_sweep.pdf"))
    plt.close()
    print("  -> Saved fig3b_bias_sweep")

    # --------------- FIGURE 4: Pareto Frontier (CTMC + SimPy) ---------------
    fig, ax = plt.subplots(figsize=(9, 6))

    fam_names_seen = set()
    for p in pareto_points:
        fam = p["family"]
        ax.scatter(p["time_ctmc"], p["mse_ctmc"], color=p["color"],
                   marker=p["marker"], s=110, alpha=0.9,
                   edgecolors='black', linewidths=0.8,
                   label=f'{fam} [CTMC]' if fam not in fam_names_seen else None)
        ax.scatter(p["time_sim"], p["mse_sim"], color=p["color"],
                   marker=p["marker"], s=110, alpha=0.5,
                   facecolors='none', edgecolors=p["color"], linewidths=2.0,
                   label=f'{fam} [SimPy]' if fam not in fam_names_seen else None)
        fam_names_seen.add(fam)

    ax.set_xlabel(r'Wall-Clock Convergence Delay $T_{\epsilon}$ [s]')
    ax.set_ylabel(r'Asymptotic Model Inaccuracy / MSE')
    ax.set_title(r'Pareto Frontier: CTMC (solid) vs SimPy (hollow)')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_pareto_frontier.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_pareto_frontier.pdf"))
    plt.close()
    print("  -> Saved fig4_pareto_frontier")

    # --------------- FIGURE 5: Multi-Job HoL Blocking ---------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    colors_jobs = ['#1f77b4', '#2ca02c', '#d62728']
    J = hol_results[0]["J"]

    for j in range(J):
        for r_idx, res in enumerate(hol_results):
            if len(res["job_time"][j]) > 1:
                ax1.plot(res["job_time"][j], res["job_mse"][j],
                         color=colors_jobs[j], alpha=0.4,
                         label=f'Job {j+1}' if r_idx == 0 else None,
                         linewidth=2.0)

    ax1.set_yscale('log')
    ax1.set_xlabel(r'Wall-Clock Time $t$ [s]')
    ax1.set_ylabel(r'Model Error / MSE')
    ax1.set_title(r'(a) Per-Job Convergence under HoL Blocking')
    ax1.legend()
    ax1.grid(True)

    mean_activation = np.zeros(J)
    for res in hol_results:
        for j in range(J):
            mean_activation[j] += res["job_activation_time"][j]
    mean_activation /= len(hol_results)

    ax2.bar(np.arange(J), mean_activation, color=colors_jobs[:J],
            edgecolor='black', alpha=0.85)
    ax2.set_xticks(np.arange(J))
    ax2.set_xticklabels([f'Job {j+1}' for j in range(J)])
    ax2.set_ylabel(r'Job Activation (Unblocked) Time [s]')
    ax2.set_title(r'(b) HoL Blocking Delay per Job')
    ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_hol_blocking.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_hol_blocking.pdf"))
    plt.close()
    print("  -> Saved fig5_hol_blocking")

    # ==========================================================================
    # 7. GENERATE LATEX TABLES
    # ==========================================================================
    print("\nGenerating LaTeX tables...")

    # Table 1: System Parameters
    t1_rows = []
    labels_g = ["Group 1 (Fast)", "Group 2 (Medium)", "Group 3 (Straggler)"]
    for i, (N_g, mu_g) in enumerate(groups):
        t1_rows.append(f"{labels_g[i]} & {N_g} & {mu_g:.1f} & {a_g[i]:.1f} & {c_g[i]:+.1f} \\\\")

    t1_code = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{System Evaluation Parameters and Group Heterogeneity}" "\n"
        r"\label{tab:system_parameters}" "\n"
        r"\begin{tabular}{lcccc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Group} & \textbf{Workers} & \textbf{Rate $\mu_g$} & \textbf{Curvature $a_g$} & \textbf{Optimum $c_g$} \\" "\n"
        r"\midrule" "\n"
        + "\n".join(t1_rows) + "\n"
        r"\midrule" "\n"
        r"\multicolumn{5}{l}{\textbf{Config:} $\eta=0.08$, $N=10$, $\sigma_g \in [0.1, 0.2]$} \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with open(os.path.join(OUTPUT_DIR, "table_system_parameters.tex"), "w") as f:
        f.write(t1_code)

    # Table 2: Validation
    t2_rows = []
    for v in val_results:
        cfg = v["config"]
        ci = v["sim"]["ci95_cycle_time"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        t2_rows.append(
            f"({cfg['b']}, {cfg['B']}, {cfg['K']}) & {v['ctmc'].num_states} & "
            f"{v['ctmc'].aggregation_rate:.4f} & {v['sim']['aggregation_rate']:.4f} & "
            f"{ci_str} & {v['err_rel']:.2f}\\% \\\\"
        )

    t2_code = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Analytical Lumped CTMC vs. Microscopic SimPy Simulation}" "\n"
        r"\label{tab:validation_comparison}" "\n"
        r"\begin{tabular}{cccccc}" "\n"
        r"\toprule" "\n"
        r"$(b,B,K)$ & $|\mathcal{S}|$ & \textbf{CTMC $A$} & \textbf{SimPy $\hat{A}$} & \textbf{SimPy 95\% CI ($\tau$)} & \textbf{Rel. Error} \\" "\n"
        r"\midrule" "\n"
        + "\n".join(t2_rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with open(os.path.join(OUTPUT_DIR, "table_validation_comparison.tex"), "w") as f:
        f.write(t2_code)

    # Table 3: Parametric Sweep
    t3_rows = []
    for key, data in sweep_data.items():
        fam = data["family"]
        n = len(data["K"])
        for idx in [0, n//2, n-1]:
            k = data["K"][idx]
            t3_rows.append(
                f"{fam['name']} & {k} & "
                f"{data['rate_ctmc'][idx]:.4f} & {data['rate_sim'][idx]:.4f} & "
                f"{data['bias_ctmc'][idx]:.5f} & {data['bias_sim'][idx]:.5f} & "
                f"{data['mse_ctmc'][idx]:.5f} & {data['mse_sim'][idx]:.5f} \\\\"
            )

    t3_code = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Parametric Sweep: CTMC vs. SimPy Comparison}" "\n"
        r"\label{tab:parametric_sweep}" "\n"
        r"\begin{tabular}{lccccccc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Family} & $K$ & $A_{\text{CTMC}}$ & $A_{\text{SimPy}}$ & "
        r"$\beta^2_{\text{bias,CTMC}}$ & $\beta^2_{\text{bias,SimPy}}$ & "
        r"$\text{MSE}_{\text{CTMC}}$ & $\text{MSE}_{\text{SimPy}}$ \\" "\n"
        r"\midrule" "\n"
        + "\n".join(t3_rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with open(os.path.join(OUTPUT_DIR, "table_parametric_sweep.tex"), "w") as f:
        f.write(t3_code)

    # Table 4: Pareto Summary
    t4_code = (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Pareto Frontier Operational Trade-off (CTMC vs. SimPy)}" "\n"
        r"\label{tab:pareto_summary}" "\n"
        r"\begin{tabular}{lcccc}" "\n"
        r"\toprule" "\n"
        r"\textbf{Regime} & \textbf{$(b, B, K)$} & \textbf{$T_\epsilon$ (CTMC)} & \textbf{$T_\epsilon$ (SimPy)} & \textbf{Asymptotic MSE} \\" "\n"
        r"\midrule" "\n"
        r"Ultra Low Latency & $(1,1,2)$ & See Fig.~4 & See Fig.~4 & High \\" "\n"
        r"Balanced K-Batch-Sync & $(3,1,6)$ & See Fig.~4 & See Fig.~4 & Low \\" "\n"
        r"High Precision & $(3,3,10)$ & See Fig.~4 & See Fig.~4 & Minimal \\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )
    with open(os.path.join(OUTPUT_DIR, "table_pareto_summary.tex"), "w") as f:
        f.write(t4_code)

    print("  -> Saved 4 LaTeX tables")

    # ==========================================================================
    # 8. EXPORT SUMMARY JSON
    # ==========================================================================
    summary_data = {
        "system_config": {
            "groups": groups,
            "N_total": N_total,
            "weights": group_weights.tolist(),
            "optimum_theta_star": sgd_probe.theta_star
        },
        "validation_results": [
            {
                "b": v["config"]["b"], "B": v["config"]["B"], "K": v["config"]["K"],
                "num_states": v["ctmc"].num_states,
                "ctmc_rate": v["ctmc"].aggregation_rate,
                "sim_rate": v["sim"]["aggregation_rate"],
                "sim_ci95_cycle": list(v["sim"]["ci95_cycle_time"]),
                "rel_error_pct": v["err_rel"],
                "ctmc_beta2_bias": v["ctmc"].beta2_bias,
                "sim_beta2_bias": v["sim"]["beta2_bias"],
            }
            for v in val_results
        ],
        "sweep_data": {
            k: {
                "K": v["K"],
                "rate_ctmc": v["rate_ctmc"],
                "rate_sim": v["rate_sim"],
                "bias_ctmc": v["bias_ctmc"],
                "bias_sim": v["bias_sim"],
                "mse_ctmc": v["mse_ctmc"],
                "mse_sim": v["mse_sim"],
            }
            for k, v in sweep_data.items()
        },
        "pareto_points": pareto_points,
    }

    with open(os.path.join(OUTPUT_DIR, "summary_numerical_results.json"), "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    print("  -> Saved summary_numerical_results.json")
    print("\n" + "=" * 80)
    print(f"ALL RESULTS SAVED IN: ./{OUTPUT_DIR}/")
    print("=" * 80)



if __name__ == "__main__":
    run_all_experiments()