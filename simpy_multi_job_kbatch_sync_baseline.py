#!/usr/bin/env python3
"""
simpy_multi_job_kbatch_sync_baseline.py
Microscopic SimPy Simulator comparing target (N, K, J, B) against
the Fully Synchronous Baseline (K=N, J=1, B=1) with Monte Carlo Confidence Bands.
"""

import argparse
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class Subtask:
    job_id: int
    round_idx: int
    worker_id: int


@dataclass
class JobState:
    job_id: int
    theta: float
    theta_star: float
    eta: float
    target_rounds: int
    current_round: int = 0
    history: List[dict] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return self.current_round >= self.target_rounds

    @property
    def current_mse(self) -> float:
        return float((self.theta - self.theta_star) ** 2)


class Worker:
    def __init__(self, env: simpy.Environment, worker_id: int, mu: float,
                 buffer_capacity: int, a_n: float, c_n: float, sigma_n: float,
                 server: 'MultiJobServer'):
        self.env = env
        self.worker_id = worker_id
        self.mu = mu
        self.buffer_capacity = buffer_capacity
        self.a_n = a_n
        self.c_n = c_n
        self.sigma_n = sigma_n
        self.server = server

        self.queue: List[Subtask] = []
        self.wake_event = env.event()
        self.process = env.process(self.run())

    def enqueue_subtask(self, subtask: Subtask) -> bool:
        if len(self.queue) < self.buffer_capacity:
            self.queue.append(subtask)
            if not self.wake_event.triggered:
                self.wake_event.succeed()
            return True
        return False

    def run(self):
        while True:
            if not self.queue:
                self.wake_event = self.env.event()
                yield self.wake_event
                continue

            current_subtask = self.queue[0]
            service_time = np.random.exponential(1.0 / self.mu)
            yield self.env.timeout(service_time)

            if not self.queue or self.queue[0] != current_subtask:
                continue

            completed_subtask = self.queue.pop(0)
            job = self.server.jobs[completed_subtask.job_id]
            grad = self.a_n * (job.theta - self.c_n) + np.random.normal(0.0, self.sigma_n)
            self.server.receive_completion(self.worker_id, completed_subtask, grad)


class MultiJobServer:
    def __init__(self, env: simpy.Environment, jobs: List[JobState],
                 K: int, B: int, N: int):
        self.env = env
        self.jobs = jobs
        self.K = K
        self.B = B
        self.N = N
        self.num_jobs = len(jobs)

        self.active_job_idx: int = 0
        self.completed_subtasks_by_job: Dict[int, List[dict]] = {j.job_id: [] for j in jobs}
        self.workers: List[Worker] = []
        self.all_done_event = env.event()

    def set_workers(self, workers: List[Worker]):
        self.workers = workers

    @property
    def all_jobs_finished(self) -> bool:
        return all(j.is_finished for j in self.jobs)

    def receive_completion(self, worker_id: int, subtask: Subtask, grad: float):
        self.completed_subtasks_by_job[subtask.job_id].append({
            'worker_id': worker_id,
            'subtask': subtask,
            'grad': grad
        })
        self.check_and_aggregate()

    def advance_to_next_job(self):
        if self.all_jobs_finished:
            if not self.all_done_event.triggered:
                self.all_done_event.succeed()
            return

        for step in range(1, self.num_jobs + 1):
            candidate = (self.active_job_idx + step) % self.num_jobs
            if not self.jobs[candidate].is_finished:
                self.active_job_idx = candidate
                break

    def check_and_aggregate(self):
        if self.all_jobs_finished:
            if not self.all_done_event.triggered:
                self.all_done_event.succeed()
            return

        active_job = self.jobs[self.active_job_idx]

        if active_job.is_finished:
            self.advance_to_next_job()
            return

        completions = self.completed_subtasks_by_job[active_job.job_id]

        if len(completions) >= self.K:
            selected = completions[:self.K]
            grads = [c['grad'] for c in selected]
            avg_grad = float(np.mean(grads))

            active_job.theta -= active_job.eta * avg_grad
            active_job.current_round += 1

            now = self.env.now
            active_job.history.append({
                'job_id': active_job.job_id,
                'round': active_job.current_round,
                'wallclock_time': now,
                'theta': active_job.theta,
                'mse': active_job.current_mse
            })

            self.completed_subtasks_by_job[active_job.job_id] = completions[self.K:]

            if not active_job.is_finished:
                participating_workers: Set[int] = set(c['worker_id'] for c in selected)
                for w_id in participating_workers:
                    new_st = Subtask(job_id=active_job.job_id,
                                     round_idx=active_job.current_round,
                                     worker_id=w_id)
                    self.workers[w_id].enqueue_subtask(new_st)

            self.advance_to_next_job()


def run_single_trajectory(traj_id: int, N: int, J: int, B: int, K: int, R: int,
                          service_rates: np.ndarray, seed: int,
                          theta_0: float = 15.0, eta: float = 0.05) -> List[dict]:
    np.random.seed(seed)
    env = simpy.Environment()
    buffer_capacity = B * J

    jobs = []
    for j in range(J):
        jobs.append(JobState(job_id=j, theta=theta_0, theta_star=0.0,
                             eta=eta, target_rounds=R))

    server = MultiJobServer(env=env, jobs=jobs, K=K, B=B, N=N)

    workers = []
    for i in range(N):
        a_n = 1.0
        c_n = 0.0
        sigma_n = 0.5
        w = Worker(env=env, worker_id=i, mu=service_rates[i],
                   buffer_capacity=buffer_capacity,
                   a_n=a_n, c_n=c_n, sigma_n=sigma_n, server=server)
        workers.append(w)

    server.set_workers(workers)

    for w in workers:
        for j in range(J):
            for _ in range(B):
                w.enqueue_subtask(Subtask(job_id=j, round_idx=0, worker_id=w.worker_id))

    env.run(until=server.all_done_event)

    records = []
    for job in jobs:
        for entry in job.history:
            entry_copy = dict(entry)
            entry_copy['trajectory_id'] = traj_id
            records.append(entry_copy)

    return records


def run_experiment_monte_carlo(N: int, J: int, B: int, K: int, R: int,
                               n_trajectories: int, service_rates: np.ndarray,
                               base_seed: int) -> pd.DataFrame:
    all_records = []
    for traj in range(n_trajectories):
        traj_seed = base_seed + 1000 * (traj + 1)
        records = run_single_trajectory(traj_id=traj, N=N, J=J, B=B, K=K, R=R,
                                         service_rates=service_rates, seed=traj_seed)
        all_records.extend(records)
    return pd.DataFrame(all_records)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.groupby(['job_id', 'round']).agg(
        mean_mse=('mse', 'mean'),
        std_mse=('mse', 'std'),
        mean_time=('wallclock_time', 'mean'),
        std_time=('wallclock_time', 'std')
    ).reset_index()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Multi-Job K-Batch-Sync vs Full-Sync Baseline")
    parser.add_argument("--N", type=int, default=10, help="Number of workers")
    parser.add_argument("--J", type=int, default=1, help="Number of concurrent jobs")
    parser.add_argument("--B", type=int, default=1, help="Batch size per job")
    parser.add_argument("--K", type=int, default=1, help="Aggregation quorum size")
    parser.add_argument("--R", type=int, default=200, help="Target SGD rounds")
    parser.add_argument("--trajectories", type=int, default=20, help="Monte Carlo trajectories")
    parser.add_argument("--seed", type=int, default=2024, help="Base random seed")
    parser.add_argument("--out", type=str, default="comparison_baseline.png", help="Output plot filename")

    args = parser.parse_args()

    # ایجاد پروفایل یکنواخت و هتروژن نرخ‌های سرویس
    np.random.seed(args.seed)
    service_rates = np.random.uniform(0.8, 2.2, size=args.N)

    # ۱. اجرای پیکربندی هدف (Target Setting)
    print(f"\n[1/2] Running Target Configuration (N={args.N}, K={args.K}, J={args.J}, B={args.B})...")
    df_target = run_experiment_monte_carlo(
        N=args.N, J=args.J, B=args.B, K=args.K, R=args.R,
        n_trajectories=args.trajectories, service_rates=service_rates, base_seed=args.seed
    )
    stats_target = compute_statistics(df_target)

    # ۲. اجرای حالت مبنا (Fully-Synchronous Baseline: N=N, K=N, J=1, B=1)
    print(f"[2/2] Running Fully-Synchronous Baseline (N={args.N}, K={args.N}, J=1, B=1)...")
    df_baseline = run_experiment_monte_carlo(
        N=args.N, J=1, B=1, K=args.N, R=args.R,
        n_trajectories=args.trajectories, service_rates=service_rates, base_seed=args.seed
    )
    stats_baseline = compute_statistics(df_baseline)

    # ۳. رسم نمودارهای همگرایی مقایسه‌ای
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # الف) رسم مبنا (Baseline: K=N)
    sub_base = stats_baseline[stats_baseline['job_id'] == 0]
    ax1.plot(sub_base['round'], sub_base['mean_mse'], color='black', linestyle='--',
             linewidth=2.0, label=f'Baseline (K={args.N}, J=1, B=1)')
    ax1.fill_between(sub_base['round'],
                     np.maximum(1e-12, sub_base['mean_mse'] - sub_base['std_mse']),
                     sub_base['mean_mse'] + sub_base['std_mse'], color='black', alpha=0.1)

    ax2.plot(sub_base['mean_time'], sub_base['mean_mse'], color='black', linestyle='--',
             linewidth=2.0, label=f'Baseline (K={args.N}, J=1, B=1)')
    ax2.fill_betweenx(sub_base['mean_mse'],
                      np.maximum(0, sub_base['mean_time'] - sub_base['std_time']),
                      sub_base['mean_time'] + sub_base['std_time'], color='black', alpha=0.1)

    # ب) رسم تنظیمات هدف (Target Configurations)
    colors = plt.cm.tab10(np.linspace(0, 1, max(args.J, 10)))
    for j in range(args.J):
        sub_tgt = stats_target[stats_target['job_id'] == j]
        lbl = f'Job {j} (K={args.K}, B={args.B})' if args.J > 1 else f'Target (K={args.K}, B={args.B})'

        ax1.plot(sub_tgt['round'], sub_tgt['mean_mse'], color=colors[j],
                 linewidth=1.8, label=lbl)
        ax1.fill_between(sub_tgt['round'],
                         np.maximum(1e-12, sub_tgt['mean_mse'] - sub_tgt['std_mse']),
                         sub_tgt['mean_mse'] + sub_tgt['std_mse'], color=colors[j], alpha=0.2)

        ax2.plot(sub_tgt['mean_time'], sub_tgt['mean_mse'], color=colors[j],
                 linewidth=1.8, label=lbl)
        ax2.fill_betweenx(sub_tgt['mean_mse'],
                          np.maximum(0, sub_tgt['mean_time'] - sub_tgt['std_time']),
                          sub_tgt['mean_time'] + sub_tgt['std_time'], color=colors[j], alpha=0.2)

    # تنظیمات محورها و استایل نمودارها
    ax1.set_yscale('log')
    ax1.set_xlabel('SGD Iteration Round (r)', fontsize=11)
    ax1.set_ylabel('Expected MSE', fontsize=11)
    ax1.set_title('Iteration Convergence vs. Baseline', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc='upper right', framealpha=0.9)

    ax2.set_yscale('log')
    ax2.set_xlabel('Wall-Clock Time (t)', fontsize=11)
    ax2.set_ylabel('Expected MSE', fontsize=11)
    ax2.set_title('Wall-Clock Time Convergence vs. Baseline', fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"\n[Success] Comparison plot saved to '{args.out}'.")


if __name__ == '__main__':
    main()
