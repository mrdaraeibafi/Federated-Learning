"""
Experimental Benchmark: Pipelined K-Batch-Sync FL vs. FedAvg Baseline
Environment: SimPy Discretized Heterogeneous Cluster (Poisson/Exp Arrivals & Service)
Dataset: MNIST (Multi-Class Logistic Regression on 28x28 Features)
Distribution: IID Uniform Partitioning across Heterogeneous Workers
Data Aggregation: Uniform Averaging (1/K) under IID Assumption
CRN Architecture: Coupled PRNG streams for fair comparison
Extensions: Parametric K-Sweep and Sensitivity Analysis vs. Wall-Clock Time
"""

import simpy
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ==========================================
# 0. Global Setup & Seed Configuration (CRN)
# ==========================================
SEED = 42
USE_CRN = True  # Common Random Numbers flag

N_WORKERS = 10
K_VAL_DEFAULT = 5      # Baseline K value
K_SWEEP_VALUES = [3, 5, 7, 10]  # Values evaluated in sensitivity study
TARGET_LOSS = 0.50     # Target test loss threshold for convergence speedup
B_SUBTASKS = 1         # B subtasks per job
J_CONCURRENCY = 1      # Concurrency backlog multiplier
TOTAL_ROUNDS = 500
BATCH_SIZE = 32
LR = 0.05

# Worker service rates: Heterogeneous speeds
MU_RATES = np.array([2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.2])

# ==========================================
# 1. Common Random Number (CRN) Manager
# ==========================================
class CRNManager:
    """
    Manages orthogonal deterministic PRNG streams:
    - Service times: identically sampled across algorithms
    - Data mini-batches: worker n processes the exact same data order in round q
    """
    def __init__(self, n_workers, total_rounds, seed=SEED):
        self.n_workers = n_workers
        self.total_rounds = total_rounds
        self.seed = seed
        
        # Stream 1: Service times
        self.service_rng = np.random.default_rng(self.seed + 100)
        self.service_times = [
            self.service_rng.exponential(scale=1.0 / MU_RATES, size=(B_SUBTASKS, self.n_workers))
            for _ in range(self.total_rounds)
        ]
        
        # Stream 2: Worker mini-batch permutations
        self.batch_rng = np.random.default_rng(self.seed + 200)
        self.batch_indices = [
            [self.batch_rng.permutation(6000 // self.n_workers) for _ in range(self.n_workers)]
            for _ in range(self.total_rounds)
        ]

    def get_service_time(self, round_idx, subtask_idx, worker_idx):
        return self.service_times[round_idx][subtask_idx, worker_idx]

    def get_worker_batch(self, round_idx, worker_idx, batch_size):
        perm = self.batch_indices[round_idx][worker_idx]
        return perm[:batch_size]

# ==========================================
# 2. Dataset Preparation (IID MNIST)
# ==========================================
def get_iid_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    X_train = train_dataset.data.float().view(-1, 784) / 255.0
    y_train = train_dataset.targets
    X_test = test_dataset.data.float().view(-1, 784) / 255.0
    y_test = test_dataset.targets

    # Uniform partition among workers
    indices = np.random.default_rng(SEED).permutation(len(X_train))
    split_size = len(X_train) // N_WORKERS
    worker_data = []
    for i in range(N_WORKERS):
        idx = indices[i * split_size:(i + 1) * split_size]
        worker_data.append((X_train[idx], y_train[idx]))
        
    return worker_data, (X_test, y_test)

# ==========================================
# 3. Model Definition
# ==========================================
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

def evaluate(model, test_data):
    X_test, y_test = test_data
    with torch.no_grad():
        out = model(X_test)
        loss = nn.CrossEntropyLoss()(out, y_test).item()
        pred = out.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
    return loss, acc

# ==========================================
# 4. Simulation Engine (SimPy FL Benchmark)
# ==========================================
def run_fl_benchmark(algorithm='kbatch_sync', k_val=K_VAL_DEFAULT, shared_data=None):
    if shared_data is None:
        worker_data, test_data = get_iid_mnist()
    else:
        worker_data, test_data = shared_data
        
    crn = CRNManager(N_WORKERS, TOTAL_ROUNDS, seed=SEED)
    
    # Model initialization
    torch.manual_seed(SEED)
    model = LogisticRegression()
    
    rounds_log = []
    time_log = []
    loss_log = []
    acc_log = []
    
    wall_clock_time = 0.0

    # ----------------------------------------------------
    # Case A: FedAvg (Full-Sync Baseline)
    # ----------------------------------------------------
    if algorithm == 'fedavg':
        for q in range(TOTAL_ROUNDS):
            # Straggler effect: dominated by slowest worker
            if USE_CRN:
                times = [sum(crn.get_service_time(q, b, n) for b in range(B_SUBTASKS)) for n in range(N_WORKERS)]
            else:
                times = [np.sum(np.random.exponential(1.0 / MU_RATES[n], B_SUBTASKS)) for n in range(N_WORKERS)]
            
            wall_clock_time += np.max(times)
            
            # Local training on all N workers
            accum_w = torch.zeros_like(model.linear.weight)
            accum_b = torch.zeros_like(model.linear.bias)
            
            for n in range(N_WORKERS):
                model.zero_grad()
                idx = crn.get_worker_batch(q, n, BATCH_SIZE) if USE_CRN else np.random.choice(len(worker_data[n][0]), BATCH_SIZE)
                X_batch, y_batch = worker_data[n][0][idx], worker_data[n][1][idx]
                
                out = model(X_batch)
                loss = nn.CrossEntropyLoss()(out, y_batch)
                grads = torch.autograd.grad(loss, [model.linear.weight, model.linear.bias])
                
                accum_w += grads[0]
                accum_b += grads[1]
                
            model.linear.weight.data -= (LR / N_WORKERS) * accum_w
            model.linear.bias.data -= (LR / N_WORKERS) * accum_b
            
            loss_val, acc_val = evaluate(model, test_data)
            rounds_log.append(q + 1)
            time_log.append(wall_clock_time)
            loss_log.append(loss_val)
            acc_log.append(acc_val)

    # ----------------------------------------------------
    # Case B: Pipelined K-Batch-Sync (SimPy Asynchronous Queue)
    # ----------------------------------------------------
    elif algorithm == 'kbatch_sync':
        env = simpy.Environment()
        
        # SimPy worker process
        class Worker:
            def __init__(self, env, wid):
                self.env = env
                self.wid = wid
                self.queue = simpy.Store(env)
                self.action = env.process(self.run())
                
            def run(self):
                while True:
                    job_round, subtask_idx, ack_event = yield self.queue.get()
                    if USE_CRN:
                        svc_time = crn.get_service_time(job_round, subtask_idx, self.wid)
                    else:
                        svc_time = np.random.exponential(1.0 / MU_RATES[self.wid])
                    yield self.env.timeout(svc_time)
                    ack_event.succeed(value=(self.wid, job_round))

        workers = [Worker(env, i) for i in range(N_WORKERS)]

        def aggregator_process(env):
            nonlocal model
            for q in range(TOTAL_ROUNDS):
                ack_events = []
                for n in range(N_WORKERS):
                    for b in range(B_SUBTASKS):
                        ack = env.event()
                        workers[n].queue.put((q, b, ack))
                        ack_events.append(ack)
                
                # Wait for the fastest k_val workers for job q
                first_k_workers = []
                while len(first_k_workers) < k_val:
                    finished = yield simpy.AnyOf(env, ack_events)
                    for ev in finished.events:
                        wid, job_id = ev.value
                        if job_id == q and wid not in first_k_workers:
                            first_k_workers.append(wid)
                        ack_events.remove(ev)
                        if len(first_k_workers) == k_val:
                            break
                
                # Model aggregation over the fastest k_val workers
                accum_w = torch.zeros_like(model.linear.weight)
                accum_b = torch.zeros_like(model.linear.bias)
                weight = 1.0 / k_val

                for n in first_k_workers:
                    model.zero_grad()
                    idx = crn.get_worker_batch(q, n, BATCH_SIZE) if USE_CRN else np.random.choice(len(worker_data[n][0]), BATCH_SIZE)
                    X_batch, y_batch = worker_data[n][0][idx], worker_data[n][1][idx]
                    
                    out = model(X_batch)
                    loss = nn.CrossEntropyLoss()(out, y_batch)
                    grads = torch.autograd.grad(loss, [model.linear.weight, model.linear.bias])
                    
                    accum_w += weight * grads[0]
                    accum_b += weight * grads[1]
                    
                model.linear.weight.data -= LR * accum_w
                model.linear.bias.data -= LR * accum_b
                
                loss_val, acc_val = evaluate(model, test_data)
                rounds_log.append(q + 1)
                time_log.append(env.now)
                loss_log.append(loss_val)
                acc_log.append(acc_val)

        env.process(aggregator_process(env))
        env.run()

    return {
        'rounds': np.array(rounds_log),
        'time': np.array(time_log),
        'loss': np.array(loss_log),
        'acc': np.array(acc_log)
    }

# ==========================================
# 5. Main Execution & Comparison Plotting
# ==========================================
if __name__ == '__main__':
    print("=" * 65)
    print(" Running FL Heterogeneous Benchmark (FedAvg vs. K-Batch-Sync)")
    print(f" Config: N={N_WORKERS}, Default K={K_VAL_DEFAULT}, B={B_SUBTASKS}, J={J_CONCURRENCY}, CRN={USE_CRN}")
    print("=" * 65)
    
    # Preload dataset once for efficiency
    print("Preloading IID MNIST dataset...")
    dataset_bundle = get_iid_mnist()

    # 1. Baseline runs
    print("\n [1/2] Running FedAvg (Full-Sync Baseline)...")
    res_fedavg = run_fl_benchmark('fedavg', shared_data=dataset_bundle)
    
    print(f"\n [2/2] Running Pipelined K-Batch-Sync (K={K_VAL_DEFAULT})...")
    res_kbatch_default = run_fl_benchmark('kbatch_sync', k_val=K_VAL_DEFAULT, shared_data=dataset_bundle)
    
    # ----------------------------------------------------
    # Figure 1: Baseline Comparison (Fig. 5 in paper)
    # ----------------------------------------------------
    print("\nGenerating baseline comparison figure...")
    plt.figure(figsize=(7, 10))
    
    c_fedavg = '#d62728'   # Red
    c_kbatch = '#2ca02c'   # Green
    
    # Subplot 1: Convergence vs. Communication Rounds
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(res_fedavg['rounds'], res_fedavg['loss'], label='FedAvg (Full-Sync)', color=c_fedavg, linestyle='--', linewidth=2)
    ax1.plot(res_kbatch_default['rounds'], res_kbatch_default['loss'], label=f'K-Batch-Sync ($K={K_VAL_DEFAULT}$)', color=c_kbatch, linewidth=2)
    ax1.set_xlabel('Global Rounds ($q$)', fontsize=11)
    ax1.set_ylabel('Test Loss', fontsize=11)
    ax1.set_title('Convergence vs. Communication Rounds', fontsize=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Convergence vs. Wall-Clock Time
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(res_fedavg['time'], res_fedavg['loss'], label='FedAvg (Full-Sync)', color=c_fedavg, linestyle='--', linewidth=2)
    ax2.plot(res_kbatch_default['time'], res_kbatch_default['loss'], label=f'K-Batch-Sync ($K={K_VAL_DEFAULT}$)', color=c_kbatch, linewidth=2)
    ax2.set_xlabel('Wall-Clock Time (s)', fontsize=11)
    ax2.set_ylabel('Test Loss', fontsize=11)
    ax2.set_title('Wall-Clock Convergence (Straggler Mitigation)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    baseline_filename = 'fig_mnist_kbatch_vs_baselines.png'
    plt.savefig(baseline_filename, dpi=300)
    print(f"Baseline comparison saved to '{baseline_filename}'.")

    # ----------------------------------------------------
    # 2. Parametric K-Sweep Evaluation
    # ----------------------------------------------------
    print("\n" + "=" * 65)
    print(" Running K-Sensitivity Sweep across values:", K_SWEEP_VALUES)
    print("=" * 65)
    
    sweep_results = {}
    time_to_target = {}

    for k in K_SWEEP_VALUES:
        print(f" -> Simulating K-Batch-Sync for K = {k} ...")
        res_k = run_fl_benchmark('kbatch_sync', k_val=k, shared_data=dataset_bundle)
        sweep_results[k] = res_k
        
        # Calculate time required to reach target test loss
        below_target = np.where(res_k['loss'] <= TARGET_LOSS)[0]
        if len(below_target) > 0:
            first_idx = below_target[0]
            time_to_target[k] = res_k['time'][first_idx]
        else:
            time_to_target[k] = res_k['time'][-1]
        print(f"    Done. Time to Loss <= {TARGET_LOSS}: {time_to_target[k]:.1f} s")

    # Also compute FedAvg target time for direct reference
    fedavg_target_idx = np.where(res_fedavg['loss'] <= TARGET_LOSS)[0]
    fedavg_time_target = res_fedavg['time'][fedavg_target_idx[0]] if len(fedavg_target_idx) > 0 else res_fedavg['time'][-1]
    print(f" -> FedAvg Time to Loss <= {TARGET_LOSS}: {fedavg_time_target:.1f} s")

    # ----------------------------------------------------
    # Figure 2: K-Sensitivity Analysis Plot (New Journal Plot)
    # ----------------------------------------------------
    print("\nGenerating K-Sensitivity figure...")
    fig_sens, (p1, p2) = plt.subplots(2, 1, figsize=(6, 9), constrained_layout=True)

    colors = {3: '#e66101', 5: '#2ca02c', 7: '#1b9e77', 10: '#7570b3'}
    styles = {3: '-.', 5: '-', 7: '-', 10: ':'}

    # Panel (a): Trajectories across K
    for k in K_SWEEP_VALUES:
        p1.plot(
            sweep_results[k]['time'],
            sweep_results[k]['loss'],
            label=f'$K={k}$',
            color=colors.get(k, '#333333'),
            linestyle=styles.get(k, '-'),
            linewidth=2
        )
    p1.axhline(TARGET_LOSS, color='gray', linestyle='--', alpha=0.7, label=f'Target Loss ({TARGET_LOSS})')
    p1.set_xlabel('Wall-Clock Time (s)', fontsize=11)
    p1.set_ylabel('Test Loss', fontsize=11)
    p1.set_title('(a) Wall-Clock Loss Trajectories vs. Quota $K$', fontsize=12)
    p1.set_xlim(left=0, right=min(14000, np.max([sweep_results[k]['time'][-1] for k in K_SWEEP_VALUES])))
    p1.grid(True, linestyle=':', alpha=0.6)
    p1.legend(fontsize=9, loc='upper right')

    # Panel (b): Time to Target vs. K
    k_coords = list(K_SWEEP_VALUES)
    t_coords = [time_to_target[k] for k in k_coords]

    p2.plot(k_coords, t_coords, marker='s', markersize=7, color='#1f77b4', linewidth=2, label='$K$-Batch-Sync')
    p2.axhline(fedavg_time_target, color=c_fedavg, linestyle='--', linewidth=1.8, label=f'FedAvg ($K=10$ Sync): {fedavg_time_target:.0f}s')

    # Highlight optimal point
    best_k = min(time_to_target, key=time_to_target.get)
    p2.scatter([best_k], [time_to_target[best_k]], color='#d95f02', s=100, zorder=5, label=f'Optimal ($K^*={best_k}$)')

    p2.set_xlabel('Synchronization Quota ($K$)', fontsize=11)
    p2.set_ylabel(f'Time to Target Loss ({TARGET_LOSS}) [s]', fontsize=11)
    p2.set_title('(b) Wall-Clock Acceleration vs. Quota $K$', fontsize=12)
    p2.set_xticks(k_coords)
    p2.grid(True, linestyle=':', alpha=0.6)
    p2.legend(fontsize=9, loc='upper left')

    sensitivity_filename = 'fig_mnist_k_sensitivity.png'
    fig_sens.savefig(sensitivity_filename, dpi=300)
    print(f"K-Sensitivity figure saved to '{sensitivity_filename}'.")
    print("=" * 65)
    print(" All runs completed successfully.")
