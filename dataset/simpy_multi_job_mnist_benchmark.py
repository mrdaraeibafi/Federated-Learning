"""
Experimental Benchmark: Pipelined K-Batch-Sync FL vs. FedAvg Baseline
Environment: SimPy Discretized Heterogeneous Cluster (Poisson/Exp Arrivals & Service)
Dataset: MNIST (Synthetic LeNet / Logistic Regression on 28x28 Features)
Distribution: IID Uniform Partitioning across Heterogeneous Workers
Data Aggregation: Uniform Averaging (1/K) under IID Assumption
CRN Architecture: Coupled PRNG streams for fair comparison
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
K_VAL = 5              # Fastest K workers aggregated
B_SUBTASKS = 1         # B subtasks per job
J_CONCURRENCY = 1      # Fixed to 1 for fair baseline comparison
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
def run_fl_benchmark(algorithm='kbatch_sync'):
    worker_data, test_data = get_iid_mnist()
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
                
                # Wait for the fastest K completions for job q
                first_k_workers = []
                while len(first_k_workers) < K_VAL:
                    finished = yield simpy.AnyOf(env, ack_events)
                    for ev in finished.events:
                        wid, job_id = ev.value
                        if job_id == q and wid not in first_k_workers:
                            first_k_workers.append(wid)
                        ack_events.remove(ev)
                        if len(first_k_workers) == K_VAL:
                            break
                
                # Model aggregation over the fastest K workers (Uniform 1/K for IID)
                accum_w = torch.zeros_like(model.linear.weight)
                accum_b = torch.zeros_like(model.linear.bias)
                weight = 1.0 / K_VAL

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
    print("=" * 60)
    print(" Running FL Heterogeneous Benchmark (FedAvg vs. K-Batch-Sync)")
    print(f" Config: N={N_WORKERS}, K={K_VAL}, B={B_SUBTASKS}, J={J_CONCURRENCY}, CRN={USE_CRN}")
    print("=" * 60)
    
    print("\n [1/2] Running FedAvg (Full-Sync Baseline)...")
    res_fedavg = run_fl_benchmark('fedavg')
    
    print("\n [2/2] Running Pipelined K-Batch-Sync...")
    res_kbatch = run_fl_benchmark('kbatch_sync')
    
    print("\nPlotting benchmark results...")
    plt.figure(figsize=(7, 12))
    
    c_fedavg = '#d62728'   # Red
    c_kbatch = '#2ca02c'   # Green
    
    # Subplot 1: Convergence vs. Communication Rounds
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(res_fedavg['rounds'], res_fedavg['loss'], label='FedAvg (Full-Sync)', color=c_fedavg, linestyle='--', linewidth=2)
    ax1.plot(res_kbatch['rounds'], res_kbatch['loss'], label=f'K-Batch-Sync ($K={K_VAL}$)', color=c_kbatch, linewidth=2)
    ax1.set_xlabel('Global Rounds ($q$)', fontsize=12)
    ax1.set_ylabel('Test Loss', fontsize=12)
    ax1.set_title('Convergence vs. Communication Rounds', fontsize=13)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Convergence vs. Wall-Clock Time
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(res_fedavg['time'], res_fedavg['loss'], label='FedAvg (Full-Sync)', color=c_fedavg, linestyle='--', linewidth=2)
    ax2.plot(res_kbatch['time'], res_kbatch['loss'], label=f'K-Batch-Sync ($K={K_VAL}$)', color=c_kbatch, linewidth=2)
    ax2.set_xlabel('Wall-Clock Time (s)', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title('Wall-Clock Convergence (Straggler Mitigation)', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    output_filename = 'fig_mnist_kbatch_vs_baselines.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Results successfully saved to '{output_filename}'.")
