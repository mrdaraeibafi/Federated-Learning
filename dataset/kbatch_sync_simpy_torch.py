"""K-Batch-Sync: real PyTorch gradients, SimPy service/cancellation.
Scope: one active round (J=1), sequential service per worker, B queued
minibatches per worker including the in-service task (b=B). No CTMC.
Time is simulated, NOT hardware runtime. Ties use SimPy event creation order.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import simpy


@dataclass
class Task:
    uid: int
    worker: int
    indices: np.ndarray


class Engine:
    """Backend provides gradient(task), update(grads), and evaluate().

    Coordinator alone changes parameters; they remain fixed throughout a round.
    Carry-over tasks keep identity/data, but draw a fresh full service duration.
    """
    def __init__(self, backend, sizes, rates, B=2, K=3, rounds=10,
                 batch_size=32, seed=42, deterministic=False):
        if not sizes or len(sizes) != len(rates):
            raise ValueError('sizes and rates must match')
        if min(sizes) < batch_size or min(rates) <= 0:
            raise ValueError('Each partition must fit a batch; rates must be positive')
        if B < 1 or not 1 <= K <= len(sizes)*B or rounds < 1:
            raise ValueError('Require B>=1, rounds>=1, 1<=K<=N*B')
        self.backend, self.sizes, self.rates = backend, sizes, rates
        self.B, self.K, self.rounds, self.batch_size = B, K, rounds, batch_size
        self.deterministic = deterministic
        streams = np.random.SeedSequence(seed).spawn(2*len(sizes))
        self.data_rng = [np.random.default_rng(s) for s in streams[:len(sizes)]]
        self.time_rng = [np.random.default_rng(s) for s in streams[len(sizes):]]
        self.env = simpy.Environment()
        self.pending = [deque() for _ in sizes]
        self.uid, self.version = 0, 0
        self.events, self.history = [], []
        self.computations = 0

    def log(self, kind, task=None, **extra):
        self.events.append(dict(time=float(self.env.now), version=self.version,
                                event=kind, task_id=None if task is None else task.uid,
                                worker=None if task is None else task.worker, **extra))

    def fill(self):
        for i, queue in enumerate(self.pending):
            while len(queue) < self.B:
                indices = self.data_rng[i].choice(self.sizes[i], self.batch_size, replace=False)
                queue.append(Task(self.uid, i, indices))
                self.uid += 1
            assert len(queue) == self.B

    def worker(self, i, version):
        while self.pending[i] and not self.closed:
            task = self.pending[i][0]
            duration = (1/self.rates[i] if self.deterministic
                        else self.time_rng[i].exponential(1/self.rates[i]))
            self.log('start', task, duration=float(duration))
            try:
                yield self.env.timeout(duration)
            except simpy.Interrupt:
                self.log('cancel_running', task)
                return
            # K may have been reached by another worker at the same timestamp.
            if self.closed or version != self.version:
                self.log('discard_at_boundary', task)
                return
            gradient = self.backend.gradient(task)
            self.computations += 1
            self.pending[i].popleft()
            self.accepted.append((task, gradient, version))
            self.counts[i] += 1
            self.log('accept', task, count=len(self.accepted))
            assert self.counts[i] <= self.B
            if len(self.accepted) == self.K:
                self.closed = True
                self.threshold.succeed()
                return

    def coordinator(self):
        initial = self.backend.evaluate()
        self.history.append(dict(round=0, time=0., **initial))
        for version in range(self.rounds):
            self.version = version
            self.fill()
            self.accepted, self.counts = [], [0]*len(self.sizes)
            self.closed = False
            self.threshold = self.env.event()
            processes = [self.env.process(self.worker(i, version)) for i in range(len(self.sizes))]
            yield self.threshold
            assert len(self.accepted) == self.K
            assert all(v == version for _, _, v in self.accepted)
            self.backend.update([g for _, g, _ in self.accepted])
            self.log('global_update', contributions=self.counts.copy())
            for process in processes:
                if process.is_alive:
                    process.interrupt('global_update')
            # Drain interrupts before launching new-version workers; also log queued work.
            yield self.env.all_of(processes)
            remaining = {t.uid for q in self.pending for t in q}
            canceled_running = {e['task_id'] for e in self.events
                                if e['version'] == version and
                                e['event'] in ('cancel_running', 'discard_at_boundary')}
            for queue in self.pending:
                for task in queue:
                    if task.uid not in canceled_running:
                        self.log('cancel_queued', task)
            self.history.append(dict(round=version+1, time=float(self.env.now),
                                     contributions=self.counts.copy(),
                                     canceled=len(remaining), **self.backend.evaluate()))

    def run(self):
        self.env.run(until=self.env.process(self.coordinator()))
        return self.history

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
        (directory/'history.json').write_text(json.dumps(self.history, indent=2))
        with (directory/'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['round', 'time', 'loss', 'accuracy'])
            writer.writeheader()
            writer.writerows({k:r[k] for k in writer.fieldnames} for r in self.history)


class TorchBackend:
    def __init__(self, partitions, test, lr, seed=42):
        import torch
        self.torch, self.partitions, self.test, self.lr = torch, partitions, test, lr
        torch.manual_seed(seed)
        self.model = torch.nn.Linear(partitions[0][0].shape[1], int(test[1].max())+1)
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.bias.zero_()

    def gradient(self, task):
        torch = self.torch
        x, y = self.partitions[task.worker]
        idx = torch.as_tensor(task.indices, dtype=torch.long)
        # No optimizer step here: all accepted gradients use the fixed round model.
        loss = torch.nn.functional.cross_entropy(self.model(x[idx]), y[idx])
        return [g.detach().clone() for g in torch.autograd.grad(loss, tuple(self.model.parameters()))]

    def update(self, gradients):
        with self.torch.no_grad():
            for p, components in zip(self.model.parameters(), zip(*gradients)):
                p.add_(self.torch.stack(components).mean(0), alpha=-self.lr)

    def evaluate(self):
        with self.torch.no_grad():
            x, y = self.test
            logits = self.model(x)
            return dict(loss=self.torch.nn.functional.cross_entropy(logits, y).item(),
                        accuracy=(logits.argmax(1)==y).float().mean().item())


def load_data(dataset, n, seed, root):
    import torch
    generator = torch.Generator().manual_seed(seed)
    if dataset == 'mnist':
        from torchvision.datasets import MNIST
        train = MNIST(root, train=True, download=True)
        test = MNIST(root, train=False, download=True)
        idx = torch.randperm(len(train.data), generator=generator)[:6000]
        x = (train.data[idx].reshape(-1,784).float()/255-.1307)/.3081
        y = train.targets[idx]
        idx = torch.randperm(len(test.data), generator=generator)[:1000]
        xt = (test.data[idx].reshape(-1,784).float()/255-.1307)/.3081
        yt = test.targets[idx]
    else:
        x = torch.randn(3600, 20, generator=generator)
        teacher = torch.randn(20, 3, generator=generator)
        y = (x @ teacher).argmax(1)
        x, xt, y, yt = x[:3000], x[3000:], y[:3000], y[3000:]
    partitions = [(x[idx], y[idx]) for idx in torch.tensor_split(torch.arange(len(x)), n)]
    return partitions, (xt, yt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['synthetic','mnist'], default='synthetic')
    p.add_argument('--rates', type=float, nargs='+', default=[1., .7, .3])
    p.add_argument('--B', type=int, default=2)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--data-root', default='./data')
    p.add_argument('--output', default='kbatch_results')
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    partitions, test = load_data(args.dataset, len(args.rates), args.seed, args.data_root)
    backend = TorchBackend(partitions, test, args.lr, args.seed)
    engine = Engine(backend, [len(y) for _,y in partitions], args.rates,
                    args.B, args.K, args.rounds, args.batch_size, args.seed, args.deterministic)
    engine.run()
    engine.save(args.output)
    Path(args.output, 'config.json').write_text(json.dumps(vars(args), indent=2))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        for ax, metric in zip(axes, ['loss','accuracy']):
            ax.plot([r['time'] for r in engine.history], [r[metric] for r in engine.history])
            ax.set(xlabel='Simulated time', ylabel=metric)
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(Path(args.output,'learning.png'), dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(engine.history[-1], indent=2))


if __name__ == '__main__':
    main()
