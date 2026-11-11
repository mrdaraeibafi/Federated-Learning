#!/usr/bin/env python3
"""Quadratic SGD with exact marked-CTMC timing (no staleness, one learner).
Dependencies: numpy scipy matplotlib. Run: python quadratic_ctmc_mse.py --config quadratic_config.json
sigma2 denotes PER-SAMPLE variance; m is samples per returned worker result.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from limited_ctmc_kbatch_sync_lumped_grouped_metrics import HeterogeneousLumpedKBatchSyncCTMC


def event_model(model):
    """All physical completions, including marked self-events."""
    events = []
    H = np.zeros_like(model.Q)  # aggregation event rate matrix, INCLUDING diagonal
    for i, state in enumerate(model.states):
        row = []
        for g, (_, mu) in enumerate(model.groups):
            for level in range(model.b):
                rate = state[g * (model.b + 1) + level] * mu
                if rate:
                    target, marked = model.transition_target(state, g, level)
                    j = model.state_to_index[target]
                    row.append((j, float(rate), bool(marked)))
                    if marked:
                        H[i, j] += rate
        events.append(row)
    return events, H


def pgf(model, H, alpha, z, times):
    """G(z,t)=alpha exp([Q+(z-1)H]t) 1; uniform grid starting at zero."""
    M = csr_matrix(model.Q + (z - 1.0) * H)
    values = expm_multiply(M, np.ones(len(alpha)), start=0.0,
                           stop=float(times[-1]), num=len(times))
    return values @ alpha


def curves(a, eta, sigma2, m, K, error0, rounds):
    q = 1.0 - eta * a
    aggregate_var = sigma2 / (m * K)
    floor = eta**2 * aggregate_var / (1 - q*q)
    dutta_floor = eta * aggregate_var / a
    s0 = error0**2
    exact = floor + (s0 - floor) * q**(2 * rounds)
    bound = dutta_floor + (s0 - dutta_floor) * q**rounds
    return q, floor, dutta_floor, exact, bound


def estimate(samples):
    mean = samples.mean(axis=0)
    half = 1.96 * samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    return mean, np.maximum(0, mean-half), mean+half


def simulate(events, alpha, times, rounds, reps, q, eta, aggregate_var, error0, seed):
    # Separate random streams: system timing is independent of SGD noise.
    streams = np.random.SeedSequence(seed).spawn(3)
    system, noise_time, noise_round = [np.random.default_rng(s) for s in streams]
    time_samples = np.empty((reps, len(times)))
    round_samples = np.empty((reps, len(rounds)))
    sd = np.sqrt(aggregate_var)  # reduction by K occurs ONCE
    for rep in range(reps):
        e = error0
        round_samples[rep, 0] = e*e
        for r in range(1, len(rounds)):
            e = q*e - eta*noise_round.normal(0, sd)
            round_samples[rep, r] = e*e
        state = int(system.choice(len(alpha), p=alpha))
        t, e, position = 0.0, error0, 0
        while position < len(times):
            row = events[state]
            rates = np.array([x[1] for x in row])
            total = rates.sum()
            next_t = t + system.exponential(1/total) if total else np.inf
            end = np.searchsorted(times, next_t, side='left')
            end = min(end, len(times))
            time_samples[rep, position:end] = e*e
            position = end
            if position == len(times):
                break
            j = int(system.choice(len(row), p=rates/total))
            state, _, marked = row[j]
            if marked:
                e = q*e - eta*noise_time.normal(0, sd)
            t = next_t
    return estimate(round_samples), estimate(time_samples)


def save_table(path, xname, x, exact, bound, stats, approximation=None):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([xname, 'exact_mse', 'dutta_upper_bound', 'sgd_mean',
                         'ci95_low', 'ci95_high'] + (['rate_approx_mse'] if approximation is not None else []))
        for i in range(len(x)):
            writer.writerow([x[i], exact[i], bound[i], *[v[i] for v in stats]] +
                             ([approximation[i]] if approximation is not None else []))


def run(config, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups = config['groups']
    a, eta = float(config['a']), float(config['eta'])
    sigma2, m = float(config['sigma2']), config['m']
    if not (a > 0 and 0 < eta <= 1/(2*a) and sigma2 >= 0 and m >= 1 and int(m) == m):
        raise ValueError('Require a>0, 0<eta<=1/(2a), sigma2>=0 and integer m>=1.')
    reps = config.get('replications', 1000)
    if reps < 2 or config['time_horizon'] <= 0 or config['time_points'] < 2 or config['rounds'] < 1:
        raise ValueError('Invalid replication/grid settings.')
    rounds = np.arange(config['rounds'] + 1)
    times = np.linspace(0, config['time_horizon'], config['time_points'])
    error0 = config['theta0'] - config['theta_star']
    fig_r, ax_r = plt.subplots(figsize=(8, 5))
    fig_t, ax_t = plt.subplots(figsize=(8, 5))
    summaries = []
    for index, cfg in enumerate(config['configurations']):
        K, B, J = (cfg[k] for k in ('K', 'B', 'J'))
        if any(not isinstance(v, int) or v < 1 for v in (K, B, J)):
            raise ValueError('K,B,J must be positive integers.')
        if K > sum(g[0] for g in groups)*B:
            raise ValueError('Infeasible K > N B.')
        model = HeterogeneousLumpedKBatchSyncCTMC(groups, K, B*J, B)
        events, H = event_model(model)
        initial = config.get('initial_system', 'stationary')
        if initial == 'stationary':
            alpha = model.solve_stationary_distribution()
            alpha = np.maximum(alpha, 0); alpha /= alpha.sum()
        elif initial == 'empty':
            alpha = np.zeros(len(model.states))
            empty = tuple(v for n, _ in groups for v in [n] + [0]*(B*J))
            alpha[model.state_to_index[empty]] = 1
        else:
            raise ValueError('initial_system must be stationary or empty.')
        pi = model.solve_stationary_distribution()
        rate = float(pi @ H @ np.ones(len(pi)))
        q, floor, dfloor, sr, dr = curves(a, eta, sigma2, m, K, error0, rounds)
        st = floor + (error0**2-floor)*pgf(model, H, alpha, q*q, times)
        dt = dfloor + (error0**2-dfloor)*pgf(model, H, alpha, q, times)
        approximate = floor + (error0**2-floor)*q**(2*rate*times)
        rs, ts = simulate(events, alpha, times, rounds, reps, q, eta,
                          sigma2/(m*K), error0, config.get('seed', 1234)+index)
        tag = f'K{K}_B{B}_J{J}'
        save_table(out/f'{index}_{tag}_round.csv', 'round', rounds, sr, dr, rs)
        save_table(out/f'{index}_{tag}_time.csv', 'time', times, st, dt, ts, approximate)
        color = f'C{index % 10}'
        for ax, x, exact, bound, stats in [(ax_r, rounds, sr, dr, rs), (ax_t, times, st, dt, ts)]:
            ax.plot(x, exact, color=color, label=f'{tag}: exact')
            ax.plot(x, bound, '--', color=color, label=f'{tag}: Dutta bound')
            stride = max(1, len(x)//25)
            ax.plot(x[::stride], stats[0][::stride], '.', color=color, label=f'{tag}: SGD')
            ax.fill_between(x, stats[1], stats[2], color=color, alpha=.15)
        if config.get('plot_rate_approximation', False):
            ax_t.plot(times, approximate, ':', color=color, label=f'{tag}: rate approx.')
        summaries.append(dict(configuration=cfg, b=B*J, states=len(pi),
            aggregation_rate=rate, aggregate_noise_var=sigma2/(m*K),
            exact_mse_floor=floor, dutta_mse_floor=dfloor,
            max_bound_violation=float(max(np.max(sr-dr), np.max(st-dt))),
            generator_row_error=float(np.max(abs(model.Q.sum(axis=1))))))
    for fig, ax, xlabel, name in [(fig_r, ax_r, 'Aggregation round', 'mse_round'),
                                  (fig_t, ax_t, 'Time', 'mse_time')]:
        ax.set(xlabel=xlabel, ylabel='Parameter MSE', yscale='log')
        ax.grid(True, which='both', alpha=.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{name}.{ext}', dpi=200)
        plt.close(fig)
    (out/'summary.json').write_text(json.dumps(dict(config=config, results=summaries,
        notes=['Single global learner; shared optimum; independent additive Gaussian noise; no staleness.',
               'CI: pointwise normal approximation across independent replications, not a simultaneous band.',
               'Time simulation uses CTMC completion channels, not independent DES validation.',
               'Dutta assumes M_G=0; sigma2 is per-sample variance; m is NOT b=BJ.']), indent=2))
    print(f'Outputs written to {out.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='quadratic_config.json')
    parser.add_argument('--out', default='quadratic_results')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.out)
