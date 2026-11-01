"""
Heterogeneous Grouped Lumped Closed K-Batch-Sync CTMC

State:
    x = (x_{0,0}, ..., x_{0,b}, x_{1,0}, ..., x_{1,b}, ..., x_{G-1,b})

where:
    x_{g,j} = number of workers in group g at level j
    G = number of groups
    b = maximum queue level

Parameters:
    groups : list of tuples (N_g, mu_g) representing worker counts and service rates
    K : aggregation threshold
    b : maximum queue level
    B : number of levels removed by each batch synchronization
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral, Real
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np

GroupedState = Tuple[int, ...]


class SimpleProgressBar:
    def __init__(self, total, desc="Progress", width=50, color='white'):
        """
        Parameters:
        -----------
        total : int
            Total number of items
        desc : str
            Description text
        width : int
            Width of the progress bar
        color : str
            Color of the progress bar ('green', 'blue', 'red', 'yellow', 'magenta', 'cyan', 'white')
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        
        # Color mapping
        self.colors = {
            'green': '\033[92m',
            'blue': '\033[94m',
            'red': '\033[91m',
            'yellow': '\033[93m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'orange': '\033[38;5;208m',  # Orange
            'purple': '\033[38;5;129m',  # Purple
        }
        self.color_code = self.colors.get(color, '\033[0m')
        self.reset = '\033[0m'
        
    def update(self, n=1):
        self.current += n
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = '█' * filled + '░' * (self.width - filled)
        # Colored bar
        print(f"\r{self.desc}: {self.color_code}|{bar}|{self.reset} {percent*100:5.1f}% [{self.current}/{self.total}]", end='')
        if self.current >= self.total:
            print()
    
    def close(self):
        if self.current < self.total:
            print()

@dataclass
class LumpedCTMCResult:
    states: List[GroupedState]
    state_to_index: Dict[GroupedState, int]
    Q: np.ndarray
    stationary: np.ndarray

    mean_occupancy: np.ndarray  # shape: (G, b + 1)
    mean_total_pending: float
    mean_aggregation_score: float
    aggregation_rate: float
    mean_aggregation_interval: float

    max_row_sum_error: float
    stationary_residual: float

    trigger_rate_by_group: np.ndarray | None = None
    trigger_group_probability: np.ndarray | None = None
    beta2_trigger: float = 0.0
    aggregation_rate_by_group: np.ndarray | None = None
    group_weights: np.ndarray | None = None
    pg: np.ndarray | None = None
    beta2: float = 0.0

    update_consumption_rate_by_group: np.ndarray | None = None
    mean_consumed_updates_by_group_per_aggregation: np.ndarray | None = None
    participation_composition_mean: np.ndarray | None = None
    participation_second_moment: np.ndarray | None = None
    participation_covariance: np.ndarray | None = None
    participation_compositions: np.ndarray | None = None
    participation_composition_probabilities: np.ndarray | None = None
    beta2_participation_bias: float = 0.0
    beta2_participation_variance: float = 0.0
    beta2_participation_total: float = 0.0
    beta2_decomposition_error: float = 0.0
    update_rate_balance_error: float = 0.0
    participation_sum_error: float = 0.0
    composition_probability_sum_error: float = 0.0

    group_utilization: np.ndarray | None = None
    overall_utilization: float = 0.0


class HeterogeneousLumpedKBatchSyncCTMC:
    def __init__(
        self,
        groups: List[Tuple[int, float]],
        K: int,
        b: int,
        B: int,
    ):
        self.groups = [(count, mu) for count, mu in groups]
        self.K = K
        self.b = b
        self.B = B

        self._validate_parameters()

        self.G = len(self.groups)
        self.N = sum(count for count, _ in self.groups)
        self.mus = np.asarray([mu for _, mu in self.groups], dtype=float)
        self.group_sizes = tuple(count for count, _ in self.groups)
        self.group_weights = np.asarray([count / self.N for count, _ in self.groups], dtype=float)
        self.state_dim = self.G * (self.b + 1)

        self.states = self._enumerate_states()
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.Q = self._build_generator()

    def _validate_parameters(self) -> None:
        if not self.groups:
            raise ValueError("Groups list cannot be empty.")
        if not isinstance(self.K, Integral) or self.K < 1:
            raise ValueError("K must be an integer >= 1.")
        if not isinstance(self.b, Integral) or self.b < 1:
            raise ValueError("b must be an integer >= 1.")
        if not isinstance(self.B, Integral) or self.B < 1:
            raise ValueError("B must be an integer >= 1.")

        for g_idx, (count, mu) in enumerate(self.groups):
            if not isinstance(count, Integral) or count < 1 or not np.isfinite(count):
                raise ValueError(f"Group {g_idx} count must be a finite integer >= 1.")
            if not isinstance(mu, Real) or not np.isfinite(mu) or mu <= 0.0:
                raise ValueError(f"Group {g_idx} mu must be finite and > 0.")

        self.K = int(self.K)
        self.b = int(self.b)
        self.B = int(self.B)
        self.groups = [(int(count), float(mu)) for count, mu in self.groups]

    def _group_occupancies(self, count: int) -> List[Tuple[int, ...]]:
        return [p for p in product(range(count + 1), repeat=self.b + 1) if sum(p) == count]

    def aggregation_score(self, state: GroupedState) -> int:
        return sum(
            min(level, self.B) * state[g * (self.b + 1) + level]
            for g in range(self.G)
            for level in range(self.b + 1)
        )

    def group_score_contributions(self, state: GroupedState) -> np.ndarray:
        contributions = np.zeros(self.G, dtype=int)
        for g in range(self.G):
            offset = g * (self.b + 1)
            contributions[g] = sum(
                min(level, self.B) * state[offset + level]
                for level in range(self.b + 1)
            )
        return contributions

    def completion_event_info(
        self, state: GroupedState, group: int, level: int
    ) -> Tuple[GroupedState, GroupedState, bool, np.ndarray, np.ndarray]:
        progressed = self.normal_transition(state, group, level)
        triggered = self.aggregation_score(state) < self.K <= self.aggregation_score(progressed)
        target = self.restart(progressed) if triggered else progressed
        update_vector = self.group_score_contributions(progressed)
        if triggered:
            total_updates = int(np.sum(update_vector))
            if total_updates != self.K:
                raise RuntimeError(
                    "Triggered event produced an invalid update total: "
                    f"sum(U)={total_updates}, expected K={self.K}, "
                    f"state={state}, group={group}, level={level}."
                )
        composition = update_vector / float(self.K)
        return progressed, target, triggered, update_vector, composition

    def _enumerate_states(self) -> List[GroupedState]:
        per_group_occupancies = [self._group_occupancies(n_g) for n_g in self.group_sizes]
        valid_states = []
        
        total_combinations = 1
        for occ in per_group_occupancies:
            total_combinations *= len(occ)
        
        progress = SimpleProgressBar(total=total_combinations, desc="Enumerating states", color='orange')
        
        for group_states in product(*per_group_occupancies):
            flat_state = tuple(value for g_state in group_states for value in g_state)
            if self.aggregation_score(flat_state) < self.K:
                valid_states.append(flat_state)
            progress.update()
        
        progress.close()
        
        if not valid_states:
            raise RuntimeError("No valid states were generated.")
        return sorted(valid_states)


    def positive_worker_count(self, state: GroupedState) -> int:
        return sum(
            state[g * (self.b + 1) + level]
            for g in range(self.G)
            for level in range(1, self.b + 1)
        )

    def normal_transition(self, state: GroupedState, group: int, level: int) -> GroupedState:
        offset = group * (self.b + 1)
        new_state = list(state)
        new_state[offset + level] -= 1
        new_state[offset + level + 1] += 1
        return tuple(new_state)

    def restart(self, state: GroupedState) -> GroupedState:
        new_state = [0] * self.state_dim
        for g in range(self.G):
            offset = g * (self.b + 1)
            for level in range(self.b + 1):
                new_level = max(level - self.B, 0)
                new_state[offset + new_level] += state[offset + level]
        return tuple(new_state)

    def transition_target(self, state: GroupedState, group: int, level: int) -> Tuple[GroupedState, bool]:
        progressed, target, triggered, _, _ = self.completion_event_info(state, group, level)
        return target, triggered

    def _build_generator(self) -> np.ndarray:
        n_states = len(self.states)
        Q = np.zeros((n_states, n_states), dtype=float)

        progress = SimpleProgressBar(total=n_states, desc="Building generator", color='blue')

        for row, state in enumerate(self.states):
            outgoing_rate = 0.0

            for g in range(self.G):
                offset = g * (self.b + 1)

                for level in range(self.b):
                    count = state[offset + level]
                    if count == 0:
                        continue

                    event_rate = count * self.mus[g]
                    target, _ = self.transition_target(state, g, level)
                    col = self.state_to_index.get(target)
                    if col is None:
                        raise RuntimeError(
                            f"Transition target {target} is missing from the state map "
                            f"(source={state}, group={g}, level={level})."
                        )

                    if col == row:
                        continue

                    Q[row, col] += event_rate
                    outgoing_rate += event_rate

            Q[row, row] = -outgoing_rate
            progress.update()

        progress.close()
        return Q

    # def solve_stationary_distribution(self) -> np.ndarray:
    #     n_states = len(self.states)
    #     A = self.Q.T.copy()
    #     rhs = np.zeros(n_states)
    #     A[-1, :] = 1.0
    #     rhs[-1] = 1.0
    #     pi = np.linalg.solve(A, rhs)
    #     negative = pi < -1e-10
    #     if np.any(negative):
    #         raise RuntimeError(
    #             "Stationary solve produced materially negative probabilities: "
    #             f"min={float(np.min(pi)):.3e}."
    #         )
    #     pi[np.abs(pi) <= 1e-10] = 0.0
    #     total = float(np.sum(pi))
    #     if not np.isfinite(total) or total <= 0.0:
    #         raise RuntimeError("Stationary probabilities have non-positive or invalid sum.")
    #     pi /= total
    #     return pi

    def solve_stationary_distribution(self) -> np.ndarray:
        n_states = len(self.states)
        
        # Show progress bar for building the system matrix
        progress = SimpleProgressBar(
            total=3,  # 3 steps: build, solve, post-process
            desc="Solving stationary distribution",
            color='yellow',
            width=40
        )
        
        # Step 1: Build system matrix
        progress.update()
        A = self.Q.T.copy()
        rhs = np.zeros(n_states)
        A[-1, :] = 1.0
        rhs[-1] = 1.0
        
        # Step 2: Solve
        progress.update()
        pi = np.linalg.solve(A, rhs)
        
        # Step 3: Post-process
        progress.update()
        negative = pi < -1e-10
        if np.any(negative):
            raise RuntimeError(
                "Stationary solve produced materially negative probabilities: "
                f"min={float(np.min(pi)):.3e}."
            )
        pi[np.abs(pi) <= 1e-10] = 0.0
        total = float(np.sum(pi))
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("Stationary probabilities have non-positive or invalid sum.")
        pi /= total
        
        progress.close()
        return pi

    def _validate_metric(self, error: float, name: str, scale: float = 1.0, tolerance: float = 1e-9) -> None:
        limit = tolerance * max(1.0, scale)
        if error > limit:
            raise RuntimeError(f"{name}={error:.3e} exceeds tolerance {limit:.3e}.")

    def compute_result(self) -> LumpedCTMCResult:
        pi = self.solve_stationary_distribution()
        mean_occupancy = np.zeros((self.G, self.b + 1), dtype=float)
        mean_aggregation_score = 0.0
        trigger_rate_by_group = np.zeros(self.G, dtype=float)
        update_consumption_rate_by_group = np.zeros(self.G, dtype=float)
        composition_mass: Dict[Tuple[int, ...], float] = {}
        participation_mean_numerator = np.zeros(self.G, dtype=float)
        participation_second_moment_numerator = np.zeros((self.G, self.G), dtype=float)
        participation_total_numerator = 0.0
        utilization_mass = np.zeros(self.G, dtype=float)
        A = 0.0

        progress = SimpleProgressBar(total=len(self.states), desc="Computing Palm metrics", color='magenta')
        for idx, state in enumerate(self.states):
            progress.update()
            prob = pi[idx]
            mean_aggregation_score += prob * self.aggregation_score(state)
            for g in range(self.G):
                offset = g * (self.b + 1)
                mean_occupancy[g, :] += prob * np.asarray(state[offset:offset + self.b + 1], dtype=float)
                utilization_mass[g] += prob * state[offset]
                for level in range(self.b):
                    count = state[offset + level]
                    if count == 0:
                        continue
                    event_flux = prob * count * self.mus[g]
                    progressed, _, triggered, update_vector, composition = self.completion_event_info(state, g, level)
                    if triggered:
                        A += event_flux
                        trigger_rate_by_group[g] += event_flux
                        update_consumption_rate_by_group += event_flux * update_vector
                        participation_mean_numerator += event_flux * composition
                        participation_second_moment_numerator += event_flux * np.outer(composition, composition)
                        participation_total_numerator += event_flux * float(np.sum((composition - self.group_weights) ** 2))
                        key = tuple(int(v) for v in update_vector.tolist())
                        composition_mass[key] = composition_mass.get(key, 0.0) + event_flux

        mean_total_pending = float(np.sum(mean_occupancy[:, 1:] * np.arange(1, self.b + 1, dtype=float)))
        mean_aggregation_score = float(mean_aggregation_score)
        aggregation_rate = float(A)
        mean_aggregation_interval = 1.0 / aggregation_rate if aggregation_rate > 0.0 else np.inf

        trigger_group_probability = (
            trigger_rate_by_group / aggregation_rate if aggregation_rate > 0.0 else np.zeros(self.G, dtype=float)
        )
        beta2_trigger = float(np.sum((trigger_group_probability - self.group_weights) ** 2)) if aggregation_rate > 0.0 else 0.0
        aggregation_rate_by_group = trigger_rate_by_group.copy()
        pg = trigger_group_probability.copy()
        beta2 = beta2_trigger

        if aggregation_rate > 0.0:
            # Normalize all Palm quantities exactly once by the common event-flux denominator A.
            mean_consumed_updates_by_group_per_aggregation = update_consumption_rate_by_group / A
            participation_composition_mean = participation_mean_numerator / A
            participation_second_moment = participation_second_moment_numerator / A
            participation_covariance = participation_second_moment - np.outer(
                participation_composition_mean, participation_composition_mean
            )
            participation_covariance = 0.5 * (participation_covariance + participation_covariance.T)
            participation_compositions = np.asarray(list(composition_mass.keys()), dtype=int) if composition_mass else np.zeros((0, self.G), dtype=int)
            composition_probabilities_raw = np.asarray(list(composition_mass.values()), dtype=float) if composition_mass else np.zeros(0, dtype=float)
            participation_composition_probabilities = composition_probabilities_raw / A if composition_mass else np.zeros(0, dtype=float)
            composition_probability_sum_error = float(abs(float(np.sum(participation_composition_probabilities)) - 1.0)) if composition_mass else 0.0
            participation_sum_error = float(abs(float(np.sum(participation_composition_mean)) - 1.0))
            beta2_participation_bias = float(np.sum((participation_composition_mean - self.group_weights) ** 2))
            beta2_participation_variance = float(np.trace(participation_covariance))
            beta2_participation_total = float(participation_total_numerator / A)
            beta2_decomposition_error = float(abs(beta2_participation_total - beta2_participation_bias - beta2_participation_variance))
            update_rate_balance_error = float(abs(float(np.sum(update_consumption_rate_by_group)) - self.K * aggregation_rate))
            participation_sum_error = float(abs(float(np.sum(participation_composition_mean)) - 1.0))
        else:
            mean_consumed_updates_by_group_per_aggregation = np.zeros(self.G, dtype=float)
            participation_composition_mean = np.zeros(self.G, dtype=float)
            participation_second_moment = np.zeros((self.G, self.G), dtype=float)
            participation_covariance = np.zeros((self.G, self.G), dtype=float)
            participation_compositions = np.zeros((0, self.G), dtype=int)
            participation_composition_probabilities = np.zeros(0, dtype=float)
            beta2_participation_bias = 0.0
            beta2_participation_variance = 0.0
            beta2_participation_total = 0.0
            beta2_decomposition_error = 0.0
            update_rate_balance_error = 0.0
            participation_sum_error = 0.0
            composition_probability_sum_error = 0.0

        progress.close()

        group_utilization = 1.0 - utilization_mass / np.asarray(self.group_sizes, dtype=float)
        overall_utilization = float(np.dot(np.asarray(self.group_sizes, dtype=float), group_utilization) / self.N)

        max_row_sum_error = float(np.max(np.abs(np.sum(self.Q, axis=1))))
        stationary_residual = float(np.max(np.abs(pi @ self.Q)))

        self._validate_metric(max_row_sum_error, "row_sum_error", scale=float(np.max(np.abs(self.Q))))
        self._validate_metric(stationary_residual, "stationary_residual", scale=float(np.max(np.abs(pi)) * np.max(np.abs(self.Q)) * max(1, self.Q.shape[0])))
        if aggregation_rate > 0.0:
            self._validate_metric(update_rate_balance_error, "update_rate_balance_error", scale=self.K * aggregation_rate)
            self._validate_metric(participation_sum_error, "participation_sum_error")
            self._validate_metric(beta2_decomposition_error, "beta2_decomposition_error", scale=max(1.0, beta2_participation_total))
            self._validate_metric(composition_probability_sum_error, "composition_probability_sum_error")

        return LumpedCTMCResult(
            states=self.states,
            state_to_index=self.state_to_index,
            Q=self.Q,
            stationary=pi,
            mean_occupancy=mean_occupancy,
            mean_total_pending=mean_total_pending,
            mean_aggregation_score=float(mean_aggregation_score),
            aggregation_rate=aggregation_rate,
            mean_aggregation_interval=float(mean_aggregation_interval),
            max_row_sum_error=max_row_sum_error,
            stationary_residual=stationary_residual,
            trigger_rate_by_group=trigger_rate_by_group,
            trigger_group_probability=trigger_group_probability,
            beta2_trigger=beta2_trigger,
            aggregation_rate_by_group=aggregation_rate_by_group,
            group_weights=self.group_weights,
            pg=pg,
            beta2=beta2,
            update_consumption_rate_by_group=update_consumption_rate_by_group,
            mean_consumed_updates_by_group_per_aggregation=mean_consumed_updates_by_group_per_aggregation,
            participation_composition_mean=participation_composition_mean,
            participation_second_moment=participation_second_moment,
            participation_covariance=participation_covariance,
            participation_compositions=participation_compositions,
            participation_composition_probabilities=participation_composition_probabilities,
            beta2_participation_bias=beta2_participation_bias,
            beta2_participation_variance=beta2_participation_variance,
            beta2_participation_total=beta2_participation_total,
            beta2_decomposition_error=beta2_decomposition_error,
            update_rate_balance_error=update_rate_balance_error,
            participation_sum_error=participation_sum_error,
            composition_probability_sum_error=composition_probability_sum_error,
            group_utilization=group_utilization,
            overall_utilization=overall_utilization,
        )

    def print_report(self, result: LumpedCTMCResult) -> None:
        print("=" * 72)
        print("Heterogeneous Grouped Lumped Closed K-Batch-Sync CTMC")
        print("=" * 72)
        print(f"Groups: {self.groups}")
        print(f"Total Workers (N): {self.N}")
        print(f"K = {self.K}, b = {self.b}, B = {self.B}")
        print(f"Number of lumped states: {len(self.states)}")
        print("-" * 72)
        print("Mean Occupancies E[x_{g,j}]:")
        for g in range(self.G):
            values = ", ".join(
                f"level {j}: {result.mean_occupancy[g, j]:.6f}"
                for j in range(self.b + 1)
            )
            print(f"  Group {g} (N_g={self.group_sizes[g]}, mu_g={self.mus[g]}): {values}")
        print("\nTrigger metrics:")
        print(f"  aggregation_rate = {result.aggregation_rate:.6f}")
        print(f"  mean_aggregation_interval = {result.mean_aggregation_interval:.6f}")
        print(f"  trigger_rate_by_group = {np.array2string(result.trigger_rate_by_group, precision=6)}")
        print(f"  trigger_group_probability = {np.array2string(result.trigger_group_probability, precision=6)}")
        print(f"  beta2_trigger = {result.beta2_trigger:.6f}")
        print(f"  backward-compatible aggregation_rate_by_group = {np.array2string(result.aggregation_rate_by_group, precision=6)}")
        print(f"  backward-compatible pg = {np.array2string(result.pg, precision=6)}")
        print(f"  backward-compatible beta2 = {result.beta2:.6f}")
        print("\nParticipation metrics:")
        print(f"  update_consumption_rate_by_group = {np.array2string(result.update_consumption_rate_by_group, precision=6)}")
        print(f"  mean_consumed_updates_by_group_per_aggregation = {np.array2string(result.mean_consumed_updates_by_group_per_aggregation, precision=6)}")
        print(f"  participation_composition_mean = {np.array2string(result.participation_composition_mean, precision=6)}")
        print(f"  participation_second_moment =\n{np.array2string(result.participation_second_moment, precision=6)}")
        print(f"  participation_covariance =\n{np.array2string(result.participation_covariance, precision=6)}")
        print(f"  participation_compositions = {np.array2string(result.participation_compositions, separator=', ') if result.participation_compositions is not None else '[]'}")
        print(f"  participation_composition_probabilities = {np.array2string(result.participation_composition_probabilities, precision=6)}")
        print(f"  beta2_participation_bias = {result.beta2_participation_bias:.6f}")
        print(f"  beta2_participation_variance = {result.beta2_participation_variance:.6f}")
        print(f"  beta2_participation_total = {result.beta2_participation_total:.6f}")
        print(f"  beta2_decomposition_error = {result.beta2_decomposition_error:.3e}")
        print("\nUtilization:")
        print(f"  group_utilization = {np.array2string(result.group_utilization, precision=6)}")
        print(f"  overall_utilization = {result.overall_utilization:.6f}")
        print("\nDiagnostics:")
        print(f"  mean_total_pending = {result.mean_total_pending:.6f}")
        print(f"  mean_aggregation_score = {result.mean_aggregation_score:.6f}")
        print(f"  update_rate_balance_error = {result.update_rate_balance_error:.3e}")
        print(f"  participation_sum_error = {result.participation_sum_error:.3e}")
        print(f"  composition_probability_sum_error = {result.composition_probability_sum_error:.3e}")
        print(f"  max_row_sum_error = {result.max_row_sum_error:.3e}")
        print(f"  stationary_residual = {result.stationary_residual:.3e}")
        print("=" * 72)


def _run_self_tests() -> None:
    model = HeterogeneousLumpedKBatchSyncCTMC(groups=[(4, 1.0), (3, 1.5), (3, 2.0)], K=10, b=1, B=1)
    result = model.compute_result()
    expected = np.array([0.4, 0.3, 0.3])
    assert np.allclose(result.participation_composition_mean, expected, atol=1e-9)
    assert np.allclose(result.beta2_participation_bias, 0.0, atol=1e-9)
    assert np.allclose(result.beta2_participation_variance, 0.0, atol=1e-9)
    assert np.allclose(result.beta2_participation_total, 0.0, atol=1e-9)
    assert result.aggregation_rate >= 0.0
    assert result.max_row_sum_error < 1e-9
    assert result.stationary_residual < 1e-9

    small = HeterogeneousLumpedKBatchSyncCTMC(groups=[(2, 1.0), (1, 2.0)], K=2, b=2, B=2)
    small_result = small.compute_result()
    assert abs(float(np.sum(small_result.update_consumption_rate_by_group)) - small.K * small_result.aggregation_rate) < 1e-9
    if small_result.aggregation_rate > 0:
        assert abs(float(np.sum(small_result.participation_composition_mean)) - 1.0) < 1e-9
        assert abs(float(np.sum(small_result.participation_composition_probabilities)) - 1.0) < 1e-9
    assert small_result.max_row_sum_error < 1e-9
    assert small_result.stationary_residual < 1e-9


def main() -> None:
    model = HeterogeneousLumpedKBatchSyncCTMC(
        groups=[(4, 1.0), (3, 1.5), (3, 2.0)],
        K=10,
        b=1,
        B=1,
    )
    result = model.compute_result()
    model.print_report(result)
    _run_self_tests()


if __name__ == "__main__":
    main()
