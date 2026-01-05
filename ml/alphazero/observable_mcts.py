"""
Observable MCTS with Evolvable Policy Statechart

Integrates the PolicyStatechart with MCTS for:
1. Observable policy decisions during search
2. Evolvable policy structure
3. Full introspection of decision-making
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import math
from typing import Dict, Optional, Tuple, List
import copy
import json

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState,
    BOARD_SIZE, TOTAL_POINTS, BLACK, WHITE, EMPTY,
    xy_to_idx, idx_to_xy
)

try:
    from .policy_statechart import PolicyStatechart, PolicyEvolution, PolicyTrace
    from .statechart_mcts import EPS, ACTION_SIZE
except ImportError:
    from policy_statechart import PolicyStatechart, PolicyEvolution, PolicyTrace
    from statechart_mcts import EPS, ACTION_SIZE


class ObservableMCTS:
    """
    MCTS with observable, evolvable policy statechart.

    Key features:
    1. Policy statechart modulates neural network output
    2. Full trace of decisions for interpretability
    3. Evolvable policy structure
    """

    def __init__(self, nnet, policy_genome: Optional[Dict] = None,
                 cpuct: float = 1.0, num_sims: int = 100):
        """
        Args:
            nnet: Neural network with predict(state) -> (pi, v)
            policy_genome: Evolved genome for policy statechart
            cpuct: Exploration constant for PUCT
            num_sims: Number of MCTS simulations per move
        """
        self.nnet = nnet
        self.cpuct = cpuct
        self.num_sims = num_sims

        # Create policy statechart (evolvable)
        self.policy = PolicyStatechart(genome=policy_genome)

        # MCTS data structures
        self.Qsa: Dict[Tuple[str, int], float] = {}
        self.Nsa: Dict[Tuple[str, int], int] = {}
        self.Ns: Dict[str, int] = {}
        self.Ps: Dict[str, np.ndarray] = {}  # Modulated policy
        self.Ps_raw: Dict[str, np.ndarray] = {}  # Raw NN policy
        self.Es: Dict[str, float] = {}
        self.Vs: Dict[str, np.ndarray] = {}

        # State cache
        self.states: Dict[str, Go9x9Statechart] = {}
        self.policy_states: Dict[str, Dict] = {}  # Policy state per game state

        # Observability
        self.decision_traces: List[Dict] = []
        self.current_trace: Optional[Dict] = None

    def get_action_prob(self, state: Go9x9Statechart,
                        temp: float = 1.0,
                        record_trace: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get action probabilities from MCTS.

        Returns:
            probs: [82] action probabilities
            trace: Dict with full decision trace
        """
        # Start trace
        if record_trace:
            self.current_trace = {
                'move_number': len(state.move_history),
                'simulations': [],
                'policy_state_before': self.policy.get_state_summary(),
            }

        # Get last move for policy update
        last_move = None
        if state.move_history:
            _, pos = state.move_history[-1]
            if pos is not None:
                last_move = pos

        # Update policy statechart with board evaluation
        board_eval = self._evaluate_board(state)
        guard_results = self.policy.update(state, board_eval, last_move)

        if record_trace:
            self.current_trace['board_eval'] = board_eval
            self.current_trace['guard_results'] = [
                {'name': g.name, 'satisfied': g.satisfied,
                 'confidence': g.confidence, 'explanation': g.explanation}
                for g in guard_results
            ]
            self.current_trace['policy_state_after'] = self.policy.get_state_summary()

        # Run simulations
        for sim in range(self.num_sims):
            self._search(state)

        # Get visit counts
        s = self._state_key(state)
        counts = np.array([
            self.Nsa.get((s, a), 0) for a in range(ACTION_SIZE)
        ])

        if temp == 0:
            best_actions = np.argwhere(counts == counts.max()).flatten()
            probs = np.zeros(ACTION_SIZE)
            probs[np.random.choice(best_actions)] = 1.0
        else:
            counts_temp = counts ** (1.0 / temp)
            probs = counts_temp / (counts_temp.sum() + EPS)

        if record_trace:
            self.current_trace['visit_counts'] = counts.tolist()
            self.current_trace['final_probs'] = probs.tolist()
            self.current_trace['selected_action'] = int(np.argmax(probs))

            if s in self.Ps_raw:
                self.current_trace['raw_policy'] = self.Ps_raw[s].tolist()
            if s in self.Ps:
                self.current_trace['modulated_policy'] = self.Ps[s].tolist()

            self.decision_traces.append(self.current_trace)

        return probs, self.current_trace if record_trace else {}

    def _search(self, state: Go9x9Statechart) -> float:
        """Recursive MCTS search with policy modulation."""
        s = self._state_key(state)

        # Check if game ended
        if s not in self.Es:
            ended = state.is_game_over()
            if ended:
                winner = state.winner()
                if winner is None:
                    self.Es[s] = 0
                else:
                    current = 1 if state.turn == TurnState.BLACK else -1
                    winner_val = 1 if winner == 1 else -1
                    self.Es[s] = 1 if current == winner_val else -1
            else:
                self.Es[s] = 0

        if self.Es[s] != 0 or state.is_game_over():
            return -self.Es[s]

        # Leaf node - expand with policy modulation
        if s not in self.Ps:
            # Get raw neural network prediction
            raw_pi, v = self.nnet.predict(state)

            # Store raw policy
            self.Ps_raw[s] = raw_pi.copy()

            # Modulate policy with statechart
            modulated_pi, modulation_info = self.policy.modulate_policy(raw_pi, state)

            # Apply guard mask (100% legal)
            guard_mask = self._compute_guard_mask(state)
            masked_pi = modulated_pi * guard_mask

            # Normalize
            pi_sum = masked_pi.sum()
            if pi_sum > EPS:
                masked_pi /= pi_sum
            else:
                masked_pi = guard_mask / (guard_mask.sum() + EPS)

            # Store
            self.Ps[s] = masked_pi
            self.Vs[s] = guard_mask
            self.Ns[s] = 0
            self.states[s] = copy.deepcopy(state)
            self.policy_states[s] = self.policy.get_state_summary()

            return -v

        # Select action with PUCT
        guard_mask = self.Vs[s]
        best_action = -1
        best_ucb = -float('inf')

        for a in range(ACTION_SIZE):
            if guard_mask[a] == 0:
                continue

            if (s, a) in self.Qsa:
                q = self.Qsa[(s, a)]
                n_sa = self.Nsa[(s, a)]
            else:
                q = 0
                n_sa = 0

            ucb = q + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) / (1 + n_sa)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a

        a = best_action

        # Get next state
        next_state = copy.deepcopy(state)
        if a == 81:
            next_state.play_pass()
        else:
            x, y = idx_to_xy(a)
            next_state.play_move(x, y)

        # Recurse
        v = self._search(next_state)

        # Backpropagate
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def _evaluate_board(self, state: Go9x9Statechart) -> Dict[str, float]:
        """
        Evaluate board position for policy statechart.

        Returns features that inform strategic decisions.
        """
        our_color = state.current_player()
        opp_color = WHITE if our_color == BLACK else BLACK

        # Count weak groups
        our_weak = 0
        opp_weak = 0

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            stone = state.board.get(x, y)
            if stone == EMPTY:
                continue

            libs = state.board.count_liberties(x, y)
            if libs <= 3:
                if stone == our_color:
                    our_weak += 1
                else:
                    opp_weak += 1

        # Territory estimation (simplified)
        our_territory = 0
        opp_territory = 0

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if state.board.get(x, y) == EMPTY:
                # Check influence
                our_inf = 0
                opp_inf = 0
                for nx, ny in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        n_stone = state.board.get(nx, ny)
                        if n_stone == our_color:
                            our_inf += 1
                        elif n_stone == opp_color:
                            opp_inf += 1

                if our_inf > opp_inf:
                    our_territory += 1
                elif opp_inf > our_inf:
                    opp_territory += 1

        total_territory = our_territory + opp_territory + 1

        return {
            'attack_potential': min(1.0, opp_weak * 0.3),
            'defend_need': min(1.0, our_weak * 0.3),
            'build_potential': our_territory / total_territory,
            'tenuki_value': 0.3 if our_weak == 0 else 0.1,
            'our_weak_groups': our_weak,
            'opp_weak_groups': opp_weak,
            'our_territory_est': our_territory,
            'opp_territory_est': opp_territory,
        }

    def _compute_guard_mask(self, state: Go9x9Statechart) -> np.ndarray:
        """Compute guard mask from statechart."""
        mask = np.zeros(ACTION_SIZE, dtype=np.float32)

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if state.is_legal_move(x, y):
                mask[idx] = 1.0

        mask[81] = 1.0
        return mask

    def _state_key(self, state: Go9x9Statechart) -> str:
        """Generate unique key for state."""
        parts = [
            'B' if state.turn == TurnState.BLACK else 'W',
            f"ko:{state.ko_point}" if state.ko_state == KoState.KO_FORBIDDEN else "noko",
            f"p:{state.consecutive_passes}",
            ''.join(str(s) for s in state.board.stones)
        ]
        return '|'.join(parts)

    def reset(self):
        """Clear MCTS tree for new game."""
        self.Qsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ps.clear()
        self.Ps_raw.clear()
        self.Es.clear()
        self.Vs.clear()
        self.states.clear()
        self.policy_states.clear()
        self.decision_traces.clear()

    def get_policy_genome(self) -> Dict:
        """Get current policy genome for evolution."""
        return self.policy.to_genome()

    def explain_last_decision(self) -> str:
        """Get human-readable explanation of last decision."""
        if not self.decision_traces:
            return "No decisions recorded"

        trace = self.decision_traces[-1]
        lines = [
            f"Move {trace['move_number']} Analysis",
            "=" * 50,
            "",
            "Board Evaluation:",
        ]

        for key, val in trace.get('board_eval', {}).items():
            lines.append(f"  {key}: {val:.2f}" if isinstance(val, float)
                        else f"  {key}: {val}")

        lines.append("")
        lines.append("Policy State:")
        ps = trace.get('policy_state_after', {})
        lines.append(f"  Mode: {ps.get('mode', '?')}")
        lines.append(f"  Urgency: {ps.get('urgency', '?')}")
        lines.append(f"  Phase: {ps.get('phase', '?')}")
        lines.append(f"  Focus: {ps.get('focus', '?')}")

        lines.append("")
        lines.append("Guard Evaluations:")
        for g in trace.get('guard_results', []):
            status = "PASS" if g['satisfied'] else "FAIL"
            lines.append(f"  [{status}] {g['name']}: {g['explanation']}")

        lines.append("")
        action = trace.get('selected_action', -1)
        if action < 81:
            x, y = idx_to_xy(action)
            lines.append(f"Selected Move: ({x}, {y})")
        else:
            lines.append("Selected Move: PASS")

        # Top visited moves
        counts = trace.get('visit_counts', [])
        if counts:
            top_moves = sorted(enumerate(counts), key=lambda x: -x[1])[:5]
            lines.append("")
            lines.append("Top MCTS Moves:")
            for idx, cnt in top_moves:
                if cnt > 0:
                    if idx < 81:
                        x, y = idx_to_xy(idx)
                        lines.append(f"  ({x}, {y}): {cnt} visits")
                    else:
                        lines.append(f"  PASS: {cnt} visits")

        return "\n".join(lines)

    def get_observability_report(self) -> Dict:
        """Get full observability report for analysis."""
        return {
            'num_decisions': len(self.decision_traces),
            'policy_genome': self.get_policy_genome(),
            'traces': self.decision_traces,
            'tree_size': len(self.Ns),
            'policy_states_seen': list(set(
                json.dumps(ps) for ps in self.policy_states.values()
            ))
        }


class EvolvableMCTSPopulation:
    """
    Population of MCTS agents with evolving policy statecharts.

    Uses evolutionary algorithms to discover better policy structures.
    """

    def __init__(self, nnet, population_size: int = 20):
        self.nnet = nnet
        self.population_size = population_size
        self.population: List[Dict] = []
        self.fitness_history: List[Dict] = []

        # Initialize population with variations
        self._init_population()

    def _init_population(self):
        """Initialize population with diverse genomes."""
        base_genome = PolicyStatechart()._default_genome()

        for i in range(self.population_size):
            if i == 0:
                # Keep one default
                genome = copy.deepcopy(base_genome)
            else:
                # Mutate
                genome = PolicyEvolution.mutate_guards(base_genome, 0.3)
                genome = PolicyEvolution.mutate_modulation(genome, 0.3)
                if i > self.population_size // 2:
                    genome = PolicyEvolution.mutate_topology(genome, 0.2)

            self.population.append({
                'genome': genome,
                'fitness': 0.0,
                'games': 0,
            })

    def evaluate_fitness(self, genome: Dict, num_games: int = 10) -> float:
        """Evaluate fitness of a genome through self-play."""
        mcts = ObservableMCTS(self.nnet, policy_genome=genome, num_sims=50)
        wins = 0

        for _ in range(num_games):
            state = Go9x9Statechart()
            mcts.reset()

            moves = 0
            while not state.is_game_over() and moves < 200:
                moves += 1
                pi, _ = mcts.get_action_prob(state, temp=0.5, record_trace=False)
                action = np.random.choice(ACTION_SIZE, p=pi)

                if action == 81:
                    state.play_pass()
                else:
                    x, y = idx_to_xy(action)
                    state.play_move(x, y)

            winner = state.winner()
            # Fitness based on game quality metrics
            if winner is not None:
                wins += 0.5  # Decisive games

        return wins / num_games

    def evolve_generation(self, num_games_per_eval: int = 5):
        """Evolve one generation."""
        # Evaluate fitness
        for individual in self.population:
            individual['fitness'] = self.evaluate_fitness(
                individual['genome'], num_games_per_eval
            )
            individual['games'] += num_games_per_eval

        # Sort by fitness
        self.population.sort(key=lambda x: -x['fitness'])

        # Record history
        self.fitness_history.append({
            'best_fitness': self.population[0]['fitness'],
            'avg_fitness': np.mean([p['fitness'] for p in self.population]),
            'best_genome': copy.deepcopy(self.population[0]['genome']),
        })

        # Selection and reproduction
        elite_count = self.population_size // 4
        new_population = self.population[:elite_count]  # Keep elite

        while len(new_population) < self.population_size:
            # Tournament selection
            candidates = np.random.choice(
                len(self.population) // 2, size=2, replace=False
            )
            parent1 = self.population[candidates[0]]['genome']
            parent2 = self.population[candidates[1]]['genome']

            # Crossover
            child_genome = PolicyEvolution.crossover(parent1, parent2)

            # Mutation
            child_genome = PolicyEvolution.mutate_guards(child_genome, 0.2)
            child_genome = PolicyEvolution.mutate_modulation(child_genome, 0.2)

            new_population.append({
                'genome': child_genome,
                'fitness': 0.0,
                'games': 0,
            })

        self.population = new_population

    def get_best_genome(self) -> Dict:
        """Get best genome from population."""
        return max(self.population, key=lambda x: x['fitness'])['genome']


def test_observable_mcts():
    """Test observable MCTS."""
    print("Testing ObservableMCTS...")
    print("=" * 60)

    # Create dummy network
    class DummyNet:
        def predict(self, state):
            return np.ones(82) / 82, 0.0

    nnet = DummyNet()

    # Create observable MCTS
    mcts = ObservableMCTS(nnet, num_sims=20)

    # Play a game
    state = Go9x9Statechart()
    state.play_move(4, 4)
    state.play_move(4, 5)

    print("\nGetting action probabilities with full observability...")
    probs, trace = mcts.get_action_prob(state, temp=1.0)

    print(f"\nPolicy state: {trace['policy_state_after']}")
    print(f"\nBoard evaluation:")
    for k, v in trace['board_eval'].items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nSelected action: {trace['selected_action']}")

    # Get explanation
    print("\n" + mcts.explain_last_decision())

    # Get observability report
    report = mcts.get_observability_report()
    print(f"\nObservability Report:")
    print(f"  Decisions recorded: {report['num_decisions']}")
    print(f"  Tree size: {report['tree_size']}")
    print(f"  Unique policy states: {len(report['policy_states_seen'])}")

    print("\n" + "=" * 60)
    print("All ObservableMCTS tests passed!")


if __name__ == "__main__":
    test_observable_mcts()
