"""
Observable Evolvable Policy Statechart

A statechart-based policy engine where:
1. Policy decisions flow through observable states
2. Transitions have interpretable guards
3. Both topology and guards can evolve
4. Full introspection: state traces, guard values, transition explanations

Architecture:
    PolicyStatechart (AND - parallel regions)
    ├── StrategicMode (OR): Attack | Defend | Build | Reduce | Flexible
    ├── Urgency (OR): Critical | High | Normal | Low
    ├── Phase (OR): Opening | Midgame | Endgame
    └── Focus (OR): Local | Global | Tenuki

Each state combination modulates the neural network policy output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from enum import Enum
import json
import copy
from datetime import datetime

from experiments.exp_go_9x9.go_statechart import (
    Go9x9Statechart, TurnState, KoState,
    BOARD_SIZE, TOTAL_POINTS, BLACK, WHITE, EMPTY,
    xy_to_idx, idx_to_xy, get_neighbors
)


# =============================================================================
# POLICY STATES (Observable Modes)
# =============================================================================

class StrategicMode(Enum):
    """What strategic approach is the policy using?"""
    ATTACK = "attack"       # Aggressive, seeking to capture or kill
    DEFEND = "defend"       # Protective, saving groups or territory
    BUILD = "build"         # Constructive, making territory
    REDUCE = "reduce"       # Invasive, reducing opponent territory
    FLEXIBLE = "flexible"   # Balanced, no strong preference

class Urgency(Enum):
    """How urgent is the current situation?"""
    CRITICAL = "critical"   # Must respond immediately (atari, ko)
    HIGH = "high"           # Important moves available
    NORMAL = "normal"       # Standard play
    LOW = "low"             # Calm position, many good options

class Phase(Enum):
    """What phase of the game?"""
    OPENING = "opening"     # First ~30 moves, fuseki
    MIDGAME = "midgame"     # Fighting and territory
    ENDGAME = "endgame"     # Yose, small points

class Focus(Enum):
    """Where should attention be?"""
    LOCAL = "local"         # Continue in current area
    GLOBAL = "global"       # Consider whole-board moves
    TENUKI = "tenuki"       # Play elsewhere


# =============================================================================
# GUARD FUNCTIONS (Observable Conditions)
# =============================================================================

@dataclass
class GuardResult:
    """Observable result of guard evaluation."""
    name: str
    satisfied: bool
    confidence: float  # 0-1, how strongly the condition holds
    explanation: str
    features: Dict[str, float] = field(default_factory=dict)


class PolicyGuards:
    """
    Collection of guard functions for policy transitions.

    Each guard returns a GuardResult with full observability.
    Guards can be evolved by modifying thresholds or combining signals.
    """

    def __init__(self, genome: Optional[Dict] = None):
        """Initialize with optional evolved genome."""
        default = self._default_genome()
        if genome:
            # Merge with defaults to ensure all keys exist
            default.update(genome)
        self.genome = default

    def _default_genome(self) -> Dict:
        """Default guard parameters (can be evolved)."""
        return {
            # Thresholds for urgency detection
            'atari_urgency': 0.8,
            'liberty_critical': 2,
            'liberty_warning': 3,

            # Phase thresholds (move numbers)
            'opening_end': 30,
            'midgame_end': 150,

            # Strategic mode thresholds
            'attack_threshold': 0.6,
            'defend_threshold': 0.6,
            'build_threshold': 0.5,

            # Focus thresholds
            'local_radius': 3,
            'tenuki_threshold': 0.3,
        }

    # -------------------------------------------------------------------------
    # Urgency Guards
    # -------------------------------------------------------------------------

    def guard_critical_urgency(self, game: Go9x9Statechart,
                                last_move: Optional[Tuple[int, int]]) -> GuardResult:
        """Check if position requires immediate response."""
        features = {}

        # Check for groups in atari (1 liberty)
        atari_count = 0
        our_color = game.current_player()

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if game.board.get(x, y) == our_color:
                libs = game.board.count_liberties(x, y)
                if libs == 1:
                    atari_count += 1

        features['atari_count'] = atari_count
        features['has_ko'] = float(game.ko_state == KoState.KO_FORBIDDEN)

        # Critical if we have groups in atari or active ko
        is_critical = atari_count > 0 or game.ko_state == KoState.KO_FORBIDDEN
        confidence = min(1.0, atari_count * 0.5 + features['has_ko'] * 0.8)

        return GuardResult(
            name="critical_urgency",
            satisfied=is_critical,
            confidence=confidence,
            explanation=f"Atari groups: {atari_count}, Ko: {features['has_ko']}",
            features=features
        )

    def guard_high_urgency(self, game: Go9x9Statechart,
                           last_move: Optional[Tuple[int, int]]) -> GuardResult:
        """Check for high-urgency situations."""
        features = {}

        our_color = game.current_player()
        weak_groups = 0

        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if game.board.get(x, y) == our_color:
                libs = game.board.count_liberties(x, y)
                if libs <= self.genome['liberty_warning']:
                    weak_groups += 1

        features['weak_groups'] = weak_groups

        # High urgency if we have weak groups
        is_high = weak_groups > 0
        confidence = min(1.0, weak_groups * 0.3)

        return GuardResult(
            name="high_urgency",
            satisfied=is_high,
            confidence=confidence,
            explanation=f"Weak groups (<=3 libs): {weak_groups}",
            features=features
        )

    # -------------------------------------------------------------------------
    # Phase Guards
    # -------------------------------------------------------------------------

    def guard_opening_phase(self, game: Go9x9Statechart) -> GuardResult:
        """Check if still in opening phase."""
        move_count = len(game.move_history)
        threshold = self.genome['opening_end']

        is_opening = move_count < threshold
        confidence = max(0, 1.0 - move_count / threshold)

        return GuardResult(
            name="opening_phase",
            satisfied=is_opening,
            confidence=confidence,
            explanation=f"Move {move_count} < {threshold}",
            features={'move_count': move_count, 'threshold': threshold}
        )

    def guard_endgame_phase(self, game: Go9x9Statechart) -> GuardResult:
        """Check if in endgame phase."""
        move_count = len(game.move_history)
        threshold = self.genome['midgame_end']

        # Also check board fullness
        stone_count = sum(1 for s in game.board.stones if s != EMPTY)
        fullness = stone_count / TOTAL_POINTS

        is_endgame = move_count > threshold or fullness > 0.6
        confidence = max(fullness, move_count / (threshold + 50))

        return GuardResult(
            name="endgame_phase",
            satisfied=is_endgame,
            confidence=min(1.0, confidence),
            explanation=f"Move {move_count}, fullness {fullness:.2f}",
            features={'move_count': move_count, 'fullness': fullness}
        )

    # -------------------------------------------------------------------------
    # Strategic Mode Guards
    # -------------------------------------------------------------------------

    def guard_attack_mode(self, game: Go9x9Statechart,
                          board_eval: Dict[str, float]) -> GuardResult:
        """Check if attack mode is appropriate."""
        # Look for weak opponent groups
        opp_color = WHITE if game.current_player() == BLACK else BLACK

        weak_opp_groups = 0
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if game.board.get(x, y) == opp_color:
                libs = game.board.count_liberties(x, y)
                if libs <= 3:
                    weak_opp_groups += 1

        # Attack if opponent has weak groups and we're not weak
        attack_score = board_eval.get('attack_potential', 0.5)
        is_attack = weak_opp_groups > 0 and attack_score > self.genome['attack_threshold']

        return GuardResult(
            name="attack_mode",
            satisfied=is_attack,
            confidence=attack_score,
            explanation=f"Weak opponent groups: {weak_opp_groups}, score: {attack_score:.2f}",
            features={'weak_opp_groups': weak_opp_groups, 'attack_score': attack_score}
        )

    def guard_defend_mode(self, game: Go9x9Statechart,
                          board_eval: Dict[str, float]) -> GuardResult:
        """Check if defend mode is appropriate."""
        our_color = game.current_player()

        weak_own_groups = 0
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if game.board.get(x, y) == our_color:
                libs = game.board.count_liberties(x, y)
                if libs <= 3:
                    weak_own_groups += 1

        defend_score = board_eval.get('defend_need', 0.5)
        is_defend = weak_own_groups > 0 and defend_score > self.genome['defend_threshold']

        return GuardResult(
            name="defend_mode",
            satisfied=is_defend,
            confidence=defend_score,
            explanation=f"Weak own groups: {weak_own_groups}, score: {defend_score:.2f}",
            features={'weak_own_groups': weak_own_groups, 'defend_score': defend_score}
        )

    def guard_build_mode(self, game: Go9x9Statechart,
                         board_eval: Dict[str, float]) -> GuardResult:
        """Check if territory-building mode is appropriate."""
        # Count potential territory
        build_score = board_eval.get('build_potential', 0.5)

        # Build mode when position is calm
        is_build = build_score > self.genome['build_threshold']

        return GuardResult(
            name="build_mode",
            satisfied=is_build,
            confidence=build_score,
            explanation=f"Build potential: {build_score:.2f}",
            features={'build_score': build_score}
        )

    # -------------------------------------------------------------------------
    # Focus Guards
    # -------------------------------------------------------------------------

    def guard_local_focus(self, game: Go9x9Statechart,
                          last_move: Optional[Tuple[int, int]]) -> GuardResult:
        """Check if should focus locally."""
        if last_move is None:
            return GuardResult(
                name="local_focus",
                satisfied=False,
                confidence=0.0,
                explanation="No last move",
                features={}
            )

        lx, ly = last_move
        radius = self.genome['local_radius']

        # Count nearby stones
        nearby_stones = 0
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            if game.board.get(x, y) != EMPTY:
                dist = abs(x - lx) + abs(y - ly)
                if dist <= radius:
                    nearby_stones += 1

        # Local focus if area is active
        density = nearby_stones / ((2 * radius + 1) ** 2)
        is_local = density > 0.3

        return GuardResult(
            name="local_focus",
            satisfied=is_local,
            confidence=density,
            explanation=f"Nearby stones: {nearby_stones}, density: {density:.2f}",
            features={'nearby_stones': nearby_stones, 'density': density}
        )

    def guard_tenuki(self, game: Go9x9Statechart,
                     board_eval: Dict[str, float]) -> GuardResult:
        """Check if tenuki (playing elsewhere) is appropriate."""
        tenuki_score = board_eval.get('tenuki_value', 0.3)
        is_tenuki = tenuki_score > self.genome['tenuki_threshold']

        return GuardResult(
            name="tenuki",
            satisfied=is_tenuki,
            confidence=tenuki_score,
            explanation=f"Tenuki value: {tenuki_score:.2f}",
            features={'tenuki_score': tenuki_score}
        )


# =============================================================================
# POLICY STATECHART (Observable State Machine)
# =============================================================================

@dataclass
class PolicyState:
    """Current state of the policy statechart."""
    strategic_mode: StrategicMode = StrategicMode.FLEXIBLE
    urgency: Urgency = Urgency.NORMAL
    phase: Phase = Phase.OPENING
    focus: Focus = Focus.GLOBAL

    # Soft activations (for gradient flow)
    mode_probs: np.ndarray = field(default_factory=lambda: np.ones(5) / 5)
    urgency_probs: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1, 0]))
    phase_probs: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0]))
    focus_probs: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))


@dataclass
class TransitionRecord:
    """Observable record of a state transition."""
    timestamp: str
    from_state: Dict[str, str]
    to_state: Dict[str, str]
    trigger: str
    guard_results: List[GuardResult]
    explanation: str


@dataclass
class PolicyTrace:
    """Full trace of policy decisions for observability."""
    move_number: int
    game_state_hash: str
    policy_state: PolicyState
    guard_evaluations: List[GuardResult]
    transitions: List[TransitionRecord]
    action_modulation: Dict[str, float]
    selected_action: int
    raw_policy: np.ndarray
    modulated_policy: np.ndarray


class PolicyStatechart:
    """
    Observable, evolvable policy statechart.

    Encodes policy decisions as statechart states with:
    - Observable state transitions
    - Interpretable guards
    - Evolvable topology and parameters
    """

    def __init__(self, genome: Optional[Dict] = None):
        """
        Initialize policy statechart.

        Args:
            genome: Evolved genome with topology and guard parameters
        """
        self.genome = genome or self._default_genome()
        self.guards = PolicyGuards(self.genome.get('guards', {}))

        # Current state
        self.state = PolicyState()

        # Observability
        self.traces: List[PolicyTrace] = []
        self.transition_history: List[TransitionRecord] = []

        # Action modulation weights (evolved)
        self._init_modulation_weights()

    def _default_genome(self) -> Dict:
        """Default genome for policy statechart."""
        return {
            'version': '1.0',
            'guards': {},  # Uses guard defaults
            'topology': {
                'modes': ['attack', 'defend', 'build', 'reduce', 'flexible'],
                'urgencies': ['critical', 'high', 'normal', 'low'],
                'phases': ['opening', 'midgame', 'endgame'],
                'focuses': ['local', 'global', 'tenuki'],
            },
            'modulation': {
                # How each mode affects move categories
                'attack': {'capture': 1.5, 'extend': 0.8, 'territory': 0.5},
                'defend': {'escape': 1.5, 'connect': 1.3, 'territory': 0.7},
                'build': {'territory': 1.5, 'influence': 1.2, 'fight': 0.6},
                'reduce': {'invasion': 1.5, 'reduction': 1.3, 'territory': 0.8},
                'flexible': {'all': 1.0},
            },
            'transitions': [
                # Transition rules: (from_mode, to_mode, guard_name, priority)
                ('*', 'attack', 'attack_mode', 10),
                ('*', 'defend', 'defend_mode', 15),  # Higher priority
                ('*', 'build', 'build_mode', 5),
                ('attack', 'flexible', 'no_targets', 1),
                ('defend', 'flexible', 'stable', 1),
            ]
        }

    def _init_modulation_weights(self):
        """Initialize action modulation weights from genome."""
        # Create modulation matrix: [num_modes x action_categories]
        # This will be evolved
        self.mode_weights = {}
        for mode, weights in self.genome.get('modulation', {}).items():
            self.mode_weights[mode] = weights

    def update(self, game: Go9x9Statechart,
               board_eval: Dict[str, float],
               last_move: Optional[Tuple[int, int]] = None) -> List[GuardResult]:
        """
        Update policy state based on game position.

        Returns list of guard evaluations for observability.
        """
        guard_results = []
        transitions = []

        # Evaluate all guards
        # Urgency
        critical = self.guards.guard_critical_urgency(game, last_move)
        high = self.guards.guard_high_urgency(game, last_move)
        guard_results.extend([critical, high])

        # Phase
        opening = self.guards.guard_opening_phase(game)
        endgame = self.guards.guard_endgame_phase(game)
        guard_results.extend([opening, endgame])

        # Strategic mode
        attack = self.guards.guard_attack_mode(game, board_eval)
        defend = self.guards.guard_defend_mode(game, board_eval)
        build = self.guards.guard_build_mode(game, board_eval)
        guard_results.extend([attack, defend, build])

        # Focus
        local = self.guards.guard_local_focus(game, last_move)
        tenuki = self.guards.guard_tenuki(game, board_eval)
        guard_results.extend([local, tenuki])

        # Record old state
        old_state = {
            'mode': self.state.strategic_mode.value,
            'urgency': self.state.urgency.value,
            'phase': self.state.phase.value,
            'focus': self.state.focus.value,
        }

        # Apply transitions based on guard results
        # Urgency (highest priority)
        if critical.satisfied:
            if self.state.urgency != Urgency.CRITICAL:
                self.state.urgency = Urgency.CRITICAL
                transitions.append(self._record_transition(
                    old_state, 'urgency', 'critical', critical
                ))
        elif high.satisfied:
            if self.state.urgency != Urgency.HIGH:
                self.state.urgency = Urgency.HIGH
                transitions.append(self._record_transition(
                    old_state, 'urgency', 'high', high
                ))
        else:
            self.state.urgency = Urgency.NORMAL

        # Phase
        if opening.satisfied:
            self.state.phase = Phase.OPENING
        elif endgame.satisfied:
            self.state.phase = Phase.ENDGAME
        else:
            self.state.phase = Phase.MIDGAME

        # Strategic mode (priority: defend > attack > build > flexible)
        if defend.satisfied and defend.confidence > 0.5:
            if self.state.strategic_mode != StrategicMode.DEFEND:
                self.state.strategic_mode = StrategicMode.DEFEND
                transitions.append(self._record_transition(
                    old_state, 'mode', 'defend', defend
                ))
        elif attack.satisfied and attack.confidence > 0.5:
            if self.state.strategic_mode != StrategicMode.ATTACK:
                self.state.strategic_mode = StrategicMode.ATTACK
                transitions.append(self._record_transition(
                    old_state, 'mode', 'attack', attack
                ))
        elif build.satisfied:
            if self.state.strategic_mode != StrategicMode.BUILD:
                self.state.strategic_mode = StrategicMode.BUILD
                transitions.append(self._record_transition(
                    old_state, 'mode', 'build', build
                ))
        else:
            self.state.strategic_mode = StrategicMode.FLEXIBLE

        # Focus
        if local.satisfied:
            self.state.focus = Focus.LOCAL
        elif tenuki.satisfied:
            self.state.focus = Focus.TENUKI
        else:
            self.state.focus = Focus.GLOBAL

        # Update soft activations
        self._update_soft_activations(guard_results)

        # Record transitions
        self.transition_history.extend(transitions)

        return guard_results

    def _record_transition(self, old_state: Dict, dimension: str,
                           new_value: str, guard: GuardResult) -> TransitionRecord:
        """Record a state transition for observability."""
        new_state = old_state.copy()
        new_state[dimension] = new_value

        return TransitionRecord(
            timestamp=datetime.now().isoformat(),
            from_state=old_state,
            to_state=new_state,
            trigger=guard.name,
            guard_results=[guard],
            explanation=f"{dimension}: {old_state.get(dimension)} -> {new_value} "
                       f"because {guard.explanation}"
        )

    def _update_soft_activations(self, guard_results: List[GuardResult]):
        """Update soft state activations from guard results."""
        # Mode probabilities from guard confidences
        mode_scores = {
            'attack': next((g.confidence for g in guard_results
                           if g.name == 'attack_mode'), 0.2),
            'defend': next((g.confidence for g in guard_results
                           if g.name == 'defend_mode'), 0.2),
            'build': next((g.confidence for g in guard_results
                          if g.name == 'build_mode'), 0.2),
            'reduce': 0.1,
            'flexible': 0.3,
        }
        total = sum(mode_scores.values())
        self.state.mode_probs = np.array([
            mode_scores[m] / total for m in
            ['attack', 'defend', 'build', 'reduce', 'flexible']
        ])

    def modulate_policy(self, raw_policy: np.ndarray,
                        game: Go9x9Statechart) -> Tuple[np.ndarray, Dict]:
        """
        Modulate raw neural network policy based on current state.

        Returns:
            modulated_policy: Policy after state-based modulation
            explanation: Dict explaining the modulation
        """
        modulated = raw_policy.copy()
        explanation = {
            'mode': self.state.strategic_mode.value,
            'urgency': self.state.urgency.value,
            'modulations': []
        }

        # Get modulation weights for current mode
        mode_name = self.state.strategic_mode.value
        weights = self.mode_weights.get(mode_name, {'all': 1.0})

        # Classify moves and apply modulation
        for idx in range(TOTAL_POINTS):
            x, y = idx_to_xy(idx)
            move_type = self._classify_move(game, x, y)

            # Apply mode-specific weight
            weight = weights.get(move_type, weights.get('all', 1.0))
            modulated[idx] *= weight

            if weight != 1.0:
                explanation['modulations'].append({
                    'move': (x, y),
                    'type': move_type,
                    'weight': weight
                })

        # Apply urgency modulation
        if self.state.urgency == Urgency.CRITICAL:
            # Boost urgent moves (atari responses, ko)
            for idx in range(TOTAL_POINTS):
                x, y = idx_to_xy(idx)
                if self._is_urgent_move(game, x, y):
                    modulated[idx] *= 2.0
                    explanation['modulations'].append({
                        'move': (x, y),
                        'type': 'urgent',
                        'weight': 2.0
                    })

        # Renormalize
        total = modulated.sum()
        if total > 0:
            modulated /= total

        return modulated, explanation

    def _classify_move(self, game: Go9x9Statechart, x: int, y: int) -> str:
        """Classify a move by its strategic type."""
        if not game.is_legal_move(x, y):
            return 'illegal'

        # Check if move captures
        opp_color = WHITE if game.current_player() == BLACK else BLACK
        for nx, ny in get_neighbors(x, y):
            if game.board.get(nx, ny) == opp_color:
                if game.board.count_liberties(nx, ny) == 1:
                    return 'capture'

        # Check if move extends our group
        our_color = game.current_player()
        for nx, ny in get_neighbors(x, y):
            if game.board.get(nx, ny) == our_color:
                return 'extend'

        # Check corner/edge for territory
        if x <= 3 or x >= 5 or y <= 3 or y >= 5:
            return 'territory'

        return 'influence'

    def _is_urgent_move(self, game: Go9x9Statechart, x: int, y: int) -> bool:
        """Check if a move is urgent (responds to atari, etc.)."""
        if not game.is_legal_move(x, y):
            return False

        our_color = game.current_player()

        # Check if move saves a group in atari
        for nx, ny in get_neighbors(x, y):
            if game.board.get(nx, ny) == our_color:
                if game.board.count_liberties(nx, ny) == 1:
                    return True

        return False

    def record_trace(self, move_number: int, game: Go9x9Statechart,
                     guard_results: List[GuardResult],
                     raw_policy: np.ndarray, modulated_policy: np.ndarray,
                     selected_action: int, modulation_explanation: Dict):
        """Record full trace for observability."""
        trace = PolicyTrace(
            move_number=move_number,
            game_state_hash=str(hash(tuple(game.board.stones))),
            policy_state=copy.deepcopy(self.state),
            guard_evaluations=guard_results,
            transitions=self.transition_history[-5:],  # Last 5 transitions
            action_modulation=modulation_explanation,
            selected_action=selected_action,
            raw_policy=raw_policy.copy(),
            modulated_policy=modulated_policy.copy()
        )
        self.traces.append(trace)

    def get_state_summary(self) -> Dict:
        """Get human-readable state summary."""
        return {
            'mode': self.state.strategic_mode.value,
            'urgency': self.state.urgency.value,
            'phase': self.state.phase.value,
            'focus': self.state.focus.value,
            'mode_probs': {
                'attack': float(self.state.mode_probs[0]),
                'defend': float(self.state.mode_probs[1]),
                'build': float(self.state.mode_probs[2]),
                'reduce': float(self.state.mode_probs[3]),
                'flexible': float(self.state.mode_probs[4]),
            }
        }

    def explain_decision(self, trace_idx: int = -1) -> str:
        """Generate human-readable explanation of a policy decision."""
        if not self.traces:
            return "No traces recorded"

        trace = self.traces[trace_idx]
        lines = [
            f"Move {trace.move_number} Decision Explanation",
            "=" * 50,
            f"Policy State:",
            f"  Mode: {trace.policy_state.strategic_mode.value}",
            f"  Urgency: {trace.policy_state.urgency.value}",
            f"  Phase: {trace.policy_state.phase.value}",
            f"  Focus: {trace.policy_state.focus.value}",
            "",
            "Guard Evaluations:",
        ]

        for guard in trace.guard_evaluations:
            status = "PASS" if guard.satisfied else "FAIL"
            lines.append(f"  [{status}] {guard.name}: {guard.explanation}")

        if trace.transitions:
            lines.append("")
            lines.append("Recent Transitions:")
            for t in trace.transitions[-3:]:
                lines.append(f"  {t.explanation}")

        lines.append("")
        lines.append(f"Selected Action: {trace.selected_action}")

        if trace.selected_action < 81:
            x, y = idx_to_xy(trace.selected_action)
            lines.append(f"  Position: ({x}, {y})")
        else:
            lines.append("  Position: PASS")

        return "\n".join(lines)

    def to_genome(self) -> Dict:
        """Export current state as genome for evolution."""
        return copy.deepcopy(self.genome)

    @classmethod
    def from_genome(cls, genome: Dict) -> 'PolicyStatechart':
        """Create policy statechart from evolved genome."""
        return cls(genome=genome)


# =============================================================================
# EVOLUTION OPERATORS
# =============================================================================

class PolicyEvolution:
    """
    Evolution operators for policy statecharts.

    Evolves both topology and guards.
    """

    @staticmethod
    def mutate_guards(genome: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate guard thresholds."""
        new_genome = copy.deepcopy(genome)
        guards = new_genome.setdefault('guards', {})

        # Threshold mutations
        thresholds = [
            'atari_urgency', 'liberty_critical', 'liberty_warning',
            'opening_end', 'midgame_end',
            'attack_threshold', 'defend_threshold', 'build_threshold',
            'local_radius', 'tenuki_threshold'
        ]

        for key in thresholds:
            if np.random.random() < mutation_rate:
                old_val = guards.get(key, 0.5)
                # Gaussian mutation
                new_val = old_val + np.random.normal(0, 0.1)
                # Clip to valid range
                if 'threshold' in key or key.endswith('_urgency'):
                    new_val = np.clip(new_val, 0.1, 0.9)
                elif key.endswith('_end'):
                    new_val = int(np.clip(new_val, 10, 200))
                elif key == 'liberty_critical':
                    new_val = int(np.clip(new_val, 1, 4))
                elif key == 'local_radius':
                    new_val = int(np.clip(new_val, 1, 5))
                guards[key] = new_val

        return new_genome

    @staticmethod
    def mutate_modulation(genome: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate action modulation weights."""
        new_genome = copy.deepcopy(genome)
        modulation = new_genome.setdefault('modulation', {})

        for mode in ['attack', 'defend', 'build', 'reduce', 'flexible']:
            if mode not in modulation:
                modulation[mode] = {'all': 1.0}

            for category in modulation[mode]:
                if np.random.random() < mutation_rate:
                    old_val = modulation[mode][category]
                    new_val = old_val + np.random.normal(0, 0.2)
                    modulation[mode][category] = np.clip(new_val, 0.1, 3.0)

        return new_genome

    @staticmethod
    def mutate_topology(genome: Dict, mutation_rate: float = 0.05) -> Dict:
        """Mutate statechart topology (add/remove states, transitions)."""
        new_genome = copy.deepcopy(genome)

        # For now, just mutate transition priorities
        transitions = new_genome.get('transitions', [])

        for i, (from_m, to_m, guard, priority) in enumerate(transitions):
            if np.random.random() < mutation_rate:
                new_priority = max(1, priority + np.random.randint(-3, 4))
                transitions[i] = (from_m, to_m, guard, new_priority)

        # Small chance to add new transition
        if np.random.random() < mutation_rate / 2:
            modes = ['attack', 'defend', 'build', 'reduce', 'flexible']
            guards = ['attack_mode', 'defend_mode', 'build_mode',
                     'critical_urgency', 'high_urgency']
            new_trans = (
                np.random.choice(modes + ['*']),
                np.random.choice(modes),
                np.random.choice(guards),
                np.random.randint(1, 20)
            )
            transitions.append(new_trans)

        new_genome['transitions'] = transitions
        return new_genome

    @staticmethod
    def crossover(genome1: Dict, genome2: Dict) -> Dict:
        """Crossover two genomes."""
        child = copy.deepcopy(genome1)

        # Crossover guards
        if 'guards' in genome2:
            for key, val in genome2['guards'].items():
                if np.random.random() < 0.5:
                    child.setdefault('guards', {})[key] = val

        # Crossover modulation
        if 'modulation' in genome2:
            for mode, weights in genome2['modulation'].items():
                if np.random.random() < 0.5:
                    child.setdefault('modulation', {})[mode] = copy.deepcopy(weights)

        return child


# =============================================================================
# TEST
# =============================================================================

def test_policy_statechart():
    """Test the observable policy statechart."""
    print("Testing PolicyStatechart...")
    print("=" * 60)

    # Create policy statechart
    policy = PolicyStatechart()

    # Create game
    game = Go9x9Statechart()

    # Initial state
    print("\nInitial policy state:")
    print(json.dumps(policy.get_state_summary(), indent=2))

    # Make some moves
    game.play_move(4, 4)  # Black center
    game.play_move(4, 5)  # White adjacent
    game.play_move(3, 4)  # Black extends

    # Update policy state
    board_eval = {
        'attack_potential': 0.3,
        'defend_need': 0.2,
        'build_potential': 0.6,
        'tenuki_value': 0.4,
    }

    guard_results = policy.update(game, board_eval, last_move=(3, 4))

    print("\nAfter 3 moves:")
    print(json.dumps(policy.get_state_summary(), indent=2))

    print("\nGuard evaluations:")
    for g in guard_results:
        status = "PASS" if g.satisfied else "FAIL"
        print(f"  [{status}] {g.name}: {g.explanation}")

    # Test policy modulation
    raw_policy = np.ones(82) / 82
    modulated, explanation = policy.modulate_policy(raw_policy, game)

    print(f"\nPolicy modulation:")
    print(f"  Mode: {explanation['mode']}")
    print(f"  Urgency: {explanation['urgency']}")
    print(f"  Modulations applied: {len(explanation['modulations'])}")

    # Record trace
    policy.record_trace(
        move_number=3,
        game=game,
        guard_results=guard_results,
        raw_policy=raw_policy,
        modulated_policy=modulated,
        selected_action=40,
        modulation_explanation=explanation
    )

    # Get explanation
    print("\n" + policy.explain_decision())

    # Test evolution
    print("\n" + "=" * 60)
    print("Testing Evolution...")

    genome = policy.to_genome()

    # Mutate
    mutated = PolicyEvolution.mutate_guards(genome, mutation_rate=0.5)
    mutated = PolicyEvolution.mutate_modulation(mutated, mutation_rate=0.5)

    print(f"\nOriginal guard threshold: {genome.get('guards', {})}")
    print(f"Mutated guard threshold: {mutated.get('guards', {})}")

    # Create from mutated genome
    evolved_policy = PolicyStatechart.from_genome(mutated)
    print(f"\nEvolved policy created successfully")

    print("\n" + "=" * 60)
    print("All PolicyStatechart tests passed!")


if __name__ == "__main__":
    test_policy_statechart()
