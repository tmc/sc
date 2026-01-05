"""
Evolvable Seeker Statechart

Modes to emerge from evolution:
- Search: Systematically explore unseen areas
- Chase: Pursue spotted hider aggressively
- Predict: Move to likely hiding spots based on learned patterns
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any
import random
import math

try:
    from .grid_world import Direction, Observation, Cell
except ImportError:
    from grid_world import Direction, Observation, Cell


class SeekerMode(Enum):
    """Strategic modes for seeker behavior."""
    SEARCH = auto()   # Explore unseen areas
    CHASE = auto()    # Pursue spotted hider
    PREDICT = auto()  # Go to likely hiding spots


@dataclass
class SeekerGenome:
    """
    Evolvable genome for seeker statechart.

    All parameters start simple and can be evolved.
    """
    # Mode transition thresholds
    search_to_chase_threshold: float = 0.5  # Confidence needed to start chase
    chase_to_predict_threshold: float = 0.3  # How long before predicting after losing sight
    predict_to_search_threshold: float = 0.5  # When to give up predicting

    # Search mode parameters
    search_spiral: bool = False  # Use spiral search pattern
    search_wall_follow: bool = False  # Follow walls
    search_random_weight: float = 0.3  # Randomness in search

    # Chase mode parameters
    chase_intercept: bool = False  # Try to intercept vs direct chase
    chase_persistence: int = 5  # How many steps to chase last known position

    # Predict mode parameters
    predict_use_shelter: bool = False  # Check shelters first
    predict_use_corners: bool = False  # Check corners
    predict_use_history: bool = False  # Use memory of past hiding spots

    # General parameters
    exploration_bonus: float = 0.1  # Bonus for visiting new cells
    memory_length: int = 10  # How many past positions to remember

    def mutate(self, rate: float = 0.2) -> 'SeekerGenome':
        """Create a mutated copy of the genome."""
        new = SeekerGenome(
            search_to_chase_threshold=self.search_to_chase_threshold,
            chase_to_predict_threshold=self.chase_to_predict_threshold,
            predict_to_search_threshold=self.predict_to_search_threshold,
            search_spiral=self.search_spiral,
            search_wall_follow=self.search_wall_follow,
            search_random_weight=self.search_random_weight,
            chase_intercept=self.chase_intercept,
            chase_persistence=self.chase_persistence,
            predict_use_shelter=self.predict_use_shelter,
            predict_use_corners=self.predict_use_corners,
            predict_use_history=self.predict_use_history,
            exploration_bonus=self.exploration_bonus,
            memory_length=self.memory_length,
        )

        # Mutate floats
        if random.random() < rate:
            new.search_to_chase_threshold = max(0, min(1,
                self.search_to_chase_threshold + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.chase_to_predict_threshold = max(0, min(1,
                self.chase_to_predict_threshold + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.predict_to_search_threshold = max(0, min(1,
                self.predict_to_search_threshold + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.search_random_weight = max(0, min(1,
                self.search_random_weight + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.exploration_bonus = max(0,
                self.exploration_bonus + random.gauss(0, 0.05))

        # Mutate booleans
        if random.random() < rate:
            new.search_spiral = not self.search_spiral
        if random.random() < rate:
            new.search_wall_follow = not self.search_wall_follow
        if random.random() < rate:
            new.chase_intercept = not self.chase_intercept
        if random.random() < rate:
            new.predict_use_shelter = not self.predict_use_shelter
        if random.random() < rate:
            new.predict_use_corners = not self.predict_use_corners
        if random.random() < rate:
            new.predict_use_history = not self.predict_use_history

        # Mutate integers
        if random.random() < rate:
            new.chase_persistence = max(1, self.chase_persistence + random.randint(-2, 2))
        if random.random() < rate:
            new.memory_length = max(1, self.memory_length + random.randint(-3, 3))

        return new

    @classmethod
    def crossover(cls, a: 'SeekerGenome', b: 'SeekerGenome') -> 'SeekerGenome':
        """Create offspring from two genomes."""
        return SeekerGenome(
            search_to_chase_threshold=random.choice([a.search_to_chase_threshold, b.search_to_chase_threshold]),
            chase_to_predict_threshold=random.choice([a.chase_to_predict_threshold, b.chase_to_predict_threshold]),
            predict_to_search_threshold=random.choice([a.predict_to_search_threshold, b.predict_to_search_threshold]),
            search_spiral=random.choice([a.search_spiral, b.search_spiral]),
            search_wall_follow=random.choice([a.search_wall_follow, b.search_wall_follow]),
            search_random_weight=random.choice([a.search_random_weight, b.search_random_weight]),
            chase_intercept=random.choice([a.chase_intercept, b.chase_intercept]),
            chase_persistence=random.choice([a.chase_persistence, b.chase_persistence]),
            predict_use_shelter=random.choice([a.predict_use_shelter, b.predict_use_shelter]),
            predict_use_corners=random.choice([a.predict_use_corners, b.predict_use_corners]),
            predict_use_history=random.choice([a.predict_use_history, b.predict_use_history]),
            exploration_bonus=random.choice([a.exploration_bonus, b.exploration_bonus]),
            memory_length=random.choice([a.memory_length, b.memory_length]),
        )


@dataclass
class SeekerState:
    """Internal state of the seeker statechart."""
    mode: SeekerMode = SeekerMode.SEARCH
    last_seen_position: Optional[Tuple[int, int]] = None
    steps_since_sight: int = 0
    visited_cells: set = field(default_factory=set)
    spiral_direction: int = 0
    spiral_steps: int = 0
    memory: List[Tuple[int, int]] = field(default_factory=list)


class SeekerStatechart:
    """
    Observable, evolvable seeker statechart.

    The statechart structure is FIXED but parameters are evolved.
    This ensures interpretability while allowing learning.
    """

    def __init__(self, genome: SeekerGenome = None):
        self.genome = genome or SeekerGenome()
        self.state = SeekerState()
        self.decision_trace: List[Dict[str, Any]] = []

    def reset(self):
        """Reset statechart state."""
        self.state = SeekerState()
        self.decision_trace = []

    def _evaluate_transitions(self, obs: Observation) -> SeekerMode:
        """
        Evaluate mode transitions based on observation.

        Returns new mode (may be same as current).
        """
        current_mode = self.state.mode

        # SEARCH -> CHASE: When hider spotted
        if current_mode == SeekerMode.SEARCH:
            if obs.visible_agents:
                confidence = 1.0 / (1 + self.state.steps_since_sight)
                if confidence >= self.genome.search_to_chase_threshold:
                    return SeekerMode.CHASE

        # CHASE -> PREDICT: When hider lost for too long
        elif current_mode == SeekerMode.CHASE:
            if not obs.visible_agents:
                self.state.steps_since_sight += 1
                if self.state.steps_since_sight > self.genome.chase_persistence:
                    return SeekerMode.PREDICT
            else:
                self.state.steps_since_sight = 0

        # PREDICT -> SEARCH: When prediction exhausted
        elif current_mode == SeekerMode.PREDICT:
            if obs.visible_agents:
                return SeekerMode.CHASE
            # Give up predicting after exploring likely spots
            if self.state.steps_since_sight > self.genome.chase_persistence * 3:
                return SeekerMode.SEARCH

        return current_mode

    def _search_action(self, obs: Observation) -> Direction:
        """Execute search mode behavior."""
        self.state.visited_cells.add(obs.own_position)

        # Calculate exploration scores for each direction
        scores: Dict[Direction, float] = {}

        for direction in [Direction.UP, Direction.DOWN,
                          Direction.LEFT, Direction.RIGHT]:
            new_x = obs.own_position[0] + direction.dx
            new_y = obs.own_position[1] + direction.dy

            # Base score
            score = 0.0

            # Exploration bonus for unvisited cells
            if (new_x, new_y) not in self.state.visited_cells:
                score += self.genome.exploration_bonus

            # Spiral pattern bonus
            if self.genome.search_spiral:
                spiral_dirs = [Direction.RIGHT, Direction.DOWN,
                               Direction.LEFT, Direction.UP]
                if direction == spiral_dirs[self.state.spiral_direction % 4]:
                    score += 0.2

            # Wall following bonus
            if self.genome.search_wall_follow:
                # Prefer directions with wall on one side
                perpendicular = {
                    Direction.UP: [Direction.LEFT, Direction.RIGHT],
                    Direction.DOWN: [Direction.LEFT, Direction.RIGHT],
                    Direction.LEFT: [Direction.UP, Direction.DOWN],
                    Direction.RIGHT: [Direction.UP, Direction.DOWN],
                }
                for perp in perpendicular[direction]:
                    wall_x = obs.own_position[0] + perp.dx
                    wall_y = obs.own_position[1] + perp.dy
                    if (wall_x, wall_y) not in obs.visible_cells:
                        score += 0.1

            # Random component
            score += random.random() * self.genome.search_random_weight

            scores[direction] = score

        # Update spiral state
        self.state.spiral_steps += 1
        if self.state.spiral_steps > self.state.spiral_direction + 1:
            self.state.spiral_direction += 1
            self.state.spiral_steps = 0

        return max(scores.keys(), key=lambda d: scores[d])

    def _chase_action(self, obs: Observation) -> Direction:
        """Execute chase mode behavior."""
        target = None

        if obs.visible_agents:
            # Update last seen position
            target = (obs.visible_agents[0].x, obs.visible_agents[0].y)
            self.state.last_seen_position = target
            self.state.steps_since_sight = 0

            # Add to memory
            self.state.memory.append(target)
            if len(self.state.memory) > self.genome.memory_length:
                self.state.memory.pop(0)
        else:
            target = self.state.last_seen_position
            self.state.steps_since_sight += 1

        if target is None:
            return self._search_action(obs)

        # Calculate direction to target
        dx = target[0] - obs.own_position[0]
        dy = target[1] - obs.own_position[1]

        if self.genome.chase_intercept and obs.visible_agents:
            # Try to predict where hider is going
            # Simple: assume hider moves away from us
            intercept_x = target[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
            intercept_y = target[1] + (1 if dy > 0 else -1 if dy < 0 else 0)
            dx = intercept_x - obs.own_position[0]
            dy = intercept_y - obs.own_position[1]

        # Move toward target (Manhattan)
        if abs(dx) >= abs(dy):
            return Direction.RIGHT if dx > 0 else Direction.LEFT
        else:
            return Direction.DOWN if dy > 0 else Direction.UP

    def _predict_action(self, obs: Observation) -> Direction:
        """Execute predict mode behavior."""
        # Build list of candidate positions to check
        candidates: List[Tuple[int, int]] = []

        # Check shelters if enabled
        if self.genome.predict_use_shelter:
            # Shelters are likely hiding spots - look for them
            for cell in obs.visible_cells:
                candidates.append(cell)

        # Check corners if enabled
        if self.genome.predict_use_corners:
            # Corners are common hiding spots
            # We don't know grid size, but can head toward edges
            corners = [(1, 1), (1, 13), (13, 1), (13, 13)]  # Assuming ~15x15
            candidates.extend(corners)

        # Use memory of past hiding spots
        if self.genome.predict_use_history and self.state.memory:
            candidates.extend(self.state.memory)

        # Last known position as fallback
        if self.state.last_seen_position:
            candidates.append(self.state.last_seen_position)

        # Find unvisited candidate closest to us
        best_candidate = None
        best_dist = float('inf')

        for cand in candidates:
            if cand in self.state.visited_cells:
                continue
            dist = abs(cand[0] - obs.own_position[0]) + \
                   abs(cand[1] - obs.own_position[1])
            if dist < best_dist:
                best_dist = dist
                best_candidate = cand

        if best_candidate:
            dx = best_candidate[0] - obs.own_position[0]
            dy = best_candidate[1] - obs.own_position[1]
            if abs(dx) >= abs(dy):
                return Direction.RIGHT if dx > 0 else Direction.LEFT
            else:
                return Direction.DOWN if dy > 0 else Direction.UP

        # Fallback to search
        return self._search_action(obs)

    def decide(self, obs: Observation) -> Direction:
        """
        Make a decision based on current observation.

        Returns direction and records decision trace for observability.
        """
        # Evaluate mode transitions
        old_mode = self.state.mode
        new_mode = self._evaluate_transitions(obs)
        self.state.mode = new_mode

        # Execute mode-specific behavior
        action_fns = {
            SeekerMode.SEARCH: self._search_action,
            SeekerMode.CHASE: self._chase_action,
            SeekerMode.PREDICT: self._predict_action,
        }
        action = action_fns[self.state.mode](obs)

        # Record decision trace
        self.decision_trace.append({
            'time_step': obs.time_step,
            'mode': self.state.mode.name,
            'mode_changed': old_mode != new_mode,
            'visible_hiders': len(obs.visible_agents),
            'action': action.name,
            'visited_count': len(self.state.visited_cells),
        })

        return action

    def explain(self) -> str:
        """Generate human-readable explanation of current state."""
        lines = [
            f"Seeker Statechart:",
            f"  Current Mode: {self.state.mode.name}",
            f"  Cells Visited: {len(self.state.visited_cells)}",
            f"  Steps Since Sight: {self.state.steps_since_sight}",
        ]
        if self.state.last_seen_position:
            lines.append(f"  Last Seen: {self.state.last_seen_position}")

        lines.append(f"\n  Genome Traits:")
        lines.append(f"    Search: spiral={self.genome.search_spiral}, "
                     f"wall_follow={self.genome.search_wall_follow}")
        lines.append(f"    Chase: intercept={self.genome.chase_intercept}, "
                     f"persistence={self.genome.chase_persistence}")
        lines.append(f"    Predict: shelter={self.genome.predict_use_shelter}, "
                     f"corners={self.genome.predict_use_corners}, "
                     f"history={self.genome.predict_use_history}")
        return '\n'.join(lines)

    def extract_strategy(self) -> Dict[str, Any]:
        """
        Extract learned strategy as interpretable statechart.

        This is the key innovation - strategies are EXTRACTABLE.
        """
        return {
            'type': 'seeker_statechart',
            'modes': ['SEARCH', 'CHASE', 'PREDICT'],
            'transitions': [
                {
                    'from': 'SEARCH',
                    'to': 'CHASE',
                    'guard': f'hider_visible AND confidence >= {self.genome.search_to_chase_threshold:.2f}',
                },
                {
                    'from': 'CHASE',
                    'to': 'PREDICT',
                    'guard': f'NOT hider_visible AND steps_since_sight > {self.genome.chase_persistence}',
                },
                {
                    'from': 'PREDICT',
                    'to': 'CHASE',
                    'guard': 'hider_visible',
                },
                {
                    'from': 'PREDICT',
                    'to': 'SEARCH',
                    'guard': f'steps_since_sight > {self.genome.chase_persistence * 3}',
                },
            ],
            'mode_behaviors': {
                'SEARCH': {
                    'strategy': 'spiral' if self.genome.search_spiral else
                               'wall_follow' if self.genome.search_wall_follow else 'random',
                    'exploration_bonus': self.genome.exploration_bonus,
                },
                'CHASE': {
                    'intercept': self.genome.chase_intercept,
                    'persistence': self.genome.chase_persistence,
                },
                'PREDICT': {
                    'check_shelters': self.genome.predict_use_shelter,
                    'check_corners': self.genome.predict_use_corners,
                    'use_memory': self.genome.predict_use_history,
                },
            },
        }
