"""
Evolvable Hider Statechart

Modes to emerge from evolution:
- Hide: Stay in cover, minimize exposure
- Evade: Move when spotted to break line of sight
- Distract: Bait and switch tactics
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


class HiderMode(Enum):
    """Strategic modes for hider behavior."""
    HIDE = auto()      # Stay hidden in cover
    EVADE = auto()     # Move to break line of sight
    DISTRACT = auto()  # Bait and switch


@dataclass
class HiderGenome:
    """
    Evolvable genome for hider statechart.

    All parameters start simple and can be evolved.
    """
    # Mode transition thresholds
    hide_to_evade_threshold: float = 0.5  # Distance at which to start evading
    evade_to_hide_threshold: float = 0.7  # How safe before returning to hide
    distract_chance: float = 0.1  # Probability of entering distract mode

    # Hide mode parameters
    hide_prefer_shelter: bool = True  # Seek shelter cells
    hide_prefer_corners: bool = False  # Prefer corner positions
    hide_stay_still: float = 0.7  # Probability of staying still when hidden

    # Evade mode parameters
    evade_away: bool = True  # Move directly away from seeker
    evade_perpendicular: bool = False  # Move perpendicular to seeker
    evade_toward_shelter: bool = False  # Move toward nearest shelter
    evade_juke: bool = False  # Random direction changes

    # Distract mode parameters
    distract_duration: int = 3  # How long to maintain distraction
    distract_reverse: bool = False  # Reverse direction mid-distract
    distract_visibility: float = 0.5  # How much to expose self

    # General parameters
    safety_distance: float = 5.0  # Distance considered safe
    memory_length: int = 10  # Remember seeker positions

    def mutate(self, rate: float = 0.2) -> 'HiderGenome':
        """Create a mutated copy of the genome."""
        new = HiderGenome(
            hide_to_evade_threshold=self.hide_to_evade_threshold,
            evade_to_hide_threshold=self.evade_to_hide_threshold,
            distract_chance=self.distract_chance,
            hide_prefer_shelter=self.hide_prefer_shelter,
            hide_prefer_corners=self.hide_prefer_corners,
            hide_stay_still=self.hide_stay_still,
            evade_away=self.evade_away,
            evade_perpendicular=self.evade_perpendicular,
            evade_toward_shelter=self.evade_toward_shelter,
            evade_juke=self.evade_juke,
            distract_duration=self.distract_duration,
            distract_reverse=self.distract_reverse,
            distract_visibility=self.distract_visibility,
            safety_distance=self.safety_distance,
            memory_length=self.memory_length,
        )

        # Mutate floats
        if random.random() < rate:
            new.hide_to_evade_threshold = max(0.1, min(1,
                self.hide_to_evade_threshold + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.evade_to_hide_threshold = max(0.1, min(1,
                self.evade_to_hide_threshold + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.distract_chance = max(0, min(0.5,
                self.distract_chance + random.gauss(0, 0.05)))
        if random.random() < rate:
            new.hide_stay_still = max(0, min(1,
                self.hide_stay_still + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.distract_visibility = max(0, min(1,
                self.distract_visibility + random.gauss(0, 0.1)))
        if random.random() < rate:
            new.safety_distance = max(2, min(10,
                self.safety_distance + random.gauss(0, 1)))

        # Mutate booleans
        if random.random() < rate:
            new.hide_prefer_shelter = not self.hide_prefer_shelter
        if random.random() < rate:
            new.hide_prefer_corners = not self.hide_prefer_corners
        if random.random() < rate:
            new.evade_away = not self.evade_away
        if random.random() < rate:
            new.evade_perpendicular = not self.evade_perpendicular
        if random.random() < rate:
            new.evade_toward_shelter = not self.evade_toward_shelter
        if random.random() < rate:
            new.evade_juke = not self.evade_juke
        if random.random() < rate:
            new.distract_reverse = not self.distract_reverse

        # Mutate integers
        if random.random() < rate:
            new.distract_duration = max(1, self.distract_duration + random.randint(-1, 2))
        if random.random() < rate:
            new.memory_length = max(1, self.memory_length + random.randint(-3, 3))

        return new

    @classmethod
    def crossover(cls, a: 'HiderGenome', b: 'HiderGenome') -> 'HiderGenome':
        """Create offspring from two genomes."""
        return HiderGenome(
            hide_to_evade_threshold=random.choice([a.hide_to_evade_threshold, b.hide_to_evade_threshold]),
            evade_to_hide_threshold=random.choice([a.evade_to_hide_threshold, b.evade_to_hide_threshold]),
            distract_chance=random.choice([a.distract_chance, b.distract_chance]),
            hide_prefer_shelter=random.choice([a.hide_prefer_shelter, b.hide_prefer_shelter]),
            hide_prefer_corners=random.choice([a.hide_prefer_corners, b.hide_prefer_corners]),
            hide_stay_still=random.choice([a.hide_stay_still, b.hide_stay_still]),
            evade_away=random.choice([a.evade_away, b.evade_away]),
            evade_perpendicular=random.choice([a.evade_perpendicular, b.evade_perpendicular]),
            evade_toward_shelter=random.choice([a.evade_toward_shelter, b.evade_toward_shelter]),
            evade_juke=random.choice([a.evade_juke, b.evade_juke]),
            distract_duration=random.choice([a.distract_duration, b.distract_duration]),
            distract_reverse=random.choice([a.distract_reverse, b.distract_reverse]),
            distract_visibility=random.choice([a.distract_visibility, b.distract_visibility]),
            safety_distance=random.choice([a.safety_distance, b.safety_distance]),
            memory_length=random.choice([a.memory_length, b.memory_length]),
        )


@dataclass
class HiderState:
    """Internal state of the hider statechart."""
    mode: HiderMode = HiderMode.HIDE
    last_seeker_position: Optional[Tuple[int, int]] = None
    steps_hidden: int = 0
    distract_steps_remaining: int = 0
    distract_direction: Optional[Direction] = None
    seeker_memory: List[Tuple[int, int]] = field(default_factory=list)
    evade_direction: Optional[Direction] = None


class HiderStatechart:
    """
    Observable, evolvable hider statechart.

    The statechart structure is FIXED but parameters are evolved.
    This ensures interpretability while allowing learning.
    """

    def __init__(self, genome: HiderGenome = None):
        self.genome = genome or HiderGenome()
        self.state = HiderState()
        self.decision_trace: List[Dict[str, Any]] = []

    def reset(self):
        """Reset statechart state."""
        self.state = HiderState()
        self.decision_trace = []

    def _distance_to_seeker(self, obs: Observation) -> float:
        """Calculate distance to nearest visible seeker."""
        if not obs.visible_agents:
            return float('inf')
        return min(
            abs(s.x - obs.own_position[0]) + abs(s.y - obs.own_position[1])
            for s in obs.visible_agents
        )

    def _evaluate_transitions(self, obs: Observation) -> HiderMode:
        """
        Evaluate mode transitions based on observation.

        Returns new mode (may be same as current).
        """
        current_mode = self.state.mode
        dist = self._distance_to_seeker(obs)

        # HIDE -> EVADE: When seeker gets too close
        if current_mode == HiderMode.HIDE:
            if obs.visible_agents:
                danger_level = 1.0 - (dist / self.genome.safety_distance)
                if danger_level >= self.genome.hide_to_evade_threshold:
                    return HiderMode.EVADE
                # Small chance to enter distract mode
                if random.random() < self.genome.distract_chance:
                    return HiderMode.DISTRACT

        # EVADE -> HIDE: When seeker is far enough
        elif current_mode == HiderMode.EVADE:
            if not obs.visible_agents or dist > self.genome.safety_distance:
                safety_level = min(1.0, dist / self.genome.safety_distance)
                if safety_level >= self.genome.evade_to_hide_threshold:
                    return HiderMode.HIDE
            # Chance to try distraction
            if obs.visible_agents and random.random() < self.genome.distract_chance:
                return HiderMode.DISTRACT

        # DISTRACT -> HIDE/EVADE: After distraction duration
        elif current_mode == HiderMode.DISTRACT:
            if self.state.distract_steps_remaining <= 0:
                if obs.visible_agents and dist < self.genome.safety_distance:
                    return HiderMode.EVADE
                return HiderMode.HIDE

        return current_mode

    def _hide_action(self, obs: Observation) -> Direction:
        """Execute hide mode behavior."""
        self.state.steps_hidden += 1

        # Stay still with high probability
        if random.random() < self.genome.hide_stay_still:
            return Direction.STAY

        # Look for better hiding spot
        scores: Dict[Direction, float] = {d: 0.0 for d in Direction}

        for direction in Direction:
            if direction == Direction.STAY:
                scores[direction] = 0.5  # Slight preference to stay
                continue

            new_x = obs.own_position[0] + direction.dx
            new_y = obs.own_position[1] + direction.dy

            # Check if cell is visible to us (and thus potentially valid)
            if (new_x, new_y) in obs.visible_cells:
                # Prefer positions away from known seeker positions
                if self.state.last_seeker_position:
                    seeker_x, seeker_y = self.state.last_seeker_position
                    old_dist = abs(seeker_x - obs.own_position[0]) + \
                               abs(seeker_y - obs.own_position[1])
                    new_dist = abs(seeker_x - new_x) + abs(seeker_y - new_y)
                    if new_dist > old_dist:
                        scores[direction] += 0.3

                # Prefer corners if enabled
                if self.genome.hide_prefer_corners:
                    # Corner proximity bonus
                    corners = [(1, 1), (1, 13), (13, 1), (13, 13)]
                    for cx, cy in corners:
                        if abs(new_x - cx) <= 2 and abs(new_y - cy) <= 2:
                            scores[direction] += 0.4
            else:
                scores[direction] = -1.0  # Can't move there

        return max(scores.keys(), key=lambda d: scores[d])

    def _evade_action(self, obs: Observation) -> Direction:
        """Execute evade mode behavior."""
        self.state.steps_hidden = 0

        if not obs.visible_agents:
            # No seeker visible - use memory or return to hiding
            if self.state.last_seeker_position:
                seeker_x, seeker_y = self.state.last_seeker_position
            else:
                return self._hide_action(obs)
        else:
            # Update seeker position
            seeker = obs.visible_agents[0]
            seeker_x, seeker_y = seeker.x, seeker.y
            self.state.last_seeker_position = (seeker_x, seeker_y)
            self.state.seeker_memory.append((seeker_x, seeker_y))
            if len(self.state.seeker_memory) > self.genome.memory_length:
                self.state.seeker_memory.pop(0)

        dx = obs.own_position[0] - seeker_x  # Direction away from seeker
        dy = obs.own_position[1] - seeker_y

        # Juke: Random direction change
        if self.genome.evade_juke and random.random() < 0.3:
            return random.choice([Direction.UP, Direction.DOWN,
                                  Direction.LEFT, Direction.RIGHT])

        # Perpendicular movement
        if self.genome.evade_perpendicular:
            if random.random() < 0.5:
                return Direction.UP if dx >= 0 else Direction.DOWN
            else:
                return Direction.LEFT if dy >= 0 else Direction.RIGHT

        # Direct away movement (default)
        if self.genome.evade_away:
            if abs(dx) >= abs(dy):
                return Direction.RIGHT if dx > 0 else Direction.LEFT
            else:
                return Direction.DOWN if dy > 0 else Direction.UP

        # Fallback
        return random.choice([Direction.UP, Direction.DOWN,
                              Direction.LEFT, Direction.RIGHT])

    def _distract_action(self, obs: Observation) -> Direction:
        """Execute distract mode behavior."""
        # Initialize distraction
        if self.state.distract_steps_remaining <= 0:
            self.state.distract_steps_remaining = self.genome.distract_duration
            # Pick initial distraction direction
            if obs.visible_agents:
                seeker = obs.visible_agents[0]
                # Move somewhat toward seeker (bait)
                dx = seeker.x - obs.own_position[0]
                dy = seeker.y - obs.own_position[1]
                if random.random() < self.genome.distract_visibility:
                    # Move toward
                    if abs(dx) >= abs(dy):
                        self.state.distract_direction = Direction.RIGHT if dx > 0 else Direction.LEFT
                    else:
                        self.state.distract_direction = Direction.DOWN if dy > 0 else Direction.UP
                else:
                    # Move perpendicular
                    self.state.distract_direction = random.choice(
                        [Direction.UP, Direction.DOWN] if abs(dx) > abs(dy)
                        else [Direction.LEFT, Direction.RIGHT]
                    )
            else:
                self.state.distract_direction = random.choice(
                    [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
                )

        self.state.distract_steps_remaining -= 1

        # Mid-distract reversal
        if (self.genome.distract_reverse and
                self.state.distract_steps_remaining == self.genome.distract_duration // 2):
            # Reverse direction
            reversal = {
                Direction.UP: Direction.DOWN,
                Direction.DOWN: Direction.UP,
                Direction.LEFT: Direction.RIGHT,
                Direction.RIGHT: Direction.LEFT,
                Direction.STAY: Direction.STAY,
            }
            self.state.distract_direction = reversal.get(
                self.state.distract_direction, Direction.STAY
            )

        return self.state.distract_direction or Direction.STAY

    def decide(self, obs: Observation) -> Direction:
        """
        Make a decision based on current observation.

        Returns direction and records decision trace for observability.
        """
        # Update memory
        for agent in obs.visible_agents:
            self.state.last_seeker_position = (agent.x, agent.y)
            self.state.seeker_memory.append((agent.x, agent.y))
            if len(self.state.seeker_memory) > self.genome.memory_length:
                self.state.seeker_memory.pop(0)

        # Evaluate mode transitions
        old_mode = self.state.mode
        new_mode = self._evaluate_transitions(obs)
        self.state.mode = new_mode

        # Execute mode-specific behavior
        action_fns = {
            HiderMode.HIDE: self._hide_action,
            HiderMode.EVADE: self._evade_action,
            HiderMode.DISTRACT: self._distract_action,
        }
        action = action_fns[self.state.mode](obs)

        # Record decision trace
        self.decision_trace.append({
            'time_step': obs.time_step,
            'mode': self.state.mode.name,
            'mode_changed': old_mode != new_mode,
            'seeker_distance': self._distance_to_seeker(obs),
            'action': action.name,
            'steps_hidden': self.state.steps_hidden,
        })

        return action

    def explain(self) -> str:
        """Generate human-readable explanation of current state."""
        lines = [
            f"Hider Statechart:",
            f"  Current Mode: {self.state.mode.name}",
            f"  Steps Hidden: {self.state.steps_hidden}",
        ]
        if self.state.last_seeker_position:
            lines.append(f"  Last Seeker Position: {self.state.last_seeker_position}")
        if self.state.distract_steps_remaining > 0:
            lines.append(f"  Distract Steps Left: {self.state.distract_steps_remaining}")

        lines.append(f"\n  Genome Traits:")
        lines.append(f"    Hide: shelter={self.genome.hide_prefer_shelter}, "
                     f"corners={self.genome.hide_prefer_corners}, "
                     f"stay_still={self.genome.hide_stay_still:.2f}")
        lines.append(f"    Evade: away={self.genome.evade_away}, "
                     f"perpendicular={self.genome.evade_perpendicular}, "
                     f"juke={self.genome.evade_juke}")
        lines.append(f"    Distract: duration={self.genome.distract_duration}, "
                     f"reverse={self.genome.distract_reverse}")
        return '\n'.join(lines)

    def extract_strategy(self) -> Dict[str, Any]:
        """
        Extract learned strategy as interpretable statechart.

        This is the key innovation - strategies are EXTRACTABLE.
        """
        return {
            'type': 'hider_statechart',
            'modes': ['HIDE', 'EVADE', 'DISTRACT'],
            'transitions': [
                {
                    'from': 'HIDE',
                    'to': 'EVADE',
                    'guard': f'seeker_visible AND danger_level >= {self.genome.hide_to_evade_threshold:.2f}',
                },
                {
                    'from': 'HIDE',
                    'to': 'DISTRACT',
                    'guard': f'random < {self.genome.distract_chance:.2f}',
                },
                {
                    'from': 'EVADE',
                    'to': 'HIDE',
                    'guard': f'safety_level >= {self.genome.evade_to_hide_threshold:.2f}',
                },
                {
                    'from': 'EVADE',
                    'to': 'DISTRACT',
                    'guard': f'seeker_visible AND random < {self.genome.distract_chance:.2f}',
                },
                {
                    'from': 'DISTRACT',
                    'to': 'EVADE',
                    'guard': f'distract_done AND seeker_close',
                },
                {
                    'from': 'DISTRACT',
                    'to': 'HIDE',
                    'guard': 'distract_done AND seeker_far',
                },
            ],
            'mode_behaviors': {
                'HIDE': {
                    'prefer_shelter': self.genome.hide_prefer_shelter,
                    'prefer_corners': self.genome.hide_prefer_corners,
                    'stay_still_prob': self.genome.hide_stay_still,
                },
                'EVADE': {
                    'move_away': self.genome.evade_away,
                    'move_perpendicular': self.genome.evade_perpendicular,
                    'toward_shelter': self.genome.evade_toward_shelter,
                    'juke': self.genome.evade_juke,
                },
                'DISTRACT': {
                    'duration': self.genome.distract_duration,
                    'mid_reverse': self.genome.distract_reverse,
                    'visibility': self.genome.distract_visibility,
                },
            },
            'parameters': {
                'safety_distance': self.genome.safety_distance,
                'memory_length': self.genome.memory_length,
            },
        }
