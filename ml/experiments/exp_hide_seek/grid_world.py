"""
Grid World Environment for Hide-and-Seek

Features:
- Partial observability via vision cones
- Obstacles that block line of sight
- Movable boxes for emergent tool use
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Set
import random
import math


class Cell(Enum):
    """Cell types in the grid."""
    EMPTY = auto()
    WALL = auto()      # Immovable obstacle
    BOX = auto()       # Movable obstacle (can be pushed)
    SHELTER = auto()   # Hiding spot bonus


class Direction(Enum):
    """Movement and facing directions."""
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]


@dataclass
class Agent:
    """Agent in the grid world."""
    x: int
    y: int
    facing: Direction = Direction.UP
    is_seeker: bool = True
    vision_range: int = 5
    vision_angle: float = 90.0  # degrees

    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Observation:
    """What an agent can observe."""
    visible_cells: Set[Tuple[int, int]]
    visible_agents: List[Agent]
    own_position: Tuple[int, int]
    own_facing: Direction
    time_step: int
    memory: List[Tuple[int, int]]  # Recently seen opponent positions


@dataclass
class GridWorld:
    """
    Grid world environment for hide-and-seek.

    Supports:
    - Line-of-sight occlusion
    - Pushable boxes
    - Multiple seekers/hiders
    """
    width: int = 15
    height: int = 15
    grid: List[List[Cell]] = field(default_factory=list)
    seekers: List[Agent] = field(default_factory=list)
    hiders: List[Agent] = field(default_factory=list)
    time_step: int = 0
    max_steps: int = 200

    def __post_init__(self):
        if not self.grid:
            self.grid = [[Cell.EMPTY for _ in range(self.width)]
                         for _ in range(self.height)]

    @classmethod
    def create_arena(cls, width: int = 15, height: int = 15,
                     num_walls: int = 10, num_boxes: int = 5,
                     num_shelters: int = 3) -> 'GridWorld':
        """Create a randomized arena."""
        world = cls(width=width, height=height)

        # Add walls along edges
        for x in range(width):
            world.grid[0][x] = Cell.WALL
            world.grid[height-1][x] = Cell.WALL
        for y in range(height):
            world.grid[y][0] = Cell.WALL
            world.grid[y][width-1] = Cell.WALL

        # Add random interior walls
        interior = [(x, y) for x in range(2, width-2)
                    for y in range(2, height-2)]
        random.shuffle(interior)

        for i in range(min(num_walls, len(interior))):
            x, y = interior[i]
            world.grid[y][x] = Cell.WALL

        # Add boxes (pushable obstacles)
        for i in range(num_walls, min(num_walls + num_boxes, len(interior))):
            x, y = interior[i]
            world.grid[y][x] = Cell.BOX

        # Add shelters (hiding spots)
        for i in range(num_walls + num_boxes,
                       min(num_walls + num_boxes + num_shelters, len(interior))):
            x, y = interior[i]
            world.grid[y][x] = Cell.SHELTER

        return world

    def add_seeker(self, x: int, y: int) -> Agent:
        """Add a seeker agent."""
        seeker = Agent(x=x, y=y, is_seeker=True,
                       vision_range=6, vision_angle=120.0)
        self.seekers.append(seeker)
        return seeker

    def add_hider(self, x: int, y: int) -> Agent:
        """Add a hider agent."""
        hider = Agent(x=x, y=y, is_seeker=False,
                      vision_range=4, vision_angle=360.0)  # Hiders see all around
        self.hiders.append(hider)
        return hider

    def spawn_agents(self, num_seekers: int = 1, num_hiders: int = 1):
        """Spawn agents at random empty positions."""
        empty = [(x, y) for x in range(self.width)
                 for y in range(self.height)
                 if self.grid[y][x] == Cell.EMPTY]
        random.shuffle(empty)

        idx = 0
        for _ in range(num_seekers):
            if idx < len(empty):
                x, y = empty[idx]
                self.add_seeker(x, y)
                idx += 1

        for _ in range(num_hiders):
            if idx < len(empty):
                x, y = empty[idx]
                self.add_hider(x, y)
                idx += 1

    def get_cell(self, x: int, y: int) -> Cell:
        """Get cell at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return Cell.WALL

    def is_blocked(self, x: int, y: int) -> bool:
        """Check if position is blocked."""
        cell = self.get_cell(x, y)
        return cell in (Cell.WALL, Cell.BOX)

    def blocks_vision(self, x: int, y: int) -> bool:
        """Check if position blocks line of sight."""
        cell = self.get_cell(x, y)
        return cell in (Cell.WALL, Cell.BOX)

    def _line_of_sight(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a clear line of sight using Bresenham's."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1
        while True:
            if x == x2 and y == y2:
                return True
            if (x, y) != (x1, y1) and self.blocks_vision(x, y):
                return False

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _in_vision_cone(self, agent: Agent, x: int, y: int) -> bool:
        """Check if position is within agent's vision cone."""
        dx = x - agent.x
        dy = y - agent.y

        # Check range
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > agent.vision_range:
            return False

        # Check angle (360 degree vision sees everything)
        if agent.vision_angle >= 360.0:
            return True

        # Calculate angle to target
        angle_to_target = math.degrees(math.atan2(dy, dx))

        # Calculate facing angle
        facing_angles = {
            Direction.UP: -90,
            Direction.DOWN: 90,
            Direction.LEFT: 180,
            Direction.RIGHT: 0,
            Direction.STAY: 0,
        }
        facing_angle = facing_angles[agent.facing]

        # Check if within cone
        angle_diff = abs(angle_to_target - facing_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= agent.vision_angle / 2

    def get_observation(self, agent: Agent,
                        memory: List[Tuple[int, int]] = None) -> Observation:
        """Get observation for an agent with partial observability."""
        visible_cells = set()
        visible_agents = []

        # Check all cells in range
        for y in range(self.height):
            for x in range(self.width):
                if self._in_vision_cone(agent, x, y):
                    if self._line_of_sight(agent.x, agent.y, x, y):
                        visible_cells.add((x, y))

        # Check for visible opponents
        opponents = self.hiders if agent.is_seeker else self.seekers
        for opp in opponents:
            if (opp.x, opp.y) in visible_cells:
                visible_agents.append(opp)

        return Observation(
            visible_cells=visible_cells,
            visible_agents=visible_agents,
            own_position=(agent.x, agent.y),
            own_facing=agent.facing,
            time_step=self.time_step,
            memory=memory or [],
        )

    def try_move(self, agent: Agent, direction: Direction) -> bool:
        """Try to move agent in direction. Returns success."""
        new_x = agent.x + direction.dx
        new_y = agent.y + direction.dy

        # Check bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            return False

        cell = self.get_cell(new_x, new_y)

        # Can't move into walls
        if cell == Cell.WALL:
            return False

        # Try to push boxes
        if cell == Cell.BOX:
            push_x = new_x + direction.dx
            push_y = new_y + direction.dy
            if (0 <= push_x < self.width and 0 <= push_y < self.height and
                    self.grid[push_y][push_x] == Cell.EMPTY):
                # Push the box
                self.grid[new_y][new_x] = Cell.EMPTY
                self.grid[push_y][push_x] = Cell.BOX
            else:
                return False

        # Check for collision with other agents
        all_agents = self.seekers + self.hiders
        for other in all_agents:
            if other != agent and other.x == new_x and other.y == new_y:
                return False

        # Move successful
        agent.x = new_x
        agent.y = new_y
        agent.facing = direction if direction != Direction.STAY else agent.facing
        return True

    def step(self, seeker_actions: List[Direction],
             hider_actions: List[Direction]) -> Tuple[bool, float, float]:
        """
        Execute one time step.

        Returns: (game_over, seeker_reward, hider_reward)
        """
        self.time_step += 1

        # Execute seeker actions
        for i, action in enumerate(seeker_actions):
            if i < len(self.seekers):
                self.try_move(self.seekers[i], action)

        # Execute hider actions
        for i, action in enumerate(hider_actions):
            if i < len(self.hiders):
                self.try_move(self.hiders[i], action)

        # Check for catches (seeker adjacent to hider)
        seeker_reward = 0.0
        hider_reward = 0.0

        for seeker in self.seekers:
            for hider in self.hiders:
                dist = abs(seeker.x - hider.x) + abs(seeker.y - hider.y)
                if dist <= 1:  # Adjacent = caught
                    seeker_reward += 10.0
                    hider_reward -= 10.0
                    return True, seeker_reward, hider_reward

        # Time penalty for seeker, survival bonus for hider
        seeker_reward -= 0.1
        hider_reward += 0.1

        # Check time limit
        if self.time_step >= self.max_steps:
            # Hider survives = hider wins
            hider_reward += 5.0
            seeker_reward -= 5.0
            return True, seeker_reward, hider_reward

        return False, seeker_reward, hider_reward

    def is_hider_visible_to_any_seeker(self, hider: Agent) -> bool:
        """Check if hider is visible to any seeker."""
        for seeker in self.seekers:
            obs = self.get_observation(seeker)
            if any(a.x == hider.x and a.y == hider.y
                   for a in obs.visible_agents):
                return True
        return False

    def get_distance_to_nearest_seeker(self, hider: Agent) -> float:
        """Get Manhattan distance to nearest seeker."""
        if not self.seekers:
            return float('inf')
        return min(abs(s.x - hider.x) + abs(s.y - hider.y)
                   for s in self.seekers)

    def render(self) -> str:
        """Render the grid as ASCII."""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                # Check for agents first
                char = None
                for s in self.seekers:
                    if s.x == x and s.y == y:
                        char = 'S'
                        break
                if char is None:
                    for h in self.hiders:
                        if h.x == x and h.y == y:
                            char = 'H'
                            break
                if char is None:
                    cell = self.grid[y][x]
                    char = {
                        Cell.EMPTY: '.',
                        Cell.WALL: '#',
                        Cell.BOX: 'B',
                        Cell.SHELTER: '_',
                    }.get(cell, '?')
                row.append(char)
            lines.append(''.join(row))
        return '\n'.join(lines)
