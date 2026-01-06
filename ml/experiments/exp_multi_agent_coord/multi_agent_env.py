"""
Multi-Agent Environment with Shared Event Bus

Environment for 4+ agents with:
- Shared event bus for communication
- Partial observability per agent
- Coordination tasks requiring cooperation
- Observable emergent communication protocols

Extends hide-seek grid world patterns to multi-agent coordination.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
from collections import deque
import random
import math


# =============================================================================
# Event System
# =============================================================================

class EventType(Enum):
    """Types of events on the shared bus."""
    # Movement signals
    MOVE_REQUEST = auto()
    POSITION_UPDATE = auto()

    # Role signals
    LEADER_CLAIM = auto()
    FOLLOWER_ACK = auto()
    ROLE_QUERY = auto()

    # Coordination signals
    GATHER = auto()
    DISPERSE = auto()
    FOLLOW_ME = auto()
    WAIT = auto()
    GO = auto()

    # Task signals
    TARGET_SPOTTED = auto()
    TASK_COMPLETE = auto()
    NEED_HELP = auto()

    # Custom evolved signals (slots for evolution)
    SIGNAL_A = auto()
    SIGNAL_B = auto()
    SIGNAL_C = auto()
    SIGNAL_D = auto()


@dataclass
class Event:
    """An event on the shared bus."""
    event_type: EventType
    sender_id: int
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    target_id: Optional[int] = None  # None = broadcast

    def is_broadcast(self) -> bool:
        return self.target_id is None


class EventBus:
    """
    Shared event bus for agent communication.

    All agents can broadcast or send targeted messages.
    Events persist for a short window (for async processing).
    """

    def __init__(self, max_history: int = 100):
        self.events: deque = deque(maxlen=max_history)
        self.current_time: float = 0.0
        self.event_window: float = 1.0  # Events visible for 1 time unit

        # Statistics
        self.total_events: int = 0
        self.events_by_type: Dict[EventType, int] = {t: 0 for t in EventType}

    def publish(self, event: Event):
        """Publish event to bus."""
        event.timestamp = self.current_time
        self.events.append(event)
        self.total_events += 1
        self.events_by_type[event.event_type] += 1

    def get_events_for_agent(self, agent_id: int) -> List[Event]:
        """Get events visible to an agent."""
        visible = []
        for event in self.events:
            # Check time window
            if self.current_time - event.timestamp > self.event_window:
                continue
            # Check if event is for this agent
            if event.is_broadcast() or event.target_id == agent_id:
                visible.append(event)
        return visible

    def get_recent_events(self, n: int = 10) -> List[Event]:
        """Get n most recent events."""
        return list(self.events)[-n:]

    def tick(self, dt: float = 0.1):
        """Advance time."""
        self.current_time += dt

    def clear(self):
        """Clear all events."""
        self.events.clear()
        self.current_time = 0.0


# =============================================================================
# Agent
# =============================================================================

@dataclass
class AgentState:
    """State of a single agent."""
    agent_id: int
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0  # radians

    # Role (evolved)
    role: str = "NONE"  # LEADER, FOLLOWER, SCOUT, etc.

    # Internal state
    energy: float = 1.0
    carrying: bool = False

    def distance_to(self, other: "AgentState") -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_point(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def to_array(self) -> mx.array:
        return mx.array([
            self.x, self.y, self.vx, self.vy, self.heading,
            float(self.role == "LEADER"),
            float(self.role == "FOLLOWER"),
            self.energy, float(self.carrying)
        ])


# =============================================================================
# Tasks
# =============================================================================

class TaskType(Enum):
    """Types of coordination tasks."""
    GATHER = auto()       # All agents gather at point
    FORMATION = auto()    # Form specific shape
    COVERAGE = auto()     # Spread to cover area
    TRANSPORT = auto()    # Move object together
    PURSUIT = auto()      # Chase moving target


@dataclass
class Task:
    """A coordination task."""
    task_type: TaskType
    target_x: float = 0.5
    target_y: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)

    def is_complete(self, agents: List[AgentState]) -> bool:
        """Check if task is complete."""
        if self.task_type == TaskType.GATHER:
            # All agents within threshold of target
            threshold = self.params.get("threshold", 0.1)
            return all(
                a.distance_to_point(self.target_x, self.target_y) < threshold
                for a in agents
            )

        elif self.task_type == TaskType.FORMATION:
            # Check formation (simplified: agents evenly spaced)
            if len(agents) < 2:
                return True
            center_x = sum(a.x for a in agents) / len(agents)
            center_y = sum(a.y for a in agents) / len(agents)
            distances = [a.distance_to_point(center_x, center_y) for a in agents]
            # Check variance is low (evenly distributed)
            mean_dist = sum(distances) / len(distances)
            variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
            return variance < 0.01

        elif self.task_type == TaskType.COVERAGE:
            # Agents spread out (pairwise distances > threshold)
            threshold = self.params.get("min_distance", 0.3)
            for i, a1 in enumerate(agents):
                for a2 in agents[i + 1:]:
                    if a1.distance_to(a2) < threshold:
                        return False
            return True

        elif self.task_type == TaskType.TRANSPORT:
            # Object at target (requires coordination)
            obj_x = self.params.get("obj_x", 0.0)
            obj_y = self.params.get("obj_y", 0.0)
            # Check if enough agents near object
            near_object = sum(
                1 for a in agents
                if a.distance_to_point(obj_x, obj_y) < 0.1
            )
            return near_object >= self.params.get("min_agents", 2)

        return False


# =============================================================================
# Multi-Agent Environment
# =============================================================================

class MultiAgentEnv:
    """
    Environment for multi-agent coordination.

    Features:
    - 4+ agents with continuous positions
    - Shared event bus for communication
    - Partial observability (vision radius)
    - Coordination tasks requiring cooperation
    """

    def __init__(
        self,
        n_agents: int = 4,
        world_size: float = 1.0,
        vision_radius: float = 0.3,
        dt: float = 0.1,
    ):
        self.n_agents = n_agents
        self.world_size = world_size
        self.vision_radius = vision_radius
        self.dt = dt

        # Agents
        self.agents: List[AgentState] = []

        # Event bus
        self.event_bus = EventBus()

        # Current task
        self.task: Optional[Task] = None

        # Episode tracking
        self.step_count = 0
        self.max_steps = 500

    def reset(self, task: Optional[Task] = None) -> Dict[int, Dict[str, Any]]:
        """Reset environment."""
        self.step_count = 0
        self.event_bus.clear()

        # Initialize agents in random positions
        self.agents = []
        for i in range(self.n_agents):
            agent = AgentState(
                agent_id=i,
                x=random.uniform(0.1, 0.9) * self.world_size,
                y=random.uniform(0.1, 0.9) * self.world_size,
                heading=random.uniform(0, 2 * math.pi),
                role="NONE",
            )
            self.agents.append(agent)

        # Set task
        self.task = task or Task(
            task_type=TaskType.GATHER,
            target_x=self.world_size / 2,
            target_y=self.world_size / 2,
        )

        return self._get_observations()

    def _get_observations(self) -> Dict[int, Dict[str, Any]]:
        """Get observations for each agent."""
        obs = {}
        for agent in self.agents:
            # Visible agents (within vision radius)
            visible_agents = [
                a for a in self.agents
                if a.agent_id != agent.agent_id
                and agent.distance_to(a) < self.vision_radius
            ]

            # Events for this agent
            events = self.event_bus.get_events_for_agent(agent.agent_id)

            obs[agent.agent_id] = {
                "self": agent,
                "visible_agents": visible_agents,
                "events": events,
                "task": self.task,
            }

        return obs

    def step(
        self,
        actions: Dict[int, Tuple[float, float, Optional[Event]]]
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, float], bool, Dict]:
        """
        Step environment.

        Args:
            actions: Dict of agent_id -> (vx, vy, optional_event)

        Returns:
            observations, rewards, done, info
        """
        self.step_count += 1

        # Process actions
        for agent_id, action in actions.items():
            vx, vy, event = action

            # Clamp velocities
            max_speed = 0.1
            vx = max(-max_speed, min(max_speed, vx))
            vy = max(-max_speed, min(max_speed, vy))

            # Update agent
            agent = self.agents[agent_id]
            agent.x = max(0, min(self.world_size, agent.x + vx * self.dt))
            agent.y = max(0, min(self.world_size, agent.y + vy * self.dt))
            agent.vx = vx
            agent.vy = vy

            if vx != 0 or vy != 0:
                agent.heading = math.atan2(vy, vx)

            # Publish event if provided
            if event is not None:
                self.event_bus.publish(event)

        # Tick event bus
        self.event_bus.tick(self.dt)

        # Compute rewards
        rewards = self._compute_rewards()

        # Check done
        done = (
            self.step_count >= self.max_steps or
            (self.task is not None and self.task.is_complete(self.agents))
        )

        # Info
        info = {
            "step": self.step_count,
            "task_complete": self.task.is_complete(self.agents) if self.task else False,
            "total_events": self.event_bus.total_events,
        }

        return self._get_observations(), rewards, done, info

    def _compute_rewards(self) -> Dict[int, float]:
        """Compute rewards for each agent."""
        rewards = {}

        if self.task is None:
            return {a.agent_id: 0.0 for a in self.agents}

        # Base reward: progress toward task
        if self.task.task_type == TaskType.GATHER:
            for agent in self.agents:
                dist = agent.distance_to_point(self.task.target_x, self.task.target_y)
                rewards[agent.agent_id] = -dist * 0.1

                # Bonus for completing task
                if self.task.is_complete(self.agents):
                    rewards[agent.agent_id] += 10.0

        elif self.task.task_type == TaskType.COVERAGE:
            # Reward for spreading out
            for agent in self.agents:
                min_dist = float('inf')
                for other in self.agents:
                    if other.agent_id != agent.agent_id:
                        min_dist = min(min_dist, agent.distance_to(other))
                rewards[agent.agent_id] = min_dist * 0.5

        else:
            rewards = {a.agent_id: 0.0 for a in self.agents}

        # Small penalty for excessive communication
        recent_events = self.event_bus.get_recent_events(20)
        comm_penalty = len(recent_events) * 0.001
        for agent_id in rewards:
            rewards[agent_id] -= comm_penalty

        return rewards

    def get_global_state(self) -> mx.array:
        """Get full global state (for centralized critic)."""
        states = []
        for agent in self.agents:
            states.append(agent.to_array())
        return mx.concatenate(states)


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate multi-agent environment."""
    print("=" * 60)
    print("Multi-Agent Coordination Environment")
    print("=" * 60)

    env = MultiAgentEnv(n_agents=4)

    # Gather task
    task = Task(
        task_type=TaskType.GATHER,
        target_x=0.5,
        target_y=0.5,
        params={"threshold": 0.15}
    )

    obs = env.reset(task)

    print(f"\nAgents: {env.n_agents}")
    print(f"Task: {task.task_type.name} at ({task.target_x}, {task.target_y})")
    print(f"\nInitial positions:")
    for agent in env.agents:
        print(f"  Agent {agent.agent_id}: ({agent.x:.2f}, {agent.y:.2f})")

    # Simple rule-based policy: move toward target
    total_rewards = {i: 0.0 for i in range(env.n_agents)}

    for step in range(100):
        actions = {}
        for agent in env.agents:
            # Move toward target
            dx = task.target_x - agent.x
            dy = task.target_y - agent.y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6

            vx = (dx / dist) * 0.1
            vy = (dy / dist) * 0.1

            # Optionally send event
            event = None
            if step == 0:
                event = Event(
                    event_type=EventType.GATHER,
                    sender_id=agent.agent_id,
                    payload={"target": (task.target_x, task.target_y)}
                )

            actions[agent.agent_id] = (vx, vy, event)

        obs, rewards, done, info = env.step(actions)

        for agent_id, r in rewards.items():
            total_rewards[agent_id] += r

        if done:
            print(f"\nDone at step {step}!")
            print(f"Task complete: {info['task_complete']}")
            break

    print(f"\nFinal positions:")
    for agent in env.agents:
        dist = agent.distance_to_point(task.target_x, task.target_y)
        print(f"  Agent {agent.agent_id}: ({agent.x:.2f}, {agent.y:.2f}), "
              f"dist={dist:.3f}")

    print(f"\nTotal rewards: {total_rewards}")
    print(f"Events published: {env.event_bus.total_events}")

    return env


if __name__ == "__main__":
    demo()
