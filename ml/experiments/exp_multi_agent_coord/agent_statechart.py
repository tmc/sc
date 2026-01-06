"""
Agent Statechart with Emergent Roles

Each agent has its own statechart that:
- Manages discrete behavior modes (IDLE, LEADING, FOLLOWING, etc.)
- Responds to events from the shared bus
- Evolves role assignment and communication patterns

Roles emerge from evolution:
- LEADER: Broadcasts commands, others follow
- FOLLOWER: Listens and executes
- SCOUT: Explores and reports
- COORDINATOR: Mediates between groups
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import math

from .multi_agent_env import Event, EventType, AgentState, Task, TaskType


# =============================================================================
# Agent Roles (Emergent)
# =============================================================================

class AgentRole(Enum):
    """Roles that can emerge through evolution."""
    NONE = auto()
    LEADER = auto()
    FOLLOWER = auto()
    SCOUT = auto()
    COORDINATOR = auto()


# =============================================================================
# Agent Behavior Modes
# =============================================================================

class BehaviorMode(Enum):
    """Discrete behavior modes for agent statechart."""
    IDLE = auto()
    EXPLORING = auto()
    MOVING_TO_TARGET = auto()
    FOLLOWING_LEADER = auto()
    LEADING = auto()
    WAITING = auto()
    SIGNALING = auto()
    COORDINATING = auto()


# =============================================================================
# Statechart Transitions
# =============================================================================

@dataclass
class AgentTransition:
    """Transition in agent statechart."""
    source: BehaviorMode
    target: BehaviorMode
    trigger: Optional[EventType] = None  # Event that triggers transition
    guard: Optional[Callable[[AgentState, Dict], bool]] = None
    priority: int = 0

    # Evolved parameters
    probability: float = 1.0  # For stochastic transitions

    def is_enabled(
        self,
        agent: AgentState,
        context: Dict[str, Any],
        events: List[Event]
    ) -> bool:
        """Check if transition is enabled."""
        # Check trigger event
        if self.trigger is not None:
            has_trigger = any(e.event_type == self.trigger for e in events)
            if not has_trigger:
                return False

        # Check guard
        if self.guard is not None:
            if not self.guard(agent, context):
                return False

        # Stochastic check
        if self.probability < 1.0:
            if random.random() > self.probability:
                return False

        return True


# =============================================================================
# Agent Statechart
# =============================================================================

class AgentStatechart:
    """
    Statechart for a single agent.

    Features:
    - Discrete behavior modes
    - Event-driven transitions
    - Emergent role assignment
    - Communication protocol execution
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.current_mode = BehaviorMode.IDLE
        self.role = AgentRole.NONE

        # Transitions (will be evolved)
        self.transitions: List[AgentTransition] = self._build_default_transitions()

        # Mode-specific controllers
        self.mode_controllers: Dict[BehaviorMode, Callable] = {
            BehaviorMode.IDLE: self._idle_action,
            BehaviorMode.EXPLORING: self._explore_action,
            BehaviorMode.MOVING_TO_TARGET: self._move_to_target_action,
            BehaviorMode.FOLLOWING_LEADER: self._follow_leader_action,
            BehaviorMode.LEADING: self._lead_action,
            BehaviorMode.WAITING: self._wait_action,
            BehaviorMode.SIGNALING: self._signal_action,
            BehaviorMode.COORDINATING: self._coordinate_action,
        }

        # Internal state
        self.target_x: float = 0.5
        self.target_y: float = 0.5
        self.leader_id: Optional[int] = None
        self.followers: Set[int] = set()
        self.signal_cooldown: float = 0.0

        # Statistics
        self.mode_time: Dict[BehaviorMode, float] = {m: 0.0 for m in BehaviorMode}
        self.transitions_taken: int = 0

    def _build_default_transitions(self) -> List[AgentTransition]:
        """Build default transition table."""
        return [
            # IDLE transitions
            AgentTransition(
                source=BehaviorMode.IDLE,
                target=BehaviorMode.EXPLORING,
                guard=lambda a, c: c.get("task") is not None,
                priority=1,
            ),
            AgentTransition(
                source=BehaviorMode.IDLE,
                target=BehaviorMode.FOLLOWING_LEADER,
                trigger=EventType.FOLLOW_ME,
                priority=2,
            ),

            # EXPLORING transitions
            AgentTransition(
                source=BehaviorMode.EXPLORING,
                target=BehaviorMode.MOVING_TO_TARGET,
                guard=lambda a, c: c.get("target_known", False),
                priority=1,
            ),
            AgentTransition(
                source=BehaviorMode.EXPLORING,
                target=BehaviorMode.LEADING,
                trigger=EventType.LEADER_CLAIM,
                guard=lambda a, c: c.get("no_leader", True),
                priority=2,
            ),

            # MOVING_TO_TARGET transitions
            AgentTransition(
                source=BehaviorMode.MOVING_TO_TARGET,
                target=BehaviorMode.WAITING,
                guard=lambda a, c: c.get("at_target", False),
                priority=1,
            ),
            AgentTransition(
                source=BehaviorMode.MOVING_TO_TARGET,
                target=BehaviorMode.FOLLOWING_LEADER,
                trigger=EventType.FOLLOW_ME,
                priority=2,
            ),

            # FOLLOWING_LEADER transitions
            AgentTransition(
                source=BehaviorMode.FOLLOWING_LEADER,
                target=BehaviorMode.WAITING,
                trigger=EventType.WAIT,
                priority=1,
            ),
            AgentTransition(
                source=BehaviorMode.FOLLOWING_LEADER,
                target=BehaviorMode.MOVING_TO_TARGET,
                trigger=EventType.GO,
                priority=1,
            ),

            # LEADING transitions
            AgentTransition(
                source=BehaviorMode.LEADING,
                target=BehaviorMode.COORDINATING,
                guard=lambda a, c: len(c.get("followers", [])) >= 2,
                priority=1,
            ),

            # WAITING transitions
            AgentTransition(
                source=BehaviorMode.WAITING,
                target=BehaviorMode.MOVING_TO_TARGET,
                trigger=EventType.GO,
                priority=1,
            ),

            # COORDINATING transitions
            AgentTransition(
                source=BehaviorMode.COORDINATING,
                target=BehaviorMode.IDLE,
                guard=lambda a, c: c.get("task_complete", False),
                priority=1,
            ),
        ]

    def step(
        self,
        agent: AgentState,
        context: Dict[str, Any],
        events: List[Event],
        dt: float = 0.1,
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """
        Execute one step of the statechart.

        Args:
            agent: Current agent state
            context: Observation context
            events: Events from bus
            dt: Time step

        Returns:
            (velocity, optional_event)
        """
        # Update cooldowns
        self.signal_cooldown = max(0, self.signal_cooldown - dt)

        # Build context for guards
        guard_context = self._build_guard_context(agent, context, events)

        # Check transitions
        enabled = []
        for trans in self.transitions:
            if trans.source == self.current_mode:
                if trans.is_enabled(agent, guard_context, events):
                    enabled.append(trans)

        # Take highest priority transition
        if enabled:
            enabled.sort(key=lambda t: -t.priority)
            trans = enabled[0]
            self.current_mode = trans.target
            self.transitions_taken += 1

        # Track mode time
        self.mode_time[self.current_mode] += dt

        # Execute mode controller
        controller = self.mode_controllers.get(self.current_mode, self._idle_action)
        return controller(agent, context, events)

    def _build_guard_context(
        self,
        agent: AgentState,
        context: Dict[str, Any],
        events: List[Event]
    ) -> Dict[str, Any]:
        """Build context for guard evaluation."""
        task = context.get("task")
        visible_agents = context.get("visible_agents", [])

        # Check if at target
        at_target = False
        if task:
            dist = agent.distance_to_point(task.target_x, task.target_y)
            at_target = dist < 0.1

        # Check if we know target
        target_known = task is not None

        # Check for leader
        no_leader = not any(
            e.event_type == EventType.LEADER_CLAIM
            for e in events
            if e.sender_id != agent.agent_id
        )

        # Count followers
        followers = [
            e.sender_id for e in events
            if e.event_type == EventType.FOLLOWER_ACK
            and e.target_id == agent.agent_id
        ]

        return {
            "task": task,
            "at_target": at_target,
            "target_known": target_known,
            "no_leader": no_leader,
            "followers": followers,
            "visible_agents": visible_agents,
            "task_complete": task.is_complete([agent] + visible_agents) if task else False,
        }

    # ==========================================================================
    # Mode Controllers
    # ==========================================================================

    def _idle_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """IDLE: Stay still, wait for task."""
        return (0.0, 0.0), None

    def _explore_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """EXPLORING: Random walk to find task target."""
        # Random direction with momentum
        vx = agent.vx + random.gauss(0, 0.02)
        vy = agent.vy + random.gauss(0, 0.02)

        # Clamp and normalize
        speed = math.sqrt(vx * vx + vy * vy) + 1e-6
        if speed > 0.1:
            vx = (vx / speed) * 0.1
            vy = (vy / speed) * 0.1

        # Maybe claim leadership
        event = None
        if self.signal_cooldown <= 0 and random.random() < 0.1:
            if self.role == AgentRole.NONE:
                self.role = AgentRole.LEADER
                event = Event(
                    event_type=EventType.LEADER_CLAIM,
                    sender_id=agent.agent_id,
                )
                self.signal_cooldown = 1.0

        return (vx, vy), event

    def _move_to_target_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """MOVING_TO_TARGET: Move directly to task target."""
        task = context.get("task")
        if task is None:
            return (0.0, 0.0), None

        # Vector to target
        dx = task.target_x - agent.x
        dy = task.target_y - agent.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6

        speed = min(0.1, dist)
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed

        return (vx, vy), None

    def _follow_leader_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """FOLLOWING_LEADER: Follow the designated leader."""
        # Find leader position from events
        leader_pos = None
        for event in events:
            if event.event_type == EventType.POSITION_UPDATE:
                if event.sender_id == self.leader_id:
                    leader_pos = event.payload.get("position")
                    break

        # Or find from visible agents
        if leader_pos is None:
            visible = context.get("visible_agents", [])
            for a in visible:
                if a.agent_id == self.leader_id:
                    leader_pos = (a.x, a.y)
                    break

        if leader_pos is None:
            # Lost leader, wander
            return (random.gauss(0, 0.02), random.gauss(0, 0.02)), None

        # Move toward leader with offset
        dx = leader_pos[0] - agent.x
        dy = leader_pos[1] - agent.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6

        # Keep some distance
        target_dist = 0.15
        if dist > target_dist:
            speed = min(0.1, dist - target_dist)
            vx = (dx / dist) * speed
            vy = (dy / dist) * speed
        else:
            vx, vy = 0.0, 0.0

        # Send follower acknowledgment
        event = None
        if self.signal_cooldown <= 0:
            event = Event(
                event_type=EventType.FOLLOWER_ACK,
                sender_id=agent.agent_id,
                target_id=self.leader_id,
            )
            self.signal_cooldown = 0.5

        return (vx, vy), event

    def _lead_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """LEADING: Lead followers to target."""
        self.role = AgentRole.LEADER

        task = context.get("task")
        if task is None:
            return (0.0, 0.0), None

        # Move to target
        dx = task.target_x - agent.x
        dy = task.target_y - agent.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6

        speed = min(0.08, dist)  # Slower to let followers keep up
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed

        # Broadcast position and follow command
        event = None
        if self.signal_cooldown <= 0:
            event = Event(
                event_type=EventType.FOLLOW_ME,
                sender_id=agent.agent_id,
                payload={"position": (agent.x, agent.y)},
            )
            self.signal_cooldown = 0.3

        return (vx, vy), event

    def _wait_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """WAITING: Wait for signal."""
        return (0.0, 0.0), None

    def _signal_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """SIGNALING: Broadcast signal to others."""
        event = Event(
            event_type=EventType.SIGNAL_A,
            sender_id=agent.agent_id,
        )
        return (0.0, 0.0), event

    def _coordinate_action(
        self, agent: AgentState, context: Dict, events: List[Event]
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """COORDINATING: Coordinate group action."""
        task = context.get("task")

        # Move to target
        vx, vy = 0.0, 0.0
        if task:
            dx = task.target_x - agent.x
            dy = task.target_y - agent.y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            speed = min(0.05, dist)
            vx = (dx / dist) * speed
            vy = (dy / dist) * speed

        # Broadcast GO signal
        event = None
        if self.signal_cooldown <= 0:
            event = Event(
                event_type=EventType.GO,
                sender_id=agent.agent_id,
            )
            self.signal_cooldown = 0.5

        return (vx, vy), event

    # ==========================================================================
    # Event Handlers
    # ==========================================================================

    def handle_event(self, event: Event, agent: AgentState):
        """Handle incoming event."""
        if event.event_type == EventType.LEADER_CLAIM:
            if self.role != AgentRole.LEADER:
                self.role = AgentRole.FOLLOWER
                self.leader_id = event.sender_id

        elif event.event_type == EventType.FOLLOWER_ACK:
            if event.target_id == agent.agent_id:
                self.followers.add(event.sender_id)

        elif event.event_type == EventType.FOLLOW_ME:
            if self.role == AgentRole.FOLLOWER:
                self.leader_id = event.sender_id

    def reset(self):
        """Reset statechart state."""
        self.current_mode = BehaviorMode.IDLE
        self.role = AgentRole.NONE
        self.leader_id = None
        self.followers.clear()
        self.signal_cooldown = 0.0
        self.mode_time = {m: 0.0 for m in BehaviorMode}
        self.transitions_taken = 0


# =============================================================================
# Multi-Agent Statechart Controller
# =============================================================================

class MultiAgentStatechartController:
    """
    Controller that runs a statechart for each agent.
    """

    def __init__(self, n_agents: int):
        self.statecharts = {
            i: AgentStatechart(agent_id=i)
            for i in range(n_agents)
        }

    def step(
        self,
        observations: Dict[int, Dict[str, Any]],
        dt: float = 0.1,
    ) -> Dict[int, Tuple[float, float, Optional[Event]]]:
        """Step all agent statecharts."""
        actions = {}

        for agent_id, obs in observations.items():
            sc = self.statecharts[agent_id]
            agent = obs["self"]
            events = obs.get("events", [])

            # Handle events
            for event in events:
                sc.handle_event(event, agent)

            # Step statechart
            velocity, event = sc.step(agent, obs, events, dt)

            actions[agent_id] = (velocity[0], velocity[1], event)

        return actions

    def reset(self):
        """Reset all statecharts."""
        for sc in self.statecharts.values():
            sc.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics across all agents."""
        stats = {
            "roles": {},
            "modes": {},
            "transitions": 0,
        }

        for agent_id, sc in self.statecharts.items():
            stats["roles"][agent_id] = sc.role.name
            stats["modes"][agent_id] = sc.current_mode.name
            stats["transitions"] += sc.transitions_taken

        return stats


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate agent statecharts."""
    print("=" * 60)
    print("Agent Statechart Controller")
    print("=" * 60)

    from .multi_agent_env import MultiAgentEnv, Task, TaskType

    env = MultiAgentEnv(n_agents=4)
    controller = MultiAgentStatechartController(n_agents=4)

    task = Task(
        task_type=TaskType.GATHER,
        target_x=0.5,
        target_y=0.5,
        params={"threshold": 0.15}
    )

    obs = env.reset(task)
    controller.reset()

    print(f"\nRunning {env.n_agents} agents with statecharts...")

    for step in range(200):
        actions = controller.step(obs, dt=env.dt)
        obs, rewards, done, info = env.step(actions)

        if step % 50 == 0:
            stats = controller.get_statistics()
            print(f"\nStep {step}:")
            print(f"  Roles: {stats['roles']}")
            print(f"  Modes: {stats['modes']}")
            print(f"  Transitions: {stats['transitions']}")

        if done:
            print(f"\nDone at step {step}!")
            print(f"Task complete: {info['task_complete']}")
            break

    # Final statistics
    stats = controller.get_statistics()
    print(f"\n--- Final Statistics ---")
    print(f"Roles: {stats['roles']}")
    print(f"Total transitions: {stats['transitions']}")
    print(f"Events published: {env.event_bus.total_events}")

    return controller, env


if __name__ == "__main__":
    demo()
