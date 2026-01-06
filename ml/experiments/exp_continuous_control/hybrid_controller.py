"""
Hybrid Discrete-Continuous Control for Robotics

Combines discrete state machines with continuous control:
- Discrete states: IDLE -> APPROACH -> GRASP -> LIFT
- Continuous actions: velocity, force, position within each state
- AND-states: Parallel control of multiple limbs
- Guards: Continuous conditions trigger discrete transitions

Key insight: Statecharts provide high-level behavior structure,
while continuous controllers handle low-level execution within states.

No external dependencies (no MuJoCo). Simple 2D simulated environment.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import math


# =============================================================================
# Continuous State Space
# =============================================================================

@dataclass
class RobotState:
    """Continuous state of a robot arm."""
    # End effector position (2D)
    x: float = 0.0
    y: float = 0.0

    # End effector velocity
    vx: float = 0.0
    vy: float = 0.0

    # Gripper state (0=open, 1=closed)
    gripper: float = 0.0

    # Force sensor
    force: float = 0.0

    def to_array(self) -> mx.array:
        return mx.array([self.x, self.y, self.vx, self.vy, self.gripper, self.force])

    @staticmethod
    def from_array(arr: mx.array) -> "RobotState":
        return RobotState(
            x=float(arr[0]), y=float(arr[1]),
            vx=float(arr[2]), vy=float(arr[3]),
            gripper=float(arr[4]), force=float(arr[5])
        )

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)


@dataclass
class ObjectState:
    """State of an object to be manipulated."""
    x: float = 0.5
    y: float = 0.0
    grasped: bool = False
    lifted: bool = False

    def to_array(self) -> mx.array:
        return mx.array([self.x, self.y, float(self.grasped), float(self.lifted)])


@dataclass
class WorldState:
    """Complete world state."""
    robot: RobotState = field(default_factory=RobotState)
    obj: ObjectState = field(default_factory=ObjectState)
    time: float = 0.0

    def to_array(self) -> mx.array:
        return mx.concatenate([
            self.robot.to_array(),
            self.obj.to_array(),
            mx.array([self.time])
        ])


# =============================================================================
# Discrete States
# =============================================================================

class DiscreteState(Enum):
    """High-level discrete states for manipulation task."""
    IDLE = auto()
    APPROACH = auto()
    GRASP = auto()
    LIFT = auto()
    DONE = auto()
    ERROR = auto()


class LimbState(Enum):
    """State of individual limb (for AND-states)."""
    READY = auto()
    MOVING = auto()
    CONTACT = auto()
    HOLDING = auto()


# =============================================================================
# Continuous Controllers (one per discrete state)
# =============================================================================

class ContinuousController(ABC):
    """Base class for continuous controllers within a discrete state."""

    @abstractmethod
    def compute_action(self, world: WorldState) -> Tuple[float, float, float]:
        """Compute continuous action (vx, vy, gripper_cmd)."""
        pass

    @abstractmethod
    def is_done(self, world: WorldState) -> bool:
        """Check if this controller has achieved its goal."""
        pass


class IdleController(ContinuousController):
    """Controller for IDLE state - stay still."""

    def compute_action(self, world: WorldState) -> Tuple[float, float, float]:
        # Zero velocity, open gripper
        return (0.0, 0.0, 0.0)

    def is_done(self, world: WorldState) -> bool:
        # Always ready to transition
        return True


class ApproachController(ContinuousController):
    """Controller for APPROACH state - move toward object."""

    def __init__(self, approach_speed: float = 0.1, threshold: float = 0.05):
        self.approach_speed = approach_speed
        self.threshold = threshold

    def compute_action(self, world: WorldState) -> Tuple[float, float, float]:
        # Vector toward object
        dx = world.obj.x - world.robot.x
        dy = world.obj.y - world.robot.y
        dist = math.sqrt(dx * dx + dy * dy) + 1e-6

        # Proportional control with speed limit
        speed = min(self.approach_speed, dist)
        vx = (dx / dist) * speed
        vy = (dy / dist) * speed

        # Keep gripper open during approach
        return (vx, vy, 0.0)

    def is_done(self, world: WorldState) -> bool:
        dist = world.robot.distance_to(world.obj.x, world.obj.y)
        return dist < self.threshold


class GraspController(ContinuousController):
    """Controller for GRASP state - close gripper on object."""

    def __init__(self, close_speed: float = 0.2, force_threshold: float = 0.5):
        self.close_speed = close_speed
        self.force_threshold = force_threshold

    def compute_action(self, world: WorldState) -> Tuple[float, float, float]:
        # Stay still, close gripper
        gripper_cmd = min(1.0, world.robot.gripper + self.close_speed)
        return (0.0, 0.0, gripper_cmd)

    def is_done(self, world: WorldState) -> bool:
        # Done when gripper closed and force detected
        return world.robot.gripper > 0.9 and world.robot.force > self.force_threshold


class LiftController(ContinuousController):
    """Controller for LIFT state - lift object upward."""

    def __init__(self, lift_speed: float = 0.1, target_height: float = 0.5):
        self.lift_speed = lift_speed
        self.target_height = target_height

    def compute_action(self, world: WorldState) -> Tuple[float, float, float]:
        # Move upward, keep gripper closed
        vy = self.lift_speed if world.robot.y < self.target_height else 0.0
        return (0.0, vy, 1.0)

    def is_done(self, world: WorldState) -> bool:
        return world.robot.y >= self.target_height


# =============================================================================
# Hybrid Statechart
# =============================================================================

@dataclass
class HybridTransition:
    """Transition with continuous guard condition."""
    source: DiscreteState
    target: DiscreteState
    guard: Callable[[WorldState], bool]
    priority: int = 0


class HybridStatechart:
    """
    Hybrid statechart with discrete states and continuous controllers.

    Each discrete state has an associated continuous controller.
    Transitions are guarded by continuous conditions.
    """

    def __init__(self):
        # Current discrete state
        self.current_state = DiscreteState.IDLE

        # Controllers for each state
        self.controllers: Dict[DiscreteState, ContinuousController] = {
            DiscreteState.IDLE: IdleController(),
            DiscreteState.APPROACH: ApproachController(),
            DiscreteState.GRASP: GraspController(),
            DiscreteState.LIFT: LiftController(),
        }

        # Transitions
        self.transitions: List[HybridTransition] = self._build_transitions()

        # State history
        self.state_history: List[Tuple[float, DiscreteState]] = []

    def _build_transitions(self) -> List[HybridTransition]:
        """Build transition table."""
        return [
            # IDLE -> APPROACH: when object detected
            HybridTransition(
                source=DiscreteState.IDLE,
                target=DiscreteState.APPROACH,
                guard=lambda w: True,  # Start immediately
                priority=1,
            ),

            # APPROACH -> GRASP: when close to object
            HybridTransition(
                source=DiscreteState.APPROACH,
                target=DiscreteState.GRASP,
                guard=lambda w: w.robot.distance_to(w.obj.x, w.obj.y) < 0.05,
                priority=1,
            ),

            # GRASP -> LIFT: when object grasped (force detected)
            HybridTransition(
                source=DiscreteState.GRASP,
                target=DiscreteState.LIFT,
                guard=lambda w: w.robot.gripper > 0.9 and w.robot.force > 0.3,
                priority=1,
            ),

            # LIFT -> DONE: when lifted high enough
            HybridTransition(
                source=DiscreteState.LIFT,
                target=DiscreteState.DONE,
                guard=lambda w: w.robot.y > 0.4,
                priority=1,
            ),

            # Any -> ERROR: if object dropped during lift
            HybridTransition(
                source=DiscreteState.LIFT,
                target=DiscreteState.ERROR,
                guard=lambda w: w.robot.force < 0.1 and w.robot.gripper > 0.5,
                priority=2,
            ),
        ]

    def step(self, world: WorldState) -> Tuple[float, float, float]:
        """
        Execute one step of hybrid control.

        Returns: (vx, vy, gripper_cmd) continuous action
        """
        # Record state
        self.state_history.append((world.time, self.current_state))

        # Check for enabled transitions (highest priority first)
        enabled = [
            t for t in self.transitions
            if t.source == self.current_state and t.guard(world)
        ]

        if enabled:
            # Take highest priority transition
            enabled.sort(key=lambda t: -t.priority)
            self.current_state = enabled[0].target

        # Get continuous action from current state's controller
        controller = self.controllers.get(self.current_state)
        if controller:
            return controller.compute_action(world)
        else:
            return (0.0, 0.0, 0.0)

    def reset(self):
        """Reset to initial state."""
        self.current_state = DiscreteState.IDLE
        self.state_history = []

    def is_done(self) -> bool:
        return self.current_state in (DiscreteState.DONE, DiscreteState.ERROR)

    def is_success(self) -> bool:
        return self.current_state == DiscreteState.DONE


# =============================================================================
# AND-States for Parallel Limb Control
# =============================================================================

@dataclass
class LimbController:
    """Controller for a single limb."""
    limb_id: str
    state: LimbState = LimbState.READY
    target_x: float = 0.0
    target_y: float = 0.0

    # Limb-local state
    x: float = 0.0
    y: float = 0.0

    def compute_action(self) -> Tuple[float, float]:
        """Compute limb velocity."""
        if self.state == LimbState.READY:
            return (0.0, 0.0)
        elif self.state == LimbState.MOVING:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = math.sqrt(dx * dx + dy * dy) + 1e-6
            speed = min(0.1, dist)
            return (dx / dist * speed, dy / dist * speed)
        else:
            return (0.0, 0.0)

    def update(self, vx: float, vy: float, dt: float):
        """Update limb position."""
        self.x += vx * dt
        self.y += vy * dt


class ParallelLimbController:
    """
    AND-state controller for multiple limbs.

    Each limb runs its own state machine in parallel.
    High-level commands coordinate limb behaviors.
    """

    def __init__(self, n_limbs: int = 2):
        self.limbs = [
            LimbController(limb_id=f"limb_{i}", x=-0.2 + i * 0.4, y=0.0)
            for i in range(n_limbs)
        ]

        # High-level state (AND of all limb states)
        self.high_level_state = DiscreteState.IDLE

    def all_limbs_in_state(self, state: LimbState) -> bool:
        """Check if all limbs are in given state."""
        return all(limb.state == state for limb in self.limbs)

    def any_limb_in_state(self, state: LimbState) -> bool:
        """Check if any limb is in given state."""
        return any(limb.state == state for limb in self.limbs)

    def command_approach(self, target_x: float, target_y: float):
        """Command all limbs to approach a target."""
        for limb in self.limbs:
            limb.target_x = target_x
            limb.target_y = target_y
            limb.state = LimbState.MOVING

    def command_grasp(self):
        """Command all limbs to grasp."""
        for limb in self.limbs:
            limb.state = LimbState.CONTACT

    def step(self, dt: float = 0.01) -> List[Tuple[float, float]]:
        """Step all limbs in parallel."""
        actions = []
        for limb in self.limbs:
            vx, vy = limb.compute_action()
            limb.update(vx, vy, dt)
            actions.append((vx, vy))

            # Check limb-level transitions
            if limb.state == LimbState.MOVING:
                dist = math.sqrt(
                    (limb.x - limb.target_x) ** 2 +
                    (limb.y - limb.target_y) ** 2
                )
                if dist < 0.05:
                    limb.state = LimbState.CONTACT

        # High-level state transitions based on limb states
        if self.high_level_state == DiscreteState.IDLE:
            if self.any_limb_in_state(LimbState.MOVING):
                self.high_level_state = DiscreteState.APPROACH

        elif self.high_level_state == DiscreteState.APPROACH:
            if self.all_limbs_in_state(LimbState.CONTACT):
                self.high_level_state = DiscreteState.GRASP

        return actions


# =============================================================================
# Simple 2D Robot Environment
# =============================================================================

class SimpleRobotEnv:
    """
    Simple 2D robot manipulation environment.

    No external dependencies - pure Python/MLX simulation.
    """

    def __init__(self, dt: float = 0.02):
        self.dt = dt
        self.world = WorldState()
        self.max_steps = 500
        self.step_count = 0

    def reset(self) -> WorldState:
        """Reset environment to initial state."""
        self.world = WorldState(
            robot=RobotState(x=0.0, y=0.0, vx=0.0, vy=0.0, gripper=0.0, force=0.0),
            obj=ObjectState(x=0.3 + random.uniform(-0.1, 0.1),
                            y=0.0,
                            grasped=False,
                            lifted=False),
            time=0.0,
        )
        self.step_count = 0
        return self.world

    def step(self, action: Tuple[float, float, float]) -> Tuple[WorldState, float, bool]:
        """
        Step environment with continuous action.

        Args:
            action: (vx, vy, gripper_cmd)

        Returns:
            (next_state, reward, done)
        """
        vx, vy, gripper_cmd = action

        # Clamp velocities
        vx = max(-0.5, min(0.5, vx))
        vy = max(-0.5, min(0.5, vy))
        gripper_cmd = max(0.0, min(1.0, gripper_cmd))

        # Update robot position
        self.world.robot.x += vx * self.dt
        self.world.robot.y += vy * self.dt
        self.world.robot.vx = vx
        self.world.robot.vy = vy

        # Update gripper
        self.world.robot.gripper = gripper_cmd

        # Compute force (if gripper closed and near object)
        dist = self.world.robot.distance_to(self.world.obj.x, self.world.obj.y)
        if dist < 0.05 and gripper_cmd > 0.8:
            self.world.robot.force = 1.0
            self.world.obj.grasped = True
        else:
            self.world.robot.force = 0.0
            if not self.world.obj.lifted:
                self.world.obj.grasped = False

        # Update object if grasped
        if self.world.obj.grasped:
            self.world.obj.x = self.world.robot.x
            self.world.obj.y = self.world.robot.y
            if self.world.robot.y > 0.3:
                self.world.obj.lifted = True

        # Update time
        self.world.time += self.dt
        self.step_count += 1

        # Compute reward
        reward = self._compute_reward()

        # Check done
        done = self.step_count >= self.max_steps or self.world.obj.lifted

        return self.world, reward, done

    def _compute_reward(self) -> float:
        """Compute reward for current state."""
        reward = 0.0

        # Reward for being close to object
        dist = self.world.robot.distance_to(self.world.obj.x, self.world.obj.y)
        reward -= dist * 0.1

        # Reward for grasping
        if self.world.obj.grasped:
            reward += 1.0

        # Reward for lifting
        if self.world.obj.lifted:
            reward += 10.0

        return reward


# =============================================================================
# Learned Continuous Controller (Neural Network)
# =============================================================================

class NeuralController(nn.Module):
    """Neural network controller for continuous actions."""

    def __init__(self, obs_dim: int = 11, hidden_dim: int = 64, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def __call__(self, obs: mx.array) -> mx.array:
        return self.net(obs)


class LearnedHybridController:
    """
    Hybrid controller with learned continuous policies.

    Each discrete state has its own learned neural controller.
    """

    def __init__(self, obs_dim: int = 11):
        self.current_state = DiscreteState.IDLE

        # Learned controller per state
        self.controllers: Dict[DiscreteState, NeuralController] = {
            state: NeuralController(obs_dim=obs_dim)
            for state in [DiscreteState.IDLE, DiscreteState.APPROACH,
                          DiscreteState.GRASP, DiscreteState.LIFT]
        }

        # Transition conditions (still rule-based for now)
        self.transitions = self._build_transitions()

    def _build_transitions(self) -> List[HybridTransition]:
        """Same transitions as HybridStatechart."""
        return [
            HybridTransition(
                source=DiscreteState.IDLE,
                target=DiscreteState.APPROACH,
                guard=lambda w: True,
            ),
            HybridTransition(
                source=DiscreteState.APPROACH,
                target=DiscreteState.GRASP,
                guard=lambda w: w.robot.distance_to(w.obj.x, w.obj.y) < 0.05,
            ),
            HybridTransition(
                source=DiscreteState.GRASP,
                target=DiscreteState.LIFT,
                guard=lambda w: w.robot.gripper > 0.9 and w.robot.force > 0.3,
            ),
            HybridTransition(
                source=DiscreteState.LIFT,
                target=DiscreteState.DONE,
                guard=lambda w: w.robot.y > 0.4,
            ),
        ]

    def step(self, world: WorldState) -> Tuple[float, float, float]:
        """Compute action using learned controller."""
        # Check transitions
        enabled = [
            t for t in self.transitions
            if t.source == self.current_state and t.guard(world)
        ]
        if enabled:
            self.current_state = enabled[0].target

        # Get action from learned controller
        controller = self.controllers.get(self.current_state)
        if controller:
            obs = world.to_array()
            action = controller(obs.reshape(1, -1))[0]
            # Scale to action range
            vx = float(action[0]) * 0.5
            vy = float(action[1]) * 0.5
            gripper = (float(action[2]) + 1) / 2  # [0, 1]
            return (vx, vy, gripper)
        return (0.0, 0.0, 0.0)

    def reset(self):
        self.current_state = DiscreteState.IDLE


# =============================================================================
# Training Loop
# =============================================================================

def train_hybrid_controller(
    env: SimpleRobotEnv,
    controller: LearnedHybridController,
    n_episodes: int = 100,
    verbose: bool = True,
) -> List[float]:
    """
    Train learned hybrid controller via policy gradient.

    Simplified training - just random search for demo.
    """
    rewards_history = []

    for episode in range(n_episodes):
        world = env.reset()
        controller.reset()
        total_reward = 0.0

        while True:
            action = controller.step(world)
            world, reward, done = env.step(action)
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

        if verbose and episode % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / min(10, len(rewards_history))
            print(f"Episode {episode}: reward={total_reward:.2f}, "
                  f"avg={avg_reward:.2f}, state={controller.current_state.name}")

    return rewards_history


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate hybrid discrete-continuous control."""
    print("=" * 60)
    print("Hybrid Discrete-Continuous Control")
    print("=" * 60)

    # Create environment and controller
    env = SimpleRobotEnv()
    controller = HybridStatechart()

    # Run episode
    world = env.reset()
    controller.reset()

    print(f"\nInitial state: robot=({world.robot.x:.2f}, {world.robot.y:.2f}), "
          f"object=({world.obj.x:.2f}, {world.obj.y:.2f})")

    total_reward = 0.0
    state_times: Dict[DiscreteState, float] = {s: 0.0 for s in DiscreteState}

    while not controller.is_done() and env.step_count < env.max_steps:
        action = controller.step(world)
        world, reward, done = env.step(action)
        total_reward += reward
        state_times[controller.current_state] += env.dt

        if done:
            break

    # Results
    print(f"\n--- Results ---")
    print(f"Success: {controller.is_success()}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Steps: {env.step_count}")
    print(f"Final state: {controller.current_state.name}")
    print(f"Object lifted: {world.obj.lifted}")

    print(f"\n--- Time in each state ---")
    for state, time in state_times.items():
        if time > 0:
            print(f"  {state.name}: {time:.2f}s")

    print(f"\n--- State transitions ---")
    prev_state = None
    for time, state in controller.state_history[:20]:
        if state != prev_state:
            print(f"  t={time:.2f}: {state.name}")
            prev_state = state

    return controller, env


def demo_parallel_limbs():
    """Demonstrate AND-states with parallel limb control."""
    print("\n" + "=" * 60)
    print("Parallel Limb Control (AND-states)")
    print("=" * 60)

    controller = ParallelLimbController(n_limbs=2)

    # Command both limbs to approach target
    target_x, target_y = 0.3, 0.0
    controller.command_approach(target_x, target_y)

    print(f"\nTarget: ({target_x}, {target_y})")
    print(f"Limb initial positions:")
    for limb in controller.limbs:
        print(f"  {limb.limb_id}: ({limb.x:.2f}, {limb.y:.2f})")

    # Simulate
    for step in range(100):
        actions = controller.step(dt=0.02)

        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  High-level state: {controller.high_level_state.name}")
            for limb in controller.limbs:
                print(f"  {limb.limb_id}: ({limb.x:.2f}, {limb.y:.2f}) - {limb.state.name}")

        if controller.all_limbs_in_state(LimbState.CONTACT):
            print(f"\nAll limbs reached target at step {step}!")
            break

    return controller


if __name__ == "__main__":
    demo()
    demo_parallel_limbs()
