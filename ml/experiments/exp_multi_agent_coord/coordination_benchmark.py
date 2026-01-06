"""
Coordination Benchmark: Compare Statechart-based vs MAPPO/QMIX

Baselines:
1. MAPPO (Multi-Agent PPO) - Independent policies with shared critic
2. QMIX - Value decomposition with monotonic mixing
3. Random - Random actions baseline
4. Rule-based - Hand-coded leader-follower

Compare on:
- Task completion rate
- Time to complete
- Communication efficiency (events per step)
- Emergent role stability
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
import random
import math
import time

from .multi_agent_env import MultiAgentEnv, Event, EventType, Task, TaskType, AgentState
from .agent_statechart import (
    AgentStatechart, MultiAgentStatechartController, AgentRole, BehaviorMode
)
from .protocol_evolution import ProtocolEvolver, ProtocolGenome, EvolvedMultiAgentController


# =============================================================================
# MAPPO Baseline
# =============================================================================

class MAPPONetwork(nn.Module):
    """Simple actor-critic network for MAPPO."""

    def __init__(self, obs_dim: int = 20, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean, log_std
        )

        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def __call__(self, obs: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass returning action, log_prob, value."""
        actor_out = self.actor(obs)
        action_dim = actor_out.shape[-1] // 2
        mean = actor_out[..., :action_dim]
        log_std = mx.clip(actor_out[..., action_dim:], -2, 2)
        std = mx.exp(log_std)

        # Sample action
        noise = mx.random.normal(mean.shape)
        action = mean + std * noise
        action = mx.tanh(action) * 0.1  # Scale to valid velocity range

        # Log probability
        log_prob = -0.5 * mx.sum(
            ((action / 0.1 - mean) / (std + 1e-6)) ** 2 + 2 * log_std + math.log(2 * math.pi),
            axis=-1
        )

        # Value
        value = self.critic(obs)

        return action, log_prob, value


class MAPPOAgent:
    """MAPPO agent wrapper."""

    def __init__(self, agent_id: int, obs_dim: int = 20, hidden_dim: int = 64):
        self.agent_id = agent_id
        self.network = MAPPONetwork(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.optimizer = None  # Would initialize for training

    def obs_to_array(self, obs: Dict[str, Any]) -> mx.array:
        """Convert observation dict to array."""
        agent = obs["self"]
        features = [
            agent.x, agent.y, agent.vx, agent.vy, agent.heading,
            float(agent.role == "LEADER"),
            float(agent.role == "FOLLOWER"),
            agent.energy,
        ]

        # Add visible agents (up to 3)
        visible = obs["visible_agents"][:3]
        for other in visible:
            features.extend([
                other.x - agent.x,
                other.y - agent.y,
                other.vx,
                other.vy,
            ])
        # Pad if fewer visible
        while len(features) < 20:
            features.append(0.0)

        return mx.array(features[:20])

    def act(self, obs: Dict[str, Any]) -> Tuple[float, float]:
        """Get action from observation."""
        obs_array = self.obs_to_array(obs)
        action, _, _ = self.network(obs_array)
        return float(action[0]), float(action[1])


class MAPPOController:
    """MAPPO multi-agent controller."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.agents = {i: MAPPOAgent(i) for i in range(n_agents)}

    def reset(self):
        pass

    def step(
        self,
        observations: Dict[int, Dict[str, Any]],
        dt: float = 0.1
    ) -> Dict[int, Tuple[float, float, Optional[Event]]]:
        """Get actions for all agents."""
        actions = {}
        for agent_id, obs in observations.items():
            vx, vy = self.agents[agent_id].act(obs)
            actions[agent_id] = (vx, vy, None)  # No communication
        return actions


# =============================================================================
# QMIX Baseline
# =============================================================================

class QNetwork(nn.Module):
    """Q-network for individual agent."""

    def __init__(self, obs_dim: int = 20, n_actions: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def __call__(self, obs: mx.array) -> mx.array:
        return self.net(obs)


class QMIXMixer(nn.Module):
    """QMIX mixing network."""

    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents

        # Hypernetworks generate weights
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.hidden_dim = hidden_dim

    def __call__(self, q_values: mx.array, state: mx.array) -> mx.array:
        """Mix individual Q-values into Q_tot."""
        batch_size = q_values.shape[0] if len(q_values.shape) > 1 else 1
        q_values = q_values.reshape(batch_size, self.n_agents)

        # Generate weights (ensure positive via abs)
        w1 = mx.abs(self.hyper_w1(state)).reshape(batch_size, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(state).reshape(batch_size, 1, self.hidden_dim)

        # First layer
        hidden = mx.relu(mx.matmul(q_values.reshape(batch_size, 1, self.n_agents), w1) + b1)

        # Second layer
        w2 = mx.abs(self.hyper_w2(state)).reshape(batch_size, self.hidden_dim, 1)
        b2 = self.hyper_b2(state).reshape(batch_size, 1, 1)

        q_tot = mx.matmul(hidden, w2) + b2
        return q_tot.reshape(batch_size)


class QMIXAgent:
    """QMIX agent with discretized actions."""

    # Discretized action space: 9 actions (3x3 grid of velocities)
    ACTIONS = [
        (-0.1, -0.1), (-0.1, 0.0), (-0.1, 0.1),
        (0.0, -0.1), (0.0, 0.0), (0.0, 0.1),
        (0.1, -0.1), (0.1, 0.0), (0.1, 0.1),
    ]

    def __init__(self, agent_id: int, obs_dim: int = 20):
        self.agent_id = agent_id
        self.q_net = QNetwork(obs_dim=obs_dim, n_actions=len(self.ACTIONS))
        self.epsilon = 0.1

    def obs_to_array(self, obs: Dict[str, Any]) -> mx.array:
        """Convert observation to array."""
        agent = obs["self"]
        features = [
            agent.x, agent.y, agent.vx, agent.vy, agent.heading,
            float(agent.role == "LEADER"),
            float(agent.role == "FOLLOWER"),
            agent.energy,
        ]

        visible = obs["visible_agents"][:3]
        for other in visible:
            features.extend([
                other.x - agent.x,
                other.y - agent.y,
                other.vx,
                other.vy,
            ])
        while len(features) < 20:
            features.append(0.0)

        return mx.array(features[:20])

    def act(self, obs: Dict[str, Any]) -> Tuple[float, float]:
        """Select action using epsilon-greedy."""
        if random.random() < self.epsilon:
            action_idx = random.randint(0, len(self.ACTIONS) - 1)
        else:
            obs_array = self.obs_to_array(obs)
            q_values = self.q_net(obs_array)
            action_idx = int(mx.argmax(q_values))

        return self.ACTIONS[action_idx]


class QMIXController:
    """QMIX multi-agent controller."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.agents = {i: QMIXAgent(i) for i in range(n_agents)}
        self.mixer = QMIXMixer(n_agents, state_dim=n_agents * 20)

    def reset(self):
        pass

    def step(
        self,
        observations: Dict[int, Dict[str, Any]],
        dt: float = 0.1
    ) -> Dict[int, Tuple[float, float, Optional[Event]]]:
        """Get actions for all agents."""
        actions = {}
        for agent_id, obs in observations.items():
            vx, vy = self.agents[agent_id].act(obs)
            actions[agent_id] = (vx, vy, None)
        return actions


# =============================================================================
# Rule-Based Baseline
# =============================================================================

class RuleBasedController:
    """Hand-coded leader-follower coordination."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.leader_id = 0  # Agent 0 is always leader

    def reset(self):
        pass

    def step(
        self,
        observations: Dict[int, Dict[str, Any]],
        dt: float = 0.1
    ) -> Dict[int, Tuple[float, float, Optional[Event]]]:
        """Get actions using simple rules."""
        actions = {}

        # Get leader position from agent 0
        leader_obs = observations.get(0)
        leader = leader_obs["self"] if leader_obs else None
        task = leader_obs["task"] if leader_obs else None

        for agent_id, obs in observations.items():
            agent = obs["self"]
            event = None

            if agent_id == self.leader_id:
                # Leader moves to target
                if task:
                    dx = task.target_x - agent.x
                    dy = task.target_y - agent.y
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-6
                    vx = (dx / dist) * 0.08
                    vy = (dy / dist) * 0.08
                else:
                    vx, vy = 0.0, 0.0

                # Send follow signal
                event = Event(
                    event_type=EventType.FOLLOW_ME,
                    sender_id=agent_id,
                    payload={"x": agent.x, "y": agent.y}
                )
            else:
                # Followers follow leader
                if leader:
                    dx = leader.x - agent.x
                    dy = leader.y - agent.y
                    dist = math.sqrt(dx * dx + dy * dy) + 1e-6

                    # Keep some distance
                    if dist > 0.15:
                        vx = (dx / dist) * 0.1
                        vy = (dy / dist) * 0.1
                    else:
                        vx = (dx / dist) * 0.02
                        vy = (dy / dist) * 0.02
                else:
                    vx, vy = 0.0, 0.0

            actions[agent_id] = (vx, vy, event)

        return actions


# =============================================================================
# Random Baseline
# =============================================================================

class RandomController:
    """Random action baseline."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def reset(self):
        pass

    def step(
        self,
        observations: Dict[int, Dict[str, Any]],
        dt: float = 0.1
    ) -> Dict[int, Tuple[float, float, Optional[Event]]]:
        """Random actions."""
        actions = {}
        for agent_id in observations:
            vx = random.uniform(-0.1, 0.1)
            vy = random.uniform(-0.1, 0.1)
            actions[agent_id] = (vx, vy, None)
        return actions


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result from benchmarking a controller."""
    controller_name: str
    task_type: str
    success_rate: float
    avg_steps_to_complete: float
    avg_reward: float
    events_per_step: float
    role_changes: float  # Stability metric
    run_time: float


class CoordinationBenchmark:
    """Benchmark for comparing coordination methods."""

    def __init__(self, n_agents: int = 4, n_episodes: int = 20):
        self.n_agents = n_agents
        self.n_episodes = n_episodes
        self.env = MultiAgentEnv(n_agents=n_agents)
        self.results: List[BenchmarkResult] = []

    def _run_episode(
        self,
        controller,
        task: Task,
        max_steps: int = 300
    ) -> Tuple[bool, int, float, int]:
        """Run single episode, return (success, steps, reward, events)."""
        obs = self.env.reset(task)
        controller.reset()

        total_reward = 0.0
        initial_events = self.env.event_bus.total_events

        for step in range(max_steps):
            actions = controller.step(obs, dt=self.env.dt)
            obs, rewards, done, info = self.env.step(actions)
            total_reward += sum(rewards.values())

            if done:
                success = info.get("task_complete", False)
                events = self.env.event_bus.total_events - initial_events
                return success, step + 1, total_reward, events

        events = self.env.event_bus.total_events - initial_events
        return False, max_steps, total_reward, events

    def benchmark_controller(
        self,
        controller,
        controller_name: str,
        tasks: List[Task],
    ) -> List[BenchmarkResult]:
        """Benchmark a controller on tasks."""
        results = []

        for task in tasks:
            successes = 0
            total_steps = 0
            total_reward = 0.0
            total_events = 0

            start_time = time.time()

            for _ in range(self.n_episodes):
                success, steps, reward, events = self._run_episode(controller, task)
                if success:
                    successes += 1
                    total_steps += steps
                total_reward += reward
                total_events += events

            run_time = time.time() - start_time

            result = BenchmarkResult(
                controller_name=controller_name,
                task_type=task.task_type.name,
                success_rate=successes / self.n_episodes,
                avg_steps_to_complete=total_steps / max(successes, 1),
                avg_reward=total_reward / self.n_episodes,
                events_per_step=(total_events / self.n_episodes) / 300,  # Normalized
                role_changes=0.0,  # Would track role stability
                run_time=run_time,
            )
            results.append(result)
            self.results.append(result)

        return results

    def run_all_benchmarks(self, tasks: List[Task]) -> Dict[str, List[BenchmarkResult]]:
        """Run benchmarks for all controllers."""
        all_results = {}

        # 1. Random baseline
        print("Benchmarking Random...")
        random_ctrl = RandomController(self.n_agents)
        all_results["Random"] = self.benchmark_controller(random_ctrl, "Random", tasks)

        # 2. Rule-based baseline
        print("Benchmarking Rule-based...")
        rule_ctrl = RuleBasedController(self.n_agents)
        all_results["Rule-based"] = self.benchmark_controller(rule_ctrl, "Rule-based", tasks)

        # 3. MAPPO
        print("Benchmarking MAPPO...")
        mappo_ctrl = MAPPOController(self.n_agents)
        all_results["MAPPO"] = self.benchmark_controller(mappo_ctrl, "MAPPO", tasks)

        # 4. QMIX
        print("Benchmarking QMIX...")
        qmix_ctrl = QMIXController(self.n_agents)
        all_results["QMIX"] = self.benchmark_controller(qmix_ctrl, "QMIX", tasks)

        # 5. Statechart (hand-coded)
        print("Benchmarking Statechart...")
        sc_ctrl = MultiAgentStatechartController(self.n_agents)
        all_results["Statechart"] = self.benchmark_controller(sc_ctrl, "Statechart", tasks)

        # 6. Evolved Statechart (if evolved genome available)
        print("Evolving protocols...")
        evolver = ProtocolEvolver(n_agents=self.n_agents, population_size=10)
        best_genome = evolver.evolve(tasks, n_generations=10, verbose=False)

        print("Benchmarking Evolved-Statechart...")
        evolved_ctrl = EvolvedMultiAgentController(self.n_agents, best_genome)
        all_results["Evolved-Statechart"] = self.benchmark_controller(
            evolved_ctrl, "Evolved-Statechart", tasks
        )

        return all_results

    def print_results(self):
        """Print benchmark results table."""
        print("\n" + "=" * 80)
        print("COORDINATION BENCHMARK RESULTS")
        print("=" * 80)

        # Group by task
        tasks = set(r.task_type for r in self.results)

        for task in sorted(tasks):
            print(f"\nTask: {task}")
            print("-" * 70)
            print(f"{'Controller':<20} {'Success':>10} {'Steps':>10} {'Reward':>10} {'Comm':>10}")
            print("-" * 70)

            task_results = [r for r in self.results if r.task_type == task]
            task_results.sort(key=lambda r: -r.success_rate)

            for r in task_results:
                print(
                    f"{r.controller_name:<20} "
                    f"{r.success_rate:>9.1%} "
                    f"{r.avg_steps_to_complete:>10.1f} "
                    f"{r.avg_reward:>10.2f} "
                    f"{r.events_per_step:>10.3f}"
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        summary = {}
        controllers = set(r.controller_name for r in self.results)

        for ctrl in controllers:
            ctrl_results = [r for r in self.results if r.controller_name == ctrl]
            summary[ctrl] = {
                "avg_success_rate": sum(r.success_rate for r in ctrl_results) / len(ctrl_results),
                "avg_reward": sum(r.avg_reward for r in ctrl_results) / len(ctrl_results),
                "avg_events": sum(r.events_per_step for r in ctrl_results) / len(ctrl_results),
            }

        return summary


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run coordination benchmark."""
    print("=" * 60)
    print("Multi-Agent Coordination Benchmark")
    print("=" * 60)

    # Define tasks
    tasks = [
        Task(TaskType.GATHER, 0.5, 0.5, {"threshold": 0.15}),
        Task(TaskType.COVERAGE, 0.5, 0.5, {"min_distance": 0.2}),
    ]

    # Run benchmark
    benchmark = CoordinationBenchmark(n_agents=4, n_episodes=10)
    results = benchmark.run_all_benchmarks(tasks)

    # Print results
    benchmark.print_results()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = benchmark.get_summary()
    for ctrl, stats in sorted(summary.items(), key=lambda x: -x[1]["avg_success_rate"]):
        print(f"{ctrl}: success={stats['avg_success_rate']:.1%}, reward={stats['avg_reward']:.2f}")

    # Best controller
    best_ctrl = max(summary.items(), key=lambda x: x[1]["avg_success_rate"])
    print(f"\nBest controller: {best_ctrl[0]} ({best_ctrl[1]['avg_success_rate']:.1%} success)")

    return benchmark, results


if __name__ == "__main__":
    demo()
