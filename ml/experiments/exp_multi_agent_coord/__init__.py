"""
Experiment: Multi-Agent Coordination with Emergent Communication

Extends statecharts to 4+ agents with:
- Shared event bus for communication
- Per-agent statecharts with evolved roles (LEADER/FOLLOWER)
- Protocol evolution: signal meanings, timing, transitions
- Coordination tasks: GATHER, COVERAGE, FORMATION

Key insight: Communication protocols emerge through co-evolution.
Agents in a team share the same protocol genome, creating
consistent conventions for role assignment and signaling.

Components:
- MultiAgentEnv: Environment with shared event bus
- AgentStatechart: Per-agent statechart with role states
- ProtocolEvolver: Evolve communication protocols
- CoordinationBenchmark: Compare to MAPPO/QMIX baselines

Builds on: exp_hide_seek_evolution (2-agent), exp_transition_priorities
"""

from .multi_agent_env import (
    EventType,
    Event,
    EventBus,
    AgentState,
    TaskType,
    Task,
    MultiAgentEnv,
)

from .agent_statechart import (
    AgentRole,
    BehaviorMode,
    AgentTransition,
    AgentStatechart,
    MultiAgentStatechartController,
)

from .protocol_evolution import (
    SignalMeaning,
    RoleCondition,
    ProtocolGenome,
    EvolvedAgentStatechart,
    EvolvedMultiAgentController,
    ProtocolEvolver,
)

from .coordination_benchmark import (
    MAPPOController,
    QMIXController,
    RuleBasedController,
    RandomController,
    BenchmarkResult,
    CoordinationBenchmark,
)

__all__ = [
    # Environment
    "EventType",
    "Event",
    "EventBus",
    "AgentState",
    "TaskType",
    "Task",
    "MultiAgentEnv",
    # Statecharts
    "AgentRole",
    "BehaviorMode",
    "AgentTransition",
    "AgentStatechart",
    "MultiAgentStatechartController",
    # Evolution
    "SignalMeaning",
    "RoleCondition",
    "ProtocolGenome",
    "EvolvedAgentStatechart",
    "EvolvedMultiAgentController",
    "ProtocolEvolver",
    # Benchmarks
    "MAPPOController",
    "QMIXController",
    "RuleBasedController",
    "RandomController",
    "BenchmarkResult",
    "CoordinationBenchmark",
]
