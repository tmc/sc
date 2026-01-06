"""
Experiment: Hybrid Discrete-Continuous Control

Combines statecharts with continuous control for robotics:
- Discrete states: IDLE -> APPROACH -> GRASP -> LIFT
- Continuous actions: velocity, force, gripper within states
- AND-states: Parallel control of multiple limbs
- Guards: Continuous conditions trigger transitions

Key insight: Statecharts provide high-level behavior structure,
continuous controllers handle low-level execution within states.

No external dependencies (no MuJoCo). Simple 2D simulated environment.

Components:
- HybridStatechart: Discrete states with continuous controllers
- SimpleRobotEnv: 2D manipulation environment
- ParallelLimbController: AND-states for multi-limb coordination
- LearnedHybridController: Neural controllers per discrete state
"""

from .hybrid_controller import (
    # States
    DiscreteState,
    LimbState,
    RobotState,
    ObjectState,
    WorldState,

    # Controllers
    ContinuousController,
    IdleController,
    ApproachController,
    GraspController,
    LiftController,

    # Hybrid system
    HybridStatechart,
    HybridTransition,
    ParallelLimbController,
    LimbController,

    # Environment
    SimpleRobotEnv,

    # Learned controllers
    NeuralController,
    LearnedHybridController,
)

__all__ = [
    "DiscreteState",
    "LimbState",
    "RobotState",
    "ObjectState",
    "WorldState",
    "ContinuousController",
    "IdleController",
    "ApproachController",
    "GraspController",
    "LiftController",
    "HybridStatechart",
    "HybridTransition",
    "ParallelLimbController",
    "LimbController",
    "SimpleRobotEnv",
    "NeuralController",
    "LearnedHybridController",
]
