from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Goal:
    pass

@dataclass
class TargetState(Goal):
    state_label: str

@dataclass
class SequenceGoal(Goal):
    states: List[str]

@dataclass
class AvoidanceGoal(Goal):
    target: str
    avoid: List[str]

def parse_goal(goal_spec: str) -> Goal:
    # Simplified parser
    if "THEN" in goal_spec:
        parts = [s.strip() for s in goal_spec.split("THEN")]
        return SequenceGoal(parts)
    elif "AVOID" in goal_spec:
        parts = goal_spec.split("AVOID")
        return AvoidanceGoal(target=parts[0].strip(), avoid=[p.strip() for p in parts[1].split(",")])
    else:
        return TargetState(goal_spec.strip())
