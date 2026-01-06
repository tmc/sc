"""
exp_stateful_behavior_induction: Induce stateful SC from I/O sequences.

Core insight: Output depends on HISTORY, not just current input.

Given: Sequence of (input, state_before) → state_after observations
Output: Induced rules that can predict novel sequences

Test Machines:
1. Stack: PUSH/POP (LIFO)
2. Queue: ENQUEUE/DEQUEUE (FIFO)
3. Counter: INC/DEC/RESET
4. Toggle with memory: remembers last N states
5. Accumulator: running sum/product
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable, Optional
from abc import ABC, abstractmethod
import random


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass
class IOStep:
    """Single I/O observation."""
    input_event: str
    input_arg: Any  # Optional argument (e.g., value for PUSH)
    state_before: Any
    state_after: Any


@dataclass
class InducedRule:
    """Induced transition rule."""
    event: str
    condition: str  # Human-readable condition
    action: str     # Human-readable action
    examples: List[IOStep] = field(default_factory=list)


@dataclass
class InducedMachine:
    """Induced state machine from observations."""
    state_type: str  # "list", "int", "tuple", etc.
    events: List[str]
    rules: List[InducedRule]
    initial_state: Any


# =============================================================================
# Abstract Machine Interface
# =============================================================================

class StatefulMachine(ABC):
    """Abstract base for stateful machines."""

    @abstractmethod
    def reset(self) -> Any:
        """Reset and return initial state."""
        pass

    @abstractmethod
    def step(self, event: str, arg: Any = None) -> Any:
        """Process event, return new state."""
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """Get current state."""
        pass

    @property
    @abstractmethod
    def events(self) -> List[str]:
        """Available events."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Machine name."""
        pass


# =============================================================================
# Test Machines
# =============================================================================

class StackMachine(StatefulMachine):
    """LIFO stack: PUSH/POP."""

    def __init__(self):
        self.stack = []

    def reset(self) -> List:
        self.stack = []
        return self.stack.copy()

    def step(self, event: str, arg: Any = None) -> List:
        if event == "PUSH" and arg is not None:
            self.stack.append(arg)
        elif event == "POP" and self.stack:
            self.stack.pop()
        return self.stack.copy()

    def get_state(self) -> List:
        return self.stack.copy()

    @property
    def events(self) -> List[str]:
        return ["PUSH", "POP"]

    @property
    def name(self) -> str:
        return "Stack"


class QueueMachine(StatefulMachine):
    """FIFO queue: ENQUEUE/DEQUEUE."""

    def __init__(self):
        self.queue = []

    def reset(self) -> List:
        self.queue = []
        return self.queue.copy()

    def step(self, event: str, arg: Any = None) -> List:
        if event == "ENQUEUE" and arg is not None:
            self.queue.append(arg)
        elif event == "DEQUEUE" and self.queue:
            self.queue.pop(0)  # Remove from front
        return self.queue.copy()

    def get_state(self) -> List:
        return self.queue.copy()

    @property
    def events(self) -> List[str]:
        return ["ENQUEUE", "DEQUEUE"]

    @property
    def name(self) -> str:
        return "Queue"


class CounterMachine(StatefulMachine):
    """Counter: INC/DEC/RESET."""

    def __init__(self, max_val: int = 10):
        self.count = 0
        self.max_val = max_val

    def reset(self) -> int:
        self.count = 0
        return self.count

    def step(self, event: str, arg: Any = None) -> int:
        if event == "INC":
            self.count = min(self.count + 1, self.max_val)
        elif event == "DEC":
            self.count = max(self.count - 1, 0)
        elif event == "RESET":
            self.count = 0
        return self.count

    def get_state(self) -> int:
        return self.count

    @property
    def events(self) -> List[str]:
        return ["INC", "DEC", "RESET"]

    @property
    def name(self) -> str:
        return "Counter"


class AccumulatorMachine(StatefulMachine):
    """Accumulator: ADD/MULT/CLEAR."""

    def __init__(self):
        self.value = 0

    def reset(self) -> int:
        self.value = 0
        return self.value

    def step(self, event: str, arg: Any = None) -> int:
        if event == "ADD" and arg is not None:
            self.value += arg
        elif event == "MULT" and arg is not None:
            self.value *= arg
        elif event == "CLEAR":
            self.value = 0
        return self.value

    def get_state(self) -> int:
        return self.value

    @property
    def events(self) -> List[str]:
        return ["ADD", "MULT", "CLEAR"]

    @property
    def name(self) -> str:
        return "Accumulator"


class ToggleMemoryMachine(StatefulMachine):
    """Toggle with last N states memory."""

    def __init__(self, memory_size: int = 3):
        self.current = "OFF"
        self.history = []
        self.memory_size = memory_size

    def reset(self) -> Tuple[str, List]:
        self.current = "OFF"
        self.history = []
        return (self.current, self.history.copy())

    def step(self, event: str, arg: Any = None) -> Tuple[str, List]:
        if event == "TOGGLE":
            # Remember current before changing
            self.history.append(self.current)
            if len(self.history) > self.memory_size:
                self.history.pop(0)
            # Toggle
            self.current = "ON" if self.current == "OFF" else "OFF"
        elif event == "RECALL":
            # Go back to most recent history if available
            if self.history:
                self.current = self.history[-1]
        return (self.current, self.history.copy())

    def get_state(self) -> Tuple[str, List]:
        return (self.current, self.history.copy())

    @property
    def events(self) -> List[str]:
        return ["TOGGLE", "RECALL"]

    @property
    def name(self) -> str:
        return "ToggleMemory"


# =============================================================================
# Trace Generation
# =============================================================================

def generate_trace(
    machine: StatefulMachine,
    num_steps: int = 10,
    seed: int = None,
) -> List[IOStep]:
    """Generate random trace from machine."""
    if seed is not None:
        random.seed(seed)

    trace = []
    machine.reset()

    for _ in range(num_steps):
        event = random.choice(machine.events)

        # Generate arg for events that need one
        arg = None
        if event in ["PUSH", "ENQUEUE", "ADD", "MULT"]:
            arg = random.randint(1, 9)

        state_before = machine.get_state()
        state_after = machine.step(event, arg)

        trace.append(IOStep(
            input_event=event,
            input_arg=arg,
            state_before=state_before,
            state_after=state_after,
        ))

    return trace


def generate_test_sequences(
    machine: StatefulMachine,
    num_train: int = 8,
    num_test: int = 4,
    steps_per_seq: int = 5,
) -> Tuple[List[List[IOStep]], List[List[IOStep]]]:
    """Generate train and test sequences."""
    train_seqs = []
    test_seqs = []

    for i in range(num_train):
        train_seqs.append(generate_trace(machine, steps_per_seq, seed=i))

    for i in range(num_test):
        test_seqs.append(generate_trace(machine, steps_per_seq, seed=100 + i))

    return train_seqs, test_seqs


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_predictions(
    predictions: List[Any],
    actuals: List[Any],
) -> Dict[str, float]:
    """Evaluate prediction accuracy."""
    if not predictions or not actuals:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    total = len(actuals)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }


# =============================================================================
# All Test Machines
# =============================================================================

ALL_MACHINES = [
    StackMachine(),
    QueueMachine(),
    CounterMachine(),
    AccumulatorMachine(),
    ToggleMemoryMachine(),
]


def demo():
    """Demonstrate stateful machines."""
    print("=" * 60)
    print("Stateful Behavior Induction Infrastructure")
    print("=" * 60)

    for machine in ALL_MACHINES:
        print(f"\n--- {machine.name} ---")
        print(f"Events: {machine.events}")

        trace = generate_trace(machine, num_steps=5, seed=42)
        print("Sample trace:")
        for i, step in enumerate(trace):
            arg_str = f" {step.input_arg}" if step.input_arg is not None else ""
            print(f"  {i+1}. ({step.input_event}{arg_str}, {step.state_before}) → {step.state_after}")


if __name__ == "__main__":
    demo()
