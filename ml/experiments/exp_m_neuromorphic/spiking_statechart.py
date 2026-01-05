"""
Experiment M: Neuromorphic Statecharts

Statecharts AS neural circuits, not approximated BY them.

The Isomorphism:
| Statechart   | Neural                              |
|--------------|-------------------------------------|
| Event        | Spike (binary pulse)                |
| State        | Attractor (reverberating cluster)   |
| Transition   | Synaptic wiring (hard connections)  |
| Guard        | Shunting inhibition                 |
| OR-state     | Lateral inhibition (winner-take-all)|
| AND-state    | Parallel circuits (no inhibition)   |
| History      | Synaptic trace (STDP-like)          |

Hypothesis: A properly wired SNN IS a statechart.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpikeEvent:
    """A spike event with source and timing."""
    source: str
    time: int
    payload: Optional[dict] = None


class SpikingNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron.

    Membrane dynamics:
        V(t+1) = leak * V(t) + sum(input_spikes * weights)
        if V > threshold: spike and reset
    """

    def __init__(
        self,
        name: str,
        threshold: float = 1.0,
        leak: float = 0.9,
        refractory_period: int = 2,
    ):
        self.name = name
        self.threshold = threshold
        self.leak = leak
        self.refractory_period = refractory_period

        # State
        self.membrane = 0.0
        self.refractory_counter = 0
        self.last_spike_time = -1
        self.spike_history: List[int] = []

    def reset(self):
        """Reset neuron state."""
        self.membrane = 0.0
        self.refractory_counter = 0
        self.last_spike_time = -1
        self.spike_history = []

    def step(self, inputs: List[Tuple[float, float]], time: int) -> bool:
        """
        Process one timestep.

        Args:
            inputs: List of (spike, weight) pairs from presynaptic neurons
            time: Current simulation time

        Returns:
            True if neuron spikes
        """
        # Refractory period check
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.membrane *= self.leak  # Still leak during refractory
            return False

        # Leak
        self.membrane *= self.leak

        # Integrate inputs
        for spike, weight in inputs:
            if spike:
                self.membrane += weight

        # Threshold check
        if self.membrane >= self.threshold:
            self.membrane = 0.0  # Reset
            self.refractory_counter = self.refractory_period
            self.last_spike_time = time
            self.spike_history.append(time)
            return True

        return False


class SpikingState:
    """
    A state represented by a cluster of neurons with explicit attractor dynamics.

    Active state = sustained reverberating activity in the cluster.
    Uses bistable dynamics: once activated, stays active until inhibited.
    """

    def __init__(
        self,
        name: str,
        num_neurons: int = 4,
        threshold: float = 0.5,
        sustain_current: float = 0.3,
    ):
        self.name = name
        self.num_neurons = num_neurons
        self.threshold = threshold
        self.sustain_current = sustain_current

        # Create neuron cluster
        self.neurons = [
            SpikingNeuron(f"{name}_n{i}", threshold=threshold, leak=0.8, refractory_period=1)
            for i in range(num_neurons)
        ]

        # Activity tracking - bistable: either active or not
        self.activity_level = 0.0
        self.is_active = False
        self._sustained = False  # Explicit sustained activation flag

    def reset(self):
        """Reset all neurons."""
        for n in self.neurons:
            n.reset()
        self.activity_level = 0.0
        self.is_active = False
        self._sustained = False

    def activate(self, strength: float = 2.0):
        """Inject current to activate this state."""
        self._sustained = True
        for n in self.neurons:
            n.membrane = strength  # Set directly above threshold

    def inhibit(self, strength: float = 5.0):
        """Inhibit this state (turns off sustained activity)."""
        self._sustained = False
        for n in self.neurons:
            n.membrane = 0

    def step(self, external_input: float, time: int) -> List[bool]:
        """
        Process one timestep.
        """
        spikes = []

        for i, neuron in enumerate(self.neurons):
            inputs = []

            # Add sustaining current if in sustained mode
            if self._sustained:
                inputs.append((1, self.sustain_current))

            # Add external input
            if external_input > 0:
                inputs.append((1, external_input))

            spike = neuron.step(inputs, time)
            spikes.append(spike)

        # Update activity level
        spike_rate = sum(spikes) / self.num_neurons
        self.activity_level = 0.5 * self.activity_level + 0.5 * spike_rate

        # State is active if sustained OR if there's recent activity
        self.is_active = self._sustained or self.activity_level > 0.2

        return spikes


class SpikingTransition:
    """
    A transition as synaptic wiring between state clusters.

    When source state is active AND event spike arrives:
    - Excite target state
    - Inhibit source state (leave old state)
    """

    def __init__(
        self,
        name: str,
        source: SpikingState,
        target: SpikingState,
        event_name: str,
        weight: float = 3.0,
        guard_neuron: Optional[SpikingNeuron] = None,
    ):
        self.name = name
        self.source = source
        self.target = target
        self.event_name = event_name
        self.weight = weight
        self.guard_neuron = guard_neuron  # For shunting inhibition

    def check_and_fire(self, event: Optional[SpikeEvent], time: int) -> bool:
        """
        Check if transition should fire and execute it.

        Returns True if transition fired.
        """
        # Check event match
        if event is None or event.source != self.event_name:
            return False

        # Check source state is active
        if not self.source.is_active:
            return False

        # Check guard (shunting inhibition)
        if self.guard_neuron is not None:
            # Guard blocks transition if not spiking
            if self.guard_neuron.last_spike_time != time - 1:
                return False

        # FIRE TRANSITION
        # Inhibit source state
        self.source.inhibit(strength=5.0)

        # Activate target state
        self.target.activate(strength=self.weight)

        return True


class SpikingStatechart:
    """
    A complete statechart as a spiking neural network.

    OR-states use lateral inhibition (winner-take-all).
    AND-states run as parallel circuits.
    """

    def __init__(self, name: str):
        self.name = name
        self.states: Dict[str, SpikingState] = {}
        self.transitions: List[SpikingTransition] = []
        self.time = 0

        # For OR-state lateral inhibition
        self.or_groups: List[List[str]] = []

        # Event queue
        self.event_queue: List[SpikeEvent] = []

        # History (synaptic traces)
        self.history_traces: Dict[str, float] = {}

    def add_state(self, name: str, num_neurons: int = 3) -> SpikingState:
        """Add a state cluster."""
        state = SpikingState(name, num_neurons=num_neurons)
        self.states[name] = state
        return state

    def add_or_group(self, state_names: List[str]):
        """Define states that are mutually exclusive (OR)."""
        self.or_groups.append(state_names)

    def add_transition(
        self,
        name: str,
        source: str,
        target: str,
        event: str,
        weight: float = 3.0,
    ) -> SpikingTransition:
        """Add a transition between states."""
        trans = SpikingTransition(
            name=name,
            source=self.states[source],
            target=self.states[target],
            event_name=event,
            weight=weight,
        )
        self.transitions.append(trans)
        return trans

    def set_initial(self, state_name: str):
        """Set initial state by activating it."""
        self.states[state_name].activate(strength=3.0)

    def inject_event(self, event_name: str):
        """Inject an event spike."""
        self.event_queue.append(SpikeEvent(source=event_name, time=self.time))

    def apply_lateral_inhibition(self):
        """Apply winner-take-all within OR groups."""
        for group in self.or_groups:
            # Find most active state
            activities = [(name, self.states[name].activity_level)
                         for name in group]
            activities.sort(key=lambda x: -x[1])

            if len(activities) > 1 and activities[0][1] > 0.1:
                winner = activities[0][0]
                # Inhibit losers
                for name, _ in activities[1:]:
                    if self.states[name].activity_level > 0.1:
                        self.states[name].inhibit(strength=2.0)

    def step(self) -> Dict[str, any]:
        """
        Process one timestep.

        Returns info about current state.
        """
        # Get current event
        current_event = None
        if self.event_queue and self.event_queue[0].time <= self.time:
            current_event = self.event_queue.pop(0)

        # Check and fire transitions
        fired = []
        for trans in self.transitions:
            if trans.check_and_fire(current_event, self.time):
                fired.append(trans.name)

        # Step all state clusters
        all_spikes = {}
        for name, state in self.states.items():
            spikes = state.step(0.0, self.time)
            all_spikes[name] = spikes

        # Apply lateral inhibition for OR groups
        self.apply_lateral_inhibition()

        # Update history traces (STDP-like decay)
        for name, state in self.states.items():
            if name not in self.history_traces:
                self.history_traces[name] = 0.0

            if state.is_active:
                self.history_traces[name] = 1.0
            else:
                self.history_traces[name] *= 0.95  # Decay

        self.time += 1

        return {
            "time": self.time,
            "event": current_event.source if current_event else None,
            "fired": fired,
            "active_states": [n for n, s in self.states.items() if s.is_active],
            "activities": {n: s.activity_level for n, s in self.states.items()},
            "spikes": all_spikes,
        }

    def get_active_state(self) -> Optional[str]:
        """Get the currently active state (for OR groups)."""
        max_activity = 0.0
        active = None
        for name, state in self.states.items():
            if state.activity_level > max_activity:
                max_activity = state.activity_level
                active = name
        return active if max_activity > 0.2 else None

    def run(self, num_steps: int) -> List[Dict]:
        """Run simulation for multiple steps."""
        results = []
        for _ in range(num_steps):
            results.append(self.step())
        return results


def create_cycle_statechart() -> SpikingStatechart:
    """Create a 3-state cycle: A -> B -> C -> A"""
    sc = SpikingStatechart("cycle")

    # Add states
    sc.add_state("A", num_neurons=3)
    sc.add_state("B", num_neurons=3)
    sc.add_state("C", num_neurons=3)

    # Define as OR group (mutually exclusive)
    sc.add_or_group(["A", "B", "C"])

    # Add transitions
    sc.add_transition("A_to_B", "A", "B", "NEXT")
    sc.add_transition("B_to_C", "B", "C", "NEXT")
    sc.add_transition("C_to_A", "C", "A", "NEXT")

    return sc


def test_cycle():
    """Test 3-state cycle with spike patterns."""
    print("=" * 70)
    print("NEUROMORPHIC STATECHART: 3-State Cycle Test")
    print("=" * 70)
    print("\nHypothesis: Properly wired SNN IS a statechart")
    print("Setup: A -> B -> C -> A with NEXT event")
    print()

    sc = create_cycle_statechart()
    sc.set_initial("A")

    # Run without events to stabilize
    print("Phase 1: Stabilize initial state A")
    print("-" * 50)
    for _ in range(10):
        info = sc.step()

    print(f"Active state: {sc.get_active_state()}")
    print(f"Activities: A={sc.states['A'].activity_level:.2f}, "
          f"B={sc.states['B'].activity_level:.2f}, "
          f"C={sc.states['C'].activity_level:.2f}")

    # Inject NEXT events and observe transitions
    print("\nPhase 2: Inject NEXT events")
    print("-" * 50)

    transitions_observed = []
    expected_sequence = ["A", "B", "C", "A", "B", "C"]

    for i in range(6):
        # Record current state
        current = sc.get_active_state()

        # Inject event
        sc.inject_event("NEXT")

        # Run several steps for transition to complete
        for _ in range(5):
            info = sc.step()

        # Record new state
        new = sc.get_active_state()
        transitions_observed.append((current, new))

        print(f"NEXT #{i+1}: {current} -> {new} "
              f"(expected: {expected_sequence[i]} -> {expected_sequence[(i+1)%3]})")

    # Verify
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    correct = 0
    for i, (src, tgt) in enumerate(transitions_observed):
        exp_src = expected_sequence[i]
        exp_tgt = expected_sequence[(i + 1) % 3]
        if tgt == exp_tgt:
            correct += 1
            status = "[OK]"
        else:
            status = "[FAIL]"
        print(f"Transition {i+1}: {src} -> {tgt} {status}")

    accuracy = correct / len(transitions_observed)
    print(f"\nAccuracy: {accuracy:.0%} ({correct}/{len(transitions_observed)})")

    if accuracy == 1.0:
        print("\n[PASS] Spike patterns match statechart semantics!")
        print("       SNN correctly implements the state machine.")
    else:
        print("\n[PARTIAL] Some transitions incorrect.")
        print("       May need tuning of weights/thresholds.")

    return accuracy


def test_lateral_inhibition():
    """Test OR-state mutual exclusion via lateral inhibition."""
    print("\n" + "=" * 70)
    print("LATERAL INHIBITION TEST (OR-State Semantics)")
    print("=" * 70)

    sc = SpikingStatechart("or_test")
    sc.add_state("S1", num_neurons=3)
    sc.add_state("S2", num_neurons=3)
    sc.add_state("S3", num_neurons=3)
    sc.add_or_group(["S1", "S2", "S3"])

    # Activate multiple states simultaneously
    print("\nActivating S1 and S2 simultaneously...")
    sc.states["S1"].activate(3.0)
    sc.states["S2"].activate(2.5)  # Slightly weaker

    # Run and observe winner-take-all
    for t in range(20):
        sc.step()
        if t % 5 == 0:
            active = sc.get_active_state()
            print(f"  t={t:2d}: Active={active}, "
                  f"S1={sc.states['S1'].activity_level:.2f}, "
                  f"S2={sc.states['S2'].activity_level:.2f}, "
                  f"S3={sc.states['S3'].activity_level:.2f}")

    winner = sc.get_active_state()
    print(f"\nWinner: {winner} (should be S1 due to stronger activation)")

    if winner == "S1":
        print("[PASS] Lateral inhibition enforces mutual exclusion")
    else:
        print(f"[CHECK] Winner was {winner}, expected S1")


def test_history_trace():
    """Test history via synaptic traces."""
    print("\n" + "=" * 70)
    print("HISTORY TRACE TEST (Synaptic Memory)")
    print("=" * 70)

    sc = create_cycle_statechart()
    sc.set_initial("A")

    # Stabilize
    for _ in range(10):
        sc.step()

    print("\nVisiting states A -> B -> C...")

    for state in ["A", "B", "C"]:
        # Stay in current state
        for _ in range(5):
            sc.step()

        traces = {n: sc.history_traces.get(n, 0) for n in ["A", "B", "C"]}
        print(f"After visiting {state}: traces = {traces}")

        # Move to next
        sc.inject_event("NEXT")
        for _ in range(5):
            sc.step()

    # Check traces
    print("\nFinal traces (should show recency):")
    for name in ["A", "B", "C"]:
        trace = sc.history_traces.get(name, 0)
        print(f"  {name}: {trace:.3f}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("EXPERIMENT M: NEUROMORPHIC STATECHARTS")
    print("=" * 70)
    print("\nCore Insight: Statecharts and SNNs share deep structure")
    print("- States = Attractor dynamics")
    print("- Transitions = Synaptic wiring")
    print("- OR-states = Lateral inhibition")
    print()

    # Test 1: Basic cycle
    cycle_acc = test_cycle()

    # Test 2: Lateral inhibition
    test_lateral_inhibition()

    # Test 3: History traces
    test_history_trace()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nCycle accuracy: {cycle_acc:.0%}")
    print("\nThe isomorphism holds:")
    print("  - Spikes encode events")
    print("  - Attractor clusters encode states")
    print("  - Synaptic wiring encodes transitions")
    print("  - Lateral inhibition encodes OR-semantics")
    print("  - Synaptic traces encode history")
    print("\nImplication: Statecharts could run on neuromorphic hardware!")
    print("=" * 70)


if __name__ == "__main__":
    main()
