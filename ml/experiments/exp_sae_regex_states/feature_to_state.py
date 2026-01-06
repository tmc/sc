"""
Feature to State Mapping.

Maps SAE features to discrete statechart states.

Key insight: SAE features ARE states, but we need to:
1. Identify which feature combinations are meaningful
2. Merge similar states (same feature set)
3. Interpret what each feature represents
4. Determine which states are accepting
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, FrozenSet
from collections import defaultdict
from .sae_regex import CharSequenceSAE, SAEConfig


@dataclass
class SAEState:
    """
    A state derived from SAE feature activations.

    The state is defined by which features are active.
    """
    # Unique identifier
    id: int

    # The active features that define this state
    active_features: FrozenSet[int]

    # Interpretation
    label: str = ""
    is_accepting: bool = False
    is_initial: bool = False

    # Statistics
    visit_count: int = 0
    prefixes: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.active_features)

    def __eq__(self, other):
        if not isinstance(other, SAEState):
            return False
        return self.active_features == other.active_features


@dataclass
class StateActivation:
    """Record of a state activation during string processing."""
    prefix: str           # The prefix that led to this state
    state: SAEState       # The state activated
    char_consumed: str    # Character that led here (empty for initial)
    feature_values: List[float]  # Actual activation values


@dataclass
class FeatureInterpretation:
    """Interpretation of what a feature represents."""
    feature_id: int
    label: str                          # Human-readable label
    activating_prefixes: List[str]      # Prefixes that activate this
    char_pattern: Optional[str] = None  # Detected char pattern
    position_pattern: Optional[str] = None  # Position pattern


class FeatureStateMapper:
    """
    Maps SAE features to statechart states.

    This is the core component that bridges:
        SAE (continuous, distributed) → Statechart (discrete, symbolic)
    """

    def __init__(self, sae: CharSequenceSAE):
        self.sae = sae
        self.states: Dict[FrozenSet[int], SAEState] = {}
        self.state_id_counter = 0
        self.feature_interpretations: Dict[int, FeatureInterpretation] = {}

    def get_or_create_state(self, active_features: Tuple[int, ...]) -> SAEState:
        """
        Get existing state or create new one for given features.

        States are uniquely identified by their active feature set.
        """
        key = frozenset(active_features)

        if key not in self.states:
            state = SAEState(
                id=self.state_id_counter,
                active_features=key,
                label=f"S{self.state_id_counter}"
            )
            self.states[key] = state
            self.state_id_counter += 1

        return self.states[key]

    def discover_states(
        self,
        strings: List[str],
        verbose: bool = False
    ) -> List[SAEState]:
        """
        Discover all states by processing strings through SAE.

        Returns list of unique states found.
        """
        all_activations: List[StateActivation] = []

        for s in strings:
            trace = self.sae.trace_states(s)

            prev_char = ""
            for prefix, feature_tuple in trace:
                state = self.get_or_create_state(feature_tuple)
                state.visit_count += 1
                state.prefixes.append(prefix)

                # Get full feature values for analysis
                features, _ = self.sae.get_feature_activation(prefix)

                activation = StateActivation(
                    prefix=prefix,
                    state=state,
                    char_consumed=prefix[-1] if prefix else "",
                    feature_values=features
                )
                all_activations.append(activation)

        # Mark initial state
        empty_state = self.get_or_create_state(
            self.sae.get_state_for_prefix("")
        )
        empty_state.is_initial = True

        if verbose:
            print(f"Discovered {len(self.states)} unique states")
            for state in sorted(self.states.values(), key=lambda s: s.id):
                print(f"  {state.label}: features={set(state.active_features)}, "
                      f"visits={state.visit_count}")

        return list(self.states.values())

    def mark_accepting_states(
        self,
        positive: List[str],
        negative: List[str],
        threshold: float = 0.3
    ) -> Set[int]:
        """
        Mark which states are accepting based on positive/negative examples.

        A state is accepting if it appears at the END of positive strings
        more often than negative strings.

        Returns set of accepting state IDs.
        """
        positive_end_states: Dict[int, int] = defaultdict(int)
        negative_end_states: Dict[int, int] = defaultdict(int)

        for s in positive:
            state = self.get_or_create_state(
                self.sae.get_state_for_prefix(s)
            )
            positive_end_states[state.id] += 1

        for s in negative:
            state = self.get_or_create_state(
                self.sae.get_state_for_prefix(s)
            )
            negative_end_states[state.id] += 1

        accepting = set()
        for state_id in positive_end_states:
            pos_count = positive_end_states[state_id]
            neg_count = negative_end_states.get(state_id, 0)
            total = pos_count + neg_count

            if total > 0:
                pos_ratio = pos_count / total
                if pos_ratio > 0.5 + threshold:  # More positive than negative
                    accepting.add(state_id)

                    # Mark the state
                    for state in self.states.values():
                        if state.id == state_id:
                            state.is_accepting = True

        return accepting

    def interpret_features(self, verbose: bool = False) -> Dict[int, FeatureInterpretation]:
        """
        Interpret what each active feature represents.

        Analyzes patterns in prefixes that activate each feature.
        """
        interpretations = {}

        for feature_id in self.sae.feature_activations:
            prefixes = self.sae.feature_activations[feature_id]

            if not prefixes:
                continue

            # Analyze character patterns
            char_counts: Dict[str, int] = defaultdict(int)
            for p in prefixes:
                for c in p:
                    char_counts[c] += 1

            # Find dominant character
            if char_counts:
                dominant_char = max(char_counts.keys(), key=lambda c: char_counts[c])
                char_pattern = f"contains '{dominant_char}'"
            else:
                char_pattern = None

            # Analyze position patterns
            lengths = [len(p) for p in prefixes]
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                if avg_len < 1.5:
                    position_pattern = "start"
                elif all(l == lengths[0] for l in lengths):
                    position_pattern = f"position {lengths[0]}"
                else:
                    position_pattern = f"avg_position {avg_len:.1f}"
            else:
                position_pattern = None

            # Create label
            label_parts = []
            if char_pattern:
                label_parts.append(char_pattern)
            if position_pattern:
                label_parts.append(position_pattern)

            label = ", ".join(label_parts) if label_parts else f"feature_{feature_id}"

            interpretation = FeatureInterpretation(
                feature_id=feature_id,
                label=label,
                activating_prefixes=prefixes[:10],  # Keep first 10
                char_pattern=char_pattern,
                position_pattern=position_pattern
            )
            interpretations[feature_id] = interpretation

            if verbose:
                print(f"Feature {feature_id}: {label}")
                print(f"  Prefixes: {prefixes[:5]}")

        self.feature_interpretations = interpretations
        return interpretations

    def label_states(self) -> None:
        """
        Generate human-readable labels for states based on feature interpretations.
        """
        if not self.feature_interpretations:
            self.interpret_features()

        for state in self.states.values():
            parts = []
            for f_id in sorted(state.active_features):
                if f_id in self.feature_interpretations:
                    interp = self.feature_interpretations[f_id]
                    if interp.char_pattern:
                        parts.append(interp.char_pattern.replace("contains '", "").replace("'", ""))
                else:
                    parts.append(f"f{f_id}")

            if parts:
                state.label = "_".join(parts[:3])  # Keep first 3
            else:
                state.label = f"S{state.id}"

            if state.is_accepting:
                state.label += "*"
            if state.is_initial:
                state.label = ">" + state.label

    def get_state_transitions(
        self,
        strings: List[str]
    ) -> Dict[Tuple[int, str], int]:
        """
        Extract transitions from state traces.

        Returns dict mapping (source_state_id, char) -> target_state_id
        """
        transitions: Dict[Tuple[int, str], Set[int]] = defaultdict(set)

        for s in strings:
            trace = self.sae.trace_states(s)

            for i in range(len(trace) - 1):
                prefix1, features1 = trace[i]
                prefix2, features2 = trace[i + 1]

                state1 = self.get_or_create_state(features1)
                state2 = self.get_or_create_state(features2)

                char = prefix2[-1] if prefix2 else ""

                transitions[(state1.id, char)].add(state2.id)

        # Convert to deterministic (pick most common)
        deterministic: Dict[Tuple[int, str], int] = {}
        for key, targets in transitions.items():
            # If multiple targets, this is non-deterministic
            # Pick the most frequently visited one
            target_counts = [(t, self.states[frozenset({t})].visit_count if frozenset({t}) in self.states else 0)
                            for t in targets]
            # Fallback: just pick first
            if target_counts:
                deterministic[key] = list(targets)[0]

        return deterministic


def extract_states_from_sae(
    sae: CharSequenceSAE,
    positive: List[str],
    negative: List[str],
    verbose: bool = False
) -> Tuple[FeatureStateMapper, List[SAEState], Set[int]]:
    """
    Extract discrete states from a trained SAE.

    Returns:
        (mapper, states, accepting_state_ids)
    """
    mapper = FeatureStateMapper(sae)

    # Discover states from all strings
    all_strings = positive + negative
    states = mapper.discover_states(all_strings, verbose=verbose)

    # Mark accepting states
    accepting = mapper.mark_accepting_states(positive, negative)

    # Interpret and label
    mapper.interpret_features(verbose=verbose)
    mapper.label_states()

    return mapper, states, accepting


def interpret_features(
    sae: CharSequenceSAE,
    verbose: bool = False
) -> Dict[int, FeatureInterpretation]:
    """
    Interpret what each SAE feature represents.

    Convenience function that creates mapper and interprets.
    """
    mapper = FeatureStateMapper(sae)
    return mapper.interpret_features(verbose=verbose)


def test_feature_to_state():
    """Test feature to state mapping."""
    from .sae_regex import train_char_sae, SAEConfig

    print("=" * 60)
    print("Testing Feature to State Mapping")
    print("=" * 60)

    # Test data
    positive = ["ab", "aab", "aaab", "aaaab"]
    negative = ["", "a", "b", "ba", "bb", "baa"]

    # Train SAE
    print("\n1. Training SAE:")
    config = SAEConfig(n_features=16, k_active=3, n_epochs=50)
    sae = train_char_sae(positive + negative, config, verbose=False)

    # Extract states
    print("\n2. Extracting states:")
    mapper, states, accepting = extract_states_from_sae(
        sae, positive, negative, verbose=True
    )

    print(f"\nFound {len(states)} states, {len(accepting)} accepting")

    # Show states
    print("\n3. State details:")
    for state in sorted(states, key=lambda s: s.id):
        acc = "ACCEPT" if state.is_accepting else ""
        init = "INITIAL" if state.is_initial else ""
        flags = " ".join(filter(None, [init, acc]))
        print(f"  {state.label}: features={set(state.active_features)} {flags}")
        print(f"    Prefixes: {state.prefixes[:5]}")

    # Show transitions
    print("\n4. Transitions:")
    transitions = mapper.get_state_transitions(positive + negative)
    for (src, char), tgt in sorted(transitions.items()):
        src_label = [s.label for s in states if s.id == src][0]
        tgt_label = [s.label for s in states if s.id == tgt][0]
        print(f"  {src_label} --'{char}'--> {tgt_label}")

    # Test matching
    print("\n5. Testing matches:")
    for s in positive + negative[:3]:
        final_state = mapper.get_or_create_state(
            sae.get_state_for_prefix(s)
        )
        matches = final_state.is_accepting
        expected = s in positive
        status = "PASS" if matches == expected else "FAIL"
        print(f"  '{s}': {matches} (expected {expected}) [{status}]")

    print("\n" + "=" * 60)
    print("Feature to State tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_to_state()
