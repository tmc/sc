"""
Reachability Predictor: BFS baseline and LLM-based prediction.

BFS provides ground truth for reachability analysis.
LLM attempts to predict reachability without exhaustive search.
"""

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_execution_prediction.starlark_eval import eval_guard


@dataclass
class ReachabilityResult:
    """Result of reachability analysis."""
    target_state: str
    is_reachable: bool
    min_steps: int  # -1 if not reachable
    required_events: List[str]
    path: List[str]  # State path to reach target
    analysis_time_ms: float = 0.0


@dataclass
class BFSState:
    """State for BFS exploration."""
    config: Set[str]
    context: Dict[str, Any]
    path: List[str]
    events: List[str]
    steps: int


class BFSReachabilityAnalyzer:
    """
    Ground truth reachability analyzer using BFS.

    Exhaustively searches the state space to find shortest path.
    """

    def __init__(self, sc_json: Dict, max_steps: int = 50):
        self.sc_json = sc_json
        self.max_steps = max_steps
        self._parse_sc()

    def _parse_sc(self):
        """Parse statechart structure."""
        self.states: Set[str] = set()
        self.transitions: List[Dict] = self.sc_json.get("transitions", [])
        self.initial_context = self.sc_json.get("initial_context", {})

        # Extract all state labels
        def extract_states(state_node: Dict):
            if "label" in state_node:
                self.states.add(state_node["label"])
            for child in state_node.get("children", []):
                extract_states(child)

        root = self.sc_json.get("root_state", {})
        extract_states(root)

        # Find initial state(s)
        self.initial_states = self._find_initial_states(root)

        # Build transition map
        self.trans_from: Dict[str, List[Dict]] = {}
        for t in self.transitions:
            for src in t.get("from", []):
                if src not in self.trans_from:
                    self.trans_from[src] = []
                self.trans_from[src].append(t)

    def _find_initial_states(self, state_node: Dict) -> Set[str]:
        """Find initial configuration."""
        initial = set()

        if state_node.get("is_initial"):
            if state_node.get("type", 1) == 1:  # BASIC
                initial.add(state_node["label"])
            else:
                # Composite: find initial children
                for child in state_node.get("children", []):
                    initial.update(self._find_initial_states(child))

        # For root, check all children
        if state_node.get("label") == "__root__":
            for child in state_node.get("children", []):
                if child.get("is_initial"):
                    initial.update(self._find_initial_states(child))
                elif len(state_node.get("children", [])) == 1:
                    # Single child is default initial
                    initial.update(self._find_initial_states(child))

        return initial if initial else set()

    def analyze(
        self,
        target_state: str,
        start_config: Optional[Set[str]] = None,
        start_context: Optional[Dict[str, Any]] = None,
    ) -> ReachabilityResult:
        """
        Analyze reachability of target state using BFS.

        Args:
            target_state: State to reach
            start_config: Starting configuration (default: initial)
            start_context: Starting context (default: initial_context)

        Returns:
            ReachabilityResult with path and events
        """
        start_time = time.time()

        # Initialize
        if start_config is None:
            start_config = self.initial_states.copy()
        if start_context is None:
            start_context = self.initial_context.copy()

        # Check if already at target
        if target_state in start_config:
            return ReachabilityResult(
                target_state=target_state,
                is_reachable=True,
                min_steps=0,
                required_events=[],
                path=[target_state],
                analysis_time_ms=(time.time() - start_time) * 1000,
            )

        # BFS
        queue = deque([BFSState(
            config=start_config,
            context=start_context,
            path=list(start_config),
            events=[],
            steps=0,
        )])

        # Track visited (config, context) pairs to avoid cycles
        # Include context in visited to handle context-dependent guards
        context_key = tuple(sorted(
            (k, v) for k, v in start_context.items()
            if isinstance(v, (int, bool)) and k not in ('values',)
        ))
        visited: Set[tuple] = {(frozenset(start_config), context_key)}

        while queue:
            state = queue.popleft()

            if state.steps >= self.max_steps:
                continue

            # Try ALL transitions from current config (not just one per event)
            for src_state in state.config:
                for trans in self.trans_from.get(src_state, []):
                    event = trans.get("event", "")

                    # Check guard
                    guard = trans.get("guard")
                    guard_result = True
                    if guard:
                        guard_expr = guard.get("expression", "") if isinstance(guard, dict) else guard
                        try:
                            guard_result = eval_guard(guard_expr, state.context)
                        except Exception:
                            guard_result = False

                    if not guard_result:
                        continue

                    # Compute new config
                    new_config = self._apply_transition(state.config, trans)

                    # Apply actions to context
                    new_context = self._apply_actions(state.context, trans)

                    # Create a key that includes context for stateful transitions
                    # Limit context to relevant numeric values to prevent explosion
                    context_key = tuple(sorted(
                        (k, v) for k, v in new_context.items()
                        if isinstance(v, (int, bool)) and k not in ('values',)
                    ))
                    state_key = (frozenset(new_config), context_key)

                    if state_key in visited:
                        continue
                    visited.add(state_key)

                    # Check if target reached
                    if target_state in new_config:
                        return ReachabilityResult(
                            target_state=target_state,
                            is_reachable=True,
                            min_steps=state.steps + 1,
                            required_events=state.events + [event],
                            path=state.path + list(new_config),
                            analysis_time_ms=(time.time() - start_time) * 1000,
                        )

                    # Enqueue
                    queue.append(BFSState(
                        config=new_config,
                        context=new_context,
                        path=state.path + list(new_config),
                        events=state.events + [event],
                        steps=state.steps + 1,
                    ))

        # Not reachable
        return ReachabilityResult(
            target_state=target_state,
            is_reachable=False,
            min_steps=-1,
            required_events=[],
            path=[],
            analysis_time_ms=(time.time() - start_time) * 1000,
        )

    def _apply_transition(self, config: Set[str], trans: Dict) -> Set[str]:
        """Apply transition to get new configuration."""
        new_config = config.copy()
        for src in trans.get("from", []):
            new_config.discard(src)
        for tgt in trans.get("to", []):
            new_config.add(tgt)
        return new_config

    def _apply_actions(self, context: Dict[str, Any], trans: Dict) -> Dict[str, Any]:
        """Apply transition actions to context (simplified)."""
        from experiments.exp_execution_prediction.starlark_eval import eval_action

        new_context = context.copy()
        for action in trans.get("actions", []):
            expr = action.get("expression", "") if isinstance(action, dict) else action
            if expr:
                try:
                    new_context = eval_action(expr, new_context)
                except Exception:
                    pass
        return new_context


class LLMReachabilityPredictor:
    """
    LLM-based reachability predictor using Qwen.

    Predicts reachability without exhaustive search.
    """

    def __init__(self, model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> bool:
        """Load MLX model."""
        try:
            from mlx_lm import load, generate
            self.model, self.tokenizer = load(self.model_name)
            self._generate_fn = generate
            self._loaded = True
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict(
        self,
        sc_json: Dict,
        current_config: Set[str],
        target_state: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReachabilityResult:
        """
        Predict reachability using LLM.

        Args:
            sc_json: Statechart definition
            current_config: Current active states
            target_state: State to reach
            context: Current context values

        Returns:
            ReachabilityResult with prediction
        """
        start_time = time.time()

        if not self._loaded:
            if not self.load_model():
                return ReachabilityResult(
                    target_state=target_state,
                    is_reachable=False,
                    min_steps=-1,
                    required_events=[],
                    path=[],
                    analysis_time_ms=0,
                )

        # Build prompt
        prompt = self._build_prompt(sc_json, current_config, target_state, context)

        # Generate
        try:
            response = self._generate_fn(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=150,
                verbose=False,
            )

            # Parse response
            result = self._parse_response(response, target_state)
            result.analysis_time_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            return ReachabilityResult(
                target_state=target_state,
                is_reachable=False,
                min_steps=-1,
                required_events=[],
                path=[],
                analysis_time_ms=(time.time() - start_time) * 1000,
            )

    def _build_prompt(
        self,
        sc_json: Dict,
        current_config: Set[str],
        target_state: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Build prompt for reachability prediction."""
        # Extract transition summary
        transitions = sc_json.get("transitions", [])
        trans_summary = []
        for t in transitions[:10]:  # Limit for prompt size
            src = t.get("from", [])
            tgt = t.get("to", [])
            event = t.get("event", "")
            guard = t.get("guard")
            guard_str = f" [{guard.get('expression', '')}]" if guard else ""
            trans_summary.append(f"  {src} --{event}{guard_str}--> {tgt}")

        # Build prompt
        prompt = f"""Analyze statechart reachability.

Current state: {list(current_config)}
Target state: {target_state}
Context: {context if context else {}}

Transitions:
{chr(10).join(trans_summary)}

Is the target state "{target_state}" reachable from {list(current_config)}?
Answer in JSON format:
{{"reachable": true/false, "min_steps": N, "events": ["EVENT1", "EVENT2"]}}

Answer:"""

        return prompt

    def _parse_response(self, response: str, target_state: str) -> ReachabilityResult:
        """Parse LLM response into ReachabilityResult."""
        import re

        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReachabilityResult(
                    target_state=target_state,
                    is_reachable=data.get("reachable", False),
                    min_steps=data.get("min_steps", -1),
                    required_events=data.get("events", []),
                    path=[],
                )
            except json.JSONDecodeError:
                pass

        # Fallback: look for keywords
        response_lower = response.lower()
        is_reachable = "true" in response_lower or "yes" in response_lower or "reachable" in response_lower

        # Try to extract number for steps
        steps_match = re.search(r'(\d+)\s*(?:step|move)', response_lower)
        min_steps = int(steps_match.group(1)) if steps_match else (1 if is_reachable else -1)

        return ReachabilityResult(
            target_state=target_state,
            is_reachable=is_reachable,
            min_steps=min_steps,
            required_events=[],
            path=[],
        )


if __name__ == "__main__":
    from experiments.exp_execution_prediction import CounterMachine, BranchingMachine

    print("Reachability Predictor Test")
    print("=" * 60)

    # Test BFS on Counter
    print("\n1. Counter Machine BFS:")
    counter = CounterMachine(target=3)
    analyzer = BFSReachabilityAnalyzer(counter.to_json())

    result = analyzer.analyze("Done")
    print(f"  Target: Done")
    print(f"  Reachable: {result.is_reachable}")
    print(f"  Min steps: {result.min_steps}")
    print(f"  Events: {result.required_events}")
    print(f"  Time: {result.analysis_time_ms:.1f}ms")

    # Test BFS on Branching
    print("\n2. Branching Machine BFS:")
    branching = BranchingMachine(threshold=50)
    analyzer = BFSReachabilityAnalyzer(branching.to_json())

    result = analyzer.analyze("Final")
    print(f"  Target: Final")
    print(f"  Reachable: {result.is_reachable}")
    print(f"  Min steps: {result.min_steps}")
    print(f"  Events: {result.required_events}")

    # Test unreachable
    result = analyzer.analyze("NonExistent")
    print(f"\n  Target: NonExistent")
    print(f"  Reachable: {result.is_reachable}")
