"""
SC Repair Engine: Fix Invalid Statecharts Using LLM

Main repair engine that:
1. Detects errors in statechart
2. Generates repair prompt
3. Uses LLM to suggest fixes
4. Applies and validates repairs
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto
import json
import copy
import re

# MLX imports
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Import error detection from sibling experiment
import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')
from experiments.exp_sc_error_patterns.error_detector import (
    ErrorDetector,
    DetectionResult,
    detect_errors,
)
from experiments.exp_sc_error_patterns.error_taxonomy import (
    ErrorCategory,
    StatechartError,
)


class RepairType(Enum):
    """Types of repairs."""
    RENAME_STATE = "rename_state"
    ADD_STATE = "add_state"
    REMOVE_STATE = "remove_state"
    ADD_TRANSITION = "add_transition"
    REMOVE_TRANSITION = "remove_transition"
    SET_INITIAL = "set_initial"
    FIX_HIERARCHY = "fix_hierarchy"
    MERGE_STATES = "merge_states"


@dataclass
class RepairConfig:
    """Configuration for repair engine."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_tokens: int = 512
    max_repair_attempts: int = 3
    use_llm: bool = True


@dataclass
class Repair:
    """A single repair action."""
    repair_type: RepairType
    description: str
    changes: Dict[str, Any]
    error_fixed: str


@dataclass
class RepairResult:
    """Result of repair attempt."""
    original: Dict[str, Any]
    repaired: Dict[str, Any]
    repairs_applied: List[Repair]
    original_errors: int
    remaining_errors: int
    success: bool
    llm_used: bool = False


class SCRepairEngine:
    """
    Main engine for repairing invalid statecharts.

    Uses Qwen2.5-Coder-0.5B-Instruct for intelligent repairs.
    """

    def __init__(self, config: Optional[RepairConfig] = None):
        self.config = config or RepairConfig()
        self.model = None
        self.tokenizer = None
        self.detector = ErrorDetector()

        if self.config.use_llm and MLX_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Load the Qwen model."""
        try:
            self.model, self.tokenizer = load(self.config.model_name)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None

    def repair(self, statechart: Dict[str, Any]) -> RepairResult:
        """
        Repair an invalid statechart.

        Args:
            statechart: Invalid statechart to repair

        Returns:
            RepairResult with repaired statechart
        """
        original = copy.deepcopy(statechart)
        current = copy.deepcopy(statechart)
        repairs = []

        # Detect initial errors
        initial_result = self.detector.detect(current)
        initial_errors = len(initial_result.errors) + len(initial_result.warnings)

        if initial_errors == 0:
            return RepairResult(
                original=original,
                repaired=current,
                repairs_applied=[],
                original_errors=0,
                remaining_errors=0,
                success=True,
            )

        # Attempt repairs
        llm_used = False
        for attempt in range(self.config.max_repair_attempts):
            # Get current errors
            result = self.detector.detect(current)
            all_errors = result.errors + result.warnings

            if not all_errors:
                break

            # Try to fix each error
            for error in all_errors:
                repair = self._generate_repair(current, error)
                if repair:
                    current = self._apply_repair(current, repair)
                    repairs.append(repair)
                    if repair.changes.get("llm_generated"):
                        llm_used = True

        # Check final state
        final_result = self.detector.detect(current)
        remaining = len(final_result.errors) + len(final_result.warnings)

        return RepairResult(
            original=original,
            repaired=current,
            repairs_applied=repairs,
            original_errors=initial_errors,
            remaining_errors=remaining,
            success=(remaining == 0),
            llm_used=llm_used,
        )

    def _generate_repair(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Generate a repair for a specific error."""
        category = error.category

        if category == ErrorCategory.DUPLICATE_ELEMENT:
            return self._repair_duplicate(sc, error)
        elif category == ErrorCategory.UNREACHABLE_STATE:
            return self._repair_unreachable(sc, error)
        elif category == ErrorCategory.HIERARCHY_VIOLATION:
            return self._repair_hierarchy(sc, error)
        elif category == ErrorCategory.MISSING_STATE:
            return self._repair_missing_state(sc, error)
        elif category == ErrorCategory.DANGLING_TRANSITION:
            return self._repair_dangling(sc, error)

        # Use LLM for complex repairs
        if self.model is not None:
            return self._llm_repair(sc, error)

        return None

    def _repair_duplicate(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Repair duplicate state labels."""
        state_name = getattr(error, 'element_name', '')
        if not state_name:
            return None

        # Find and rename duplicates
        counter = [0]

        def rename_duplicates(state: Dict, first_found: List[bool]):
            label = state.get("label", "")
            if label == state_name:
                if first_found[0]:
                    # Rename subsequent occurrences
                    counter[0] += 1
                    state["label"] = f"{state_name}_{counter[0]}"
                else:
                    first_found[0] = True

            for child in state.get("children", []):
                rename_duplicates(child, first_found)

        return Repair(
            repair_type=RepairType.RENAME_STATE,
            description=f"Rename duplicate state '{state_name}'",
            changes={
                "action": "rename_duplicates",
                "state": state_name,
            },
            error_fixed=str(error.category.value),
        )

    def _repair_unreachable(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Repair unreachable state by adding transition or removing."""
        state_name = getattr(error, 'unreachable_state', '')
        if not state_name:
            return None

        # Find an existing state to connect from
        defined_states = self._get_defined_states(sc)
        initial_states = self._get_initial_states(sc)

        if initial_states:
            source = list(initial_states)[0]
            return Repair(
                repair_type=RepairType.ADD_TRANSITION,
                description=f"Add transition from '{source}' to '{state_name}'",
                changes={
                    "action": "add_transition",
                    "from": source,
                    "to": state_name,
                    "event": f"TO_{state_name.upper()}",
                },
                error_fixed=str(error.category.value),
            )

        # If no initial state, remove the unreachable state
        return Repair(
            repair_type=RepairType.REMOVE_STATE,
            description=f"Remove unreachable state '{state_name}'",
            changes={
                "action": "remove_state",
                "state": state_name,
            },
            error_fixed=str(error.category.value),
        )

    def _repair_hierarchy(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Repair hierarchy violations."""
        violation_type = getattr(error, 'violation_type', '')
        state_label = getattr(error, 'state_label', '')

        # Missing or multiple initial states
        if 'initial' in violation_type.lower():
            return Repair(
                repair_type=RepairType.SET_INITIAL,
                description=f"Set initial state in '{state_label}'",
                changes={
                    "action": "set_first_child_initial",
                    "parent": state_label,
                },
                error_fixed=str(error.category.value),
            )

        # Empty composite state
        if 'empty' in violation_type.lower():
            return Repair(
                repair_type=RepairType.ADD_STATE,
                description=f"Add default child to '{state_label}'",
                changes={
                    "action": "add_default_child",
                    "parent": state_label,
                },
                error_fixed=str(error.category.value),
            )

        return None

    def _repair_missing_state(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Repair missing state reference."""
        missing = getattr(error, 'missing_state', '')
        if not missing:
            return None

        # Check for typo - find similar state
        defined = self._get_defined_states(sc)
        similar = self._find_similar(missing, defined)

        if similar:
            return Repair(
                repair_type=RepairType.RENAME_STATE,
                description=f"Fix typo: '{missing}' -> '{similar}'",
                changes={
                    "action": "fix_reference",
                    "old": missing,
                    "new": similar,
                },
                error_fixed=str(error.category.value),
            )

        # Add the missing state
        return Repair(
            repair_type=RepairType.ADD_STATE,
            description=f"Add missing state '{missing}'",
            changes={
                "action": "add_state",
                "state": missing,
            },
            error_fixed=str(error.category.value),
        )

    def _repair_dangling(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Repair dangling transition."""
        event = getattr(error, 'transition_event', '')

        return Repair(
            repair_type=RepairType.REMOVE_TRANSITION,
            description=f"Remove dangling transition '{event}'",
            changes={
                "action": "remove_transition",
                "event": event,
            },
            error_fixed=str(error.category.value),
        )

    def _llm_repair(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> Optional[Repair]:
        """Use LLM to generate repair."""
        if not self.model or not self.tokenizer:
            return None

        prompt = self._build_repair_prompt(sc, error)

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            output = generate(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=self.config.max_tokens,
            )

            repair = self._parse_llm_repair(output, error)
            if repair:
                repair.changes["llm_generated"] = True
            return repair

        except Exception as e:
            print(f"LLM repair error: {e}")
            return None

    def _build_repair_prompt(
        self,
        sc: Dict[str, Any],
        error: StatechartError,
    ) -> str:
        """Build prompt for LLM repair."""
        sc_json = json.dumps(sc, indent=2)[:1000]  # Truncate for prompt

        return f"""Fix this statechart error.

ERROR: {error.message}
CATEGORY: {error.category.value}

STATECHART:
{sc_json}

Provide a JSON repair with format:
{{"action": "...", "details": {{...}}}}

Actions: add_state, remove_state, rename_state, add_transition, remove_transition, set_initial

REPAIR JSON:"""

    def _parse_llm_repair(
        self,
        output: str,
        error: StatechartError,
    ) -> Optional[Repair]:
        """Parse LLM output into repair."""
        try:
            # Find JSON in output
            json_match = re.search(r'\{[^{}]*\}', output)
            if json_match:
                repair_data = json.loads(json_match.group())
                action = repair_data.get("action", "")

                return Repair(
                    repair_type=RepairType.FIX_HIERARCHY,  # Generic
                    description=f"LLM suggested: {action}",
                    changes=repair_data,
                    error_fixed=str(error.category.value),
                )
        except:
            pass

        return None

    def _apply_repair(
        self,
        sc: Dict[str, Any],
        repair: Repair,
    ) -> Dict[str, Any]:
        """Apply a repair to the statechart."""
        sc = copy.deepcopy(sc)
        action = repair.changes.get("action", "")

        if action == "rename_duplicates":
            state = repair.changes.get("state", "")
            counter = [0]
            first_found = [False]

            def rename(s):
                if s.get("label") == state:
                    if first_found[0]:
                        counter[0] += 1
                        s["label"] = f"{state}_{counter[0]}"
                    else:
                        first_found[0] = True
                for c in s.get("children", []):
                    rename(c)

            rename(sc.get("root_state", {}))

        elif action == "add_transition":
            sc.setdefault("transitions", []).append({
                "from": [repair.changes["from"]],
                "to": [repair.changes["to"]],
                "event": repair.changes.get("event", ""),
            })

        elif action == "remove_transition":
            event = repair.changes.get("event", "")
            sc["transitions"] = [
                t for t in sc.get("transitions", [])
                if t.get("event") != event
            ]

        elif action == "add_state":
            state_name = repair.changes.get("state", "")
            root = sc.get("root_state", {})
            if "children" not in root:
                root["children"] = []
            root["children"].append({
                "label": state_name,
                "type": 1,
            })

        elif action == "remove_state":
            state_name = repair.changes.get("state", "")

            def remove(s):
                s["children"] = [
                    c for c in s.get("children", [])
                    if c.get("label") != state_name
                ]
                for c in s.get("children", []):
                    remove(c)

            remove(sc.get("root_state", {}))
            # Also remove transitions involving this state
            sc["transitions"] = [
                t for t in sc.get("transitions", [])
                if state_name not in t.get("from", []) and state_name not in t.get("to", [])
            ]

        elif action == "set_first_child_initial":
            parent = repair.changes.get("parent", "")

            def set_initial(s):
                if s.get("label") == parent or (not parent and s.get("label", "").startswith("__")):
                    children = s.get("children", [])
                    # Clear all initial flags first
                    for c in children:
                        c["is_initial"] = False
                    # Set first child as initial
                    if children:
                        children[0]["is_initial"] = True
                    return True
                for c in s.get("children", []):
                    if set_initial(c):
                        return True
                return False

            set_initial(sc.get("root_state", {}))

        elif action == "fix_reference":
            old = repair.changes.get("old", "")
            new = repair.changes.get("new", "")

            for t in sc.get("transitions", []):
                t["from"] = [new if s == old else s for s in t.get("from", [])]
                t["to"] = [new if s == old else s for s in t.get("to", [])]

        elif action == "add_default_child":
            parent = repair.changes.get("parent", "")

            def add_child(s):
                label = s.get("label", "")
                if label == parent or (label.startswith("__") and not parent):
                    # Add default child state
                    child_name = f"{parent}_default" if parent and not parent.startswith("__") else "Default"
                    s.setdefault("children", []).append({
                        "label": child_name,
                        "type": 1,
                        "is_initial": True,
                    })
                    return True
                for c in s.get("children", []):
                    if add_child(c):
                        return True
                return False

            add_child(sc.get("root_state", {}))

        return sc

    def _get_defined_states(self, sc: Dict) -> Set[str]:
        """Get all defined state names."""
        states = set()

        def collect(s):
            label = s.get("label", "")
            if label and not label.startswith("__"):
                states.add(label)
            for c in s.get("children", []):
                collect(c)

        collect(sc.get("root_state", {}))
        return states

    def _get_initial_states(self, sc: Dict) -> Set[str]:
        """Get initial state names."""
        initial = set()

        def collect(s):
            if s.get("is_initial") and not s.get("label", "").startswith("__"):
                initial.add(s.get("label", ""))
            for c in s.get("children", []):
                collect(c)

        collect(sc.get("root_state", {}))
        return initial

    def _find_similar(self, target: str, candidates: Set[str]) -> Optional[str]:
        """Find similar state name (typo detection)."""
        target_lower = target.lower()

        for c in candidates:
            c_lower = c.lower()
            # Check for simple edits
            if len(target) == len(c):
                diffs = sum(1 for a, b in zip(target_lower, c_lower) if a != b)
                if diffs <= 2:
                    return c
            # Check for missing/extra character
            if abs(len(target) - len(c)) == 1:
                if target_lower in c_lower or c_lower in target_lower:
                    return c

        return None


def repair_statechart(statechart: Dict[str, Any]) -> RepairResult:
    """Convenience function to repair a statechart."""
    engine = SCRepairEngine()
    return engine.repair(statechart)


def demo():
    """Demonstrate statechart repair."""
    print("=" * 60)
    print("SC REPAIR ENGINE: Fix Invalid Statecharts")
    print("=" * 60)

    # Test case: multiple errors
    broken_sc = {
        "name": "Broken Machine",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Start", "type": 1, "is_initial": True},
                {"label": "Start", "type": 1},  # Duplicate!
                {"label": "Middle", "type": 1},
                {"label": "End", "type": 1},
                {"label": "Orphan", "type": 1},  # Unreachable
            ]
        },
        "transitions": [
            {"from": ["Start"], "to": ["Midle"], "event": "GO"},  # Typo
            {"from": ["Middle"], "to": ["End"], "event": "FINISH"},
        ]
    }

    print("\n--- Original (Broken) ---")
    print(f"States: {[c.get('label') for c in broken_sc['root_state']['children']]}")

    engine = SCRepairEngine()
    result = engine.repair(broken_sc)

    print(f"\n--- Repair Result ---")
    print(f"Original errors: {result.original_errors}")
    print(f"Remaining errors: {result.remaining_errors}")
    print(f"Success: {result.success}")
    print(f"LLM used: {result.llm_used}")
    print(f"Repairs applied: {len(result.repairs_applied)}")

    for repair in result.repairs_applied:
        print(f"  - {repair.description}")

    print(f"\n--- Repaired Statechart ---")
    print(f"States: {[c.get('label') for c in result.repaired['root_state']['children']]}")

    return result


if __name__ == "__main__":
    demo()
