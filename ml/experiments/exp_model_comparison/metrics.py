"""
Metrics for statechart generation quality evaluation.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ValidationLevel(Enum):
    """Validation strictness levels."""
    SYNTAX = "syntax"      # Valid JSON
    STRUCTURE = "structure"  # Has required fields
    SEMANTIC = "semantic"   # Follows SC semantics
    COMPLETE = "complete"   # Fully valid SC


@dataclass
class SCMetrics:
    """Metrics for a single SC generation."""
    # Identification
    prompt_name: str
    model_size: str

    # Validity
    is_valid_json: bool = False
    is_valid_structure: bool = False
    is_valid_semantic: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # Completeness
    has_root_state: bool = False
    has_transitions: bool = False
    state_count: int = 0
    transition_count: int = 0
    expected_states: int = 0
    expected_transitions: int = 0

    # Quality
    complexity_score: float = 0.0
    completeness_score: float = 0.0

    # Performance
    generation_time_ms: float = 0.0
    token_count: int = 0
    tokens_per_second: float = 0.0

    # Raw output
    raw_output: str = ""
    parsed_sc: Optional[Dict] = None

    @property
    def validity_score(self) -> float:
        """Overall validity score 0-1."""
        score = 0.0
        if self.is_valid_json:
            score += 0.25
        if self.is_valid_structure:
            score += 0.25
        if self.is_valid_semantic:
            score += 0.5
        return score

    @property
    def state_accuracy(self) -> float:
        """How close to expected state count."""
        if self.expected_states == 0:
            return 1.0 if self.state_count > 0 else 0.0
        return min(self.state_count / self.expected_states, 1.0)

    @property
    def transition_accuracy(self) -> float:
        """How close to expected transition count."""
        if self.expected_transitions == 0:
            return 1.0 if self.transition_count > 0 else 0.0
        return min(self.transition_count / self.expected_transitions, 1.0)

    @property
    def overall_score(self) -> float:
        """Combined quality score."""
        return (
            self.validity_score * 0.4 +
            self.state_accuracy * 0.2 +
            self.transition_accuracy * 0.2 +
            self.completeness_score * 0.2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt_name": self.prompt_name,
            "model_size": self.model_size,
            "validity": {
                "json": self.is_valid_json,
                "structure": self.is_valid_structure,
                "semantic": self.is_valid_semantic,
                "score": self.validity_score,
                "errors": self.validation_errors,
            },
            "completeness": {
                "has_root": self.has_root_state,
                "has_transitions": self.has_transitions,
                "states": self.state_count,
                "transitions": self.transition_count,
                "expected_states": self.expected_states,
                "expected_transitions": self.expected_transitions,
                "score": self.completeness_score,
            },
            "performance": {
                "time_ms": self.generation_time_ms,
                "tokens": self.token_count,
                "tokens_per_sec": self.tokens_per_second,
            },
            "scores": {
                "validity": self.validity_score,
                "state_accuracy": self.state_accuracy,
                "transition_accuracy": self.transition_accuracy,
                "overall": self.overall_score,
            },
        }


def validate_json(output: str) -> Tuple[bool, Optional[Dict], List[str]]:
    """Validate JSON syntax."""
    errors = []

    # Try to extract JSON from output
    json_str = output.strip()

    # Handle markdown code blocks
    if "```json" in json_str:
        start = json_str.find("```json") + 7
        end = json_str.find("```", start)
        if end > start:
            json_str = json_str[start:end].strip()
    elif "```" in json_str:
        start = json_str.find("```") + 3
        end = json_str.find("```", start)
        if end > start:
            json_str = json_str[start:end].strip()

    # Try to find JSON object
    if not json_str.startswith("{"):
        brace_idx = json_str.find("{")
        if brace_idx >= 0:
            json_str = json_str[brace_idx:]

    try:
        parsed = json.loads(json_str)
        return True, parsed, []
    except json.JSONDecodeError as e:
        errors.append(f"JSON parse error: {e}")
        return False, None, errors


def validate_structure(sc: Dict) -> Tuple[bool, List[str]]:
    """Validate SC structure has required fields."""
    errors = []

    # Check root_state
    if "root_state" not in sc:
        errors.append("Missing root_state")
        return False, errors

    root = sc["root_state"]
    if not isinstance(root, dict):
        errors.append("root_state must be an object")
        return False, errors

    # Check root has label
    if "label" not in root:
        errors.append("root_state missing label")

    # Check for children or type
    if "children" not in root and "type" not in root:
        errors.append("root_state missing children or type")

    # Check transitions
    if "transitions" in sc:
        if not isinstance(sc["transitions"], list):
            errors.append("transitions must be an array")

    return len(errors) == 0, errors


def validate_semantic(sc: Dict) -> Tuple[bool, List[str]]:
    """Validate SC semantic correctness."""
    errors = []

    # Collect all state labels
    state_labels = set()

    def collect_states(state: Dict, path: str = ""):
        label = state.get("label", "")
        if label:
            state_labels.add(label)
        for child in state.get("children", []):
            collect_states(child, f"{path}/{label}")

    if "root_state" in sc:
        collect_states(sc["root_state"])

    # Validate transitions reference existing states
    for i, trans in enumerate(sc.get("transitions", [])):
        # Check from states
        for from_state in trans.get("from", []):
            if from_state not in state_labels:
                errors.append(f"Transition {i}: 'from' references unknown state '{from_state}'")

        # Check to states
        for to_state in trans.get("to", []):
            if to_state not in state_labels:
                errors.append(f"Transition {i}: 'to' references unknown state '{to_state}'")

    # Check for initial state
    def has_initial(state: Dict) -> bool:
        if state.get("is_initial"):
            return True
        for child in state.get("children", []):
            if has_initial(child):
                return True
        return False

    if "root_state" in sc and sc["root_state"].get("children"):
        if not has_initial(sc["root_state"]):
            errors.append("No initial state defined")

    return len(errors) == 0, errors


def count_states(sc: Dict) -> int:
    """Count total states in SC."""
    count = 0

    def count_recursive(state: Dict):
        nonlocal count
        count += 1
        for child in state.get("children", []):
            count_recursive(child)

    if "root_state" in sc:
        count_recursive(sc["root_state"])

    return count


def count_transitions(sc: Dict) -> int:
    """Count transitions in SC."""
    return len(sc.get("transitions", []))


def compute_metrics(
    output: str,
    prompt_name: str,
    model_size: str,
    expected_states: int = 0,
    expected_transitions: int = 0,
    generation_time_ms: float = 0.0,
    token_count: int = 0,
) -> SCMetrics:
    """Compute all metrics for a generation."""
    metrics = SCMetrics(
        prompt_name=prompt_name,
        model_size=model_size,
        expected_states=expected_states,
        expected_transitions=expected_transitions,
        generation_time_ms=generation_time_ms,
        token_count=token_count,
        raw_output=output,
    )

    # Validate JSON
    is_valid_json, parsed, json_errors = validate_json(output)
    metrics.is_valid_json = is_valid_json
    metrics.validation_errors.extend(json_errors)

    if not is_valid_json or parsed is None:
        return metrics

    metrics.parsed_sc = parsed

    # Validate structure
    is_valid_struct, struct_errors = validate_structure(parsed)
    metrics.is_valid_structure = is_valid_struct
    metrics.validation_errors.extend(struct_errors)

    if not is_valid_struct:
        return metrics

    # Basic completeness
    metrics.has_root_state = "root_state" in parsed
    metrics.has_transitions = len(parsed.get("transitions", [])) > 0
    metrics.state_count = count_states(parsed)
    metrics.transition_count = count_transitions(parsed)

    # Validate semantics
    is_valid_sem, sem_errors = validate_semantic(parsed)
    metrics.is_valid_semantic = is_valid_sem
    metrics.validation_errors.extend(sem_errors)

    # Compute completeness score
    state_ratio = metrics.state_count / max(expected_states, 1) if expected_states > 0 else 1.0
    trans_ratio = metrics.transition_count / max(expected_transitions, 1) if expected_transitions > 0 else 1.0
    metrics.completeness_score = min((state_ratio + trans_ratio) / 2, 1.0)

    # Performance metrics
    if generation_time_ms > 0 and token_count > 0:
        metrics.tokens_per_second = (token_count / generation_time_ms) * 1000

    return metrics


@dataclass
class AggregatedResults:
    """Aggregated results across multiple prompts for a model."""
    model_size: str
    total_prompts: int = 0
    valid_json_count: int = 0
    valid_structure_count: int = 0
    valid_semantic_count: int = 0

    avg_validity_score: float = 0.0
    avg_completeness_score: float = 0.0
    avg_overall_score: float = 0.0

    avg_generation_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0

    by_complexity: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "counts": {
                "total": self.total_prompts,
                "valid_json": self.valid_json_count,
                "valid_structure": self.valid_structure_count,
                "valid_semantic": self.valid_semantic_count,
            },
            "rates": {
                "json_rate": self.valid_json_count / max(self.total_prompts, 1),
                "structure_rate": self.valid_structure_count / max(self.total_prompts, 1),
                "semantic_rate": self.valid_semantic_count / max(self.total_prompts, 1),
            },
            "scores": {
                "validity": self.avg_validity_score,
                "completeness": self.avg_completeness_score,
                "overall": self.avg_overall_score,
            },
            "performance": {
                "avg_time_ms": self.avg_generation_time_ms,
                "avg_tokens_per_sec": self.avg_tokens_per_second,
            },
            "by_complexity": self.by_complexity,
        }


def aggregate_results(metrics_list: List[SCMetrics]) -> AggregatedResults:
    """Aggregate metrics across prompts."""
    if not metrics_list:
        return AggregatedResults(model_size="unknown")

    model_size = metrics_list[0].model_size
    result = AggregatedResults(model_size=model_size, total_prompts=len(metrics_list))

    validity_scores = []
    completeness_scores = []
    overall_scores = []
    gen_times = []
    tokens_per_sec = []

    for m in metrics_list:
        if m.is_valid_json:
            result.valid_json_count += 1
        if m.is_valid_structure:
            result.valid_structure_count += 1
        if m.is_valid_semantic:
            result.valid_semantic_count += 1

        validity_scores.append(m.validity_score)
        completeness_scores.append(m.completeness_score)
        overall_scores.append(m.overall_score)

        if m.generation_time_ms > 0:
            gen_times.append(m.generation_time_ms)
        if m.tokens_per_second > 0:
            tokens_per_sec.append(m.tokens_per_second)

    result.avg_validity_score = sum(validity_scores) / len(validity_scores)
    result.avg_completeness_score = sum(completeness_scores) / len(completeness_scores)
    result.avg_overall_score = sum(overall_scores) / len(overall_scores)

    if gen_times:
        result.avg_generation_time_ms = sum(gen_times) / len(gen_times)
    if tokens_per_sec:
        result.avg_tokens_per_second = sum(tokens_per_sec) / len(tokens_per_sec)

    return result
