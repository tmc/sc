"""
Domain Test Suite for Statechart Synthesis Benchmark

Provides standardized test domains for fair comparison:
1. GAMES: Board games, video games (Go, Chess, Zelda)
2. REGEX: Regular expression patterns
3. DIALOGUE: Conversation state machines
4. CODE: Programming language syntax

Each domain provides:
- Training examples (event sequences → final states)
- Test examples (held out for evaluation)
- Domain-specific metrics
- Difficulty levels
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum, auto
import random
import re


# =============================================================================
# Domain Types
# =============================================================================

class DomainType(Enum):
    """Types of test domains."""
    GAMES = auto()
    REGEX = auto()
    DIALOGUE = auto()
    CODE = auto()


class Difficulty(Enum):
    """Difficulty levels."""
    EASY = auto()
    MEDIUM = auto()
    HARD = auto()


@dataclass
class DomainExample:
    """A single example in a domain."""
    events: List[str]          # Input event sequence
    final_state: str           # Expected final state
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Domain:
    """A test domain with examples."""
    name: str
    domain_type: DomainType
    difficulty: Difficulty
    description: str

    train_examples: List[DomainExample]
    test_examples: List[DomainExample]

    # Domain properties
    n_events: int = 0
    n_states: int = 0
    event_names: List[str] = field(default_factory=list)
    state_names: List[str] = field(default_factory=list)

    # Expected performance bounds
    baseline_accuracy: float = 0.0    # Random baseline
    target_accuracy: float = 1.0       # Ideal performance

    def get_train_pairs(self) -> List[Tuple[List[str], str]]:
        """Get training examples as (events, state) pairs."""
        return [(ex.events, ex.final_state) for ex in self.train_examples]

    def get_test_pairs(self) -> List[Tuple[List[str], str]]:
        """Get test examples as (events, state) pairs."""
        return [(ex.events, ex.final_state) for ex in self.test_examples]


# =============================================================================
# Game Domains
# =============================================================================

def create_tictactoe_domain() -> Domain:
    """Tic-tac-toe game state machine."""
    train = [
        DomainExample(["x_move", "o_move", "x_move", "o_move", "x_wins"], "x_victory"),
        DomainExample(["x_move", "o_move", "x_move", "o_move", "o_wins"], "o_victory"),
        DomainExample(["x_move", "o_move"] * 4 + ["draw"], "draw"),
        DomainExample(["x_move", "o_move", "x_move", "x_wins"], "x_victory"),
        DomainExample(["x_move", "o_move", "o_wins"], "o_victory"),
    ]
    test = [
        DomainExample(["x_move", "x_move", "x_wins"], "x_victory"),
        DomainExample(["x_move", "o_move", "x_move", "o_move", "draw"], "draw"),
    ]

    return Domain(
        name="tictactoe",
        domain_type=DomainType.GAMES,
        difficulty=Difficulty.EASY,
        description="Tic-tac-toe game outcomes",
        train_examples=train,
        test_examples=test,
        n_events=4,
        n_states=4,
        event_names=["x_move", "o_move", "x_wins", "o_wins", "draw"],
        state_names=["playing", "x_victory", "o_victory", "draw"],
        baseline_accuracy=0.25,
        target_accuracy=1.0
    )


def create_go_ko_domain() -> Domain:
    """Go game Ko rule detection."""
    train = [
        DomainExample(["capture", "recapture"], "ko_violation"),
        DomainExample(["capture", "other_move", "recapture"], "legal"),
        DomainExample(["capture", "pass", "recapture"], "legal"),
        DomainExample(["move", "move", "capture"], "legal"),
        DomainExample(["capture", "recapture_attempt"], "ko_violation"),
    ]
    test = [
        DomainExample(["capture", "move", "move", "recapture"], "legal"),
        DomainExample(["capture", "immediate_recapture"], "ko_violation"),
    ]

    return Domain(
        name="go_ko",
        domain_type=DomainType.GAMES,
        difficulty=Difficulty.MEDIUM,
        description="Go Ko rule state machine",
        train_examples=train,
        test_examples=test,
        n_events=5,
        n_states=3,
        event_names=["capture", "recapture", "other_move", "pass", "move"],
        state_names=["normal", "ko_threat", "ko_violation", "legal"],
        baseline_accuracy=0.33,
        target_accuracy=1.0
    )


def create_zelda_module_domain() -> Domain:
    """Zelda game module transitions."""
    train = [
        DomainExample(["reset", "title_screen", "start"], "overworld"),
        DomainExample(["enter_dungeon"], "dungeon"),
        DomainExample(["enter_dungeon", "boss_room", "defeat_boss"], "victory"),
        DomainExample(["enter_cave"], "cave"),
        DomainExample(["enter_shop"], "shop"),
        DomainExample(["game_over"], "game_over"),
        DomainExample(["enter_dungeon", "die"], "game_over"),
    ]
    test = [
        DomainExample(["reset", "title_screen", "start", "enter_dungeon"], "dungeon"),
        DomainExample(["enter_dungeon", "exit_dungeon"], "overworld"),
    ]

    return Domain(
        name="zelda_modules",
        domain_type=DomainType.GAMES,
        difficulty=Difficulty.HARD,
        description="Zelda game module state machine",
        train_examples=train,
        test_examples=test,
        n_events=10,
        n_states=8,
        event_names=["reset", "title_screen", "start", "enter_dungeon", "exit_dungeon",
                     "enter_cave", "enter_shop", "boss_room", "defeat_boss", "die", "game_over"],
        state_names=["init", "title", "overworld", "dungeon", "cave", "shop",
                     "boss", "victory", "game_over"],
        baseline_accuracy=0.12,
        target_accuracy=0.95
    )


# =============================================================================
# Regex Domains
# =============================================================================

def create_email_regex_domain() -> Domain:
    """Email pattern recognition."""
    train = [
        DomainExample(["letter", "at", "letter", "dot", "letter"], "valid_email"),
        DomainExample(["letter", "letter", "at", "letter", "dot", "letter", "letter"], "valid_email"),
        DomainExample(["at", "letter"], "invalid"),
        DomainExample(["letter", "letter"], "incomplete"),
        DomainExample(["letter", "at", "at"], "invalid"),
    ]
    test = [
        DomainExample(["letter", "digit", "at", "letter", "dot", "letter"], "valid_email"),
        DomainExample(["dot", "at", "letter"], "invalid"),
    ]

    return Domain(
        name="email_regex",
        domain_type=DomainType.REGEX,
        difficulty=Difficulty.MEDIUM,
        description="Email address pattern matching",
        train_examples=train,
        test_examples=test,
        n_events=5,
        n_states=6,
        event_names=["letter", "digit", "at", "dot", "other"],
        state_names=["start", "local", "at_seen", "domain", "dot_seen", "valid_email", "invalid", "incomplete"],
        baseline_accuracy=0.25,
        target_accuracy=1.0
    )


def create_number_regex_domain() -> Domain:
    """Number pattern recognition."""
    train = [
        DomainExample(["digit"], "integer"),
        DomainExample(["digit", "digit", "digit"], "integer"),
        DomainExample(["digit", "dot", "digit"], "float"),
        DomainExample(["minus", "digit"], "negative_integer"),
        DomainExample(["minus", "digit", "dot", "digit"], "negative_float"),
        DomainExample(["dot", "digit"], "float"),
    ]
    test = [
        DomainExample(["digit", "digit"], "integer"),
        DomainExample(["minus", "digit", "digit"], "negative_integer"),
    ]

    return Domain(
        name="number_regex",
        domain_type=DomainType.REGEX,
        difficulty=Difficulty.EASY,
        description="Number pattern matching",
        train_examples=train,
        test_examples=test,
        n_events=4,
        n_states=5,
        event_names=["digit", "dot", "minus", "other"],
        state_names=["start", "integer", "float", "negative_integer", "negative_float", "invalid"],
        baseline_accuracy=0.2,
        target_accuracy=1.0
    )


# =============================================================================
# Dialogue Domains
# =============================================================================

def create_customer_service_domain() -> Domain:
    """Customer service dialogue state machine."""
    train = [
        DomainExample(["greeting", "query", "answer", "thanks"], "resolved"),
        DomainExample(["greeting", "complaint", "apologize", "offer_solution"], "escalated"),
        DomainExample(["greeting", "query", "clarify", "answer"], "resolved"),
        DomainExample(["greeting", "complaint", "apologize", "accept"], "resolved"),
        DomainExample(["greeting", "hangup"], "abandoned"),
    ]
    test = [
        DomainExample(["greeting", "query", "answer"], "resolved"),
        DomainExample(["greeting", "complaint", "escalate"], "escalated"),
    ]

    return Domain(
        name="customer_service",
        domain_type=DomainType.DIALOGUE,
        difficulty=Difficulty.MEDIUM,
        description="Customer service conversation states",
        train_examples=train,
        test_examples=test,
        n_events=10,
        n_states=6,
        event_names=["greeting", "query", "complaint", "answer", "clarify",
                     "apologize", "offer_solution", "accept", "escalate", "thanks", "hangup"],
        state_names=["init", "greeted", "query_phase", "complaint_phase",
                     "resolution_phase", "resolved", "escalated", "abandoned"],
        baseline_accuracy=0.15,
        target_accuracy=0.9
    )


def create_booking_domain() -> Domain:
    """Booking assistant dialogue."""
    train = [
        DomainExample(["greet", "select_date", "select_time", "confirm"], "booked"),
        DomainExample(["greet", "select_date", "cancel"], "cancelled"),
        DomainExample(["greet", "select_date", "select_time", "change_date", "select_time", "confirm"], "booked"),
        DomainExample(["greet", "check_availability"], "checking"),
    ]
    test = [
        DomainExample(["greet", "select_date", "check_availability", "confirm"], "booked"),
    ]

    return Domain(
        name="booking_assistant",
        domain_type=DomainType.DIALOGUE,
        difficulty=Difficulty.EASY,
        description="Booking assistant conversation",
        train_examples=train,
        test_examples=test,
        n_events=7,
        n_states=5,
        event_names=["greet", "select_date", "select_time", "confirm",
                     "cancel", "change_date", "check_availability"],
        state_names=["init", "date_selected", "time_selected", "booked", "cancelled", "checking"],
        baseline_accuracy=0.2,
        target_accuracy=1.0
    )


# =============================================================================
# Code Domains
# =============================================================================

def create_python_syntax_domain() -> Domain:
    """Python syntax state machine."""
    train = [
        DomainExample(["def", "identifier", "lparen", "rparen", "colon"], "func_body"),
        DomainExample(["class", "identifier", "colon"], "class_body"),
        DomainExample(["if", "expr", "colon"], "if_body"),
        DomainExample(["for", "identifier", "in", "expr", "colon"], "for_body"),
        DomainExample(["identifier", "equals", "expr"], "assignment"),
    ]
    test = [
        DomainExample(["while", "expr", "colon"], "while_body"),
        DomainExample(["return", "expr"], "return_stmt"),
    ]

    return Domain(
        name="python_syntax",
        domain_type=DomainType.CODE,
        difficulty=Difficulty.MEDIUM,
        description="Python syntax state machine",
        train_examples=train,
        test_examples=test,
        n_events=12,
        n_states=8,
        event_names=["def", "class", "if", "for", "while", "return",
                     "identifier", "lparen", "rparen", "colon", "in", "expr", "equals"],
        state_names=["start", "func_def", "class_def", "if_cond", "for_target",
                     "func_body", "class_body", "if_body", "for_body", "while_body",
                     "assignment", "return_stmt"],
        baseline_accuracy=0.1,
        target_accuracy=0.95
    )


def create_json_syntax_domain() -> Domain:
    """JSON syntax state machine."""
    train = [
        DomainExample(["lbrace", "string", "colon", "number", "rbrace"], "valid_object"),
        DomainExample(["lbracket", "number", "comma", "number", "rbracket"], "valid_array"),
        DomainExample(["lbrace", "rbrace"], "empty_object"),
        DomainExample(["lbracket", "rbracket"], "empty_array"),
        DomainExample(["lbrace", "string", "colon", "lbrace", "rbrace", "rbrace"], "nested_object"),
    ]
    test = [
        DomainExample(["lbracket", "string", "rbracket"], "valid_array"),
        DomainExample(["string"], "bare_value"),
    ]

    return Domain(
        name="json_syntax",
        domain_type=DomainType.CODE,
        difficulty=Difficulty.EASY,
        description="JSON syntax state machine",
        train_examples=train,
        test_examples=test,
        n_events=8,
        n_states=6,
        event_names=["lbrace", "rbrace", "lbracket", "rbracket",
                     "string", "number", "colon", "comma"],
        state_names=["start", "in_object", "in_array", "valid_object",
                     "valid_array", "empty_object", "empty_array", "nested_object", "bare_value"],
        baseline_accuracy=0.15,
        target_accuracy=1.0
    )


# =============================================================================
# Domain Registry
# =============================================================================

def get_all_domains() -> List[Domain]:
    """Get all available test domains."""
    return [
        # Games
        create_tictactoe_domain(),
        create_go_ko_domain(),
        create_zelda_module_domain(),
        # Regex
        create_email_regex_domain(),
        create_number_regex_domain(),
        # Dialogue
        create_customer_service_domain(),
        create_booking_domain(),
        # Code
        create_python_syntax_domain(),
        create_json_syntax_domain(),
    ]


def get_domains_by_type(domain_type: DomainType) -> List[Domain]:
    """Get domains of a specific type."""
    return [d for d in get_all_domains() if d.domain_type == domain_type]


def get_domains_by_difficulty(difficulty: Difficulty) -> List[Domain]:
    """Get domains of a specific difficulty."""
    return [d for d in get_all_domains() if d.difficulty == difficulty]


# =============================================================================
# Domain Suite
# =============================================================================

@dataclass
class DomainSuite:
    """Collection of domains for comprehensive testing."""
    name: str
    domains: List[Domain]
    description: str = ""

    def total_examples(self) -> int:
        return sum(len(d.train_examples) + len(d.test_examples) for d in self.domains)

    def by_type(self) -> Dict[DomainType, List[Domain]]:
        result = {}
        for d in self.domains:
            if d.domain_type not in result:
                result[d.domain_type] = []
            result[d.domain_type].append(d)
        return result


def create_full_suite() -> DomainSuite:
    """Create the full benchmark suite."""
    return DomainSuite(
        name="full_benchmark",
        domains=get_all_domains(),
        description="Complete benchmark across all domains"
    )


def create_quick_suite() -> DomainSuite:
    """Create a quick test suite (easy domains only)."""
    return DomainSuite(
        name="quick_test",
        domains=get_domains_by_difficulty(Difficulty.EASY),
        description="Quick test with easy domains"
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate domain suite."""
    print("=" * 60)
    print("DOMAIN SUITE DEMO")
    print("=" * 60)

    suite = create_full_suite()

    print(f"\nSuite: {suite.name}")
    print(f"Total domains: {len(suite.domains)}")
    print(f"Total examples: {suite.total_examples()}")

    print(f"\n{'Domain':<20} {'Type':<12} {'Difficulty':<10} {'Train':>6} {'Test':>6}")
    print("-" * 60)

    for domain in suite.domains:
        print(f"{domain.name:<20} {domain.domain_type.name:<12} "
              f"{domain.difficulty.name:<10} "
              f"{len(domain.train_examples):>6} {len(domain.test_examples):>6}")

    print(f"\n--- By Type ---")
    for dtype, domains in suite.by_type().items():
        print(f"{dtype.name}: {[d.name for d in domains]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
