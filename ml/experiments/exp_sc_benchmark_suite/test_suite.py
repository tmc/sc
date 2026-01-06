"""
Test Suite: 50+ Statechart Generation Test Cases.

Organized by complexity:
1. SIMPLE (15): 2-3 states, basic transitions
2. MEDIUM (20): 3-5 states, guards, hierarchy
3. COMPLEX (20): 5+ states, parallel, history, deep nesting

Each test case includes:
- Prompt: Natural language description
- Expected properties: min states, transitions, features
- Validation criteria
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from enum import Enum, auto


class Complexity(Enum):
    """Test case complexity level."""
    SIMPLE = auto()
    MEDIUM = auto()
    COMPLEX = auto()


class Feature(Enum):
    """Statechart features to test."""
    BASIC = auto()           # Simple states and transitions
    GUARDS = auto()          # Transition guards
    ACTIONS = auto()         # Entry/exit/transition actions
    HIERARCHY = auto()       # Nested states
    PARALLEL = auto()        # Orthogonal regions
    HISTORY = auto()         # History states
    INITIAL = auto()         # Initial states
    FINAL = auto()           # Final states
    SELF_LOOP = auto()       # Self-transitions
    MULTI_TARGET = auto()    # Multiple targets


@dataclass
class TestCase:
    """A single test case for benchmark."""
    id: str
    prompt: str
    complexity: Complexity
    min_states: int
    min_transitions: int
    expected_features: Set[Feature] = field(default_factory=set)
    expected_events: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)


# =============================================================================
# SIMPLE TEST CASES (15) - 2-3 states, basic transitions
# =============================================================================

SIMPLE_TESTS = [
    TestCase(
        id="simple_01",
        prompt="Create a simple on/off switch",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["TOGGLE", "TURN_ON", "TURN_OFF"],
        description="Basic toggle switch",
        tags=["toggle", "binary"],
    ),
    TestCase(
        id="simple_02",
        prompt="Create a light bulb state machine with on and off states",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["SWITCH"],
        description="Light bulb on/off",
        tags=["light", "binary"],
    ),
    TestCase(
        id="simple_03",
        prompt="Create a door that can be open or closed",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["OPEN", "CLOSE"],
        description="Simple door",
        tags=["door", "binary"],
    ),
    TestCase(
        id="simple_04",
        prompt="Create a pause/play button state machine",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["PLAY", "PAUSE"],
        description="Play/pause control",
        tags=["media", "binary"],
    ),
    TestCase(
        id="simple_05",
        prompt="Create a mute toggle for audio",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["MUTE", "UNMUTE", "TOGGLE"],
        description="Audio mute toggle",
        tags=["audio", "binary"],
    ),
    TestCase(
        id="simple_06",
        prompt="Create a lock/unlock mechanism",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["LOCK", "UNLOCK"],
        description="Lock mechanism",
        tags=["security", "binary"],
    ),
    TestCase(
        id="simple_07",
        prompt="Create a start/stop button",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["START", "STOP"],
        description="Start/stop control",
        tags=["control", "binary"],
    ),
    TestCase(
        id="simple_08",
        prompt="Create a traffic light with red, yellow, green states",
        complexity=Complexity.SIMPLE,
        min_states=3,
        min_transitions=3,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["TIMER", "NEXT"],
        description="Traffic light cycle",
        tags=["traffic", "cycle"],
    ),
    TestCase(
        id="simple_09",
        prompt="Create a simple fan with off, low, high speeds",
        complexity=Complexity.SIMPLE,
        min_states=3,
        min_transitions=3,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["SPEED_UP", "SPEED_DOWN", "TOGGLE"],
        description="Fan speed control",
        tags=["appliance", "cycle"],
    ),
    TestCase(
        id="simple_10",
        prompt="Create a washing machine with idle, washing, done states",
        complexity=Complexity.SIMPLE,
        min_states=3,
        min_transitions=3,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["START", "FINISH", "RESET"],
        description="Basic washing cycle",
        tags=["appliance", "sequence"],
    ),
    TestCase(
        id="simple_11",
        prompt="Create a checkbox with checked and unchecked states",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["TOGGLE", "CHECK", "UNCHECK"],
        description="Checkbox UI",
        tags=["ui", "binary"],
    ),
    TestCase(
        id="simple_12",
        prompt="Create a connection state: disconnected, connecting, connected",
        complexity=Complexity.SIMPLE,
        min_states=3,
        min_transitions=3,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["CONNECT", "CONNECTED", "DISCONNECT"],
        description="Connection states",
        tags=["network", "sequence"],
    ),
    TestCase(
        id="simple_13",
        prompt="Create a simple timer with idle and running states",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["START", "STOP", "TICK"],
        description="Simple timer",
        tags=["timer", "binary"],
    ),
    TestCase(
        id="simple_14",
        prompt="Create a visibility toggle: visible, hidden",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["SHOW", "HIDE", "TOGGLE"],
        description="Visibility control",
        tags=["ui", "binary"],
    ),
    TestCase(
        id="simple_15",
        prompt="Create a power button state machine",
        complexity=Complexity.SIMPLE,
        min_states=2,
        min_transitions=2,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["PRESS", "POWER"],
        description="Power button",
        tags=["hardware", "binary"],
    ),
]


# =============================================================================
# MEDIUM TEST CASES (20) - 3-5 states, guards, hierarchy
# =============================================================================

MEDIUM_TESTS = [
    TestCase(
        id="medium_01",
        prompt="Create a media player with idle, playing, paused, and stopped states",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["PLAY", "PAUSE", "STOP", "RESUME"],
        description="Media player controls",
        tags=["media", "player"],
    ),
    TestCase(
        id="medium_02",
        prompt="Create a door with locked, unlocked, open, and closed states",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=4,
        expected_features={Feature.BASIC, Feature.GUARDS},
        expected_events=["LOCK", "UNLOCK", "OPEN", "CLOSE"],
        description="Door with lock",
        tags=["security", "door"],
    ),
    TestCase(
        id="medium_03",
        prompt="Create a phone call state machine: idle, dialing, connected, on hold",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["DIAL", "CONNECT", "HOLD", "RESUME", "HANGUP"],
        description="Phone call states",
        tags=["telecom", "call"],
    ),
    TestCase(
        id="medium_04",
        prompt="Create an order status: pending, processing, shipped, delivered, cancelled",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL, Feature.FINAL},
        expected_events=["PROCESS", "SHIP", "DELIVER", "CANCEL"],
        description="Order lifecycle",
        tags=["ecommerce", "order"],
    ),
    TestCase(
        id="medium_05",
        prompt="Create a user authentication flow: logged_out, logging_in, logged_in, locked",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.GUARDS},
        expected_events=["LOGIN", "LOGOUT", "LOCK", "UNLOCK", "FAIL"],
        description="Auth flow",
        tags=["auth", "security"],
    ),
    TestCase(
        id="medium_06",
        prompt="Create a download manager: idle, downloading, paused, completed, failed",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.INITIAL, Feature.FINAL},
        expected_events=["START", "PAUSE", "RESUME", "COMPLETE", "FAIL", "RETRY"],
        description="Download states",
        tags=["download", "transfer"],
    ),
    TestCase(
        id="medium_07",
        prompt="Create a microwave with door open, door closed, cooking, done states",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.GUARDS},
        expected_events=["OPEN_DOOR", "CLOSE_DOOR", "START", "STOP", "TIMER_DONE"],
        description="Microwave control",
        tags=["appliance", "cooking"],
    ),
    TestCase(
        id="medium_08",
        prompt="Create a form validation state machine: empty, editing, validating, valid, invalid",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["TYPE", "BLUR", "VALIDATE", "CLEAR"],
        description="Form validation",
        tags=["form", "validation"],
    ),
    TestCase(
        id="medium_09",
        prompt="Create a printer queue: idle, receiving, printing, error, paper_jam",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["RECEIVE", "PRINT", "ERROR", "CLEAR_JAM", "RESET"],
        description="Printer states",
        tags=["printer", "queue"],
    ),
    TestCase(
        id="medium_10",
        prompt="Create an elevator with floors: ground, 1, 2, 3, and moving states",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=8,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["CALL", "ARRIVE", "MOVE_UP", "MOVE_DOWN"],
        description="Elevator control",
        tags=["elevator", "transport"],
    ),
    TestCase(
        id="medium_11",
        prompt="Create a game character with idle, walking, running, jumping states",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["WALK", "RUN", "JUMP", "LAND", "STOP"],
        description="Character movement",
        tags=["game", "character"],
    ),
    TestCase(
        id="medium_12",
        prompt="Create a meeting room booking: available, reserved, in_use, maintenance",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["RESERVE", "START", "END", "MAINTAIN", "RELEASE"],
        description="Room booking",
        tags=["booking", "resource"],
    ),
    TestCase(
        id="medium_13",
        prompt="Create a coffee machine: idle, brewing, ready, cleaning, error",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.INITIAL},
        expected_events=["BREW", "DONE", "TAKE", "CLEAN", "ERROR", "RESET"],
        description="Coffee machine",
        tags=["appliance", "beverage"],
    ),
    TestCase(
        id="medium_14",
        prompt="Create a subscription: trial, active, expired, cancelled, suspended",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.INITIAL, Feature.FINAL},
        expected_events=["ACTIVATE", "EXPIRE", "RENEW", "CANCEL", "SUSPEND"],
        description="Subscription lifecycle",
        tags=["billing", "subscription"],
    ),
    TestCase(
        id="medium_15",
        prompt="Create a nested state machine for a button with hover and pressed substates",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=4,
        expected_features={Feature.BASIC, Feature.HIERARCHY},
        expected_events=["MOUSE_ENTER", "MOUSE_LEAVE", "MOUSE_DOWN", "MOUSE_UP"],
        description="Button with hover",
        tags=["ui", "button", "hierarchy"],
    ),
    TestCase(
        id="medium_16",
        prompt="Create a wizard form with steps: info, details, review, submit, complete",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL, Feature.FINAL},
        expected_events=["NEXT", "BACK", "SUBMIT"],
        description="Wizard form",
        tags=["form", "wizard"],
    ),
    TestCase(
        id="medium_17",
        prompt="Create a document state: draft, review, approved, published, archived",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL, Feature.FINAL},
        expected_events=["SUBMIT", "APPROVE", "REJECT", "PUBLISH", "ARCHIVE"],
        description="Document workflow",
        tags=["document", "workflow"],
    ),
    TestCase(
        id="medium_18",
        prompt="Create a game level with intro, playing, boss_fight, victory, game_over",
        complexity=Complexity.MEDIUM,
        min_states=5,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.INITIAL, Feature.FINAL},
        expected_events=["START", "BOSS", "WIN", "LOSE", "RETRY"],
        description="Game level flow",
        tags=["game", "level"],
    ),
    TestCase(
        id="medium_19",
        prompt="Create a traffic light with pedestrian crossing button",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=5,
        expected_features={Feature.BASIC, Feature.GUARDS},
        expected_events=["TIMER", "PEDESTRIAN_BUTTON", "WALK", "DONT_WALK"],
        description="Traffic with pedestrian",
        tags=["traffic", "pedestrian"],
    ),
    TestCase(
        id="medium_20",
        prompt="Create a thermostat with off, heating, cooling, and idle states",
        complexity=Complexity.MEDIUM,
        min_states=4,
        min_transitions=6,
        expected_features={Feature.BASIC, Feature.GUARDS},
        expected_events=["TOO_COLD", "TOO_HOT", "TARGET_REACHED", "POWER"],
        description="Thermostat control",
        tags=["hvac", "temperature"],
    ),
]


# =============================================================================
# COMPLEX TEST CASES (20) - 5+ states, parallel, history, deep nesting
# =============================================================================

COMPLEX_TESTS = [
    TestCase(
        id="complex_01",
        prompt="Create a parallel state machine for a phone with call and data connections running simultaneously",
        complexity=Complexity.COMPLEX,
        min_states=6,
        min_transitions=8,
        expected_features={Feature.PARALLEL, Feature.INITIAL},
        expected_events=["CALL", "HANGUP", "CONNECT", "DISCONNECT"],
        description="Parallel call/data",
        tags=["telecom", "parallel"],
    ),
    TestCase(
        id="complex_02",
        prompt="Create a video player with history state that remembers playback position when interrupted",
        complexity=Complexity.COMPLEX,
        min_states=5,
        min_transitions=6,
        expected_features={Feature.HISTORY, Feature.INITIAL},
        expected_events=["PLAY", "PAUSE", "STOP", "INTERRUPT", "RESUME"],
        description="Video with history",
        tags=["media", "history"],
    ),
    TestCase(
        id="complex_03",
        prompt="Create a hierarchical ATM machine with authentication and transaction substates",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=10,
        expected_features={Feature.HIERARCHY, Feature.INITIAL, Feature.FINAL},
        expected_events=["INSERT_CARD", "ENTER_PIN", "WITHDRAW", "DEPOSIT", "EJECT", "CANCEL"],
        description="ATM hierarchy",
        tags=["banking", "hierarchy"],
    ),
    TestCase(
        id="complex_04",
        prompt="Create a smart home controller with parallel heating, lighting, and security regions",
        complexity=Complexity.COMPLEX,
        min_states=9,
        min_transitions=12,
        expected_features={Feature.PARALLEL, Feature.INITIAL},
        expected_events=["HEAT_ON", "HEAT_OFF", "LIGHT_ON", "LIGHT_OFF", "ARM", "DISARM"],
        description="Smart home parallel",
        tags=["smarthome", "parallel"],
    ),
    TestCase(
        id="complex_05",
        prompt="Create a workflow engine with nested approval states and rollback capability",
        complexity=Complexity.COMPLEX,
        min_states=7,
        min_transitions=10,
        expected_features={Feature.HIERARCHY, Feature.HISTORY},
        expected_events=["SUBMIT", "APPROVE", "REJECT", "ROLLBACK", "COMPLETE"],
        description="Workflow with rollback",
        tags=["workflow", "history"],
    ),
    TestCase(
        id="complex_06",
        prompt="Create a game character with combat substates: attacking, defending, dodging, and special moves",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=12,
        expected_features={Feature.HIERARCHY, Feature.INITIAL},
        expected_events=["ATTACK", "DEFEND", "DODGE", "SPECIAL", "HIT", "RECOVER"],
        description="Combat system",
        tags=["game", "combat", "hierarchy"],
    ),
    TestCase(
        id="complex_07",
        prompt="Create a robot vacuum with cleaning, charging, and error handling with deep history",
        complexity=Complexity.COMPLEX,
        min_states=7,
        min_transitions=10,
        expected_features={Feature.HIERARCHY, Feature.HISTORY},
        expected_events=["START", "DOCK", "LOW_BATTERY", "ERROR", "RESUME", "RESET"],
        description="Robot vacuum",
        tags=["robot", "history"],
    ),
    TestCase(
        id="complex_08",
        prompt="Create a checkout process with parallel payment and shipping configuration",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=10,
        expected_features={Feature.PARALLEL, Feature.INITIAL, Feature.FINAL},
        expected_events=["SELECT_PAYMENT", "SELECT_SHIPPING", "CONFIRM", "CANCEL"],
        description="Checkout parallel",
        tags=["ecommerce", "parallel"],
    ),
    TestCase(
        id="complex_09",
        prompt="Create an IDE with parallel editor, debugger, and terminal regions",
        complexity=Complexity.COMPLEX,
        min_states=9,
        min_transitions=12,
        expected_features={Feature.PARALLEL, Feature.INITIAL},
        expected_events=["OPEN", "CLOSE", "RUN", "DEBUG", "STOP", "TERMINAL"],
        description="IDE parallel regions",
        tags=["ide", "parallel"],
    ),
    TestCase(
        id="complex_10",
        prompt="Create a microservices circuit breaker with open, half-open, closed states and retry logic",
        complexity=Complexity.COMPLEX,
        min_states=5,
        min_transitions=8,
        expected_features={Feature.BASIC, Feature.GUARDS, Feature.SELF_LOOP},
        expected_events=["REQUEST", "SUCCESS", "FAILURE", "TIMEOUT", "RETRY"],
        description="Circuit breaker",
        tags=["microservices", "resilience"],
    ),
    TestCase(
        id="complex_11",
        prompt="Create a booking system with parallel seat selection and payment processing",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=10,
        expected_features={Feature.PARALLEL, Feature.INITIAL, Feature.FINAL},
        expected_events=["SELECT", "PAY", "CONFIRM", "CANCEL", "TIMEOUT"],
        description="Booking parallel",
        tags=["booking", "parallel"],
    ),
    TestCase(
        id="complex_12",
        prompt="Create a game save system with auto-save, manual save, and cloud sync states",
        complexity=Complexity.COMPLEX,
        min_states=6,
        min_transitions=8,
        expected_features={Feature.PARALLEL, Feature.INITIAL},
        expected_events=["AUTO_SAVE", "MANUAL_SAVE", "SYNC", "CONFLICT", "RESOLVE"],
        description="Save system",
        tags=["game", "save", "parallel"],
    ),
    TestCase(
        id="complex_13",
        prompt="Create a deployment pipeline with build, test, stage, and production environments",
        complexity=Complexity.COMPLEX,
        min_states=7,
        min_transitions=10,
        expected_features={Feature.HIERARCHY, Feature.INITIAL, Feature.FINAL},
        expected_events=["BUILD", "TEST", "DEPLOY", "ROLLBACK", "APPROVE"],
        description="Deployment pipeline",
        tags=["devops", "pipeline"],
    ),
    TestCase(
        id="complex_14",
        prompt="Create an authentication system with MFA, session management, and token refresh",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=12,
        expected_features={Feature.HIERARCHY, Feature.GUARDS},
        expected_events=["LOGIN", "MFA", "VERIFY", "REFRESH", "LOGOUT", "EXPIRE"],
        description="Auth with MFA",
        tags=["auth", "security", "hierarchy"],
    ),
    TestCase(
        id="complex_15",
        prompt="Create a file sync system with upload, download, conflict resolution, and offline mode",
        complexity=Complexity.COMPLEX,
        min_states=7,
        min_transitions=10,
        expected_features={Feature.HIERARCHY, Feature.HISTORY},
        expected_events=["UPLOAD", "DOWNLOAD", "CONFLICT", "RESOLVE", "OFFLINE", "ONLINE"],
        description="File sync",
        tags=["sync", "files", "history"],
    ),
    TestCase(
        id="complex_16",
        prompt="Create a multiplayer game lobby with matchmaking, team formation, and ready states",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=12,
        expected_features={Feature.HIERARCHY, Feature.PARALLEL},
        expected_events=["JOIN", "LEAVE", "READY", "UNREADY", "MATCH", "START"],
        description="Game lobby",
        tags=["game", "multiplayer", "hierarchy"],
    ),
    TestCase(
        id="complex_17",
        prompt="Create a vehicle control system with engine, transmission, and safety parallel regions",
        complexity=Complexity.COMPLEX,
        min_states=10,
        min_transitions=15,
        expected_features={Feature.PARALLEL, Feature.GUARDS},
        expected_events=["START", "STOP", "SHIFT", "BRAKE", "ACCELERATE", "EMERGENCY"],
        description="Vehicle control",
        tags=["vehicle", "parallel", "safety"],
    ),
    TestCase(
        id="complex_18",
        prompt="Create an order fulfillment with picking, packing, shipping, and return handling",
        complexity=Complexity.COMPLEX,
        min_states=8,
        min_transitions=12,
        expected_features={Feature.HIERARCHY, Feature.INITIAL, Feature.FINAL},
        expected_events=["PICK", "PACK", "SHIP", "DELIVER", "RETURN", "REFUND"],
        description="Order fulfillment",
        tags=["ecommerce", "fulfillment"],
    ),
    TestCase(
        id="complex_19",
        prompt="Create a chat application with connection, presence, and message states running in parallel",
        complexity=Complexity.COMPLEX,
        min_states=9,
        min_transitions=12,
        expected_features={Feature.PARALLEL, Feature.INITIAL},
        expected_events=["CONNECT", "DISCONNECT", "ONLINE", "AWAY", "SEND", "RECEIVE"],
        description="Chat parallel",
        tags=["chat", "parallel"],
    ),
    TestCase(
        id="complex_20",
        prompt="Create a deep hierarchical menu system with submenus, modals, and navigation history",
        complexity=Complexity.COMPLEX,
        min_states=10,
        min_transitions=15,
        expected_features={Feature.HIERARCHY, Feature.HISTORY},
        expected_events=["OPEN", "CLOSE", "SELECT", "BACK", "HOME"],
        description="Menu hierarchy",
        tags=["ui", "menu", "hierarchy", "history"],
    ),
]


@dataclass
class TestSuite:
    """Collection of test cases."""
    name: str
    tests: List[TestCase]
    description: str = ""

    @property
    def total(self) -> int:
        return len(self.tests)

    def by_complexity(self, complexity: Complexity) -> List[TestCase]:
        return [t for t in self.tests if t.complexity == complexity]

    def by_tag(self, tag: str) -> List[TestCase]:
        return [t for t in self.tests if tag in t.tags]

    def by_feature(self, feature: Feature) -> List[TestCase]:
        return [t for t in self.tests if feature in t.expected_features]


def get_test_suite() -> TestSuite:
    """Get the complete test suite with 55 test cases."""
    all_tests = SIMPLE_TESTS + MEDIUM_TESTS + COMPLEX_TESTS
    return TestSuite(
        name="SC Benchmark Suite",
        tests=all_tests,
        description="55 test cases: 15 simple, 20 medium, 20 complex"
    )


def test_suite():
    """Test the test suite."""
    print("=" * 60)
    print("Test Suite Validation")
    print("=" * 60)

    suite = get_test_suite()
    print(f"\nTotal tests: {suite.total}")
    print(f"  Simple: {len(suite.by_complexity(Complexity.SIMPLE))}")
    print(f"  Medium: {len(suite.by_complexity(Complexity.MEDIUM))}")
    print(f"  Complex: {len(suite.by_complexity(Complexity.COMPLEX))}")

    # Feature coverage
    print("\nFeature coverage:")
    for feature in Feature:
        count = len(suite.by_feature(feature))
        print(f"  {feature.name}: {count} tests")

    # Tag distribution
    all_tags = set()
    for test in suite.tests:
        all_tags.update(test.tags)
    print(f"\nUnique tags: {len(all_tags)}")
    print(f"  {sorted(all_tags)}")

    # Validate all tests have required fields
    print("\nValidation:")
    issues = []
    for test in suite.tests:
        if not test.prompt:
            issues.append(f"{test.id}: missing prompt")
        if test.min_states < 2:
            issues.append(f"{test.id}: min_states < 2")
        if test.min_transitions < 1:
            issues.append(f"{test.id}: min_transitions < 1")

    if issues:
        for issue in issues:
            print(f"  ERROR: {issue}")
    else:
        print("  All tests valid!")

    print("\n" + "=" * 60)
    print(f"Test suite ready: {suite.total} test cases")
    print("=" * 60)


if __name__ == "__main__":
    test_suite()
