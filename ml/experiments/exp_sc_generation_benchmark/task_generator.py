"""
Task Generator: 100-Task Benchmark Suite for Statechart Generation.

Categories:
1. Simple Toggle (20 tasks): 2-3 state machines
2. Hierarchical (20 tasks): Nested/composite states
3. Parallel Regions (20 tasks): Orthogonal states
4. Guards (20 tasks): Conditional transitions
5. History States (20 tasks): Shallow/deep history

Each task includes:
- prompt: Natural language description
- expected_properties: What the output should have
- difficulty: easy/medium/hard
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path


class TaskCategory(str, Enum):
    SIMPLE_TOGGLE = "simple_toggle"
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    GUARDS = "guards"
    HISTORY = "history"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ExpectedProperties:
    """Expected properties of generated statechart."""
    min_states: int = 2
    max_states: int = 10
    min_transitions: int = 1
    has_hierarchy: bool = False
    has_parallel: bool = False
    has_guards: bool = False
    has_history: bool = False
    required_states: List[str] = field(default_factory=list)
    required_events: List[str] = field(default_factory=list)


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    id: str
    category: TaskCategory
    difficulty: Difficulty
    prompt: str
    expected: ExpectedProperties
    hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "prompt": self.prompt,
            "expected": asdict(self.expected),
            "hints": self.hints,
        }


# =============================================================================
# SIMPLE TOGGLE TASKS (20)
# =============================================================================

SIMPLE_TOGGLE_TEMPLATES = [
    # Easy (7)
    ("toggle_switch", Difficulty.EASY,
     "Generate a statechart for a light switch with On and Off states",
     ExpectedProperties(min_states=2, max_states=3, required_states=["On", "Off"])),

    ("door_lock", Difficulty.EASY,
     "Create a statechart for a door lock with Locked and Unlocked states",
     ExpectedProperties(min_states=2, max_states=3, required_states=["Locked", "Unlocked"])),

    ("power_button", Difficulty.EASY,
     "Build a state machine for a power button (On/Off toggle)",
     ExpectedProperties(min_states=2, max_states=3)),

    ("pause_play", Difficulty.EASY,
     "Generate a statechart for media playback with Playing and Paused states",
     ExpectedProperties(min_states=2, max_states=4, required_states=["Playing", "Paused"])),

    ("mute_unmute", Difficulty.EASY,
     "Create a state machine for audio mute/unmute toggle",
     ExpectedProperties(min_states=2, max_states=3)),

    ("wifi_toggle", Difficulty.EASY,
     "Build a statechart for WiFi on/off toggle",
     ExpectedProperties(min_states=2, max_states=3)),

    ("dark_mode", Difficulty.EASY,
     "Generate a state machine for dark mode toggle (Light/Dark)",
     ExpectedProperties(min_states=2, max_states=3)),

    # Medium (7)
    ("traffic_light", Difficulty.MEDIUM,
     "Create a statechart for a traffic light with Red, Yellow, and Green states",
     ExpectedProperties(min_states=3, max_states=4, required_states=["Red", "Yellow", "Green"])),

    ("fan_speed", Difficulty.MEDIUM,
     "Build a state machine for a 3-speed fan (Off, Low, Medium, High)",
     ExpectedProperties(min_states=4, max_states=5)),

    ("washing_machine", Difficulty.MEDIUM,
     "Generate a statechart for a washing machine cycle (Idle, Wash, Rinse, Spin, Done)",
     ExpectedProperties(min_states=4, max_states=6)),

    ("microwave", Difficulty.MEDIUM,
     "Create a state machine for a microwave (Idle, Cooking, Paused, Done)",
     ExpectedProperties(min_states=3, max_states=5)),

    ("elevator_simple", Difficulty.MEDIUM,
     "Build a statechart for an elevator with Idle, MovingUp, MovingDown states",
     ExpectedProperties(min_states=3, max_states=5)),

    ("connection_status", Difficulty.MEDIUM,
     "Generate a state machine for connection status (Disconnected, Connecting, Connected)",
     ExpectedProperties(min_states=3, max_states=4)),

    ("auth_flow", Difficulty.MEDIUM,
     "Create a statechart for authentication flow (LoggedOut, LoggingIn, LoggedIn)",
     ExpectedProperties(min_states=3, max_states=5)),

    # Hard (6)
    ("order_processing", Difficulty.HARD,
     "Build a state machine for order processing (Pending, Processing, Shipped, Delivered, Cancelled)",
     ExpectedProperties(min_states=4, max_states=7)),

    ("document_workflow", Difficulty.HARD,
     "Generate a statechart for document workflow (Draft, Review, Approved, Published, Archived)",
     ExpectedProperties(min_states=4, max_states=7)),

    ("build_pipeline", Difficulty.HARD,
     "Create a state machine for CI/CD pipeline (Queued, Building, Testing, Deploying, Success, Failed)",
     ExpectedProperties(min_states=5, max_states=8)),

    ("payment_flow", Difficulty.HARD,
     "Build a statechart for payment processing (Initiated, Validating, Processing, Completed, Failed, Refunded)",
     ExpectedProperties(min_states=5, max_states=8)),

    ("interview_process", Difficulty.HARD,
     "Generate a state machine for interview process (Applied, Screening, Interview, Offer, Hired, Rejected)",
     ExpectedProperties(min_states=5, max_states=8)),

    ("ticket_lifecycle", Difficulty.HARD,
     "Create a statechart for support ticket lifecycle (New, Assigned, InProgress, Resolved, Closed, Reopened)",
     ExpectedProperties(min_states=5, max_states=8)),
]


# =============================================================================
# HIERARCHICAL TASKS (20)
# =============================================================================

HIERARCHICAL_TEMPLATES = [
    # Easy (7)
    ("nested_player", Difficulty.EASY,
     "Generate a hierarchical statechart for a media player with Playing state containing Normal and FastForward substates",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    ("mode_submodes", Difficulty.EASY,
     "Create a statechart with Active state containing Ready and Busy substates",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    ("connection_retry", Difficulty.EASY,
     "Build a hierarchical state machine with Connected state having Stable and Reconnecting substates",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    ("editing_states", Difficulty.EASY,
     "Generate a statechart with Editing composite state containing Insert and Overwrite modes",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    ("power_modes", Difficulty.EASY,
     "Create a state machine with On state containing Normal and PowerSave substates",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    ("view_modes", Difficulty.EASY,
     "Build a statechart with Viewing state containing List and Grid substates",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    ("auth_substates", Difficulty.EASY,
     "Generate a hierarchical state machine with Authenticating state containing Username and Password substates",
     ExpectedProperties(min_states=4, max_states=6, has_hierarchy=True)),

    # Medium (7)
    ("game_character", Difficulty.MEDIUM,
     "Create a hierarchical statechart for a game character with Alive state containing Idle, Walking, Running, and Combat substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    ("form_wizard", Difficulty.MEDIUM,
     "Build a state machine for a form wizard with Filling state containing Step1, Step2, Step3 substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    ("device_modes", Difficulty.MEDIUM,
     "Generate a hierarchical statechart for a device with Active mode containing Normal, Silent, and DoNotDisturb substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    ("order_states", Difficulty.MEDIUM,
     "Create a state machine with Processing composite state containing Validating, Charging, and Fulfilling substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    ("navigation_app", Difficulty.MEDIUM,
     "Build a hierarchical statechart for navigation with Navigating state containing Driving, Walking, Transit substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    ("call_handling", Difficulty.MEDIUM,
     "Generate a state machine for phone calls with InCall state containing Active, Hold, Muted substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    ("editor_modes", Difficulty.MEDIUM,
     "Create a hierarchical statechart for editor with EditMode containing Visual, Command, and Insert substates",
     ExpectedProperties(min_states=5, max_states=8, has_hierarchy=True)),

    # Hard (6)
    ("deep_nested", Difficulty.HARD,
     "Build a statechart with 3 levels of nesting: App > Module > Feature substates",
     ExpectedProperties(min_states=6, max_states=10, has_hierarchy=True)),

    ("game_state_machine", Difficulty.HARD,
     "Generate a hierarchical game state machine with Playing containing Level (Tutorial, Stage1, Boss) substates",
     ExpectedProperties(min_states=7, max_states=12, has_hierarchy=True)),

    ("workflow_engine", Difficulty.HARD,
     "Create a state machine for workflow engine with Executing composite containing Initialize, Process, Finalize substates each with substates",
     ExpectedProperties(min_states=8, max_states=15, has_hierarchy=True)),

    ("ui_navigation", Difficulty.HARD,
     "Build a hierarchical statechart for UI navigation with tabs, each tab containing multiple views",
     ExpectedProperties(min_states=8, max_states=15, has_hierarchy=True)),

    ("device_lifecycle", Difficulty.HARD,
     "Generate a state machine for device lifecycle with Operating state containing Boot, Running (Normal, Diagnostic, Recovery), Shutdown substates",
     ExpectedProperties(min_states=8, max_states=15, has_hierarchy=True)),

    ("transaction_flow", Difficulty.HARD,
     "Create a hierarchical statechart for transaction with Active containing Prepare (Validate, Lock), Execute (Apply, Commit), Cleanup substates",
     ExpectedProperties(min_states=8, max_states=15, has_hierarchy=True)),
]


# =============================================================================
# PARALLEL REGION TASKS (20)
# =============================================================================

PARALLEL_TEMPLATES = [
    # Easy (7)
    ("av_player", Difficulty.EASY,
     "Generate a statechart with parallel Audio and Video regions, each with Playing/Paused states",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    ("alarm_clock", Difficulty.EASY,
     "Create a parallel statechart with Time display and Alarm regions running concurrently",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    ("keyboard_mouse", Difficulty.EASY,
     "Build a state machine with parallel Keyboard and Mouse input regions",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    ("network_status", Difficulty.EASY,
     "Generate a statechart with parallel WiFi and Cellular connection regions",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    ("cpu_gpu", Difficulty.EASY,
     "Create a parallel state machine for CPU and GPU operation states",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    ("mic_camera", Difficulty.EASY,
     "Build a statechart with parallel Microphone and Camera permission states",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    ("read_write", Difficulty.EASY,
     "Generate a state machine with parallel Read and Write operation regions",
     ExpectedProperties(min_states=4, max_states=8, has_parallel=True)),

    # Medium (7)
    ("vehicle_systems", Difficulty.MEDIUM,
     "Create a statechart with parallel Engine, Transmission, and Brakes regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    ("media_center", Difficulty.MEDIUM,
     "Build a parallel state machine for media center with Audio, Video, and Subtitle regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    ("robot_control", Difficulty.MEDIUM,
     "Generate a statechart for robot with parallel Motion, Sensors, and Communication regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    ("smart_home", Difficulty.MEDIUM,
     "Create a parallel statechart for smart home with Lighting, Climate, and Security regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    ("game_systems", Difficulty.MEDIUM,
     "Build a state machine for game with parallel Physics, Rendering, and Audio regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    ("db_connection", Difficulty.MEDIUM,
     "Generate a statechart with parallel Connection pool, Query execution, and Transaction regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    ("file_sync", Difficulty.MEDIUM,
     "Create a parallel state machine for file sync with Upload, Download, and Conflict regions",
     ExpectedProperties(min_states=6, max_states=12, has_parallel=True)),

    # Hard (6)
    ("drone_control", Difficulty.HARD,
     "Build a statechart for drone with parallel Flight, Navigation, Camera, and Battery regions",
     ExpectedProperties(min_states=8, max_states=16, has_parallel=True)),

    ("os_kernel", Difficulty.HARD,
     "Generate a parallel state machine for OS kernel with Process, Memory, IO, and Scheduler regions",
     ExpectedProperties(min_states=8, max_states=16, has_parallel=True)),

    ("iot_device", Difficulty.HARD,
     "Create a statechart for IoT device with parallel Sensors, Actuators, Network, and Power regions",
     ExpectedProperties(min_states=8, max_states=16, has_parallel=True)),

    ("trading_system", Difficulty.HARD,
     "Build a parallel state machine for trading with Market data, Order management, Risk, and Reporting regions",
     ExpectedProperties(min_states=8, max_states=16, has_parallel=True)),

    ("video_conference", Difficulty.HARD,
     "Generate a statechart for video conference with parallel Audio, Video, Screen share, Chat, and Reactions regions",
     ExpectedProperties(min_states=10, max_states=20, has_parallel=True)),

    ("game_engine", Difficulty.HARD,
     "Create a parallel state machine for game engine with Input, Physics, Rendering, Audio, and Network regions each with substates",
     ExpectedProperties(min_states=10, max_states=25, has_parallel=True, has_hierarchy=True)),
]


# =============================================================================
# GUARD TASKS (20)
# =============================================================================

GUARD_TEMPLATES = [
    # Easy (7)
    ("login_attempts", Difficulty.EASY,
     "Generate a statechart for login with guard [attempts < 3] on retry transition",
     ExpectedProperties(min_states=3, max_states=5, has_guards=True)),

    ("age_gate", Difficulty.EASY,
     "Create a state machine with guard [age >= 18] for content access",
     ExpectedProperties(min_states=2, max_states=4, has_guards=True)),

    ("stock_check", Difficulty.EASY,
     "Build a statechart for ordering with guard [stock > 0] on purchase transition",
     ExpectedProperties(min_states=3, max_states=5, has_guards=True)),

    ("balance_check", Difficulty.EASY,
     "Generate a state machine for ATM with guard [balance >= amount] on withdraw",
     ExpectedProperties(min_states=3, max_states=5, has_guards=True)),

    ("permission_check", Difficulty.EASY,
     "Create a statechart with guard [hasPermission] on action transition",
     ExpectedProperties(min_states=3, max_states=5, has_guards=True)),

    ("temperature_guard", Difficulty.EASY,
     "Build a state machine for thermostat with guard [temp > threshold] on heating/cooling transitions",
     ExpectedProperties(min_states=3, max_states=5, has_guards=True)),

    ("size_limit", Difficulty.EASY,
     "Generate a statechart for upload with guard [size < maxSize] on upload transition",
     ExpectedProperties(min_states=3, max_states=5, has_guards=True)),

    # Medium (7)
    ("rate_limiter", Difficulty.MEDIUM,
     "Create a state machine for API with guard [requests < limit && !blocked] on request transition",
     ExpectedProperties(min_states=4, max_states=6, has_guards=True)),

    ("multi_factor", Difficulty.MEDIUM,
     "Build a statechart for 2FA with guards [passwordValid] and [codeValid] on sequential transitions",
     ExpectedProperties(min_states=4, max_states=7, has_guards=True)),

    ("shopping_cart", Difficulty.MEDIUM,
     "Generate a state machine for checkout with guards [cartNotEmpty], [addressValid], [paymentValid]",
     ExpectedProperties(min_states=4, max_states=7, has_guards=True)),

    ("workflow_approval", Difficulty.MEDIUM,
     "Create a statechart for document approval with guards [level1Approved], [level2Approved]",
     ExpectedProperties(min_states=4, max_states=7, has_guards=True)),

    ("resource_allocation", Difficulty.MEDIUM,
     "Build a state machine with guards [cpuAvailable && memoryAvailable] on allocation transition",
     ExpectedProperties(min_states=4, max_states=6, has_guards=True)),

    ("priority_queue", Difficulty.MEDIUM,
     "Generate a statechart for task processing with guards [isHighPriority], [isMediumPriority]",
     ExpectedProperties(min_states=4, max_states=7, has_guards=True)),

    ("subscription_tiers", Difficulty.MEDIUM,
     "Create a state machine with guards [tier == 'premium'], [tier == 'pro'] for feature access",
     ExpectedProperties(min_states=4, max_states=7, has_guards=True)),

    # Hard (6)
    ("complex_auth", Difficulty.HARD,
     "Build a statechart for authentication with guards combining [credentialsValid && !accountLocked && mfaVerified]",
     ExpectedProperties(min_states=5, max_states=9, has_guards=True)),

    ("auction_system", Difficulty.HARD,
     "Generate a state machine for auction with guards [bid > currentBid], [time < deadline], [bidder != owner]",
     ExpectedProperties(min_states=5, max_states=9, has_guards=True)),

    ("booking_system", Difficulty.HARD,
     "Create a statechart for booking with guards [slotAvailable], [paymentSuccessful], [noConflicts]",
     ExpectedProperties(min_states=5, max_states=9, has_guards=True)),

    ("trade_execution", Difficulty.HARD,
     "Build a state machine for trading with guards [marketOpen], [sufficientFunds], [riskWithinLimits], [orderValid]",
     ExpectedProperties(min_states=6, max_states=10, has_guards=True)),

    ("deployment_pipeline", Difficulty.HARD,
     "Generate a statechart for deployment with guards [testsPass], [approvalGranted], [targetHealthy], [rollbackAvailable]",
     ExpectedProperties(min_states=6, max_states=10, has_guards=True)),

    ("access_control", Difficulty.HARD,
     "Create a state machine for access control with guards combining role, permission, time window, and IP whitelist checks",
     ExpectedProperties(min_states=6, max_states=10, has_guards=True)),
]


# =============================================================================
# HISTORY STATE TASKS (20)
# =============================================================================

HISTORY_TEMPLATES = [
    # Easy (7)
    ("player_resume", Difficulty.EASY,
     "Generate a statechart for media player with shallow history to resume play position",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    ("editor_undo", Difficulty.EASY,
     "Create a state machine with history state for returning to previous editing mode",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    ("tab_memory", Difficulty.EASY,
     "Build a statechart that remembers last active tab using shallow history",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    ("mode_restore", Difficulty.EASY,
     "Generate a state machine that restores previous mode after interruption using history",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    ("view_history", Difficulty.EASY,
     "Create a statechart for app with history state to return to last viewed screen",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    ("call_hold", Difficulty.EASY,
     "Build a state machine for phone call that remembers state before hold using history",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    ("game_pause", Difficulty.EASY,
     "Generate a statechart for game that returns to exact state after pause using history",
     ExpectedProperties(min_states=4, max_states=6, has_history=True, has_hierarchy=True)),

    # Medium (7)
    ("wizard_back", Difficulty.MEDIUM,
     "Create a state machine for wizard with shallow history for back navigation",
     ExpectedProperties(min_states=5, max_states=8, has_history=True, has_hierarchy=True)),

    ("session_restore", Difficulty.MEDIUM,
     "Build a statechart that uses deep history to restore full session state after timeout",
     ExpectedProperties(min_states=5, max_states=9, has_history=True, has_hierarchy=True)),

    ("interrupt_resume", Difficulty.MEDIUM,
     "Generate a state machine that handles interrupts and resumes via deep history",
     ExpectedProperties(min_states=5, max_states=9, has_history=True, has_hierarchy=True)),

    ("error_recovery", Difficulty.MEDIUM,
     "Create a statechart with history for recovering to last good state after error",
     ExpectedProperties(min_states=5, max_states=8, has_history=True, has_hierarchy=True)),

    ("navigation_history", Difficulty.MEDIUM,
     "Build a state machine for navigation with both shallow and deep history options",
     ExpectedProperties(min_states=6, max_states=10, has_history=True, has_hierarchy=True)),

    ("workflow_suspend", Difficulty.MEDIUM,
     "Generate a statechart for workflow that can suspend and resume with history",
     ExpectedProperties(min_states=5, max_states=9, has_history=True, has_hierarchy=True)),

    ("power_save_resume", Difficulty.MEDIUM,
     "Create a state machine for device that saves full state with deep history before sleep",
     ExpectedProperties(min_states=5, max_states=9, has_history=True, has_hierarchy=True)),

    # Hard (6)
    ("deep_nested_history", Difficulty.HARD,
     "Build a statechart with 3-level nesting and deep history that preserves full substate configuration",
     ExpectedProperties(min_states=8, max_states=15, has_history=True, has_hierarchy=True)),

    ("transaction_checkpoint", Difficulty.HARD,
     "Generate a state machine for transactions with checkpoints using history for rollback",
     ExpectedProperties(min_states=7, max_states=12, has_history=True, has_hierarchy=True)),

    ("parallel_history", Difficulty.HARD,
     "Create a statechart with parallel regions each having their own history states",
     ExpectedProperties(min_states=8, max_states=16, has_history=True, has_hierarchy=True, has_parallel=True)),

    ("multi_undo", Difficulty.HARD,
     "Build a state machine with multiple history levels for undo/redo functionality",
     ExpectedProperties(min_states=7, max_states=12, has_history=True, has_hierarchy=True)),

    ("context_switch", Difficulty.HARD,
     "Generate a statechart for multi-context system with per-context deep history",
     ExpectedProperties(min_states=8, max_states=14, has_history=True, has_hierarchy=True)),

    ("game_checkpoint", Difficulty.HARD,
     "Create a state machine for game save system with deep history across multiple game substates",
     ExpectedProperties(min_states=8, max_states=16, has_history=True, has_hierarchy=True)),
]


# =============================================================================
# TASK GENERATOR
# =============================================================================

def generate_tasks(category: TaskCategory) -> List[BenchmarkTask]:
    """Generate all tasks for a category."""
    templates = {
        TaskCategory.SIMPLE_TOGGLE: SIMPLE_TOGGLE_TEMPLATES,
        TaskCategory.HIERARCHICAL: HIERARCHICAL_TEMPLATES,
        TaskCategory.PARALLEL: PARALLEL_TEMPLATES,
        TaskCategory.GUARDS: GUARD_TEMPLATES,
        TaskCategory.HISTORY: HISTORY_TEMPLATES,
    }

    tasks = []
    for i, (name, difficulty, prompt, expected) in enumerate(templates[category]):
        task = BenchmarkTask(
            id=f"{category.value}_{i:02d}_{name}",
            category=category,
            difficulty=difficulty,
            prompt=prompt,
            expected=expected,
        )
        tasks.append(task)

    return tasks


def generate_all_tasks() -> List[BenchmarkTask]:
    """Generate all 100 benchmark tasks."""
    all_tasks = []
    for category in TaskCategory:
        tasks = generate_tasks(category)
        all_tasks.extend(tasks)
    return all_tasks


def save_tasks_to_file(tasks: List[BenchmarkTask], filepath: Path):
    """Save tasks to JSON file."""
    data = {
        "version": "1.0",
        "total_tasks": len(tasks),
        "categories": {cat.value: 0 for cat in TaskCategory},
        "tasks": [task.to_dict() for task in tasks],
    }

    for task in tasks:
        data["categories"][task.category.value] += 1

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_tasks_from_file(filepath: Path) -> List[BenchmarkTask]:
    """Load tasks from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    tasks = []
    for t in data["tasks"]:
        task = BenchmarkTask(
            id=t["id"],
            category=TaskCategory(t["category"]),
            difficulty=Difficulty(t["difficulty"]),
            prompt=t["prompt"],
            expected=ExpectedProperties(**t["expected"]),
            hints=t.get("hints", []),
        )
        tasks.append(task)

    return tasks


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate and save all benchmark tasks."""
    tasks = generate_all_tasks()

    print(f"Generated {len(tasks)} benchmark tasks:")
    for category in TaskCategory:
        count = sum(1 for t in tasks if t.category == category)
        print(f"  {category.value}: {count} tasks")

    # Save to file
    output_dir = Path(__file__).parent / "tasks"
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / "benchmark_tasks.json"
    save_tasks_to_file(tasks, filepath)
    print(f"\nSaved to {filepath}")

    # Also save per-category files
    for category in TaskCategory:
        cat_tasks = [t for t in tasks if t.category == category]
        cat_filepath = output_dir / f"{category.value}_tasks.json"
        save_tasks_to_file(cat_tasks, cat_filepath)
        print(f"Saved {category.value} to {cat_filepath}")


if __name__ == "__main__":
    main()
