"""
Generation Demo for Live Statechart Visualization.

Demonstrates token-by-token statechart generation with live
visualization updates via WebSocket.

Components:
1. MockLLM: Simulates token-by-token generation
2. LiveGenerator: Coordinates generation with visualization
3. GenerationDemo: Main demo runner

Usage:
    from ml.experiments.exp_live_generation_viz import run_demo
    run_demo("Create a traffic light state machine")
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto

from .partial_parser import PartialParser, PartialParseResult, ParseState

# Try to import visualizer bridge
try:
    import sys
    sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')
    from utils.visualizer_bridge import VisualizerBridge, TokenUpdate, CompletionUpdate
    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False

# Use wrapper dataclasses for internal use that map to bridge format
@dataclass
class LocalTokenUpdate:
    """Local token update with states_found for easy access."""
    token: str
    token_index: int
    partial_json: Optional[Dict] = None
    states_found: List[str] = field(default_factory=list)
    parse_success: bool = False


@dataclass
class LocalCompletionUpdate:
    """Local completion update."""
    statechart: Dict
    total_tokens: int
    generation_time: float
    validation_passed: bool


class GenerationSpeed(Enum):
    """Speed presets for token generation."""
    SLOW = auto()      # 200ms per token
    MEDIUM = auto()    # 50ms per token
    FAST = auto()      # 10ms per token
    INSTANT = auto()   # No delay


@dataclass
class DemoConfig:
    """Configuration for generation demo."""
    speed: GenerationSpeed = GenerationSpeed.MEDIUM
    ws_url: str = "ws://localhost:8080/ws"
    show_partial: bool = True
    validate_on_complete: bool = True
    use_mock_llm: bool = True
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


class MockLLM:
    """
    Mock LLM for testing token-by-token generation.

    Provides controllable token streams for testing visualization
    without requiring actual LLM inference.
    """

    # Predefined statechart templates
    TEMPLATES = {
        "traffic_light": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Yellow", "type": 1},
                    {"label": "Green", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
                {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
                {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"}
            ]
        },
        "toggle": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
            ]
        },
        "player": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Playing", "type": 1},
                    {"label": "Paused", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Playing"], "event": "RESUME"},
                {"from": ["Playing"], "to": ["Idle"], "event": "STOP"},
                {"from": ["Paused"], "to": ["Idle"], "event": "STOP"}
            ]
        }
    }

    def __init__(self, template_name: str = "traffic_light"):
        self.template_name = template_name
        self.tokens: List[str] = []
        self._prepare_tokens()

    def _prepare_tokens(self):
        """Prepare token stream from template."""
        template = self.TEMPLATES.get(self.template_name, self.TEMPLATES["toggle"])
        json_str = json.dumps(template, indent=2)
        # Tokenize by chunks that simulate LLM output
        self.tokens = self._tokenize(json_str)

    def _tokenize(self, text: str) -> List[str]:
        """
        Split JSON into tokens simulating LLM generation.

        Produces chunks that mimic how an LLM generates JSON:
        - Opening brackets alone
        - Keys with colons
        - Values
        - Closing brackets
        """
        tokens = []
        current = ""
        i = 0

        while i < len(text):
            char = text[i]

            if char in '{}[]':
                if current.strip():
                    tokens.append(current)
                    current = ""
                tokens.append(char)
            elif char == ':':
                current += char
                tokens.append(current.strip())
                current = ""
            elif char == ',':
                if current.strip():
                    tokens.append(current.strip())
                    current = ""
                tokens.append(char)
            elif char == '"':
                # Collect full string
                j = i + 1
                while j < len(text) and text[j] != '"':
                    if text[j] == '\\':
                        j += 2
                    else:
                        j += 1
                current += text[i:j+1]
                i = j
            elif char in ' \n\t':
                if current and not current.isspace():
                    pass  # Keep building
            else:
                current += char

            i += 1

        if current.strip():
            tokens.append(current.strip())

        return [t for t in tokens if t.strip()]

    async def generate_tokens(
        self,
        prompt: str,
        delay: float = 0.05
    ) -> AsyncIterator[str]:
        """
        Generate tokens one at a time.

        Args:
            prompt: Input prompt (used for template selection)
            delay: Delay between tokens in seconds

        Yields:
            Individual tokens
        """
        # Select template based on prompt keywords
        if "traffic" in prompt.lower() or "light" in prompt.lower():
            self.template_name = "traffic_light"
        elif "toggle" in prompt.lower() or "switch" in prompt.lower():
            self.template_name = "toggle"
        elif "player" in prompt.lower() or "media" in prompt.lower():
            self.template_name = "player"

        self._prepare_tokens()

        for token in self.tokens:
            if delay > 0:
                await asyncio.sleep(delay)
            yield token

    def get_full_output(self) -> str:
        """Get complete output without streaming."""
        return "".join(self.tokens)


class LiveGenerator:
    """
    Coordinates token generation with live visualization.

    Integrates:
    - Token-by-token LLM output
    - Partial JSON parsing
    - WebSocket visualization updates
    """

    def __init__(
        self,
        config: DemoConfig = None,
        bridge: VisualizerBridge = None
    ):
        self.config = config or DemoConfig()
        self.bridge = bridge or VisualizerBridge(self.config.ws_url)
        self.parser = PartialParser()
        self.llm: Optional[MockLLM] = None

        # Metrics
        self.tokens_generated = 0
        self.parse_attempts = 0
        self.successful_parses = 0
        self.start_time = 0.0

    def _get_delay(self) -> float:
        """Get delay based on speed setting."""
        delays = {
            GenerationSpeed.SLOW: 0.2,
            GenerationSpeed.MEDIUM: 0.05,
            GenerationSpeed.FAST: 0.01,
            GenerationSpeed.INSTANT: 0.0,
        }
        return delays.get(self.config.speed, 0.05)

    async def generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[LocalTokenUpdate], None]] = None,
        on_complete: Optional[Callable[[LocalCompletionUpdate], None]] = None
    ) -> Dict:
        """
        Generate statechart with live updates.

        Args:
            prompt: Description of statechart to generate
            on_token: Callback for each token
            on_complete: Callback on completion

        Returns:
            Generated statechart
        """
        self.start_time = time.time()
        self.tokens_generated = 0
        self.parse_attempts = 0
        self.successful_parses = 0

        # Initialize LLM
        self.llm = MockLLM()

        # Connect to visualizer
        if HAS_BRIDGE and self.bridge:
            try:
                await self.bridge.connect()
            except Exception:
                pass  # Continue without visualization

        # Accumulate output
        accumulated = ""
        delay = self._get_delay()

        async for token in self.llm.generate_tokens(prompt, delay):
            accumulated += token
            self.tokens_generated += 1

            # Parse partial result
            self.parse_attempts += 1
            parse_result = self.parser.parse(accumulated)

            if parse_result.success:
                self.successful_parses += 1

            # Create local update for callbacks
            local_update = LocalTokenUpdate(
                token=token,
                token_index=self.tokens_generated,
                partial_json=parse_result.partial_json,
                states_found=parse_result.states_found,
                parse_success=parse_result.success
            )

            # Send update to bridge using correct format
            if HAS_BRIDGE and self.bridge and self.bridge.connected:
                try:
                    self.bridge.on_token(
                        token=token,
                        partial_statechart=parse_result.partial_json,
                        token_id=self.tokens_generated
                    )
                except Exception:
                    pass

            if on_token:
                on_token(local_update)

        # Final parse
        final_result = self.parser.parse(accumulated)
        generation_time = time.time() - self.start_time

        # Create local completion update
        local_completion = LocalCompletionUpdate(
            statechart=final_result.partial_json or {},
            total_tokens=self.tokens_generated,
            generation_time=generation_time,
            validation_passed=final_result.success and final_result.has_root
        )

        # Send completion to bridge
        if HAS_BRIDGE and self.bridge and self.bridge.connected:
            try:
                self.bridge.on_complete(
                    statechart=final_result.partial_json or {}
                )
            except Exception:
                pass

        if on_complete:
            on_complete(local_completion)

        # Disconnect
        if HAS_BRIDGE and self.bridge:
            try:
                await self.bridge.disconnect()
            except Exception:
                pass

        return final_result.partial_json or {}

    def get_metrics(self) -> Dict:
        """Get generation metrics."""
        return {
            "tokens_generated": self.tokens_generated,
            "parse_attempts": self.parse_attempts,
            "successful_parses": self.successful_parses,
            "parse_success_rate": self.successful_parses / max(1, self.parse_attempts),
            "generation_time": time.time() - self.start_time if self.start_time else 0,
        }


class GenerationDemo:
    """
    Main demo class for live statechart generation.

    Shows token-by-token generation with partial parsing
    and visualization updates.
    """

    def __init__(self, config: DemoConfig = None):
        self.config = config or DemoConfig()
        self.generator = LiveGenerator(self.config)
        self.results: List[Dict] = []

    async def run_async(self, prompt: str, verbose: bool = True) -> Dict:
        """
        Run demo asynchronously.

        Args:
            prompt: Description of statechart
            verbose: Print progress

        Returns:
            Generated statechart
        """
        if verbose:
            print("=" * 60)
            print("LIVE GENERATION DEMO")
            print("=" * 60)
            print(f"Prompt: {prompt}")
            print(f"Speed: {self.config.speed.name}")
            print("-" * 60)

        tokens_shown = []

        def on_token(update: LocalTokenUpdate):
            if verbose:
                tokens_shown.append(update.token)
                status = "OK" if update.parse_success else "..."
                states = ", ".join(update.states_found) if update.states_found else "-"
                print(f"  [{update.token_index:3d}] {update.token:20s} | {status:3s} | States: {states}")

        def on_complete(update: LocalCompletionUpdate):
            if verbose:
                print("-" * 60)
                print(f"COMPLETE: {update.total_tokens} tokens in {update.generation_time:.2f}s")
                print(f"Validation: {'PASSED' if update.validation_passed else 'FAILED'}")

        result = await self.generator.generate(
            prompt,
            on_token=on_token,
            on_complete=on_complete
        )

        if verbose:
            metrics = self.generator.get_metrics()
            print(f"\nMetrics:")
            print(f"  Parse success rate: {metrics['parse_success_rate']:.1%}")
            print(f"  States found: {len(result.get('root_state', {}).get('children', []))}")
            print(f"  Transitions: {len(result.get('transitions', []))}")
            print("=" * 60)

        self.results.append({
            "prompt": prompt,
            "result": result,
            "metrics": self.generator.get_metrics()
        })

        return result

    def run(self, prompt: str, verbose: bool = True) -> Dict:
        """
        Run demo synchronously.

        Args:
            prompt: Description of statechart
            verbose: Print progress

        Returns:
            Generated statechart
        """
        return asyncio.run(self.run_async(prompt, verbose))


def run_demo(prompt: str = "Create a traffic light state machine") -> Dict:
    """
    Convenience function to run demo.

    Args:
        prompt: Description of statechart

    Returns:
        Generated statechart
    """
    config = DemoConfig(speed=GenerationSpeed.MEDIUM)
    demo = GenerationDemo(config)
    return demo.run(prompt)


def test_generation_demo():
    """Test the generation demo."""
    print("=" * 60)
    print("Testing Generation Demo")
    print("=" * 60)

    # Test 1: Basic generation
    print("\n1. Traffic light generation:")
    config = DemoConfig(speed=GenerationSpeed.FAST)
    demo = GenerationDemo(config)
    result = demo.run("Create a traffic light")

    assert result is not None
    assert "root_state" in result
    print(f"   States: {[c['label'] for c in result['root_state'].get('children', [])]}")

    # Test 2: Different template
    print("\n2. Toggle switch generation:")
    result2 = demo.run("Create a toggle switch")
    assert "root_state" in result2

    # Test 3: Player template
    print("\n3. Media player generation:")
    result3 = demo.run("Create a media player")
    assert "root_state" in result3

    # Test 4: Mock LLM directly
    print("\n4. Mock LLM token stream:")
    llm = MockLLM("toggle")
    tokens = []

    async def collect():
        async for t in llm.generate_tokens("", delay=0):
            tokens.append(t)

    asyncio.run(collect())
    print(f"   Tokens: {len(tokens)}")
    print(f"   Sample: {tokens[:5]}...")

    # Test 5: Metrics
    print("\n5. Metrics check:")
    metrics = demo.generator.get_metrics()
    print(f"   Tokens: {metrics['tokens_generated']}")
    print(f"   Parse rate: {metrics['parse_success_rate']:.1%}")

    print("\n" + "=" * 60)
    print("Generation demo tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_generation_demo()
