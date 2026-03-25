"""
Bridge for streaming generation progress to web visualizer.

Enables real-time visualization of statechart generation:
- Token-by-token updates
- Partial statechart rendering
- Validation error highlighting
- Attention pattern overlay (with mlux)

Usage:
    from utils.visualizer_bridge import VisualizerBridge

    bridge = VisualizerBridge("ws://localhost:8080/ws/generate")

    for token in generation_loop():
        partial_sc = parse_partial(tokens_so_far)
        bridge.on_token(token, partial_sc, activations)

    bridge.on_complete(final_statechart)
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    websocket = None
    WEBSOCKET_AVAILABLE = False


class MessageType(str, Enum):
    """Types of messages sent to visualizer."""
    TOKEN = "token"
    COMPLETE = "complete"
    ERROR = "error"
    VALIDATION = "validation"
    ATTENTION = "attention"
    STEERING = "steering"


@dataclass
class ValidationResult:
    """Result of statechart validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TokenUpdate:
    """Update for a single generated token."""
    type: str = MessageType.TOKEN
    token: str = ""
    token_id: int = 0
    position: int = 0
    partial_statechart: Optional[Dict] = None
    validation: Optional[ValidationResult] = None
    activations: Optional[Dict[str, List[float]]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CompletionUpdate:
    """Update when generation completes."""
    type: str = MessageType.COMPLETE
    statechart: Dict = field(default_factory=dict)
    total_tokens: int = 0
    generation_time: float = 0.0
    validation: Optional[ValidationResult] = None


class VisualizerBridge:
    """
    Stream generation progress to web visualizer.

    Connects via WebSocket to the visualizer server and sends
    real-time updates as tokens are generated.
    """

    def __init__(
        self,
        visualizer_url: str = "ws://localhost:8080/ws/generate",
        auto_connect: bool = True,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize bridge.

        Args:
            visualizer_url: WebSocket URL for visualizer
            auto_connect: Connect immediately on init
            retry_attempts: Number of connection retries
            retry_delay: Delay between retries (seconds)
        """
        self.url = visualizer_url
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay

        self._ws: Optional[Any] = None
        self._connected = False
        self._token_count = 0
        self._start_time: Optional[float] = None

        # Callbacks for local processing
        self._on_token_callbacks: List[Callable] = []
        self._on_complete_callbacks: List[Callable] = []

        if auto_connect and WEBSOCKET_AVAILABLE:
            self.connect()

    def connect(self) -> bool:
        """Connect to visualizer server."""
        if not WEBSOCKET_AVAILABLE:
            print("websocket-client not installed, running in offline mode")
            return False

        for attempt in range(self.retry_attempts):
            try:
                self._ws = websocket.create_connection(
                    self.url,
                    timeout=5.0,
                )
                self._connected = True
                print(f"Connected to visualizer at {self.url}")
                return True
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)

        print("Could not connect to visualizer, running in offline mode")
        return False

    def disconnect(self):
        """Disconnect from visualizer."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether connected to visualizer."""
        return self._connected

    def on_token(
        self,
        token: str,
        partial_statechart: Optional[Dict] = None,
        activations: Optional[Dict[str, Any]] = None,
        validation: Optional[ValidationResult] = None,
        token_id: int = 0,
    ):
        """
        Send token update to visualizer.

        Args:
            token: Generated token string
            partial_statechart: Current partial statechart (parsed from tokens)
            activations: Activation cache from mlux (optional)
            validation: Validation result for partial SC
            token_id: Token ID from vocabulary
        """
        if self._start_time is None:
            self._start_time = time.time()

        self._token_count += 1

        # Convert activations to serializable format
        serializable_activations = None
        if activations:
            serializable_activations = self._serialize_activations(activations)

        update = TokenUpdate(
            token=token,
            token_id=token_id,
            position=self._token_count,
            partial_statechart=partial_statechart,
            validation=validation,
            activations=serializable_activations,
        )

        self._send(update)

        # Call local callbacks
        for callback in self._on_token_callbacks:
            callback(update)

    def on_complete(
        self,
        statechart: Dict,
        validation: Optional[ValidationResult] = None,
    ):
        """
        Send completion update to visualizer.

        Args:
            statechart: Final generated statechart
            validation: Final validation result
        """
        generation_time = time.time() - self._start_time if self._start_time else 0.0

        update = CompletionUpdate(
            statechart=statechart,
            total_tokens=self._token_count,
            generation_time=generation_time,
            validation=validation,
        )

        self._send(update)

        # Reset for next generation
        self._token_count = 0
        self._start_time = None

        # Call local callbacks
        for callback in self._on_complete_callbacks:
            callback(update)

    def on_error(self, error: str, partial_statechart: Optional[Dict] = None):
        """Send error update to visualizer."""
        self._send({
            "type": MessageType.ERROR,
            "error": error,
            "partial_statechart": partial_statechart,
            "timestamp": time.time(),
        })

    def send_attention_patterns(
        self,
        patterns: Dict[int, Any],
        token_labels: List[str],
    ):
        """
        Send attention patterns for visualization.

        Args:
            patterns: {layer: attention_matrix}
            token_labels: Labels for each token position
        """
        serialized = {}
        for layer, pattern in patterns.items():
            if hasattr(pattern, "tolist"):
                serialized[str(layer)] = pattern.tolist()
            else:
                serialized[str(layer)] = pattern

        self._send({
            "type": MessageType.ATTENTION,
            "patterns": serialized,
            "token_labels": token_labels,
            "timestamp": time.time(),
        })

    def send_steering_info(
        self,
        layer: int,
        alpha: float,
        positive_example: str,
        negative_example: str,
    ):
        """Send steering configuration to visualizer."""
        self._send({
            "type": MessageType.STEERING,
            "layer": layer,
            "alpha": alpha,
            "positive": positive_example,
            "negative": negative_example,
            "timestamp": time.time(),
        })

    def add_token_callback(self, callback: Callable[[TokenUpdate], None]):
        """Add callback for token updates."""
        self._on_token_callbacks.append(callback)

    def add_complete_callback(self, callback: Callable[[CompletionUpdate], None]):
        """Add callback for completion updates."""
        self._on_complete_callbacks.append(callback)

    def _send(self, data: Any):
        """Send data to visualizer."""
        if not self._connected or not self._ws:
            return

        try:
            if hasattr(data, "__dataclass_fields__"):
                payload = asdict(data)
            else:
                payload = data

            self._ws.send(json.dumps(payload, default=str))
        except Exception as e:
            print(f"Failed to send to visualizer: {e}")
            self._connected = False

    def _serialize_activations(self, activations: Dict[str, Any]) -> Dict[str, List]:
        """Convert activation arrays to serializable format."""
        result = {}
        for key, value in activations.items():
            if hasattr(value, "tolist"):
                # MLX or numpy array
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            else:
                result[key] = value
        return result


class NullBridge:
    """No-op bridge for when visualization is disabled."""

    def __init__(self, *args, **kwargs):
        pass

    @property
    def connected(self) -> bool:
        return False

    def connect(self) -> bool:
        return False

    def disconnect(self):
        pass

    def on_token(self, *args, **kwargs):
        pass

    def on_complete(self, *args, **kwargs):
        pass

    def on_error(self, *args, **kwargs):
        pass

    def send_attention_patterns(self, *args, **kwargs):
        pass

    def send_steering_info(self, *args, **kwargs):
        pass

    def add_token_callback(self, callback):
        pass

    def add_complete_callback(self, callback):
        pass


def create_bridge(
    url: Optional[str] = None,
    enabled: bool = True,
) -> VisualizerBridge:
    """
    Create visualizer bridge.

    Args:
        url: WebSocket URL (uses default if None)
        enabled: Whether visualization is enabled

    Returns:
        VisualizerBridge or NullBridge
    """
    if not enabled:
        return NullBridge()

    return VisualizerBridge(
        visualizer_url=url or "ws://localhost:8080/ws/generate",
        auto_connect=True,
    )


__all__ = [
    "VisualizerBridge",
    "NullBridge",
    "create_bridge",
    "TokenUpdate",
    "CompletionUpdate",
    "ValidationResult",
    "MessageType",
    "WEBSOCKET_AVAILABLE",
]
