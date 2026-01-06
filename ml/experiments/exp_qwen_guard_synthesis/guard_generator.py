"""
Guard Expression Generator using Qwen2.5-Coder

Uses few-shot prompting with Qwen2.5-Coder-0.5B-Instruct to generate
boolean guard expressions from natural language descriptions.

Key approach:
1. Few-shot examples teach the model guard syntax
2. NL description is parsed for intent
3. Model generates candidate expressions
4. Validator checks syntax
5. Best candidates returned

Target: 95%+ syntax valid, 80%+ semantic correct.

NO HARDCODING: Model learns from few-shot examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import re
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for guard generation."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_tokens: int = 100
    temperature: float = 0.3  # Lower for more deterministic guards
    top_p: float = 0.9
    num_candidates: int = 5
    verbose: bool = True


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

FEW_SHOT_EXAMPLES = [
    # Basic comparisons
    {
        "description": "counter is greater than 5",
        "variables": ["counter", "limit"],
        "guard": "counter > 5",
    },
    {
        "description": "the player is not moving",
        "variables": ["is_moving", "speed", "direction"],
        "guard": "not is_moving",
    },
    {
        "description": "health is zero or less",
        "variables": ["health", "max_health", "damage"],
        "guard": "health <= 0",
    },
    # Compound conditions
    {
        "description": "player is alive and has enough mana",
        "variables": ["is_alive", "mana", "mana_cost"],
        "guard": "is_alive and (mana >= mana_cost)",
    },
    {
        "description": "either timeout or user cancelled",
        "variables": ["timeout", "cancelled", "running"],
        "guard": "timeout or cancelled",
    },
    # Game-specific
    {
        "description": "move to same position as last capture (Ko rule)",
        "variables": ["move_x", "move_y", "last_capture_x", "last_capture_y"],
        "guard": "(move_x == last_capture_x) and (move_y == last_capture_y)",
    },
    {
        "description": "king has not moved and rook has not moved",
        "variables": ["king_moved", "rook_moved", "in_check"],
        "guard": "(not king_moved) and (not rook_moved)",
    },
    {
        "description": "buffer is not empty",
        "variables": ["buffer_count", "max_buffer"],
        "guard": "buffer_count > 0",
    },
]


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are a guard expression generator for state machines.
You convert natural language descriptions into boolean guard expressions.

Rules:
- Use Python-like syntax: and, or, not, ==, !=, <, >, <=, >=
- Use parentheses for clarity
- Only use the provided variables
- Output ONLY the expression, nothing else
- Keep expressions simple and readable"""

def build_few_shot_prompt(examples: List[Dict]) -> str:
    """Build few-shot examples section."""
    lines = []
    for ex in examples:
        lines.append(f"Description: {ex['description']}")
        lines.append(f"Variables: {', '.join(ex['variables'])}")
        lines.append(f"Guard: {ex['guard']}")
        lines.append("")
    return "\n".join(lines)


def build_generation_prompt(
    description: str,
    variables: List[str],
    few_shot_examples: List[Dict] = None,
) -> str:
    """Build the full prompt for guard generation."""
    if few_shot_examples is None:
        few_shot_examples = FEW_SHOT_EXAMPLES

    prompt = f"""{SYSTEM_PROMPT}

Examples:
{build_few_shot_prompt(few_shot_examples)}
Now generate a guard expression:

Description: {description}
Variables: {', '.join(variables)}
Guard:"""

    return prompt


# =============================================================================
# QWEN GUARD GENERATOR
# =============================================================================

class QwenGuardGenerator:
    """
    Generate guard expressions using Qwen2.5-Coder.

    Uses few-shot prompting to teach the model guard syntax.
    """

    def __init__(self, config: GeneratorConfig = None):
        self.config = config or GeneratorConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self):
        """Load the Qwen model using mlx_lm."""
        if self._loaded:
            return

        try:
            from mlx_lm import load
            print(f"Loading model: {self.config.model_name}")
            self.model, self.tokenizer = load(self.config.model_name)
            self._loaded = True
            print("Model loaded successfully")
        except ImportError:
            print("mlx_lm not available, using mock generation")
            self._loaded = False
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._loaded = False

    def generate(
        self,
        description: str,
        variables: List[str],
        num_candidates: int = None,
    ) -> List[str]:
        """
        Generate guard expression candidates.

        Args:
            description: Natural language description of the guard
            variables: List of available variable names
            num_candidates: Number of candidates to generate

        Returns:
            List of generated guard expression strings
        """
        if num_candidates is None:
            num_candidates = self.config.num_candidates

        if not self._loaded:
            self.load_model()

        prompt = build_generation_prompt(description, variables)

        candidates = []

        if self._loaded and self.model is not None:
            # Real generation using mlx_lm
            candidates = self._generate_with_model(prompt, num_candidates)
        else:
            # Mock generation using pattern matching
            candidates = self._generate_mock(description, variables, num_candidates)

        return candidates

    def _generate_with_model(
        self,
        prompt: str,
        num_candidates: int,
    ) -> List[str]:
        """Generate using the actual model."""
        from mlx_lm import generate

        candidates = []

        for _ in range(num_candidates):
            try:
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temp=self.config.temperature,
                )

                # Extract the guard expression (first line of response)
                lines = response.strip().split('\n')
                if lines:
                    guard = lines[0].strip()
                    # Clean up common artifacts
                    guard = self._clean_guard(guard)
                    if guard:
                        candidates.append(guard)

            except Exception as e:
                if self.config.verbose:
                    print(f"Generation error: {e}")

        return candidates

    def _generate_mock(
        self,
        description: str,
        variables: List[str],
        num_candidates: int,
    ) -> List[str]:
        """Mock generation using pattern matching (fallback)."""
        candidates = []
        desc_lower = description.lower()

        # Pattern matching for common phrases
        patterns = [
            # Comparisons
            (r"greater than (\d+)", lambda m, v: f"{v[0]} > {m.group(1)}"),
            (r"less than (\d+)", lambda m, v: f"{v[0]} < {m.group(1)}"),
            (r"equal to (\d+)", lambda m, v: f"{v[0]} == {m.group(1)}"),
            (r"at least (\d+)", lambda m, v: f"{v[0]} >= {m.group(1)}"),
            (r"at most (\d+)", lambda m, v: f"{v[0]} <= {m.group(1)}"),

            # Negation
            (r"not (.+)", lambda m, v: f"not {v[0]}"),
            (r"is false", lambda m, v: f"not {v[0]}"),
            (r"is true", lambda m, v: f"{v[0]}"),

            # Boolean logic
            (r"(.+) and (.+)", lambda m, v: f"({v[0]}) and ({v[1] if len(v) > 1 else 'True'})"),
            (r"(.+) or (.+)", lambda m, v: f"({v[0]}) or ({v[1] if len(v) > 1 else 'False'})"),
            (r"either (.+) or (.+)", lambda m, v: f"({v[0]}) or ({v[1] if len(v) > 1 else 'False'})"),

            # State checks
            (r"is empty", lambda m, v: f"{v[0]} == 0"),
            (r"not empty", lambda m, v: f"{v[0]} > 0"),
            (r"has (.+)", lambda m, v: f"{v[0]} > 0"),

            # Game patterns
            (r"same position", lambda m, v: f"({v[0]} == {v[2]}) and ({v[1]} == {v[3]})" if len(v) >= 4 else f"{v[0]} == {v[1]}"),
            (r"ko|recapture", lambda m, v: f"({v[0]} == {v[2]}) and ({v[1]} == {v[3]})" if len(v) >= 4 else f"{v[0]}"),
            (r"castle|king.*not.*moved", lambda m, v: f"not {v[0]}"),
        ]

        # Try patterns
        for pattern, generator in patterns:
            match = re.search(pattern, desc_lower)
            if match:
                try:
                    guard = generator(match, variables)
                    candidates.append(guard)
                except (IndexError, KeyError):
                    pass

        # Add simple variable-based guards
        if len(candidates) < num_candidates:
            for var in variables[:3]:
                candidates.append(f"{var}")
                candidates.append(f"not {var}")
                candidates.append(f"{var} > 0")
                candidates.append(f"{var} == True")

        return candidates[:num_candidates]

    def _clean_guard(self, guard: str) -> str:
        """Clean up a generated guard expression."""
        # Remove quotes
        guard = guard.strip('"\'')

        # Remove trailing comments
        if '#' in guard:
            guard = guard.split('#')[0]

        # Remove leading/trailing whitespace
        guard = guard.strip()

        # Remove markdown artifacts
        guard = re.sub(r'^```\w*\s*', '', guard)
        guard = re.sub(r'\s*```$', '', guard)

        # Remove "Guard:" prefix if present
        guard = re.sub(r'^Guard:\s*', '', guard, flags=re.IGNORECASE)

        return guard

    def generate_with_validation(
        self,
        description: str,
        variables: List[str],
        validator: 'GuardValidator' = None,
        num_candidates: int = None,
    ) -> List[Tuple[str, bool, str]]:
        """
        Generate candidates with validation.

        Returns:
            List of (guard, is_valid, error_message) tuples
        """
        if validator is None:
            from .validator import GuardValidator
            validator = GuardValidator(variables)

        candidates = self.generate(description, variables, num_candidates)

        results = []
        for guard in candidates:
            is_valid, error = validator.validate(guard)
            results.append((guard, is_valid, error or ""))

        return results


# =============================================================================
# BATCH GENERATION
# =============================================================================

@dataclass
class GenerationRequest:
    """A request for guard generation."""
    description: str
    variables: List[str]
    expected: Optional[str] = None  # For evaluation


@dataclass
class GenerationResult:
    """Result of guard generation."""
    request: GenerationRequest
    candidates: List[str]
    best_candidate: Optional[str] = None
    syntax_valid: int = 0
    semantic_correct: int = 0
    elapsed_time: float = 0.0


def batch_generate(
    generator: QwenGuardGenerator,
    requests: List[GenerationRequest],
    validator: 'GuardValidator' = None,
    verbose: bool = True,
) -> List[GenerationResult]:
    """
    Generate guards for multiple requests.

    Returns list of results with validation stats.
    """
    results = []

    for i, req in enumerate(requests):
        if verbose:
            print(f"\n[{i+1}/{len(requests)}] {req.description[:40]}...")

        start_time = time.time()
        candidates = generator.generate(req.description, req.variables)
        elapsed = time.time() - start_time

        result = GenerationResult(
            request=req,
            candidates=candidates,
            elapsed_time=elapsed,
        )

        # Validate candidates
        if validator:
            for cand in candidates:
                is_valid, _ = validator.validate(cand)
                if is_valid:
                    result.syntax_valid += 1
                    if result.best_candidate is None:
                        result.best_candidate = cand

        # Check semantic correctness if expected provided
        if req.expected and result.best_candidate:
            # Simple check: does the best candidate match expected?
            if _normalize_guard(result.best_candidate) == _normalize_guard(req.expected):
                result.semantic_correct = 1

        if verbose:
            valid_rate = result.syntax_valid / max(len(candidates), 1)
            print(f"  Generated {len(candidates)}, valid: {result.syntax_valid} ({valid_rate:.0%})")
            if result.best_candidate:
                print(f"  Best: {result.best_candidate[:50]}")

        results.append(result)

    return results


def _normalize_guard(guard: str) -> str:
    """Normalize a guard for comparison."""
    # Remove whitespace
    guard = re.sub(r'\s+', '', guard)
    # Lowercase
    guard = guard.lower()
    # Remove outer parens
    guard = guard.strip('()')
    return guard


# =============================================================================
# TESTING
# =============================================================================

def test_guard_generator():
    """Test the guard generator."""
    print("=" * 60)
    print("QWEN GUARD GENERATOR TEST")
    print("=" * 60)

    config = GeneratorConfig(verbose=True)
    generator = QwenGuardGenerator(config)

    # Test cases
    tests = [
        ("counter is greater than 10", ["counter", "limit", "step"]),
        ("player is alive and has mana", ["is_alive", "mana", "health"]),
        ("not in check and king has not moved", ["in_check", "king_moved", "can_castle"]),
        ("buffer is not empty", ["buffer_count", "max_size"]),
    ]

    print("\nGenerating guards (mock mode if model unavailable)...")

    for desc, vars in tests:
        print(f"\n[TEST] {desc}")
        print(f"  Variables: {vars}")

        candidates = generator.generate(desc, vars, num_candidates=3)

        print(f"  Candidates:")
        for i, cand in enumerate(candidates[:3]):
            print(f"    {i+1}. {cand}")

    print("\n" + "=" * 60)
    print("GUARD GENERATOR TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_guard_generator()
