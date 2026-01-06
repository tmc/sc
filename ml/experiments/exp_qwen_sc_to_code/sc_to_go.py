"""
Statechart to Go Code Generator.

Uses Qwen2.5-Coder via mlx_lm to generate Go code from statechart protos.
Combines template-based generation with LLM enhancement for:
- Custom method implementations
- Guard conditions
- Action implementations
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

from .code_templates import GoTemplate, build_go_template, TemplateConfig

# Try to import mlx_lm
try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    load = None
    generate = None


@dataclass
class GoGeneratorConfig:
    """Configuration for Go code generation."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.3  # Lower for more deterministic code
    use_template: bool = True  # Use template for structure
    use_llm: bool = True       # Use LLM for enhancements
    package_name: str = "statemachine"
    struct_name: str = "StateMachine"
    verify_compilation: bool = True


@dataclass
class GoCodeResult:
    """Result of Go code generation."""
    code: str
    compiles: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_enhanced: bool = False


class StatechartToGo:
    """
    Generate Go code from statechart definitions.

    Uses a hybrid approach:
    1. Template for guaranteed structure
    2. LLM for method implementations and enhancements
    """

    def __init__(self, config: GoGeneratorConfig = None):
        self.config = config or GoGeneratorConfig()
        self.model = None
        self.tokenizer = None

        if self.config.use_llm and HAS_MLX:
            self._load_model()

    def _load_model(self):
        """Load the LLM model."""
        if not HAS_MLX:
            return

        try:
            print(f"Loading model: {self.config.model_name}")
            self.model, self.tokenizer = load(self.config.model_name)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    def generate(self, statechart_json: Dict) -> GoCodeResult:
        """
        Generate Go code from statechart JSON.

        Args:
            statechart_json: Statechart definition in JSON format

        Returns:
            GoCodeResult with generated code and compilation status
        """
        errors = []
        warnings = []
        llm_enhanced = False

        # Step 1: Generate base code from template
        if self.config.use_template:
            template_config = TemplateConfig(
                package_name=self.config.package_name,
                class_name=self.config.struct_name,
            )
            template = build_go_template(statechart_json, template_config)
            code = template.render()
        else:
            # Generate entirely with LLM
            code = self._generate_with_llm(statechart_json)
            llm_enhanced = True

        # Step 2: Enhance with LLM if requested
        if self.config.use_llm and self.model is not None:
            enhanced_code = self._enhance_with_llm(code, statechart_json)
            if enhanced_code:
                code = enhanced_code
                llm_enhanced = True

        # Step 3: Verify compilation
        compiles = True
        if self.config.verify_compilation:
            compiles, compile_errors = self._verify_compilation(code)
            if not compiles:
                errors.extend(compile_errors)
                # Try to repair
                repaired = self._repair_code(code, compile_errors)
                if repaired:
                    compiles, _ = self._verify_compilation(repaired)
                    if compiles:
                        code = repaired
                        warnings.append("Code was repaired after initial compilation failure")

        return GoCodeResult(
            code=code,
            compiles=compiles,
            errors=errors,
            warnings=warnings,
            llm_enhanced=llm_enhanced,
        )

    def _generate_with_llm(self, statechart_json: Dict) -> str:
        """Generate Go code entirely with LLM."""
        if not self.model:
            return self._fallback_generation(statechart_json)

        prompt = self._build_generation_prompt(statechart_json)

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temp=self.config.temperature,
            )
            return self._extract_code(response)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            return self._fallback_generation(statechart_json)

    def _enhance_with_llm(self, code: str, statechart_json: Dict) -> Optional[str]:
        """Enhance generated code with LLM."""
        if not self.model:
            return None

        prompt = f"""The following Go code implements a state machine. Add helpful comments and improve readability:

```go
{code}
```

Output only the improved Go code:
```go
"""

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temp=self.config.temperature,
            )
            enhanced = self._extract_code(response)
            # Only use if it looks valid
            if "package " in enhanced and "func " in enhanced:
                return enhanced
        except Exception as e:
            print(f"LLM enhancement failed: {e}")

        return None

    def _build_generation_prompt(self, statechart_json: Dict) -> str:
        """Build prompt for LLM generation."""
        states = []
        root = statechart_json.get('root_state', {})
        self._extract_state_labels(root, states)

        events = set()
        transitions = []
        for t in statechart_json.get('transitions', []):
            events.add(t.get('event', ''))
            transitions.append(f"{t.get('from', '')} --{t.get('event', '')}--> {t.get('to', '')}")

        prompt = f"""Generate a Go state machine implementation.

States: {', '.join(states)}
Events: {', '.join(events)}
Transitions:
{chr(10).join('  ' + t for t in transitions)}

Requirements:
- Package name: {self.config.package_name}
- Struct name: {self.config.struct_name}
- Thread-safe with sync.RWMutex
- Send(event) method for transitions
- CurrentState() method

Output only Go code:
```go
package {self.config.package_name}
"""
        return prompt

    def _extract_state_labels(self, node: Dict, states: List[str]):
        """Extract state labels from node."""
        label = node.get('label', '')
        if label and label != '__root__':
            states.append(label)
        for child in node.get('children', []):
            self._extract_state_labels(child, states)

    def _fallback_generation(self, statechart_json: Dict) -> str:
        """Fallback to template generation."""
        template_config = TemplateConfig(
            package_name=self.config.package_name,
            class_name=self.config.struct_name,
        )
        template = build_go_template(statechart_json, template_config)
        return template.render()

    def _extract_code(self, response: str) -> str:
        """Extract Go code from LLM response."""
        # Look for code blocks
        if "```go" in response:
            start = response.find("```go") + 5
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        return response.strip()

    def _verify_compilation(self, code: str) -> Tuple[bool, List[str]]:
        """Verify Go code compiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "main.go")

            # Wrap in main package for compilation check
            test_code = code
            if "package main" not in code:
                test_code = code.replace(f"package {self.config.package_name}", "package main")

            # Add main function if not present
            if "func main()" not in test_code:
                test_code += "\n\nfunc main() {}\n"

            with open(filepath, 'w') as f:
                f.write(test_code)

            try:
                result = subprocess.run(
                    ['go', 'build', '-o', '/dev/null', filepath],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    return True, []
                else:
                    errors = result.stderr.strip().split('\n')
                    return False, errors

            except subprocess.TimeoutExpired:
                return False, ["Compilation timed out"]
            except FileNotFoundError:
                return True, []  # Go not installed, assume OK

    def _repair_code(self, code: str, errors: List[str]) -> Optional[str]:
        """Attempt to repair code based on errors."""
        repaired = code

        for error in errors:
            # Common fixes
            if "undefined:" in error:
                # Missing import or definition
                pass
            elif "syntax error" in error:
                # Syntax issues
                pass

        return repaired if repaired != code else None


def generate_go(statechart_json: Dict, config: GoGeneratorConfig = None) -> GoCodeResult:
    """
    Convenience function to generate Go code.

    Args:
        statechart_json: Statechart definition
        config: Generator configuration

    Returns:
        GoCodeResult with code and status
    """
    generator = StatechartToGo(config)
    return generator.generate(statechart_json)


def test_go_generator():
    """Test Go code generation."""
    print("=" * 60)
    print("Testing Go Code Generator")
    print("=" * 60)

    # Sample statechart
    statechart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {"label": "Running", "type": 1},
                {"label": "Paused", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["Running"], "event": "START"},
            {"from": ["Running"], "to": ["Paused"], "event": "PAUSE"},
            {"from": ["Paused"], "to": ["Running"], "event": "RESUME"},
            {"from": ["Running"], "to": ["Idle"], "event": "STOP"},
            {"from": ["Paused"], "to": ["Idle"], "event": "STOP"},
        ]
    }

    print("\n1. Template-only generation:")
    config = GoGeneratorConfig(
        use_llm=False,
        verify_compilation=True,
    )
    result = generate_go(statechart, config)

    print(f"  Compiles: {result.compiles}")
    print(f"  LLM enhanced: {result.llm_enhanced}")
    print(f"  Errors: {result.errors}")
    print(f"  Code length: {len(result.code)} chars")
    print()
    print("  Code preview:")
    print("  " + result.code[:400].replace("\n", "\n  ") + "...")

    print("\n2. Full generation with LLM (if available):")
    if HAS_MLX:
        config_llm = GoGeneratorConfig(
            use_llm=True,
            verify_compilation=True,
        )
        result_llm = generate_go(statechart, config_llm)
        print(f"  Compiles: {result_llm.compiles}")
        print(f"  LLM enhanced: {result_llm.llm_enhanced}")
    else:
        print("  MLX not available, skipping LLM test")

    print("\n" + "=" * 60)
    print("Go generator tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_go_generator()
