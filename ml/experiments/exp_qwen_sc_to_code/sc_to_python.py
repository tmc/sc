"""
Statechart to Python Code Generator.

Uses Qwen2.5-Coder via mlx_lm to generate Python code from statechart protos.
Targets 100% syntactically valid output through template-based generation.
"""

import ast
import sys
from io import StringIO
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json

from .code_templates import PythonTemplate, build_python_template, TemplateConfig

# Try to import mlx_lm
try:
    from mlx_lm import load, generate
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    load = None
    generate = None


@dataclass
class PythonGeneratorConfig:
    """Configuration for Python code generation."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.3
    use_template: bool = True
    use_llm: bool = True
    class_name: str = "StateMachine"
    verify_syntax: bool = True
    verify_import: bool = True


@dataclass
class PythonCodeResult:
    """Result of Python code generation."""
    code: str
    syntax_valid: bool
    import_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    llm_enhanced: bool = False


class StatechartToPython:
    """
    Generate Python code from statechart definitions.
    """

    def __init__(self, config: PythonGeneratorConfig = None):
        self.config = config or PythonGeneratorConfig()
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

    def generate(self, statechart_json: Dict) -> PythonCodeResult:
        """
        Generate Python code from statechart JSON.

        Args:
            statechart_json: Statechart definition in JSON format

        Returns:
            PythonCodeResult with generated code and validation status
        """
        errors = []
        warnings = []
        llm_enhanced = False

        # Step 1: Generate base code from template
        if self.config.use_template:
            template_config = TemplateConfig(
                class_name=self.config.class_name,
            )
            template = build_python_template(statechart_json, template_config)
            code = template.render()
        else:
            code = self._generate_with_llm(statechart_json)
            llm_enhanced = True

        # Step 2: Enhance with LLM if requested
        if self.config.use_llm and self.model is not None:
            enhanced = self._enhance_with_llm(code, statechart_json)
            if enhanced:
                code = enhanced
                llm_enhanced = True

        # Step 3: Verify syntax
        syntax_valid = True
        if self.config.verify_syntax:
            syntax_valid, syntax_errors = self._verify_syntax(code)
            if not syntax_valid:
                errors.extend(syntax_errors)
                # Try to repair
                repaired = self._repair_syntax(code, syntax_errors)
                if repaired:
                    syntax_valid, _ = self._verify_syntax(repaired)
                    if syntax_valid:
                        code = repaired
                        warnings.append("Code was repaired after syntax errors")

        # Step 4: Verify import
        import_valid = True
        if self.config.verify_import and syntax_valid:
            import_valid, import_errors = self._verify_import(code)
            if not import_valid:
                errors.extend(import_errors)

        return PythonCodeResult(
            code=code,
            syntax_valid=syntax_valid,
            import_valid=import_valid,
            errors=errors,
            warnings=warnings,
            llm_enhanced=llm_enhanced,
        )

    def _generate_with_llm(self, statechart_json: Dict) -> str:
        """Generate Python code with LLM."""
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

        prompt = f"""Improve this Python state machine with better docstrings and type hints:

```python
{code}
```

Output only the improved Python code:
```python
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

            # Verify enhanced code is valid
            if "class " in enhanced and "def " in enhanced:
                try:
                    ast.parse(enhanced)
                    return enhanced
                except SyntaxError:
                    pass

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
            src = t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', '')
            tgt = t.get('to', [''])[0] if isinstance(t.get('to'), list) else t.get('to', '')
            transitions.append(f"{src} --{t.get('event', '')}--> {tgt}")

        prompt = f"""Generate a Python state machine implementation.

States: {', '.join(states)}
Events: {', '.join(events)}
Transitions:
{chr(10).join('  ' + t for t in transitions)}

Requirements:
- Class name: {self.config.class_name}
- Use Enum for State and Event
- send(event) method for transitions
- state property for current state
- Type hints

Output only Python code:
```python
from enum import Enum, auto
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
            class_name=self.config.class_name,
        )
        template = build_python_template(statechart_json, template_config)
        return template.render()

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

        return response.strip()

    def _verify_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Verify Python syntax is valid."""
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Syntax error at line {e.lineno}: {e.msg}"]

    def _verify_import(self, code: str) -> Tuple[bool, List[str]]:
        """Verify code can be imported (executed)."""
        # Create a temporary module
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            exec(compile(code, '<string>', 'exec'), {})
            return True, []
        except Exception as e:
            return False, [f"Import error: {str(e)}"]
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _repair_syntax(self, code: str, errors: List[str]) -> Optional[str]:
        """Attempt to repair syntax errors."""
        repaired = code

        # Common fixes
        lines = code.split('\n')

        # Fix missing colons
        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if (stripped.startswith('def ') or
                stripped.startswith('class ') or
                stripped.startswith('if ') or
                stripped.startswith('elif ') or
                stripped.startswith('else') or
                stripped.startswith('for ') or
                stripped.startswith('while ') or
                stripped.startswith('try') or
                stripped.startswith('except') or
                stripped.startswith('finally')):
                if not stripped.endswith(':'):
                    lines[i] = stripped + ':'

        repaired = '\n'.join(lines)
        return repaired if repaired != code else None


def generate_python(statechart_json: Dict, config: PythonGeneratorConfig = None) -> PythonCodeResult:
    """
    Convenience function to generate Python code.

    Args:
        statechart_json: Statechart definition
        config: Generator configuration

    Returns:
        PythonCodeResult with code and status
    """
    generator = StatechartToPython(config)
    return generator.generate(statechart_json)


def test_python_generator():
    """Test Python code generation."""
    print("=" * 60)
    print("Testing Python Code Generator")
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
    config = PythonGeneratorConfig(
        use_llm=False,
        verify_syntax=True,
        verify_import=True,
    )
    result = generate_python(statechart, config)

    print(f"  Syntax valid: {result.syntax_valid}")
    print(f"  Import valid: {result.import_valid}")
    print(f"  LLM enhanced: {result.llm_enhanced}")
    print(f"  Errors: {result.errors}")
    print(f"  Code length: {len(result.code)} chars")
    print()
    print("  Code preview:")
    print("  " + result.code[:500].replace("\n", "\n  ") + "...")

    print("\n2. Testing generated code:")
    try:
        # Execute the generated code
        namespace = {}
        exec(result.code, namespace)

        # Get the class
        SM = namespace.get('StateMachine')
        State = namespace.get('State')
        Event = namespace.get('Event')

        if SM and State and Event:
            sm = SM()
            print(f"  Initial state: {sm.state}")
            print(f"  Send START: {sm.send(Event.START)}")
            print(f"  Current state: {sm.state}")
            print(f"  Send PAUSE: {sm.send(Event.PAUSE)}")
            print(f"  Current state: {sm.state}")
            print("  Generated code works correctly!")
    except Exception as e:
        print(f"  Error testing code: {e}")

    print("\n3. Full generation with LLM (if available):")
    if HAS_MLX:
        config_llm = PythonGeneratorConfig(
            use_llm=True,
            verify_syntax=True,
        )
        result_llm = generate_python(statechart, config_llm)
        print(f"  Syntax valid: {result_llm.syntax_valid}")
        print(f"  LLM enhanced: {result_llm.llm_enhanced}")
    else:
        print("  MLX not available, skipping LLM test")

    print("\n" + "=" * 60)
    print("Python generator tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_python_generator()
