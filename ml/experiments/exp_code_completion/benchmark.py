#!/usr/bin/env python3
"""
Benchmark for Statechart-Guided Code Generation

Measures syntactic validity rate with and without constraints.
Target: 99%+ validity with constraints vs ~60-80% unconstrained.

Metrics:
- Parse success rate (ast.parse for Python, go/parser for Go)
- Bracket matching rate
- Indentation validity rate
- Average tokens blocked per generation
- Generation throughput (tokens/sec)

Usage:
    python benchmark.py --samples 100 --language python
"""

import ast
import re
import time
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from .guided_generation import GuidedCodeGenerator, GenerationConfig, SyntaxStateTracker
from .token_masker import TokenMasker


@dataclass
class ValidationResult:
    """Result of validating a single code sample."""
    parse_success: bool = False
    brackets_balanced: bool = False
    indent_valid: bool = False
    keyword_valid: bool = False
    error_message: str = ""
    error_line: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Overall validity."""
        return self.parse_success and self.brackets_balanced


@dataclass
class GenerationSample:
    """A single generation sample with metadata."""
    prompt: str
    generated: str
    constrained: bool
    validation: ValidationResult
    tokens_generated: int
    tokens_blocked: int
    generation_time_ms: float
    final_state: str


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""
    language: str
    total_samples: int
    constrained: bool
    
    # Validity rates
    parse_success_rate: float = 0.0
    bracket_match_rate: float = 0.0
    indent_valid_rate: float = 0.0
    overall_valid_rate: float = 0.0
    
    # Performance metrics
    avg_tokens_per_sample: float = 0.0
    avg_tokens_blocked: float = 0.0
    avg_generation_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Error analysis
    error_types: Dict[str, int] = field(default_factory=dict)
    samples_with_errors: List[GenerationSample] = field(default_factory=list)


class CodeValidator:
    """
    Validates generated code for syntactic correctness.
    
    Supports Python and Go validation.
    """
    
    def __init__(self, language: str = "python"):
        self.language = language
    
    def validate(self, code: str) -> ValidationResult:
        """Validate code and return detailed result."""
        result = ValidationResult()
        
        # Check bracket balance (fast)
        result.brackets_balanced = self._check_brackets(code)
        
        # Check indentation (Python only)
        if self.language == "python":
            result.indent_valid = self._check_python_indent(code)
        else:
            result.indent_valid = True  # Go doesn't have significant whitespace
        
        # Try to parse
        if self.language == "python":
            result.parse_success, result.error_message, result.error_line = \
                self._parse_python(code)
        else:
            result.parse_success, result.error_message, result.error_line = \
                self._parse_go(code)
        
        # Check keyword usage
        result.keyword_valid = self._check_keywords(code)
        
        return result
    
    def _check_brackets(self, code: str) -> bool:
        """Check if all brackets are balanced."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        in_string = False
        string_char = None
        
        i = 0
        while i < len(code):
            char = code[i]
            
            # Handle string boundaries
            if char in '"\'':
                # Check for triple quotes
                if code[i:i+3] in ('"""', "'''"):
                    if in_string and code[i:i+3] == string_char:
                        in_string = False
                        string_char = None
                    elif not in_string:
                        in_string = True
                        string_char = code[i:i+3]
                    i += 3
                    continue
                elif not in_string:
                    in_string = True
                    string_char = char
                elif string_char == char:
                    in_string = False
                    string_char = None
            
            # Skip brackets inside strings
            if in_string:
                i += 1
                continue
            
            # Handle comment
            if char == '#' and self.language == "python":
                # Skip to end of line
                while i < len(code) and code[i] != '\n':
                    i += 1
                continue
            
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if not stack or stack.pop() != char:
                    return False
            
            i += 1
        
        return len(stack) == 0
    
    def _check_python_indent(self, code: str) -> bool:
        """Check Python indentation validity."""
        lines = code.split('\n')
        indent_stack = [0]
        
        for i, line in enumerate(lines):
            # Skip empty lines and comments
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Calculate indent
            indent = len(line) - len(stripped)
            
            # Check indent is multiple of 4 (or consistent)
            if indent > 0 and indent % 4 != 0:
                # Allow 2-space indent as well
                if indent % 2 != 0:
                    return False
            
            # Check indent change is valid
            current_indent = indent_stack[-1]
            
            if indent > current_indent:
                # Indent increase - should follow colon
                if i > 0:
                    prev_line = lines[i-1].rstrip()
                    if prev_line and not prev_line.endswith(':'):
                        # Allow multiline statements
                        if not any(c in prev_line for c in '([{'):
                            pass  # Might be continuation
                indent_stack.append(indent)
            elif indent < current_indent:
                # Dedent - pop until we find matching level
                while indent_stack and indent_stack[-1] > indent:
                    indent_stack.pop()
                if indent_stack and indent_stack[-1] != indent:
                    return False
        
        return True
    
    def _parse_python(self, code: str) -> Tuple[bool, str, int]:
        """Try to parse Python code with ast."""
        try:
            ast.parse(code)
            return True, "", 0
        except SyntaxError as e:
            return False, str(e.msg), e.lineno or 0
        except Exception as e:
            return False, str(e), 0
    
    def _parse_go(self, code: str) -> Tuple[bool, str, int]:
        """
        Validate Go syntax.
        
        Since we can't use go/parser directly from Python,
        we do basic structural validation.
        """
        # Check for required package declaration
        if not re.search(r'^package\s+\w+', code, re.MULTILINE):
            return False, "missing package declaration", 1
        
        # Check function syntax
        func_pattern = r'func\s+(\w+)?\s*\([^)]*\)\s*(\([^)]*\)|[\w*]+)?\s*\{'
        if 'func' in code and not re.search(func_pattern, code):
            return False, "malformed function declaration", 0
        
        # Check type syntax
        if 'type ' in code:
            type_pattern = r'type\s+\w+\s+(struct|interface)\s*\{'
            if not re.search(type_pattern, code) and 'type ' in code:
                # Might be type alias
                if not re.search(r'type\s+\w+\s+=?\s+\w+', code):
                    return False, "malformed type declaration", 0
        
        # Check if/for/switch have braces
        for keyword in ['if', 'for', 'switch', 'select']:
            pattern = rf'{keyword}\s+[^{{]+\{{'
            if re.search(rf'\b{keyword}\b', code):
                if not re.search(pattern, code):
                    # Might be valid without condition
                    if keyword not in ['for', 'select']:
                        return False, f"malformed {keyword} statement", 0
        
        return True, "", 0
    
    def _check_keywords(self, code: str) -> bool:
        """Check keyword usage is valid."""
        if self.language == "python":
            # Check return/yield outside function
            lines = code.split('\n')
            in_function = False
            indent_at_func = -1
            
            for line in lines:
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())
                
                if stripped.startswith('def ') or stripped.startswith('async def '):
                    in_function = True
                    indent_at_func = indent
                elif in_function and indent <= indent_at_func and stripped:
                    in_function = False
                    indent_at_func = -1
                
                if stripped.startswith('return ') or stripped == 'return':
                    if not in_function:
                        return False
                        
                if stripped.startswith('yield ') or stripped == 'yield':
                    if not in_function:
                        return False
                        
                if stripped.startswith('break') or stripped.startswith('continue'):
                    # Should be in loop - simplified check
                    pass
        
        return True


class CodeCompletionBenchmark:
    """
    Benchmark for code completion with syntax constraints.
    
    Compares constrained vs unconstrained generation.
    """
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.validator = CodeValidator(language)
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        
        # Create generator
        self.generator = GuidedCodeGenerator(self.vocab, language=language)
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for benchmark."""
        if self.language == "python":
            return self._python_vocab()
        else:
            return self._go_vocab()
    
    def _python_vocab(self) -> Dict[str, int]:
        """Python vocabulary."""
        vocab = {}
        idx = 0
        
        # Keywords
        for kw in ['def', 'class', 'if', 'elif', 'else', 'for', 'while',
                   'try', 'except', 'finally', 'with', 'as', 'return',
                   'yield', 'raise', 'import', 'from', 'pass', 'break',
                   'continue', 'and', 'or', 'not', 'in', 'is', 'lambda',
                   'True', 'False', 'None', 'async', 'await', 'assert',
                   'global', 'nonlocal', 'del']:
            vocab[kw] = idx
            idx += 1
        
        # Operators
        for op in ['+', '-', '*', '/', '//', '%', '**', '@',
                   '==', '!=', '<', '>', '<=', '>=',
                   '=', '+=', '-=', '*=', '/=',
                   '&', '|', '^', '~', '<<', '>>', '->']:
            vocab[op] = idx
            idx += 1
        
        # Delimiters
        for d in ['(', ')', '[', ']', '{', '}', ':', ',', '.', ';', '...']:
            vocab[d] = idx
            idx += 1
        
        # Whitespace
        vocab['\n'] = idx; idx += 1
        vocab['    '] = idx; idx += 1
        vocab[' '] = idx; idx += 1
        
        # Common identifiers
        for name in ['self', 'cls', 'args', 'kwargs', 'x', 'y', 'z', 'i', 'j', 'n',
                     'foo', 'bar', 'baz', 'func', 'main', 'init', 'data', 'result',
                     'value', 'item', 'key', 'index', 'count', 'total', 'name',
                     'path', 'file', 'obj', 'arr', 'lst', 'dct', 'tmp', 'ret']:
            vocab[name] = idx
            idx += 1
        
        # Built-in functions
        for fn in ['print', 'len', 'range', 'str', 'int', 'float', 'list',
                   'dict', 'set', 'tuple', 'type', 'isinstance', 'hasattr',
                   'getattr', 'setattr', 'open', 'sum', 'max', 'min', 'abs',
                   'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter']:
            vocab[fn] = idx
            idx += 1
        
        # Numbers
        for n in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                  '100', '1000', '0.0', '0.5', '1.0']:
            vocab[n] = idx
            idx += 1
        
        # Common strings
        for s in ['""', "''", '"hello"', "'test'", '"error"', "'name'",
                  '"__main__"', "'utf-8'"]:
            vocab[s] = idx
            idx += 1
        
        # Special
        vocab['<EOS>'] = idx; idx += 1
        vocab['\n\n'] = idx; idx += 1
        vocab['#'] = idx; idx += 1
        
        return vocab
    
    def _go_vocab(self) -> Dict[str, int]:
        """Go vocabulary."""
        vocab = {}
        idx = 0
        
        # Keywords
        for kw in ['package', 'import', 'func', 'type', 'struct', 'interface',
                   'var', 'const', 'if', 'else', 'for', 'range', 'switch',
                   'case', 'default', 'select', 'go', 'defer', 'return',
                   'break', 'continue', 'fallthrough', 'chan', 'map',
                   'true', 'false', 'nil', 'make', 'new', 'len', 'cap',
                   'append', 'copy', 'delete', 'close', 'panic', 'recover']:
            vocab[kw] = idx
            idx += 1
        
        # Operators
        for op in ['+', '-', '*', '/', '%', '&', '|', '^', '<<', '>>',
                   '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!',
                   '=', ':=', '+=', '-=', '*=', '/=', '<-', '...']:
            vocab[op] = idx
            idx += 1
        
        # Delimiters
        for d in ['(', ')', '[', ']', '{', '}', ':', ',', '.', ';']:
            vocab[d] = idx
            idx += 1
        
        # Whitespace
        vocab['\n'] = idx; idx += 1
        vocab['\t'] = idx; idx += 1
        vocab[' '] = idx; idx += 1
        
        # Types
        for t in ['int', 'int8', 'int16', 'int32', 'int64',
                  'uint', 'uint8', 'uint16', 'uint32', 'uint64',
                  'float32', 'float64', 'complex64', 'complex128',
                  'bool', 'string', 'byte', 'rune', 'error']:
            vocab[t] = idx
            idx += 1
        
        # Common identifiers
        for name in ['main', 'err', 'ok', 'ctx', 'req', 'res', 'resp',
                     'i', 'j', 'k', 'n', 'x', 'y', 'v', 's', 't',
                     'data', 'result', 'value', 'name', 'msg', 'buf']:
            vocab[name] = idx
            idx += 1
        
        # Common packages
        for pkg in ['fmt', 'os', 'io', 'net', 'http', 'json', 'time',
                    'context', 'sync', 'errors', 'strings', 'strconv']:
            vocab[pkg] = idx
            idx += 1
        
        # Numbers and strings
        for lit in ['0', '1', '2', '10', '100', '""', '"hello"', '`test`']:
            vocab[lit] = idx
            idx += 1
        
        # Special
        vocab['<EOS>'] = idx; idx += 1
        vocab['\n\n'] = idx; idx += 1
        vocab['//'] = idx; idx += 1
        
        return vocab
    
    def _get_test_prompts(self) -> List[str]:
        """Get test prompts for benchmark."""
        if self.language == "python":
            return [
                "def factorial(n):",
                "def fibonacci(n):",
                "class Calculator:",
                "def process_data(items):",
                "for i in range(10):",
                "if x > 0:",
                "while count < 100:",
                "def __init__(self):",
                "try:",
                "with open(path) as f:",
                "async def fetch_data(url):",
                "lambda x:",
                "def sort_list(arr):",
                "class Node:",
                "def binary_search(arr, target):",
            ]
        else:
            return [
                "package main",
                "func main() {",
                "func factorial(n int) int {",
                "type Config struct {",
                "func (s *Server) Start() error {",
                "for i := 0; i < 10; i++ {",
                "if err != nil {",
                "switch value {",
                "select {",
                "go func() {",
                "defer file.Close()",
                "type Handler interface {",
                "func NewClient(addr string) *Client {",
                "for _, item := range items {",
                "func (c *Client) Get(key string) (string, error) {",
            ]
    
    def run(self, num_samples: int = 100, 
            max_tokens: int = 50,
            compare_unconstrained: bool = True) -> Dict[str, BenchmarkResult]:
        """
        Run benchmark.
        
        Args:
            num_samples: Number of samples to generate
            max_tokens: Max tokens per generation
            compare_unconstrained: Also test without constraints
            
        Returns:
            Dict with 'constrained' and optionally 'unconstrained' results
        """
        results = {}
        
        prompts = self._get_test_prompts()
        
        # Run constrained
        print(f"\n{'='*60}")
        print(f"Running CONSTRAINED benchmark ({num_samples} samples)...")
        print(f"{'='*60}")
        
        constrained_samples = self._run_generation(
            prompts, num_samples, max_tokens, constrained=True
        )
        results['constrained'] = self._compute_results(
            constrained_samples, constrained=True
        )
        
        # Run unconstrained
        if compare_unconstrained:
            print(f"\n{'='*60}")
            print(f"Running UNCONSTRAINED benchmark ({num_samples} samples)...")
            print(f"{'='*60}")
            
            unconstrained_samples = self._run_generation(
                prompts, num_samples, max_tokens, constrained=False
            )
            results['unconstrained'] = self._compute_results(
                unconstrained_samples, constrained=False
            )
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _run_generation(self, prompts: List[str], num_samples: int,
                       max_tokens: int, constrained: bool) -> List[GenerationSample]:
        """Run generation for given prompts."""
        samples = []
        
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=0.8,
            top_k=50,
            enforce_constraints=constrained,
            language=self.language,
            track_blocked=True
        )
        
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]
            
            start_time = time.time()
            code, stats = self.generator.generate_with_stats(prompt, config)
            gen_time = (time.time() - start_time) * 1000  # ms
            
            validation = self.validator.validate(code)
            
            sample = GenerationSample(
                prompt=prompt,
                generated=code,
                constrained=constrained,
                validation=validation,
                tokens_generated=stats['tokens_generated'],
                tokens_blocked=stats['tokens_blocked'],
                generation_time_ms=gen_time,
                final_state=stats['final_state']
            )
            samples.append(sample)
            
            if (i + 1) % 10 == 0:
                valid_so_far = sum(1 for s in samples if s.validation.is_valid) / len(samples)
                print(f"  Progress: {i+1}/{num_samples} ({valid_so_far*100:.1f}% valid)")
        
        return samples
    
    def _compute_results(self, samples: List[GenerationSample], 
                        constrained: bool) -> BenchmarkResult:
        """Compute aggregate results from samples."""
        result = BenchmarkResult(
            language=self.language,
            total_samples=len(samples),
            constrained=constrained
        )
        
        if not samples:
            return result
        
        # Validity rates
        result.parse_success_rate = sum(
            1 for s in samples if s.validation.parse_success
        ) / len(samples)
        
        result.bracket_match_rate = sum(
            1 for s in samples if s.validation.brackets_balanced
        ) / len(samples)
        
        result.indent_valid_rate = sum(
            1 for s in samples if s.validation.indent_valid
        ) / len(samples)
        
        result.overall_valid_rate = sum(
            1 for s in samples if s.validation.is_valid
        ) / len(samples)
        
        # Performance
        result.avg_tokens_per_sample = sum(
            s.tokens_generated for s in samples
        ) / len(samples)
        
        result.avg_tokens_blocked = sum(
            s.tokens_blocked for s in samples
        ) / len(samples)
        
        result.avg_generation_time_ms = sum(
            s.generation_time_ms for s in samples
        ) / len(samples)
        
        total_tokens = sum(s.tokens_generated for s in samples)
        total_time_sec = sum(s.generation_time_ms for s in samples) / 1000
        result.tokens_per_second = total_tokens / total_time_sec if total_time_sec > 0 else 0
        
        # Error analysis
        for s in samples:
            if not s.validation.is_valid:
                error_type = s.validation.error_message.split(':')[0] if s.validation.error_message else 'unknown'
                result.error_types[error_type] = result.error_types.get(error_type, 0) + 1
                if len(result.samples_with_errors) < 10:
                    result.samples_with_errors.append(s)
        
        return result
    
    def _print_comparison(self, results: Dict[str, BenchmarkResult]):
        """Print comparison of results."""
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        
        print(f"\nLanguage: {self.language.upper()}")
        
        headers = ["Metric", "Constrained", "Unconstrained", "Delta"]
        
        rows = []
        
        c = results['constrained']
        u = results.get('unconstrained')
        
        def fmt_pct(v):
            return f"{v*100:.1f}%"
        
        def fmt_delta(c_val, u_val):
            if u_val is None:
                return "-"
            delta = (c_val - u_val) * 100
            sign = "+" if delta >= 0 else ""
            return f"{sign}{delta:.1f}pp"
        
        rows.append([
            "Parse Success",
            fmt_pct(c.parse_success_rate),
            fmt_pct(u.parse_success_rate) if u else "-",
            fmt_delta(c.parse_success_rate, u.parse_success_rate if u else None)
        ])
        
        rows.append([
            "Brackets Balanced",
            fmt_pct(c.bracket_match_rate),
            fmt_pct(u.bracket_match_rate) if u else "-",
            fmt_delta(c.bracket_match_rate, u.bracket_match_rate if u else None)
        ])
        
        rows.append([
            "Overall Valid",
            fmt_pct(c.overall_valid_rate),
            fmt_pct(u.overall_valid_rate) if u else "-",
            fmt_delta(c.overall_valid_rate, u.overall_valid_rate if u else None)
        ])
        
        rows.append([
            "Avg Tokens/Sample",
            f"{c.avg_tokens_per_sample:.1f}",
            f"{u.avg_tokens_per_sample:.1f}" if u else "-",
            "-"
        ])
        
        rows.append([
            "Avg Tokens Blocked",
            f"{c.avg_tokens_blocked:.1f}",
            f"{u.avg_tokens_blocked:.1f}" if u else "-",
            "-"
        ])
        
        rows.append([
            "Tokens/sec",
            f"{c.tokens_per_second:.1f}",
            f"{u.tokens_per_second:.1f}" if u else "-",
            "-"
        ])
        
        # Print table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(4)]
        
        def print_row(row):
            print("| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(4)) + " |")
        
        print("-" * (sum(col_widths) + 13))
        print_row(headers)
        print("-" * (sum(col_widths) + 13))
        for row in rows:
            print_row(row)
        print("-" * (sum(col_widths) + 13))
        
        # Target check
        print(f"\n{'='*60}")
        if c.overall_valid_rate >= 0.99:
            print("TARGET ACHIEVED: 99%+ syntactic validity!")
        elif c.overall_valid_rate >= 0.95:
            print(f"CLOSE TO TARGET: {c.overall_valid_rate*100:.1f}% validity (target: 99%)")
        else:
            print(f"BELOW TARGET: {c.overall_valid_rate*100:.1f}% validity (target: 99%)")
        print(f"{'='*60}")
        
        # Error examples
        if c.samples_with_errors:
            print(f"\nError examples (constrained):")
            for s in c.samples_with_errors[:3]:
                print(f"  Prompt: {s.prompt}")
                print(f"  Error: {s.validation.error_message}")
                print()


def main():
    parser = argparse.ArgumentParser(description='Benchmark statechart-guided code generation')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--max-tokens', type=int, default=30, help='Max tokens per generation')
    parser.add_argument('--language', choices=['python', 'go'], default='python')
    parser.add_argument('--no-compare', action='store_true', help='Skip unconstrained comparison')
    parser.add_argument('--output', type=str, help='Save results to JSON file')
    
    args = parser.parse_args()
    
    benchmark = CodeCompletionBenchmark(language=args.language)
    results = benchmark.run(
        num_samples=args.samples,
        max_tokens=args.max_tokens,
        compare_unconstrained=not args.no_compare
    )
    
    if args.output:
        # Serialize results (convert dataclasses to dicts)
        output_data = {}
        for key, result in results.items():
            output_data[key] = {
                'language': result.language,
                'total_samples': result.total_samples,
                'constrained': result.constrained,
                'parse_success_rate': result.parse_success_rate,
                'bracket_match_rate': result.bracket_match_rate,
                'overall_valid_rate': result.overall_valid_rate,
                'avg_tokens_per_sample': result.avg_tokens_per_sample,
                'avg_tokens_blocked': result.avg_tokens_blocked,
                'tokens_per_second': result.tokens_per_second,
                'error_types': result.error_types,
            }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
