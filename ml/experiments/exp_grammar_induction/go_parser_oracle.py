"""
Go Parser Oracle - Ground Truth Comparison

Uses go/parser via subprocess to validate learned grammar decisions.
Measures accept/reject accuracy against the real Go compiler.

Key Functions:
- Validate code snippets against go/parser
- Generate valid/invalid code pairs for testing
- Compare induced grammar decisions with oracle
"""

import subprocess
import tempfile
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class ParseResult(Enum):
    """Result of parsing a code snippet."""
    VALID = "valid"           # Parsed successfully
    INVALID = "invalid"       # Parse error
    ERROR = "error"           # System error (not a parse result)


@dataclass
class ParseAttempt:
    """A single parse attempt with result."""
    code: str
    result: ParseResult
    error_message: str = ""
    error_line: int = 0
    error_column: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'code': self.code,
            'result': self.result.value,
            'error_message': self.error_message,
            'error_line': self.error_line,
            'error_column': self.error_column,
        }


# Go program for parsing validation
PARSER_GO_CODE = '''
package main

import (
    "encoding/json"
    "fmt"
    "go/parser"
    "go/token"
    "io"
    "os"
    "strings"
)

type ParseResult struct {
    Valid   bool   `json:"valid"`
    Error   string `json:"error,omitempty"`
    Line    int    `json:"line,omitempty"`
    Column  int    `json:"column,omitempty"`
}

func main() {
    // Read code from stdin
    code, err := io.ReadAll(os.Stdin)
    if err != nil {
        result := ParseResult{Valid: false, Error: fmt.Sprintf("read error: %v", err)}
        json.NewEncoder(os.Stdout).Encode(result)
        os.Exit(0)
    }

    // Create file set
    fset := token.NewFileSet()
    
    // Try to parse
    _, err = parser.ParseFile(fset, "input.go", code, parser.AllErrors)
    
    if err != nil {
        // Extract position info from error
        errStr := err.Error()
        line, col := 0, 0
        
        // Parse error position (format: "input.go:line:col: message")
        if strings.HasPrefix(errStr, "input.go:") {
            parts := strings.SplitN(errStr[9:], ":", 3)
            if len(parts) >= 2 {
                fmt.Sscanf(parts[0], "%d", &line)
                fmt.Sscanf(parts[1], "%d", &col)
            }
        }
        
        result := ParseResult{
            Valid:  false,
            Error:  errStr,
            Line:   line,
            Column: col,
        }
        json.NewEncoder(os.Stdout).Encode(result)
    } else {
        result := ParseResult{Valid: true}
        json.NewEncoder(os.Stdout).Encode(result)
    }
}
'''


class GoParserOracle:
    """
    Oracle for validating Go code using the real parser.
    
    Provides ground truth for induced grammar validation.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix='go_oracle_')
        self.parser_path = os.path.join(self.cache_dir, 'parser_oracle')
        self._build_parser()
        
        # Statistics
        self.total_checks = 0
        self.valid_count = 0
        self.invalid_count = 0
        self.error_count = 0
    
    def _build_parser(self):
        """Build the Go parser binary."""
        go_file = os.path.join(self.cache_dir, 'parser_oracle.go')
        with open(go_file, 'w') as f:
            f.write(PARSER_GO_CODE)
        
        result = subprocess.run(
            ['go', 'build', '-o', self.parser_path, go_file],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build parser oracle: {result.stderr}")
    
    def parse(self, code: str) -> ParseAttempt:
        """
        Parse a code snippet and return the result.
        
        Args:
            code: Go source code to parse
        
        Returns:
            ParseAttempt with result and error info
        """
        self.total_checks += 1
        
        try:
            result = subprocess.run(
                [self.parser_path],
                input=code,
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode != 0:
                self.error_count += 1
                return ParseAttempt(
                    code=code,
                    result=ParseResult.ERROR,
                    error_message=f"Parser failed: {result.stderr}",
                )
            
            try:
                data = json.loads(result.stdout)
                
                if data.get('valid'):
                    self.valid_count += 1
                    return ParseAttempt(
                        code=code,
                        result=ParseResult.VALID,
                    )
                else:
                    self.invalid_count += 1
                    return ParseAttempt(
                        code=code,
                        result=ParseResult.INVALID,
                        error_message=data.get('error', ''),
                        error_line=data.get('line', 0),
                        error_column=data.get('column', 0),
                    )
            except json.JSONDecodeError:
                self.error_count += 1
                return ParseAttempt(
                    code=code,
                    result=ParseResult.ERROR,
                    error_message=f"Invalid JSON: {result.stdout}",
                )
        
        except subprocess.TimeoutExpired:
            self.error_count += 1
            return ParseAttempt(
                code=code,
                result=ParseResult.ERROR,
                error_message="Parse timeout",
            )
        except Exception as e:
            self.error_count += 1
            return ParseAttempt(
                code=code,
                result=ParseResult.ERROR,
                error_message=str(e),
            )
    
    def is_valid(self, code: str) -> bool:
        """Check if code is valid Go."""
        return self.parse(code).result == ParseResult.VALID
    
    def batch_parse(self, codes: List[str]) -> List[ParseAttempt]:
        """Parse multiple code snippets."""
        return [self.parse(code) for code in codes]
    
    def validate_statement(self, stmt: str) -> ParseAttempt:
        """
        Validate a single statement in a minimal context.
        
        Wraps the statement in a valid package and function.
        """
        wrapped = f'''package main

func main() {{
    {stmt}
}}
'''
        return self.parse(wrapped)
    
    def validate_expression(self, expr: str) -> ParseAttempt:
        """
        Validate an expression in a minimal context.
        
        Wraps the expression in an assignment.
        """
        wrapped = f'''package main

func main() {{
    _ = {expr}
}}
'''
        return self.parse(wrapped)
    
    def validate_declaration(self, decl: str) -> ParseAttempt:
        """
        Validate a declaration in a minimal context.
        
        Places the declaration at package level.
        """
        wrapped = f'''package main

{decl}
'''
        return self.parse(wrapped)
    
    def generate_invalid_variants(self, valid_code: str) -> List[Tuple[str, str]]:
        """
        Generate invalid variants of valid code.
        
        Useful for testing grammar's ability to reject invalid code.
        
        Args:
            valid_code: Known-valid Go code
        
        Returns:
            List of (invalid_code, description) tuples
        """
        variants = []
        
        # Missing semicolons (Go is semicolon-free but has rules)
        if '\n' in valid_code:
            lines = valid_code.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().endswith(('{', '}', ',')):
                    broken = lines.copy()
                    broken[i] = line + ' ' + (lines[i+1] if i+1 < len(lines) else '')
                    if i + 1 < len(broken):
                        broken.pop(i + 1)
                    variants.append(('\n'.join(broken), f"merged lines {i} and {i+1}"))
                    break
        
        # Missing braces
        for char, repl in [('{', ''), ('}', '')]:
            if char in valid_code:
                idx = valid_code.index(char)
                broken = valid_code[:idx] + repl + valid_code[idx+1:]
                variants.append((broken, f"removed '{char}'"))
                break
        
        # Typos in keywords
        keywords = ['func', 'return', 'if', 'for', 'package', 'import']
        for kw in keywords:
            if f' {kw} ' in valid_code or valid_code.startswith(f'{kw} '):
                broken = valid_code.replace(kw, kw[:-1] + 'x', 1)
                variants.append((broken, f"typo in '{kw}'"))
                break
        
        # Wrong operators
        operators = [':=', '==', '!=', '<=', '>=']
        for op in operators:
            if op in valid_code:
                broken = valid_code.replace(op, op[0], 1)
                variants.append((broken, f"truncated '{op}'"))
                break
        
        return variants
    
    def get_statistics(self) -> Dict:
        """Get parsing statistics."""
        return {
            'total_checks': self.total_checks,
            'valid_count': self.valid_count,
            'invalid_count': self.invalid_count,
            'error_count': self.error_count,
            'valid_rate': self.valid_count / max(self.total_checks, 1),
        }


class GrammarValidator:
    """
    Validates an induced grammar against the oracle.
    
    Compares the grammar's accept/reject decisions with go/parser.
    """
    
    def __init__(self, oracle: Optional[GoParserOracle] = None):
        self.oracle = oracle or GoParserOracle()
        
        # Validation results
        self.true_positives = 0   # Grammar accepts, oracle accepts
        self.true_negatives = 0  # Grammar rejects, oracle rejects
        self.false_positives = 0 # Grammar accepts, oracle rejects
        self.false_negatives = 0 # Grammar rejects, oracle accepts
    
    def validate(
        self,
        code: str,
        grammar_accepts: bool,
    ) -> bool:
        """
        Validate a grammar decision against the oracle.
        
        Args:
            code: Code snippet
            grammar_accepts: Whether the induced grammar accepts this code
        
        Returns:
            True if grammar agrees with oracle
        """
        oracle_result = self.oracle.parse(code)
        oracle_accepts = oracle_result.result == ParseResult.VALID
        
        if grammar_accepts and oracle_accepts:
            self.true_positives += 1
            return True
        elif not grammar_accepts and not oracle_accepts:
            self.true_negatives += 1
            return True
        elif grammar_accepts and not oracle_accepts:
            self.false_positives += 1
            return False
        else:  # not grammar_accepts and oracle_accepts
            self.false_negatives += 1
            return False
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        total = (self.true_positives + self.true_negatives + 
                 self.false_positives + self.false_negatives)
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1(self) -> float:
        """F1 score."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)
    
    def get_metrics(self) -> Dict:
        """Get all validation metrics."""
        return {
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
        }


def demo():
    """Demonstrate the parser oracle."""
    print("=" * 60)
    print("GO PARSER ORACLE DEMO")
    print("=" * 60)
    
    oracle = GoParserOracle()
    
    # Test valid code
    valid_codes = [
        'package main\n\nfunc main() {}',
        'package main\n\nimport "fmt"\n\nfunc main() { fmt.Println("hello") }',
        'package main\n\nvar x = 42',
        'package main\n\ntype Point struct { X, Y int }',
        'package main\n\nfunc add(a, b int) int { return a + b }',
    ]
    
    print("\n=== VALID CODE TESTS ===")
    for code in valid_codes:
        result = oracle.parse(code)
        status = "✓" if result.result == ParseResult.VALID else "✗"
        print(f"{status} {code[:40].replace(chr(10), ' ')!r}...")
    
    # Test invalid code
    invalid_codes = [
        'package main\n\nfunc main( {}',  # Missing )
        'package main\n\nfunc main) {}',  # Missing (
        'package main\n\nfunc main() }',  # Missing {
        'package main\n\nfunc { }',        # Missing name
        'packag main',                      # Typo
    ]
    
    print("\n=== INVALID CODE TESTS ===")
    for code in invalid_codes:
        result = oracle.parse(code)
        status = "✓" if result.result == ParseResult.INVALID else "✗"
        print(f"{status} {code[:40].replace(chr(10), ' ')!r}...")
        if result.error_message:
            print(f"    Error: {result.error_message[:60]}...")
    
    # Test statement validation
    print("\n=== STATEMENT VALIDATION ===")
    statements = [
        'x := 42',
        'if x > 0 { return }',
        'for i := 0; i < 10; i++ { }',
        'x := ',  # Invalid
    ]
    
    for stmt in statements:
        result = oracle.validate_statement(stmt)
        status = "✓" if result.result == ParseResult.VALID else "✗"
        print(f"{status} {stmt!r}")
    
    # Test expression validation
    print("\n=== EXPRESSION VALIDATION ===")
    expressions = [
        '1 + 2',
        'foo.bar()',
        'map[string]int{}',
        '+ +',  # Invalid
    ]
    
    for expr in expressions:
        result = oracle.validate_expression(expr)
        status = "✓" if result.result == ParseResult.VALID else "✗"
        print(f"{status} {expr!r}")
    
    # Generate invalid variants
    print("\n=== INVALID VARIANT GENERATION ===")
    valid = 'package main\n\nfunc main() {\n\tx := 42\n}'
    variants = oracle.generate_invalid_variants(valid)
    print(f"Generated {len(variants)} invalid variants from valid code")
    for broken, desc in variants[:3]:
        result = oracle.parse(broken)
        status = "✓" if result.result == ParseResult.INVALID else "✗"
        print(f"{status} {desc}: {result.result.value}")
    
    # Statistics
    print("\n=== STATISTICS ===")
    stats = oracle.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
