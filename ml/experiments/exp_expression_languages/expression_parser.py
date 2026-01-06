"""
Expression Parser - Multi-language guard expression parsing.

Supports:
- RAW: Simple Python-like expressions
- CEL: Common Expression Language
- STARLARK: Google's configuration language
- JAVASCRIPT: JS expressions (simulated)
- GO: Go expressions (simulated)

Each parser extracts:
- Variables referenced
- Operators used
- Function calls
- Expression AST structure
"""

import re
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto
from abc import ABC, abstractmethod


class ExpressionLanguage(Enum):
    """Supported expression languages."""
    RAW = auto()        # Simple Python-like
    CEL = auto()        # Common Expression Language
    STARLARK = auto()   # Starlark/Skylark
    JAVASCRIPT = auto() # JavaScript expressions
    GO = auto()         # Go expressions


@dataclass
class ParsedExpression:
    """Parsed representation of a guard expression."""
    source: str
    language: ExpressionLanguage
    variables: Set[str] = field(default_factory=set)
    operators: Set[str] = field(default_factory=set)
    function_calls: List[str] = field(default_factory=list)
    literals: List[Any] = field(default_factory=list)
    depth: int = 0
    is_valid: bool = True
    error: Optional[str] = None
    ast_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'language': self.language.name,
            'variables': list(self.variables),
            'operators': list(self.operators),
            'function_calls': self.function_calls,
            'literals': [str(l) for l in self.literals],
            'depth': self.depth,
            'is_valid': self.is_valid,
            'error': self.error,
        }


class ExpressionParser(ABC):
    """Abstract base for language-specific parsers."""

    @abstractmethod
    def parse(self, source: str) -> ParsedExpression:
        """Parse expression and extract components."""
        pass

    @abstractmethod
    def evaluate(self, source: str, context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Evaluate expression in context. Returns (result, error)."""
        pass


class RawExpressionParser(ExpressionParser):
    """Parser for RAW (Python-like) expressions."""

    OPERATORS = {
        'and', 'or', 'not', '==', '!=', '<', '>', '<=', '>=',
        '+', '-', '*', '/', '%', 'in', 'is'
    }

    def parse(self, source: str) -> ParsedExpression:
        result = ParsedExpression(source=source, language=ExpressionLanguage.RAW)

        try:
            tree = ast.parse(source, mode='eval')
            self._walk_ast(tree.body, result, depth=0)
            result.depth = self._compute_depth(tree.body)
        except SyntaxError as e:
            result.is_valid = False
            result.error = str(e)

        return result

    def _walk_ast(self, node: ast.AST, result: ParsedExpression, depth: int):
        """Walk AST and extract components."""
        result.ast_nodes.append(type(node).__name__)

        if isinstance(node, ast.Name):
            result.variables.add(node.id)
        elif isinstance(node, ast.Constant):
            result.literals.append(node.value)
        elif isinstance(node, ast.BoolOp):
            op_name = 'and' if isinstance(node.op, ast.And) else 'or'
            result.operators.add(op_name)
            for value in node.values:
                self._walk_ast(value, result, depth + 1)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                result.operators.add('not')
            self._walk_ast(node.operand, result, depth + 1)
        elif isinstance(node, ast.Compare):
            self._walk_ast(node.left, result, depth + 1)
            for op, comparator in zip(node.ops, node.comparators):
                result.operators.add(self._op_to_str(op))
                self._walk_ast(comparator, result, depth + 1)
        elif isinstance(node, ast.BinOp):
            result.operators.add(self._op_to_str(node.op))
            self._walk_ast(node.left, result, depth + 1)
            self._walk_ast(node.right, result, depth + 1)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                result.function_calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                result.function_calls.append(f"{self._get_attr_name(node.func)}")
            for arg in node.args:
                self._walk_ast(arg, result, depth + 1)
        elif isinstance(node, ast.Attribute):
            result.variables.add(self._get_attr_name(node))
        elif isinstance(node, ast.Subscript):
            self._walk_ast(node.value, result, depth + 1)
            self._walk_ast(node.slice, result, depth + 1)
        elif isinstance(node, ast.IfExp):
            self._walk_ast(node.test, result, depth + 1)
            self._walk_ast(node.body, result, depth + 1)
            self._walk_ast(node.orelse, result, depth + 1)

    def _op_to_str(self, op: ast.AST) -> str:
        """Convert AST operator to string."""
        op_map = {
            ast.Eq: '==', ast.NotEq: '!=', ast.Lt: '<', ast.Gt: '>',
            ast.LtE: '<=', ast.GtE: '>=', ast.In: 'in', ast.Is: 'is',
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.And: 'and', ast.Or: 'or', ast.Not: 'not',
        }
        return op_map.get(type(op), str(type(op).__name__))

    def _get_attr_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'obj.attr')."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attr_name(node.value)}.{node.attr}"
        return node.attr

    def _compute_depth(self, node: ast.AST, current: int = 0) -> int:
        """Compute expression depth."""
        max_depth = current
        for child in ast.iter_child_nodes(node):
            child_depth = self._compute_depth(child, current + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth

    def evaluate(self, source: str, context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Evaluate expression safely."""
        try:
            tree = ast.parse(source, mode='eval')
            # Compile and evaluate with restricted context
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, context)
            return result, None
        except Exception as e:
            return None, str(e)


class CELExpressionParser(ExpressionParser):
    """Parser for CEL (Common Expression Language) expressions.

    CEL syntax differences from Python:
    - Ternary: condition ? true_val : false_val
    - String membership: 'abc'.contains('a')
    - List membership: x in [1, 2, 3]
    - Type checking: type(x) == int
    """

    def parse(self, source: str) -> ParsedExpression:
        result = ParsedExpression(source=source, language=ExpressionLanguage.CEL)

        try:
            # Convert CEL to Python-like for parsing
            python_source = self._cel_to_python(source)
            tree = ast.parse(python_source, mode='eval')
            self._walk_ast(tree.body, result, depth=0)
            result.depth = self._compute_depth(tree.body)
        except SyntaxError as e:
            result.is_valid = False
            result.error = str(e)

        return result

    def _cel_to_python(self, source: str) -> str:
        """Convert CEL syntax to Python-like for parsing."""
        result = source

        # Convert logical operators first
        result = result.replace('&&', ' and ')
        result = result.replace('||', ' or ')
        result = re.sub(r'!(?!=)', ' not ', result)

        # Convert true/false/null
        result = re.sub(r'\btrue\b', 'True', result)
        result = re.sub(r'\bfalse\b', 'False', result)
        result = re.sub(r'\bnull\b', 'None', result)

        # Handle ternary (simplified - doesn't handle nested)
        ternary_pattern = r'(\([^)]+\)|[^?]+)\s*\?\s*([^:]+)\s*:\s*(.+)'
        match = re.match(ternary_pattern, result)
        if match:
            cond, true_val, false_val = match.groups()
            result = f"({true_val.strip()}) if ({cond.strip()}) else ({false_val.strip()})"

        # Convert .contains() to 'in' (simplified)
        result = re.sub(r"'([^']+)'\.contains\(([^)]+)\)", r'\2 in "\1"', result)

        return result

    def _walk_ast(self, node: ast.AST, result: ParsedExpression, depth: int):
        """Walk AST (reuse RAW parser logic)."""
        raw_parser = RawExpressionParser()
        raw_parser._walk_ast(node, result, depth)

    def _compute_depth(self, node: ast.AST, current: int = 0) -> int:
        """Compute depth (reuse RAW parser logic)."""
        raw_parser = RawExpressionParser()
        return raw_parser._compute_depth(node, current)

    def evaluate(self, source: str, context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Evaluate CEL expression."""
        try:
            python_source = self._cel_to_python(source)
            tree = ast.parse(python_source, mode='eval')
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, context)
            return result, None
        except Exception as e:
            return None, str(e)


class StarlarkExpressionParser(ExpressionParser):
    """Parser for Starlark expressions.

    Starlark is Python-like but with restrictions:
    - No while loops (only for)
    - No global variables
    - No recursion
    - Deterministic iteration order
    """

    ALLOWED_BUILTINS = {
        'len', 'str', 'int', 'float', 'bool', 'list', 'dict',
        'range', 'enumerate', 'zip', 'sorted', 'reversed',
        'min', 'max', 'sum', 'all', 'any', 'hasattr', 'getattr',
    }

    def parse(self, source: str) -> ParsedExpression:
        result = ParsedExpression(source=source, language=ExpressionLanguage.STARLARK)

        try:
            tree = ast.parse(source, mode='eval')
            self._walk_ast(tree.body, result, depth=0)
            result.depth = self._compute_depth(tree.body)

            # Check for disallowed constructs
            self._check_starlark_restrictions(tree.body, result)
        except SyntaxError as e:
            result.is_valid = False
            result.error = str(e)

        return result

    def _walk_ast(self, node: ast.AST, result: ParsedExpression, depth: int):
        """Walk AST (reuse RAW parser logic)."""
        raw_parser = RawExpressionParser()
        raw_parser._walk_ast(node, result, depth)

    def _compute_depth(self, node: ast.AST, current: int = 0) -> int:
        """Compute depth."""
        raw_parser = RawExpressionParser()
        return raw_parser._compute_depth(node, current)

    def _check_starlark_restrictions(self, node: ast.AST, result: ParsedExpression):
        """Check for Starlark-disallowed constructs."""
        # Check function calls against whitelist
        for func in result.function_calls:
            base_func = func.split('.')[0]
            if base_func not in self.ALLOWED_BUILTINS and base_func not in result.variables:
                result.is_valid = False
                result.error = f"Function '{func}' not allowed in Starlark"

    def evaluate(self, source: str, context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Evaluate Starlark expression."""
        try:
            tree = ast.parse(source, mode='eval')
            # Starlark uses restricted builtins
            import builtins
            safe_builtins = {k: getattr(builtins, k) for k in self.ALLOWED_BUILTINS
                           if hasattr(builtins, k)}
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": safe_builtins}, context)
            return result, None
        except Exception as e:
            return None, str(e)


class JavaScriptExpressionParser(ExpressionParser):
    """Parser for JavaScript expressions (simulated via translation to Python)."""

    def parse(self, source: str) -> ParsedExpression:
        result = ParsedExpression(source=source, language=ExpressionLanguage.JAVASCRIPT)

        try:
            python_source = self._js_to_python(source)
            tree = ast.parse(python_source, mode='eval')
            self._walk_ast(tree.body, result, depth=0)
            result.depth = self._compute_depth(tree.body)
        except SyntaxError as e:
            result.is_valid = False
            result.error = str(e)

        return result

    def _js_to_python(self, source: str) -> str:
        """Convert JavaScript syntax to Python-like."""
        result = source

        # Convert === to ==
        result = result.replace('===', '==')
        result = result.replace('!==', '!=')

        # Convert && to and, || to or
        result = result.replace('&&', ' and ')
        result = result.replace('||', ' or ')

        # Convert ! to not (careful with !=)
        result = re.sub(r'!(?!=)', ' not ', result)

        # Convert ternary: condition ? a : b
        ternary_pattern = r'([^?]+)\s*\?\s*([^:]+)\s*:\s*(.+)'
        match = re.match(ternary_pattern, result)
        if match:
            cond, true_val, false_val = match.groups()
            result = f"({true_val.strip()}) if ({cond.strip()}) else ({false_val.strip()})"

        # Convert true/false to True/False
        result = re.sub(r'\btrue\b', 'True', result)
        result = re.sub(r'\bfalse\b', 'False', result)
        result = re.sub(r'\bnull\b', 'None', result)

        return result

    def _walk_ast(self, node: ast.AST, result: ParsedExpression, depth: int):
        raw_parser = RawExpressionParser()
        raw_parser._walk_ast(node, result, depth)

    def _compute_depth(self, node: ast.AST, current: int = 0) -> int:
        raw_parser = RawExpressionParser()
        return raw_parser._compute_depth(node, current)

    def evaluate(self, source: str, context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Evaluate JavaScript expression (via Python translation)."""
        try:
            python_source = self._js_to_python(source)
            tree = ast.parse(python_source, mode='eval')
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, context)
            return result, None
        except Exception as e:
            return None, str(e)


class GoExpressionParser(ExpressionParser):
    """Parser for Go expressions (simulated via translation to Python)."""

    def parse(self, source: str) -> ParsedExpression:
        result = ParsedExpression(source=source, language=ExpressionLanguage.GO)

        try:
            python_source = self._go_to_python(source)
            tree = ast.parse(python_source, mode='eval')
            self._walk_ast(tree.body, result, depth=0)
            result.depth = self._compute_depth(tree.body)
        except SyntaxError as e:
            result.is_valid = False
            result.error = str(e)

        return result

    def _go_to_python(self, source: str) -> str:
        """Convert Go syntax to Python-like."""
        result = source

        # Convert && to and, || to or
        result = result.replace('&&', ' and ')
        result = result.replace('||', ' or ')

        # Convert ! to not (careful with !=)
        result = re.sub(r'!(?!=)', ' not ', result)

        # Convert true/false
        result = re.sub(r'\btrue\b', 'True', result)
        result = re.sub(r'\bfalse\b', 'False', result)
        result = re.sub(r'\bnil\b', 'None', result)

        return result

    def _walk_ast(self, node: ast.AST, result: ParsedExpression, depth: int):
        raw_parser = RawExpressionParser()
        raw_parser._walk_ast(node, result, depth)

    def _compute_depth(self, node: ast.AST, current: int = 0) -> int:
        raw_parser = RawExpressionParser()
        return raw_parser._compute_depth(node, current)

    def evaluate(self, source: str, context: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        """Evaluate Go expression (via Python translation)."""
        try:
            python_source = self._go_to_python(source)
            tree = ast.parse(python_source, mode='eval')
            code = compile(tree, '<string>', 'eval')
            result = eval(code, {"__builtins__": {}}, context)
            return result, None
        except Exception as e:
            return None, str(e)


class MultiLanguageParser:
    """Unified parser supporting all expression languages."""

    def __init__(self):
        self.parsers: Dict[ExpressionLanguage, ExpressionParser] = {
            ExpressionLanguage.RAW: RawExpressionParser(),
            ExpressionLanguage.CEL: CELExpressionParser(),
            ExpressionLanguage.STARLARK: StarlarkExpressionParser(),
            ExpressionLanguage.JAVASCRIPT: JavaScriptExpressionParser(),
            ExpressionLanguage.GO: GoExpressionParser(),
        }

    def parse(self, source: str, language: ExpressionLanguage) -> ParsedExpression:
        """Parse expression in specified language."""
        parser = self.parsers.get(language)
        if not parser:
            return ParsedExpression(
                source=source,
                language=language,
                is_valid=False,
                error=f"Unsupported language: {language}"
            )
        return parser.parse(source)

    def evaluate(
        self,
        source: str,
        language: ExpressionLanguage,
        context: Dict[str, Any]
    ) -> Tuple[Any, Optional[str]]:
        """Evaluate expression in specified language."""
        parser = self.parsers.get(language)
        if not parser:
            return None, f"Unsupported language: {language}"
        return parser.evaluate(source, context)

    def parse_all_languages(self, expressions: Dict[ExpressionLanguage, str]) -> Dict[ExpressionLanguage, ParsedExpression]:
        """Parse same logical expression in multiple languages."""
        return {lang: self.parse(source, lang) for lang, source in expressions.items()}


def demo():
    """Demonstrate multi-language parsing."""
    print("=" * 60)
    print("MULTI-LANGUAGE EXPRESSION PARSER")
    print("=" * 60)

    parser = MultiLanguageParser()

    # Same guard in different languages
    guards = {
        ExpressionLanguage.RAW: "x > 0 and y < 10",
        ExpressionLanguage.CEL: "x > 0 && y < 10",
        ExpressionLanguage.STARLARK: "x > 0 and y < 10",
        ExpressionLanguage.JAVASCRIPT: "x > 0 && y < 10",
        ExpressionLanguage.GO: "x > 0 && y < 10",
    }

    print("\nParsing guard: 'x > 0 AND y < 10'")
    print("-" * 60)

    for lang, source in guards.items():
        parsed = parser.parse(source, lang)
        print(f"\n{lang.name}:")
        print(f"  Source: {source}")
        print(f"  Variables: {parsed.variables}")
        print(f"  Operators: {parsed.operators}")
        print(f"  Depth: {parsed.depth}")
        print(f"  Valid: {parsed.is_valid}")

    # Evaluate in context
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    context = {"x": 5, "y": 3}
    print(f"\nContext: {context}")

    for lang, source in guards.items():
        result, error = parser.evaluate(source, lang, context)
        status = f"Result: {result}" if error is None else f"Error: {error}"
        print(f"  {lang.name}: {status}")

    return parser


if __name__ == "__main__":
    demo()
