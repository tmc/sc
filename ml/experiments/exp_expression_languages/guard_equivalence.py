"""
Guard Equivalence - Semantic comparison of guards across expression languages.

Tests whether guards written in different languages produce identical behavior:
1. Structural equivalence: Same AST structure after normalization
2. Semantic equivalence: Same results for all test inputs
3. Behavioral equivalence: Same results on property-based random inputs

Key insight: Guards are semantically equivalent if they:
- Reference the same variables
- Use equivalent operators
- Produce the same output for all possible inputs
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
from itertools import product

from .expression_parser import (
    ExpressionLanguage,
    ParsedExpression,
    MultiLanguageParser,
)


class EquivalenceLevel(Enum):
    """Level of equivalence between expressions."""
    NOT_EQUIVALENT = auto()      # Different behavior
    STRUCTURALLY_SIMILAR = auto()  # Same variables/operators
    SEMANTICALLY_EQUIVALENT = auto()  # Same results on test inputs
    PROVABLY_EQUIVALENT = auto()  # Formally proven equivalent


@dataclass
class EquivalenceResult:
    """Result of equivalence comparison."""
    expr1_lang: ExpressionLanguage
    expr1_source: str
    expr2_lang: ExpressionLanguage
    expr2_source: str
    level: EquivalenceLevel
    structural_similarity: float  # 0-1 score
    semantic_match_rate: float    # % of test cases that match
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def is_equivalent(self) -> bool:
        return self.level in (EquivalenceLevel.SEMANTICALLY_EQUIVALENT,
                              EquivalenceLevel.PROVABLY_EQUIVALENT)

    def to_dict(self) -> Dict:
        return {
            'expr1': f"{self.expr1_lang.name}: {self.expr1_source}",
            'expr2': f"{self.expr2_lang.name}: {self.expr2_source}",
            'level': self.level.name,
            'structural_similarity': self.structural_similarity,
            'semantic_match_rate': self.semantic_match_rate,
            'n_counterexamples': len(self.counterexamples),
            'is_equivalent': self.is_equivalent(),
        }


@dataclass
class GuardEquivalenceSet:
    """A set of equivalent guards across languages."""
    canonical_form: str  # Normalized representation
    guards: Dict[ExpressionLanguage, str] = field(default_factory=dict)
    variables: Set[str] = field(default_factory=set)
    verified_equivalent: bool = False


class StructuralComparator:
    """Compare expressions structurally."""

    def __init__(self, parser: MultiLanguageParser):
        self.parser = parser

    def compare_structure(
        self,
        parsed1: ParsedExpression,
        parsed2: ParsedExpression,
    ) -> float:
        """Compute structural similarity score (0-1)."""
        if not parsed1.is_valid or not parsed2.is_valid:
            return 0.0

        scores = []

        # Variable overlap
        if parsed1.variables or parsed2.variables:
            var_overlap = len(parsed1.variables & parsed2.variables)
            var_total = len(parsed1.variables | parsed2.variables)
            scores.append(var_overlap / var_total if var_total > 0 else 1.0)

        # Operator overlap
        if parsed1.operators or parsed2.operators:
            op_overlap = len(parsed1.operators & parsed2.operators)
            op_total = len(parsed1.operators | parsed2.operators)
            scores.append(op_overlap / op_total if op_total > 0 else 1.0)

        # Function call overlap
        funcs1 = set(parsed1.function_calls)
        funcs2 = set(parsed2.function_calls)
        if funcs1 or funcs2:
            func_overlap = len(funcs1 & funcs2)
            func_total = len(funcs1 | funcs2)
            scores.append(func_overlap / func_total if func_total > 0 else 1.0)

        # Depth similarity (penalize large differences)
        depth_diff = abs(parsed1.depth - parsed2.depth)
        max_depth = max(parsed1.depth, parsed2.depth, 1)
        scores.append(1.0 - (depth_diff / max_depth))

        return sum(scores) / len(scores) if scores else 0.0


class SemanticComparator:
    """Compare expressions semantically via evaluation."""

    def __init__(self, parser: MultiLanguageParser, seed: int = 42):
        self.parser = parser
        self.rng = random.Random(seed)

    def generate_test_inputs(
        self,
        variables: Set[str],
        n_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """Generate test inputs covering variable space."""
        inputs = []

        # Boolean domain
        bool_vars = [v for v in variables if v.startswith('is_') or v.startswith('has_')]
        other_vars = [v for v in variables if v not in bool_vars]

        # Enumerate boolean combinations (up to limit)
        if bool_vars:
            n_bool_combos = min(2 ** len(bool_vars), n_samples // 2)
            for i in range(n_bool_combos):
                context = {}
                for j, var in enumerate(bool_vars):
                    context[var] = bool((i >> j) & 1)
                # Add numeric defaults for other vars
                for var in other_vars:
                    context[var] = 0
                inputs.append(context)

        # Random numeric sampling
        n_numeric = n_samples - len(inputs)
        for _ in range(n_numeric):
            context = {}
            for var in bool_vars:
                context[var] = self.rng.choice([True, False])
            for var in other_vars:
                # Mix of integers and edge cases
                context[var] = self.rng.choice([
                    0, 1, -1, 10, -10, 100,
                    self.rng.randint(-100, 100),
                    self.rng.uniform(-10, 10),
                ])
            inputs.append(context)

        return inputs

    def compare_semantic(
        self,
        expr1: str,
        lang1: ExpressionLanguage,
        expr2: str,
        lang2: ExpressionLanguage,
        test_inputs: List[Dict[str, Any]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Compare expressions semantically.
        Returns (match_rate, counterexamples).
        """
        # Get variables from both expressions
        parsed1 = self.parser.parse(expr1, lang1)
        parsed2 = self.parser.parse(expr2, lang2)

        all_vars = parsed1.variables | parsed2.variables

        if test_inputs is None:
            test_inputs = self.generate_test_inputs(all_vars)

        matches = 0
        counterexamples = []

        for context in test_inputs:
            result1, err1 = self.parser.evaluate(expr1, lang1, context)
            result2, err2 = self.parser.evaluate(expr2, lang2, context)

            # Both error = match (undefined behavior)
            if err1 is not None and err2 is not None:
                matches += 1
                continue

            # One error, one success = mismatch
            if (err1 is None) != (err2 is None):
                counterexamples.append({
                    'context': context,
                    'result1': result1 if err1 is None else f"Error: {err1}",
                    'result2': result2 if err2 is None else f"Error: {err2}",
                })
                continue

            # Compare results (with type coercion for booleans)
            if self._results_match(result1, result2):
                matches += 1
            else:
                counterexamples.append({
                    'context': context,
                    'result1': result1,
                    'result2': result2,
                })

        match_rate = matches / len(test_inputs) if test_inputs else 1.0
        return match_rate, counterexamples[:10]  # Limit counterexamples

    def _results_match(self, r1: Any, r2: Any) -> bool:
        """Check if results match (with type coercion)."""
        # Direct equality
        if r1 == r2:
            return True

        # Boolean coercion
        if isinstance(r1, bool) or isinstance(r2, bool):
            return bool(r1) == bool(r2)

        # Numeric tolerance
        if isinstance(r1, (int, float)) and isinstance(r2, (int, float)):
            return abs(r1 - r2) < 1e-9

        return False


class GuardEquivalenceChecker:
    """Main class for checking guard equivalence across languages."""

    def __init__(self, seed: int = 42):
        self.parser = MultiLanguageParser()
        self.structural = StructuralComparator(self.parser)
        self.semantic = SemanticComparator(self.parser, seed)

    def check_equivalence(
        self,
        expr1: str,
        lang1: ExpressionLanguage,
        expr2: str,
        lang2: ExpressionLanguage,
        n_test_samples: int = 100,
    ) -> EquivalenceResult:
        """Check if two expressions are equivalent."""
        # Parse both
        parsed1 = self.parser.parse(expr1, lang1)
        parsed2 = self.parser.parse(expr2, lang2)

        # Start with result
        result = EquivalenceResult(
            expr1_lang=lang1,
            expr1_source=expr1,
            expr2_lang=lang2,
            expr2_source=expr2,
            level=EquivalenceLevel.NOT_EQUIVALENT,
            structural_similarity=0.0,
            semantic_match_rate=0.0,
        )

        # Check validity
        if not parsed1.is_valid:
            result.notes.append(f"Expression 1 invalid: {parsed1.error}")
            return result
        if not parsed2.is_valid:
            result.notes.append(f"Expression 2 invalid: {parsed2.error}")
            return result

        # Structural comparison
        result.structural_similarity = self.structural.compare_structure(parsed1, parsed2)

        if result.structural_similarity > 0.5:
            result.level = EquivalenceLevel.STRUCTURALLY_SIMILAR
            result.notes.append("Expressions share similar structure")

        # Semantic comparison
        all_vars = parsed1.variables | parsed2.variables
        test_inputs = self.semantic.generate_test_inputs(all_vars, n_test_samples)

        match_rate, counterexamples = self.semantic.compare_semantic(
            expr1, lang1, expr2, lang2, test_inputs
        )

        result.semantic_match_rate = match_rate
        result.counterexamples = counterexamples

        if match_rate == 1.0:
            result.level = EquivalenceLevel.SEMANTICALLY_EQUIVALENT
            result.notes.append(f"All {n_test_samples} test cases matched")
        elif match_rate > 0.95:
            result.notes.append(f"Near-equivalent: {match_rate*100:.1f}% match rate")

        return result

    def check_multi_language_equivalence(
        self,
        expressions: Dict[ExpressionLanguage, str],
        n_test_samples: int = 100,
    ) -> Dict[Tuple[ExpressionLanguage, ExpressionLanguage], EquivalenceResult]:
        """Check pairwise equivalence across multiple languages."""
        results = {}
        langs = list(expressions.keys())

        for i, lang1 in enumerate(langs):
            for lang2 in langs[i+1:]:
                result = self.check_equivalence(
                    expressions[lang1], lang1,
                    expressions[lang2], lang2,
                    n_test_samples,
                )
                results[(lang1, lang2)] = result

        return results

    def find_equivalent_forms(
        self,
        base_expr: str,
        base_lang: ExpressionLanguage,
        target_langs: List[ExpressionLanguage] = None,
    ) -> GuardEquivalenceSet:
        """
        Given a base expression, find equivalent forms in other languages.
        """
        if target_langs is None:
            target_langs = list(ExpressionLanguage)

        parsed = self.parser.parse(base_expr, base_lang)

        equiv_set = GuardEquivalenceSet(
            canonical_form=base_expr,
            variables=parsed.variables,
        )
        equiv_set.guards[base_lang] = base_expr

        # Generate candidates for other languages
        for lang in target_langs:
            if lang == base_lang:
                continue

            candidate = self._translate_expression(base_expr, base_lang, lang)
            if candidate:
                result = self.check_equivalence(base_expr, base_lang, candidate, lang)
                if result.is_equivalent():
                    equiv_set.guards[lang] = candidate

        equiv_set.verified_equivalent = len(equiv_set.guards) > 1
        return equiv_set

    def _translate_expression(
        self,
        expr: str,
        from_lang: ExpressionLanguage,
        to_lang: ExpressionLanguage,
    ) -> Optional[str]:
        """Translate expression between languages (best effort)."""
        # Simple translation rules
        result = expr

        if from_lang == ExpressionLanguage.RAW:
            if to_lang in (ExpressionLanguage.CEL, ExpressionLanguage.JAVASCRIPT, ExpressionLanguage.GO):
                result = result.replace(' and ', ' && ')
                result = result.replace(' or ', ' || ')
                result = result.replace('not ', '!')
                result = result.replace('True', 'true')
                result = result.replace('False', 'false')
                result = result.replace('None', 'null' if to_lang == ExpressionLanguage.JAVASCRIPT else 'nil')

        elif from_lang in (ExpressionLanguage.CEL, ExpressionLanguage.JAVASCRIPT, ExpressionLanguage.GO):
            if to_lang == ExpressionLanguage.RAW or to_lang == ExpressionLanguage.STARLARK:
                result = result.replace('&&', ' and ')
                result = result.replace('||', ' or ')
                result = result.replace('!', 'not ')
                result = result.replace('true', 'True')
                result = result.replace('false', 'False')
                result = result.replace('null', 'None')
                result = result.replace('nil', 'None')

        return result


class EquivalenceTestSuite:
    """Pre-defined test cases for guard equivalence."""

    # Common guard patterns in multiple languages
    TEST_CASES = [
        {
            'name': 'simple_comparison',
            ExpressionLanguage.RAW: 'x > 0',
            ExpressionLanguage.CEL: 'x > 0',
            ExpressionLanguage.STARLARK: 'x > 0',
            ExpressionLanguage.JAVASCRIPT: 'x > 0',
            ExpressionLanguage.GO: 'x > 0',
        },
        {
            'name': 'logical_and',
            ExpressionLanguage.RAW: 'x > 0 and y < 10',
            ExpressionLanguage.CEL: 'x > 0 && y < 10',
            ExpressionLanguage.STARLARK: 'x > 0 and y < 10',
            ExpressionLanguage.JAVASCRIPT: 'x > 0 && y < 10',
            ExpressionLanguage.GO: 'x > 0 && y < 10',
        },
        {
            'name': 'logical_or',
            ExpressionLanguage.RAW: 'x == 0 or y == 0',
            ExpressionLanguage.CEL: 'x == 0 || y == 0',
            ExpressionLanguage.STARLARK: 'x == 0 or y == 0',
            ExpressionLanguage.JAVASCRIPT: 'x === 0 || y === 0',
            ExpressionLanguage.GO: 'x == 0 || y == 0',
        },
        {
            'name': 'negation',
            ExpressionLanguage.RAW: 'not is_active',
            ExpressionLanguage.CEL: '!is_active',
            ExpressionLanguage.STARLARK: 'not is_active',
            ExpressionLanguage.JAVASCRIPT: '!is_active',
            ExpressionLanguage.GO: '!is_active',
        },
        {
            'name': 'complex_condition',
            ExpressionLanguage.RAW: '(x > 0 and y > 0) or z == 0',
            ExpressionLanguage.CEL: '(x > 0 && y > 0) || z == 0',
            ExpressionLanguage.STARLARK: '(x > 0 and y > 0) or z == 0',
            ExpressionLanguage.JAVASCRIPT: '(x > 0 && y > 0) || z === 0',
            ExpressionLanguage.GO: '(x > 0 && y > 0) || z == 0',
        },
        {
            'name': 'boolean_literal',
            ExpressionLanguage.RAW: 'is_ready == True',
            ExpressionLanguage.CEL: 'is_ready == true',
            ExpressionLanguage.STARLARK: 'is_ready == True',
            ExpressionLanguage.JAVASCRIPT: 'is_ready === true',
            ExpressionLanguage.GO: 'is_ready == true',
        },
    ]

    def __init__(self):
        self.checker = GuardEquivalenceChecker()

    def run_all_tests(self, verbose: bool = True) -> Dict[str, bool]:
        """Run all equivalence test cases."""
        results = {}

        for test_case in self.TEST_CASES:
            name = test_case['name']
            expressions = {k: v for k, v in test_case.items()
                         if isinstance(k, ExpressionLanguage)}

            if verbose:
                print(f"\nTest: {name}")

            pairwise = self.checker.check_multi_language_equivalence(expressions)

            all_equivalent = all(r.is_equivalent() for r in pairwise.values())
            results[name] = all_equivalent

            if verbose:
                status = "PASS" if all_equivalent else "FAIL"
                print(f"  Result: {status}")
                if not all_equivalent:
                    for (l1, l2), result in pairwise.items():
                        if not result.is_equivalent():
                            print(f"    {l1.name} vs {l2.name}: {result.semantic_match_rate*100:.1f}% match")

        return results


def demo():
    """Demonstrate guard equivalence checking."""
    print("=" * 60)
    print("GUARD EQUIVALENCE CHECKER")
    print("=" * 60)

    checker = GuardEquivalenceChecker()

    # Test same guard in different languages
    guards = {
        ExpressionLanguage.RAW: "x > 0 and y < 10",
        ExpressionLanguage.CEL: "x > 0 && y < 10",
        ExpressionLanguage.JAVASCRIPT: "x > 0 && y < 10",
        ExpressionLanguage.GO: "x > 0 && y < 10",
    }

    print("\nChecking equivalence of: 'x > 0 AND y < 10'")
    print("-" * 60)

    results = checker.check_multi_language_equivalence(guards)

    for (lang1, lang2), result in results.items():
        print(f"\n{lang1.name} vs {lang2.name}:")
        print(f"  Structural similarity: {result.structural_similarity:.2f}")
        print(f"  Semantic match rate: {result.semantic_match_rate*100:.1f}%")
        print(f"  Level: {result.level.name}")
        print(f"  Equivalent: {result.is_equivalent()}")

    # Run test suite
    print("\n" + "=" * 60)
    print("EQUIVALENCE TEST SUITE")
    print("=" * 60)

    suite = EquivalenceTestSuite()
    test_results = suite.run_all_tests(verbose=True)

    print("\n" + "-" * 60)
    passed = sum(1 for v in test_results.values() if v)
    print(f"Results: {passed}/{len(test_results)} tests passed")

    return results


if __name__ == "__main__":
    demo()
