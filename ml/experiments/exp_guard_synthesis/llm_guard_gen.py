"""
LLM-Based Guard and Action Generation

Use language models to propose candidate guard expressions,
then verify/refine them through evolution.

Key insight: LLMs are good at proposing PLAUSIBLE guards from natural
language descriptions, but may not get them exactly right. Evolution
fixes the details.

Pipeline:
1. User describes transition: "block recapturing at same position"
2. LLM proposes candidates: "move.x == last_capture.x && ..."
3. Evolution refines against test cases
4. Extract final verified guard
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import json
import re
from guard_synthesizer import (
    Expr, Const, Var, BinOp, UnaryOp, FuncCall,
    ExprType, GuardGenome, GuardSynthesizer
)


@dataclass
class GuardTemplate:
    """A candidate guard from LLM or template library."""
    description: str
    expression_str: str
    variables_used: List[str]
    confidence: float = 0.5


# =============================================================================
# TEMPLATE LIBRARY (Domain Knowledge)
# =============================================================================

KO_TEMPLATES = [
    GuardTemplate(
        description="Same position as last capture",
        expression_str="(move_x == last_capture_x) and (move_y == last_capture_y)",
        variables_used=['move_x', 'move_y', 'last_capture_x', 'last_capture_y'],
        confidence=0.7,
    ),
    GuardTemplate(
        description="Would recapture single stone",
        expression_str="(move_x == last_capture_x) and (move_y == last_capture_y) and (stones_removed == 1)",
        variables_used=['move_x', 'move_y', 'last_capture_x', 'last_capture_y', 'stones_removed'],
        confidence=0.9,
    ),
    GuardTemplate(
        description="Full Ko check",
        expression_str="(move_x == last_capture_x) and (move_y == last_capture_y) and would_capture_single",
        variables_used=['move_x', 'move_y', 'last_capture_x', 'last_capture_y', 'would_capture_single'],
        confidence=0.95,
    ),
]

EN_PASSANT_TEMPLATES = [
    GuardTemplate(
        description="Pawn double advance flag",
        expression_str="en_passant_file == target_file",
        variables_used=['en_passant_file', 'target_file'],
        confidence=0.8,
    ),
    GuardTemplate(
        description="Full en passant check",
        expression_str="(en_passant_file == target_file) and (piece_type == PAWN) and (abs(move_dy) == 1)",
        variables_used=['en_passant_file', 'target_file', 'piece_type', 'move_dy'],
        confidence=0.9,
    ),
]

CASTLING_TEMPLATES = [
    GuardTemplate(
        description="King hasn't moved",
        expression_str="not king_moved",
        variables_used=['king_moved'],
        confidence=0.6,
    ),
    GuardTemplate(
        description="Kingside castle available",
        expression_str="(not king_moved) and (not rook_h_moved) and (not in_check)",
        variables_used=['king_moved', 'rook_h_moved', 'in_check'],
        confidence=0.85,
    ),
    GuardTemplate(
        description="Full castling guard",
        expression_str="(not king_moved) and (not rook_moved) and (not in_check) and (path_clear)",
        variables_used=['king_moved', 'rook_moved', 'in_check', 'path_clear'],
        confidence=0.95,
    ),
]


# =============================================================================
# EXPRESSION PARSER
# =============================================================================

def parse_expression(expr_str: str, variables: List[str]) -> Optional[Expr]:
    """
    Parse a string expression into AST.

    Supports: and, or, not, ==, !=, <, >, <=, >=, (), identifiers, numbers, booleans
    """
    # Tokenize
    tokens = tokenize(expr_str)
    if not tokens:
        return None

    # Parse
    try:
        expr, remaining = parse_or_expr(tokens, variables)
        return expr
    except:
        return None


def tokenize(s: str) -> List[str]:
    """Tokenize expression string."""
    # Pattern for tokens
    pattern = r'(\(|\)|and|or|not|==|!=|<=|>=|<|>|True|False|\d+|[a-zA-Z_][a-zA-Z0-9_]*)'
    tokens = re.findall(pattern, s, re.IGNORECASE)
    return tokens


def parse_or_expr(tokens: List[str], variables: List[str]) -> Tuple[Expr, List[str]]:
    """Parse OR expression."""
    left, tokens = parse_and_expr(tokens, variables)

    while tokens and tokens[0].lower() == 'or':
        tokens = tokens[1:]  # consume 'or'
        right, tokens = parse_and_expr(tokens, variables)
        left = BinOp(op='or', left=left, right=right, expr_type=ExprType.BOOL)

    return left, tokens


def parse_and_expr(tokens: List[str], variables: List[str]) -> Tuple[Expr, List[str]]:
    """Parse AND expression."""
    left, tokens = parse_not_expr(tokens, variables)

    while tokens and tokens[0].lower() == 'and':
        tokens = tokens[1:]  # consume 'and'
        right, tokens = parse_not_expr(tokens, variables)
        left = BinOp(op='and', left=left, right=right, expr_type=ExprType.BOOL)

    return left, tokens


def parse_not_expr(tokens: List[str], variables: List[str]) -> Tuple[Expr, List[str]]:
    """Parse NOT expression."""
    if tokens and tokens[0].lower() == 'not':
        tokens = tokens[1:]
        operand, tokens = parse_not_expr(tokens, variables)
        return UnaryOp(op='not', operand=operand, expr_type=ExprType.BOOL), tokens

    return parse_comparison(tokens, variables)


def parse_comparison(tokens: List[str], variables: List[str]) -> Tuple[Expr, List[str]]:
    """Parse comparison expression."""
    left, tokens = parse_primary(tokens, variables)

    if tokens and tokens[0] in ['==', '!=', '<', '>', '<=', '>=']:
        op = tokens[0]
        tokens = tokens[1:]
        right, tokens = parse_primary(tokens, variables)
        return BinOp(op=op, left=left, right=right, expr_type=ExprType.BOOL), tokens

    return left, tokens


def parse_primary(tokens: List[str], variables: List[str]) -> Tuple[Expr, List[str]]:
    """Parse primary expression (atom or parenthesized)."""
    if not tokens:
        return Const(True, expr_type=ExprType.BOOL), []

    token = tokens[0]

    if token == '(':
        tokens = tokens[1:]  # consume '('
        expr, tokens = parse_or_expr(tokens, variables)
        if tokens and tokens[0] == ')':
            tokens = tokens[1:]  # consume ')'
        return expr, tokens

    if token.lower() == 'true':
        return Const(True, expr_type=ExprType.BOOL), tokens[1:]

    if token.lower() == 'false':
        return Const(False, expr_type=ExprType.BOOL), tokens[1:]

    if token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
        return Const(int(token), expr_type=ExprType.INT), tokens[1:]

    # Variable
    expr_type = ExprType.BOOL if token in variables else ExprType.INT
    return Var(token, expr_type=expr_type), tokens[1:]


# =============================================================================
# LLM GUARD GENERATOR (Mock)
# =============================================================================

class LLMGuardGenerator:
    """
    Generate guard candidates using an LLM.

    In practice, this would call Claude/GPT/Llama with a prompt like:
    "Given the game state variables {vars}, write a boolean expression
    that checks if {description}."

    For now, we simulate this with a template library + perturbation.
    """

    def __init__(self, domain: str = "generic"):
        self.domain = domain

        # Select templates based on domain
        self.templates = {
            'ko': KO_TEMPLATES,
            'en_passant': EN_PASSANT_TEMPLATES,
            'castling': CASTLING_TEMPLATES,
            'generic': [],
        }.get(domain, [])

    def generate_candidates(
        self,
        description: str,
        variables: List[str],
        n_candidates: int = 5,
    ) -> List[GuardGenome]:
        """
        Generate candidate guard expressions.

        In a real implementation, this would:
        1. Format a prompt with the description and variables
        2. Call the LLM to generate expression strings
        3. Parse the strings into AST
        4. Return as GuardGenome objects
        """
        candidates = []

        # Use templates if available
        for template in self.templates:
            # Check if all required variables are available
            if all(v in variables for v in template.variables_used):
                expr = parse_expression(template.expression_str, variables)
                if expr:
                    candidates.append(GuardGenome(
                        expr=expr,
                        fitness=template.confidence,
                    ))

        # Generate random variations (simulating LLM creativity)
        synth = GuardSynthesizer(variables, [0, 1, -1, True, False])
        while len(candidates) < n_candidates:
            candidates.append(GuardGenome(expr=synth.random_expr()))

        return candidates[:n_candidates]

    def generate_from_natural_language(
        self,
        nl_description: str,
        variables: List[str],
    ) -> List[str]:
        """
        Convert natural language to candidate expressions.

        This is where an actual LLM call would go.

        Example prompt:
        ```
        You are a guard expression generator for state machines.

        Given these variables: {variables}

        Generate 3 boolean expressions that check: "{nl_description}"

        Output each expression on a new line, using Python-like syntax.
        Use 'and', 'or', 'not' for boolean operators.
        ```
        """
        # Mock implementation - map keywords to templates
        expressions = []

        nl_lower = nl_description.lower()

        if 'ko' in nl_lower or 'recapture' in nl_lower:
            expressions.append(
                "(move_x == last_capture_x) and (move_y == last_capture_y)"
            )
            expressions.append(
                "(move_x == last_capture_x) and (move_y == last_capture_y) and would_capture_single"
            )

        if 'castle' in nl_lower or 'king' in nl_lower:
            expressions.append("(not king_moved) and (not rook_moved)")
            expressions.append("(not king_moved) and (not in_check)")

        if 'passant' in nl_lower:
            expressions.append("en_passant_file == target_file")

        # Add a generic fallback
        if not expressions:
            # Generate simple comparisons
            for var in variables[:3]:
                expressions.append(f"{var} == True")
                expressions.append(f"{var} > 0")

        return expressions


# =============================================================================
# COMBINED PIPELINE
# =============================================================================

def llm_guided_guard_synthesis(
    description: str,
    variables: List[str],
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    domain: str = "generic",
) -> GuardGenome:
    """
    Full pipeline: LLM proposes → Evolution refines → Verified guard

    This is the main entry point for guard synthesis.
    """
    print("=" * 60)
    print(f"LLM-GUIDED GUARD SYNTHESIS")
    print(f"Description: {description}")
    print("=" * 60)

    # Step 1: LLM generates candidates
    print("\n[1] Generating candidates from LLM...")
    generator = LLMGuardGenerator(domain)
    candidates = generator.generate_candidates(description, variables, n_candidates=10)

    print(f"    Generated {len(candidates)} candidates")
    for i, cand in enumerate(candidates[:3]):
        print(f"    {i+1}. {cand.expr.to_string()[:50]}...")

    # Step 2: Evaluate candidates on examples
    print("\n[2] Evaluating candidates on examples...")
    synth = GuardSynthesizer(variables, [0, 1, -1, True, False])

    for cand in candidates:
        synth.evaluate_fitness(cand, positive_examples, negative_examples)

    candidates.sort(key=lambda c: c.fitness, reverse=True)

    print(f"    Best initial candidate: F1={candidates[0].f1:.3f}")
    print(f"    {candidates[0].expr.to_string()[:60]}...")

    # Step 3: Evolve from best candidates
    print("\n[3] Evolving from best candidates...")

    # Seed population with LLM candidates
    population = candidates.copy()

    # Add random variations
    while len(population) < 30:
        base = candidates[0] if candidates else GuardGenome(expr=synth.random_expr())
        mutated = synth.mutate(base)
        population.append(mutated)

    # Evolve
    best = synth.evolve(
        positive_examples,
        negative_examples,
        population_size=30,
        n_generations=30,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("FINAL GUARD:")
    print(f"  {best.expr.to_string()}")
    print(f"\nMetrics: F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
    print("=" * 60)

    return best


# =============================================================================
# DEMO
# =============================================================================

def demo_llm_guard_synthesis():
    """Demonstrate LLM-guided guard synthesis."""

    # Ko rule synthesis
    print("\n" + "=" * 70)
    print(" DEMO: LLM-Guided Ko Guard Synthesis")
    print("=" * 70)

    variables = [
        'move_x', 'move_y',
        'last_capture_x', 'last_capture_y',
        'stones_removed', 'would_capture_single',
    ]

    positive = [
        {'move_x': 3, 'move_y': 4, 'last_capture_x': 3, 'last_capture_y': 4,
         'stones_removed': 1, 'would_capture_single': True},
        {'move_x': 7, 'move_y': 7, 'last_capture_x': 7, 'last_capture_y': 7,
         'stones_removed': 1, 'would_capture_single': True},
    ]

    negative = [
        {'move_x': 3, 'move_y': 5, 'last_capture_x': 3, 'last_capture_y': 4,
         'stones_removed': 1, 'would_capture_single': True},
        {'move_x': 3, 'move_y': 4, 'last_capture_x': 3, 'last_capture_y': 4,
         'stones_removed': 3, 'would_capture_single': False},
    ]

    best = llm_guided_guard_synthesis(
        description="Block immediate recapture at same position (Ko rule)",
        variables=variables,
        positive_examples=positive,
        negative_examples=negative,
        domain='ko',
    )

    return best


if __name__ == "__main__":
    demo_llm_guard_synthesis()
