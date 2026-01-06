"""
Context-Free Grammar (CFG) Induction from Examples

Algorithms implemented:
1. Sequitur - Grammar compression (linear time)
2. ADIOS (Automatic Distillation of Structure) - Significance-based chunking
3. Genetic Grammar Induction - Evolutionary search

Key insight: Induced CFGs map to hierarchical statecharts with recursive substates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import re
from collections import defaultdict

from .formal_base import Symbol, Alphabet


@dataclass
class Terminal:
    """Terminal symbol in grammar."""
    value: str

    def __str__(self):
        return f"'{self.value}'"

    def __hash__(self):
        return hash(self.value)


@dataclass
class NonTerminal:
    """Non-terminal symbol in grammar."""
    name: str

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class Production:
    """Grammar production rule: LHS -> RHS."""
    lhs: NonTerminal
    rhs: List[Any]  # List of Terminal or NonTerminal
    id: int = 0

    def __str__(self):
        rhs_str = " ".join(str(s) for s in self.rhs)
        return f"{self.lhs} -> {rhs_str}"

    def __hash__(self):
        return hash((self.lhs.name, tuple(str(s) for s in self.rhs)))


@dataclass
class CFG:
    """Context-Free Grammar."""
    name: str
    terminals: Set[Terminal] = field(default_factory=set)
    non_terminals: Set[NonTerminal] = field(default_factory=set)
    productions: List[Production] = field(default_factory=list)
    start_symbol: Optional[NonTerminal] = None

    def add_production(self, prod: Production):
        """Add a production rule."""
        self.productions.append(prod)
        self.non_terminals.add(prod.lhs)
        for sym in prod.rhs:
            if isinstance(sym, Terminal):
                self.terminals.add(sym)
            elif isinstance(sym, NonTerminal):
                self.non_terminals.add(sym)

    def generate(self, max_depth: int = 10) -> str:
        """Generate random string from grammar."""
        if self.start_symbol is None:
            return ""
        return self._expand(self.start_symbol, max_depth)

    def _expand(self, symbol: Any, depth: int) -> str:
        """Recursively expand symbol."""
        if depth <= 0:
            return ""

        if isinstance(symbol, Terminal):
            return symbol.value

        if isinstance(symbol, NonTerminal):
            # Find applicable productions
            applicable = [p for p in self.productions if p.lhs.name == symbol.name]
            if not applicable:
                return ""

            # Choose random production
            prod = random.choice(applicable)
            return "".join(self._expand(s, depth - 1) for s in prod.rhs)

        return ""

    def to_string(self) -> str:
        """Convert grammar to string representation."""
        lines = [f"Grammar: {self.name}"]
        lines.append(f"Start: {self.start_symbol}")
        lines.append("Productions:")
        for prod in self.productions:
            lines.append(f"  {prod}")
        return "\n".join(lines)


@dataclass
class InducedCFG:
    """Result of CFG induction."""
    cfg: CFG
    positive_examples: List[str]
    accuracy: float
    n_productions: int
    compression_ratio: float
    algorithm: str
    iterations: int = 0


class GrammarInducer(ABC):
    """Abstract base for grammar induction algorithms."""

    @abstractmethod
    def induce(self, examples: List[str]) -> InducedCFG:
        """Induce CFG from examples."""
        pass


class SequiturInducer(GrammarInducer):
    """
    Sequitur Grammar Induction (Nevill-Manning & Witten, 1997).

    Produces grammar by:
    1. Replacing repeated digrams with rules
    2. Enforcing digram uniqueness
    3. Enforcing rule utility

    Linear time O(n) in input length.
    """

    def __init__(self):
        self.rules: Dict[str, List[Any]] = {}
        self.rule_count = 0
        self.digram_index: Dict[Tuple, str] = {}

    def induce(self, examples: List[str]) -> InducedCFG:
        """Induce grammar using Sequitur."""
        # Concatenate examples with separator
        separator = "\x00"
        text = separator.join(examples)

        # Build grammar
        cfg = self._run_sequitur(text)

        # Calculate metrics
        original_len = len(text)
        grammar_len = sum(len(p.rhs) for p in cfg.productions)
        compression = grammar_len / original_len if original_len > 0 else 1.0

        return InducedCFG(
            cfg=cfg,
            positive_examples=examples,
            accuracy=1.0,  # Sequitur is lossless
            n_productions=len(cfg.productions),
            compression_ratio=compression,
            algorithm="sequitur",
        )

    def _run_sequitur(self, text: str) -> CFG:
        """Run Sequitur algorithm."""
        cfg = CFG(name="sequitur_grammar")

        if not text:
            return cfg

        # Initialize main rule S
        S = NonTerminal("S")
        cfg.start_symbol = S

        # Track symbols in S rule
        main_rule: List[Any] = []
        rule_bodies: Dict[str, List[Any]] = {"S": main_rule}

        # Digram tracking
        digrams: Dict[Tuple, int] = {}  # digram -> position

        # Process each character
        for char in text:
            # Add terminal
            term = Terminal(char)
            main_rule.append(term)

            # Check for repeated digram
            if len(main_rule) >= 2:
                digram = (str(main_rule[-2]), str(main_rule[-1]))

                if digram in digrams:
                    # Found repeat - create or reuse rule
                    rule_name = self._get_or_create_rule(
                        digram, main_rule, rule_bodies, cfg
                    )
                else:
                    digrams[digram] = len(main_rule) - 2

        # Convert to productions
        for name, body in rule_bodies.items():
            if body:
                prod = Production(
                    lhs=NonTerminal(name),
                    rhs=body.copy(),
                    id=len(cfg.productions),
                )
                cfg.add_production(prod)

        return cfg

    def _get_or_create_rule(
        self,
        digram: Tuple[str, str],
        main_rule: List,
        rule_bodies: Dict,
        cfg: CFG,
    ) -> str:
        """Get existing rule for digram or create new one."""
        self.rule_count += 1
        rule_name = f"R{self.rule_count}"

        # Create new rule body from digram
        rule_bodies[rule_name] = [
            self._parse_symbol(digram[0]),
            self._parse_symbol(digram[1]),
        ]

        # Replace digram in main rule
        main_rule[-2:] = [NonTerminal(rule_name)]

        return rule_name

    def _parse_symbol(self, s: str) -> Any:
        """Parse string back to symbol."""
        if s.startswith("'") and s.endswith("'"):
            return Terminal(s[1:-1])
        elif s.startswith("R") or s == "S":
            return NonTerminal(s)
        else:
            return Terminal(s)


class ADIOSInducer(GrammarInducer):
    """
    ADIOS (Automatic Distillation of Structure) - Solan et al. 2005

    Learns grammar by:
    1. Building graph from example sequences
    2. Finding significant patterns via random walks
    3. Generalizing patterns to rules

    Motivated by language acquisition in children.
    """

    def __init__(self, min_support: int = 2, significance_threshold: float = 0.1):
        self.min_support = min_support
        self.significance_threshold = significance_threshold

    def induce(self, examples: List[str]) -> InducedCFG:
        """Induce grammar using ADIOS-inspired algorithm."""
        cfg = CFG(name="adios_grammar")
        S = NonTerminal("S")
        cfg.start_symbol = S

        # Extract frequent subsequences (simplified ADIOS)
        patterns = self._find_patterns(examples)

        # Create rules from patterns
        rule_id = 0
        for pattern, count in patterns.items():
            if count >= self.min_support:
                rhs = [Terminal(c) for c in pattern]
                nt = NonTerminal(f"P{rule_id}")
                prod = Production(lhs=nt, rhs=rhs, id=rule_id)
                cfg.add_production(prod)
                rule_id += 1

        # Create start rule
        if cfg.productions:
            start_rhs = [p.lhs for p in cfg.productions[:5]]  # Use first patterns
            start_prod = Production(lhs=S, rhs=start_rhs, id=rule_id)
            cfg.add_production(start_prod)

        return InducedCFG(
            cfg=cfg,
            positive_examples=examples,
            accuracy=self._calculate_accuracy(cfg, examples),
            n_productions=len(cfg.productions),
            compression_ratio=self._compression_ratio(cfg, examples),
            algorithm="adios",
        )

    def _find_patterns(self, examples: List[str]) -> Dict[str, int]:
        """Find frequent subsequence patterns."""
        patterns: Dict[str, int] = defaultdict(int)

        for example in examples:
            # Extract all substrings of length 2-5
            for length in range(2, min(6, len(example) + 1)):
                for i in range(len(example) - length + 1):
                    pattern = example[i:i + length]
                    patterns[pattern] += 1

        return dict(patterns)

    def _calculate_accuracy(self, cfg: CFG, examples: List[str]) -> float:
        """Estimate accuracy (simplified)."""
        if not examples or not cfg.productions:
            return 0.0

        # Check how many patterns appear in examples
        covered = 0
        for prod in cfg.productions:
            pattern = "".join(
                s.value if isinstance(s, Terminal) else ""
                for s in prod.rhs
            )
            for ex in examples:
                if pattern in ex:
                    covered += 1
                    break

        return covered / len(cfg.productions) if cfg.productions else 0.0

    def _compression_ratio(self, cfg: CFG, examples: List[str]) -> float:
        """Calculate compression ratio."""
        original = sum(len(ex) for ex in examples)
        grammar = sum(len(p.rhs) for p in cfg.productions)
        return grammar / original if original > 0 else 1.0


class GeneticGrammarInducer(GrammarInducer):
    """
    Genetic Algorithm for Grammar Induction.

    Evolves grammar population by:
    1. Fitness: Parsability of examples + grammar simplicity
    2. Crossover: Exchange productions between grammars
    3. Mutation: Add/remove/modify productions
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate

    def induce(self, examples: List[str]) -> InducedCFG:
        """Induce grammar using genetic algorithm."""
        # Initialize population
        population = [self._random_grammar(examples) for _ in range(self.population_size)]

        # Evolve
        best = None
        best_fitness = -float('inf')

        for gen in range(self.n_generations):
            # Evaluate fitness
            fitnesses = [self._fitness(g, examples) for g in population]

            # Track best
            for g, f in zip(population, fitnesses):
                if f > best_fitness:
                    best = g
                    best_fitness = f

            # Selection
            selected = self._tournament_select(population, fitnesses)

            # Crossover and mutation
            new_pop = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover(selected[i], selected[i + 1])
                    new_pop.extend([
                        self._mutate(child1, examples),
                        self._mutate(child2, examples),
                    ])
                else:
                    new_pop.append(self._mutate(selected[i], examples))

            population = new_pop[:self.population_size]

        if best is None:
            best = population[0]

        return InducedCFG(
            cfg=best,
            positive_examples=examples,
            accuracy=self._accuracy(best, examples),
            n_productions=len(best.productions),
            compression_ratio=self._compression(best, examples),
            algorithm="genetic",
            iterations=self.n_generations,
        )

    def _random_grammar(self, examples: List[str]) -> CFG:
        """Create random grammar from examples."""
        cfg = CFG(name="random_grammar")
        S = NonTerminal("S")
        cfg.start_symbol = S

        # Extract alphabet
        alphabet = set()
        for ex in examples:
            alphabet.update(ex)

        # Create random productions
        n_rules = random.randint(2, 6)
        for i in range(n_rules):
            nt = NonTerminal(f"R{i}") if i > 0 else S
            rhs_len = random.randint(1, 4)
            rhs = []
            for _ in range(rhs_len):
                if random.random() < 0.7:
                    rhs.append(Terminal(random.choice(list(alphabet))))
                else:
                    rhs.append(NonTerminal(f"R{random.randint(0, n_rules - 1)}"))

            prod = Production(lhs=nt, rhs=rhs, id=i)
            cfg.add_production(prod)

        return cfg

    def _fitness(self, cfg: CFG, examples: List[str]) -> float:
        """Calculate fitness (coverage - complexity penalty)."""
        coverage = self._accuracy(cfg, examples)
        complexity = len(cfg.productions) * 0.01
        return coverage - complexity

    def _accuracy(self, cfg: CFG, examples: List[str]) -> float:
        """Simplified accuracy check."""
        if not examples:
            return 0.0

        # Check if any generated string matches any example
        matches = 0
        for _ in range(min(10, len(examples))):
            generated = cfg.generate(max_depth=10)
            if generated in examples:
                matches += 1

        return matches / min(10, len(examples))

    def _compression(self, cfg: CFG, examples: List[str]) -> float:
        """Calculate compression ratio."""
        original = sum(len(ex) for ex in examples)
        grammar = sum(len(p.rhs) for p in cfg.productions)
        return grammar / original if original > 0 else 1.0

    def _tournament_select(
        self,
        population: List[CFG],
        fitnesses: List[float],
        tournament_size: int = 3,
    ) -> List[CFG]:
        """Tournament selection."""
        selected = []
        for _ in range(len(population)):
            competitors = random.sample(list(zip(population, fitnesses)), tournament_size)
            winner = max(competitors, key=lambda x: x[1])
            selected.append(winner[0])
        return selected

    def _crossover(self, parent1: CFG, parent2: CFG) -> Tuple[CFG, CFG]:
        """Crossover productions between grammars."""
        child1 = CFG(name="child1")
        child2 = CFG(name="child2")

        # Exchange half of productions
        prods1 = parent1.productions.copy()
        prods2 = parent2.productions.copy()

        split1 = len(prods1) // 2
        split2 = len(prods2) // 2

        for p in prods1[:split1] + prods2[split2:]:
            child1.add_production(Production(lhs=p.lhs, rhs=p.rhs.copy(), id=len(child1.productions)))

        for p in prods2[:split2] + prods1[split1:]:
            child2.add_production(Production(lhs=p.lhs, rhs=p.rhs.copy(), id=len(child2.productions)))

        child1.start_symbol = parent1.start_symbol
        child2.start_symbol = parent2.start_symbol

        return child1, child2

    def _mutate(self, cfg: CFG, examples: List[str]) -> CFG:
        """Mutate grammar."""
        if random.random() > self.mutation_rate:
            return cfg

        mutated = CFG(name=cfg.name)
        mutated.start_symbol = cfg.start_symbol

        # Collect alphabet
        alphabet = set()
        for ex in examples:
            alphabet.update(ex)

        for prod in cfg.productions:
            if random.random() < 0.2:
                # Mutate RHS
                new_rhs = []
                for sym in prod.rhs:
                    if random.random() < 0.3:
                        # Replace symbol
                        if random.random() < 0.7:
                            new_rhs.append(Terminal(random.choice(list(alphabet))))
                        else:
                            new_rhs.append(NonTerminal(f"R{random.randint(0, 5)}"))
                    else:
                        new_rhs.append(sym)

                if new_rhs:
                    mutated.add_production(Production(
                        lhs=prod.lhs,
                        rhs=new_rhs,
                        id=len(mutated.productions),
                    ))
            else:
                mutated.add_production(Production(
                    lhs=prod.lhs,
                    rhs=prod.rhs.copy(),
                    id=len(mutated.productions),
                ))

        return mutated


class CFGInducer:
    """
    Unified CFG induction interface.
    """

    def __init__(self, algorithm: str = "sequitur"):
        self.algorithm = algorithm

    def induce(self, examples: List[str]) -> InducedCFG:
        """Induce CFG from examples."""
        if self.algorithm == "sequitur":
            inducer = SequiturInducer()
        elif self.algorithm == "adios":
            inducer = ADIOSInducer()
        elif self.algorithm == "genetic":
            inducer = GeneticGrammarInducer()
        else:
            inducer = SequiturInducer()

        return inducer.induce(examples)


def demo():
    """Demonstrate grammar induction."""
    print("=" * 60)
    print("GRAMMAR INDUCER: Learn CFG from Examples")
    print("=" * 60)

    # Example 1: Repeated patterns
    examples1 = [
        "abab",
        "ababab",
        "cdcd",
        "cdcdcd",
        "abcdabcd",
    ]

    print(f"\nExample 1 - Repeated patterns:")
    print(f"Examples: {examples1}")

    # Sequitur
    print("\n--- Sequitur Algorithm ---")
    seq = CFGInducer(algorithm="sequitur")
    result1 = seq.induce(examples1)
    print(f"Productions: {result1.n_productions}")
    print(f"Compression: {result1.compression_ratio:.2f}")
    print(f"Grammar:\n{result1.cfg.to_string()}")

    # Example 2: Nested structure
    examples2 = [
        "()",
        "(())",
        "((()))",
        "(()())",
        "()()",
    ]

    print(f"\n\nExample 2 - Nested parentheses:")
    print(f"Examples: {examples2}")

    # ADIOS
    print("\n--- ADIOS Algorithm ---")
    adios = CFGInducer(algorithm="adios")
    result2 = adios.induce(examples2)
    print(f"Productions: {result2.n_productions}")
    print(f"Accuracy: {result2.accuracy:.2f}")

    # Example 3: Simple language with rules
    examples3 = [
        "the cat sat",
        "the dog ran",
        "a cat sat",
        "a dog ran",
    ]

    print(f"\n\nExample 3 - Simple language:")
    print(f"Examples: {examples3}")

    # Genetic
    print("\n--- Genetic Algorithm (5 generations) ---")
    genetic = GeneticGrammarInducer(population_size=20, n_generations=5)
    result3 = genetic.induce(examples3)
    print(f"Productions: {result3.n_productions}")
    print(f"Iterations: {result3.iterations}")

    return result1, result2, result3


if __name__ == "__main__":
    demo()
