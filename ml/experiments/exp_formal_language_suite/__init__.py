"""
exp_formal_language_suite: Unified Formal Language Learning

GOAL: Common framework for DFA/NFA/CFG induction from examples.

Unifies:
- exp_regex_statechart: Regex → Statechart conversion
- exp_grammar_induction: CFG learning from sequences
- exp_active_regex_learning: Active learning for regex

Hierarchy:
- FormalLanguage (abstract base)
  ├── RegularLanguage (DFA/NFA/Regex)
  │   ├── DFA (deterministic finite automaton)
  │   ├── NFA (non-deterministic finite automaton)
  │   └── Regex (regular expression)
  └── ContextFreeLanguage (CFG/PDA)
      ├── CFG (context-free grammar)
      └── PDA (pushdown automaton)

Key insight: Statecharts generalize all of these:
- DFA = flat statechart
- NFA = statechart with ε-transitions
- CFG = hierarchical statechart (recursive)
- PDA = statechart with stack context
"""

from .formal_base import (
    FormalLanguage,
    Automaton,
    State,
    Transition,
    Symbol,
    Alphabet,
    AcceptResult,
)
from .regex_inducer import (
    RegexInducer,
    DFAInducer,
    NFAInducer,
    InducedDFA,
    InducedNFA,
    InducedRegex,
)
from .grammar_inducer import (
    GrammarInducer,
    CFGInducer,
    InducedCFG,
    Production,
    NonTerminal,
    Terminal,
)
from .unified_learner import (
    UnifiedLearner,
    LanguageType,
    LearningResult,
    learn_from_examples,
)
from .suite_benchmark import (
    BenchmarkSuite,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Base classes
    'FormalLanguage',
    'Automaton',
    'State',
    'Transition',
    'Symbol',
    'Alphabet',
    'AcceptResult',
    # Regex/DFA/NFA
    'RegexInducer',
    'DFAInducer',
    'NFAInducer',
    'InducedDFA',
    'InducedNFA',
    'InducedRegex',
    # Grammar/CFG
    'GrammarInducer',
    'CFGInducer',
    'InducedCFG',
    'Production',
    'NonTerminal',
    'Terminal',
    # Unified
    'UnifiedLearner',
    'LanguageType',
    'LearningResult',
    'learn_from_examples',
    # Benchmark
    'BenchmarkSuite',
    'BenchmarkResult',
    'run_benchmark',
]
