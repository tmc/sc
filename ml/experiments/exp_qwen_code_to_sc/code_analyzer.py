"""
Code Analyzer - AST-based analysis for statechart extraction.

Analyzes code patterns that represent state machines:
1. Switch/case on state variable
2. If/else chains on state
3. State transition patterns (state = NEW_STATE)
4. Event handlers with state checks

Supports Python, C-like pseudocode, and Go patterns.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto


class CodeLanguage(Enum):
    """Supported code languages."""
    PYTHON = auto()
    C_LIKE = auto()
    GO = auto()
    PSEUDOCODE = auto()


@dataclass
class StateVariable:
    """A variable that represents state."""
    name: str
    possible_values: Set[str] = field(default_factory=set)
    initial_value: Optional[str] = None
    line_declared: int = 0


@dataclass
class StateTransition:
    """A state transition found in code."""
    from_state: Optional[str]  # None = any/unknown
    to_state: str
    condition: Optional[str] = None  # Guard condition
    event: Optional[str] = None  # Triggering event
    action: Optional[str] = None  # Action on transition
    line_number: int = 0


@dataclass
class StateBlock:
    """A block of code associated with a state."""
    state_name: str
    entry_actions: List[str] = field(default_factory=list)
    exit_actions: List[str] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class CodeAnalysis:
    """Complete analysis of code for state machine patterns."""
    language: CodeLanguage
    state_variables: List[StateVariable] = field(default_factory=list)
    states: Set[str] = field(default_factory=set)
    transitions: List[StateTransition] = field(default_factory=list)
    state_blocks: List[StateBlock] = field(default_factory=list)
    events: Set[str] = field(default_factory=set)
    entry_actions: Dict[str, List[str]] = field(default_factory=dict)
    exit_actions: Dict[str, List[str]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'language': self.language.name,
            'states': list(self.states),
            'n_transitions': len(self.transitions),
            'events': list(self.events),
            'state_variables': [sv.name for sv in self.state_variables],
        }


class PythonAnalyzer:
    """Analyze Python code for state machine patterns."""

    STATE_VAR_PATTERNS = ['state', 'current_state', '_state', 'status', 'mode']

    def __init__(self):
        self.analysis = CodeAnalysis(language=CodeLanguage.PYTHON)

    def analyze(self, source: str) -> CodeAnalysis:
        """Analyze Python source code."""
        try:
            tree = ast.parse(source)
            self._find_state_variables(tree)
            self._find_state_values(tree)
            self._find_transitions(tree)
            self._find_match_statements(tree)
        except SyntaxError as e:
            self.analysis.errors.append(f"Syntax error: {e}")

        return self.analysis

    def _find_state_variables(self, tree: ast.AST):
        """Find variables likely to be state."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if any(p in name.lower() for p in self.STATE_VAR_PATTERNS):
                            sv = StateVariable(
                                name=name,
                                line_declared=node.lineno,
                            )
                            if isinstance(node.value, ast.Constant):
                                sv.initial_value = str(node.value.value)
                                sv.possible_values.add(str(node.value.value))
                            self.analysis.state_variables.append(sv)

    def _find_state_values(self, tree: ast.AST):
        """Find all values assigned to state variables."""
        state_var_names = {sv.name for sv in self.analysis.state_variables}

        for node in ast.walk(tree):
            # Direct assignment: state = "new_value"
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in state_var_names:
                        if isinstance(node.value, ast.Constant):
                            val = str(node.value.value)
                            self.analysis.states.add(val)
                            for sv in self.analysis.state_variables:
                                if sv.name == target.id:
                                    sv.possible_values.add(val)

            # Comparison: state == "value"
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Name) and node.left.id in state_var_names:
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Constant):
                            self.analysis.states.add(str(comparator.value))

    def _find_transitions(self, tree: ast.AST):
        """Find state transitions in code."""
        state_var_names = {sv.name for sv in self.analysis.state_variables}

        class TransitionFinder(ast.NodeVisitor):
            def __init__(self, analysis):
                self.analysis = analysis
                self.current_state = None
                self.current_condition = None

            def visit_If(self, node):
                # Check if condition is state comparison
                old_state = self.current_state
                old_condition = self.current_condition

                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name):
                        if node.test.left.id in state_var_names:
                            for comp in node.test.comparators:
                                if isinstance(comp, ast.Constant):
                                    self.current_state = str(comp.value)

                self.current_condition = ast.unparse(node.test) if hasattr(ast, 'unparse') else None

                # Visit body
                for stmt in node.body:
                    self.visit(stmt)

                self.current_state = old_state
                self.current_condition = old_condition

                # Visit else
                for stmt in node.orelse:
                    self.visit(stmt)

            def visit_Assign(self, node):
                # State transition: state = "new_state"
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in state_var_names:
                        if isinstance(node.value, ast.Constant):
                            transition = StateTransition(
                                from_state=self.current_state,
                                to_state=str(node.value.value),
                                condition=self.current_condition,
                                line_number=node.lineno,
                            )
                            self.analysis.transitions.append(transition)

                self.generic_visit(node)

        finder = TransitionFinder(self.analysis)
        finder.visit(tree)

    def _find_match_statements(self, tree: ast.AST):
        """Find Python 3.10+ match statements (state machines)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Match):
                subject = node.subject
                if isinstance(subject, ast.Name):
                    state_var_names = {sv.name for sv in self.analysis.state_variables}
                    if subject.id in state_var_names:
                        for case in node.cases:
                            if isinstance(case.pattern, ast.MatchValue):
                                if isinstance(case.pattern.value, ast.Constant):
                                    state = str(case.pattern.value.value)
                                    self.analysis.states.add(state)
                                    # Create state block
                                    block = StateBlock(
                                        state_name=state,
                                        line_start=case.pattern.lineno if hasattr(case.pattern, 'lineno') else 0,
                                    )
                                    self.analysis.state_blocks.append(block)


class CLikeAnalyzer:
    """Analyze C-like code (C, C++, Java, Go) for state patterns."""

    # Patterns for switch/case state machines
    SWITCH_PATTERN = re.compile(
        r'switch\s*\(\s*(\w+)\s*\)\s*\{([^}]+)\}',
        re.MULTILINE | re.DOTALL
    )
    CASE_PATTERN = re.compile(
        r'case\s+([^\s:]+)\s*:([^}]+?)(?=case|default|$)',
        re.MULTILINE | re.DOTALL
    )
    STATE_ASSIGN_PATTERN = re.compile(
        r'(\w*state\w*)\s*=\s*([^\s;]+)',
        re.IGNORECASE
    )
    IF_STATE_PATTERN = re.compile(
        r'if\s*\(\s*(\w*state\w*)\s*==\s*([^\s)]+)\s*\)',
        re.IGNORECASE
    )

    def __init__(self):
        self.analysis = CodeAnalysis(language=CodeLanguage.C_LIKE)

    def analyze(self, source: str) -> CodeAnalysis:
        """Analyze C-like source code."""
        self._find_switch_cases(source)
        self._find_if_states(source)
        self._find_state_assignments(source)
        return self.analysis

    def _find_switch_cases(self, source: str):
        """Find switch/case patterns."""
        for match in self.SWITCH_PATTERN.finditer(source):
            var_name = match.group(1)
            body = match.group(2)

            # Check if switch is on state variable
            if 'state' in var_name.lower():
                sv = StateVariable(name=var_name)
                self.analysis.state_variables.append(sv)

                # Extract cases
                for case_match in self.CASE_PATTERN.finditer(body):
                    state_val = case_match.group(1).strip()
                    case_body = case_match.group(2)

                    self.analysis.states.add(state_val)
                    sv.possible_values.add(state_val)

                    # Find transitions within case
                    for assign in self.STATE_ASSIGN_PATTERN.finditer(case_body):
                        new_state = assign.group(2).strip().rstrip(';')
                        transition = StateTransition(
                            from_state=state_val,
                            to_state=new_state,
                        )
                        self.analysis.transitions.append(transition)
                        self.analysis.states.add(new_state)

    def _find_if_states(self, source: str):
        """Find if-based state checks."""
        for match in self.IF_STATE_PATTERN.finditer(source):
            var_name = match.group(1)
            state_val = match.group(2).strip()
            self.analysis.states.add(state_val)

    def _find_state_assignments(self, source: str):
        """Find all state assignments."""
        for match in self.STATE_ASSIGN_PATTERN.finditer(source):
            var_name = match.group(1)
            new_state = match.group(2).strip().rstrip(';')
            self.analysis.states.add(new_state)


class GoAnalyzer(CLikeAnalyzer):
    """Analyze Go code for state machine patterns."""

    # Go-specific patterns
    SELECT_PATTERN = re.compile(
        r'select\s*\{([^}]+)\}',
        re.MULTILINE | re.DOTALL
    )
    CHANNEL_CASE_PATTERN = re.compile(
        r'case\s+(\w+)\s*(?:<-|:=\s*<-)([^:]+):',
        re.MULTILINE
    )

    def __init__(self):
        super().__init__()
        self.analysis.language = CodeLanguage.GO

    def analyze(self, source: str) -> CodeAnalysis:
        """Analyze Go source code."""
        super().analyze(source)
        self._find_select_cases(source)
        return self.analysis

    def _find_select_cases(self, source: str):
        """Find Go select statements (event handling)."""
        for match in self.SELECT_PATTERN.finditer(source):
            body = match.group(1)
            for case_match in self.CHANNEL_CASE_PATTERN.finditer(body):
                event_var = case_match.group(1)
                self.analysis.events.add(event_var)


class MultiLanguageAnalyzer:
    """Unified analyzer supporting multiple languages."""

    def __init__(self):
        self.python_analyzer = PythonAnalyzer()
        self.c_analyzer = CLikeAnalyzer()
        self.go_analyzer = GoAnalyzer()

    def detect_language(self, source: str) -> CodeLanguage:
        """Detect language from source code."""
        # Python indicators
        if 'def ' in source and ':' in source:
            return CodeLanguage.PYTHON
        if 'import ' in source and 'from ' in source:
            return CodeLanguage.PYTHON

        # Go indicators
        if 'func ' in source and 'package ' in source:
            return CodeLanguage.GO
        if 'chan ' in source or '<-' in source:
            return CodeLanguage.GO

        # C-like (default for switch/case)
        if 'switch' in source or 'case ' in source:
            return CodeLanguage.C_LIKE

        return CodeLanguage.PSEUDOCODE

    def analyze(self, source: str, language: CodeLanguage = None) -> CodeAnalysis:
        """Analyze source code."""
        if language is None:
            language = self.detect_language(source)

        if language == CodeLanguage.PYTHON:
            return PythonAnalyzer().analyze(source)
        elif language == CodeLanguage.GO:
            return GoAnalyzer().analyze(source)
        else:
            return CLikeAnalyzer().analyze(source)


# Example code snippets for testing
EXAMPLE_PYTHON = '''
state = "idle"

def handle_event(event):
    global state
    if state == "idle":
        if event == "start":
            state = "running"
            print("Starting")
    elif state == "running":
        if event == "pause":
            state = "paused"
        elif event == "stop":
            state = "idle"
    elif state == "paused":
        if event == "resume":
            state = "running"
        elif event == "stop":
            state = "idle"
'''

EXAMPLE_C = '''
typedef enum { IDLE, RUNNING, PAUSED } State;
State state = IDLE;

void handle_event(Event event) {
    switch (state) {
        case IDLE:
            if (event == START) {
                state = RUNNING;
                on_start();
            }
            break;
        case RUNNING:
            if (event == PAUSE) {
                state = PAUSED;
            } else if (event == STOP) {
                state = IDLE;
            }
            break;
        case PAUSED:
            if (event == RESUME) {
                state = RUNNING;
            } else if (event == STOP) {
                state = IDLE;
            }
            break;
    }
}
'''


def demo():
    """Demonstrate code analysis."""
    print("=" * 60)
    print("CODE ANALYZER")
    print("=" * 60)

    analyzer = MultiLanguageAnalyzer()

    # Analyze Python
    print("\n--- Python Code ---")
    print(EXAMPLE_PYTHON[:200] + "...")
    analysis = analyzer.analyze(EXAMPLE_PYTHON, CodeLanguage.PYTHON)
    print(f"\nAnalysis:")
    print(f"  States: {analysis.states}")
    print(f"  State variables: {[sv.name for sv in analysis.state_variables]}")
    print(f"  Transitions: {len(analysis.transitions)}")
    for t in analysis.transitions[:5]:
        print(f"    {t.from_state} -> {t.to_state}")

    # Analyze C-like
    print("\n--- C-like Code ---")
    print(EXAMPLE_C[:200] + "...")
    analysis = analyzer.analyze(EXAMPLE_C, CodeLanguage.C_LIKE)
    print(f"\nAnalysis:")
    print(f"  States: {analysis.states}")
    print(f"  Transitions: {len(analysis.transitions)}")
    for t in analysis.transitions[:5]:
        print(f"    {t.from_state} -> {t.to_state}")

    return analyzer


if __name__ == "__main__":
    demo()
