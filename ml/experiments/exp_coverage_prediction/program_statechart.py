"""
Program to Statechart Converter

Converts Python/Starlark programs into statechart control flow graphs (CFGs).
Each basic block becomes a state, control flow edges become transitions.

This representation is key to the coverage prediction task:
- Coverage = which states are visited during execution
- Predicting coverage = simulating the statechart with abstract input

Reference: exp_code_completion/syntax_statechart.py for syntax states
"""

import ast
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import json


class BlockType(Enum):
    """Types of basic blocks in the CFG."""
    ENTRY = auto()         # Function entry point
    EXIT = auto()          # Function exit point
    BASIC = auto()         # Regular statement block
    BRANCH = auto()        # If/while condition
    LOOP_HEADER = auto()   # For loop header
    LOOP_BODY = auto()     # Loop body
    EXCEPT = auto()        # Exception handler
    FINALLY = auto()       # Finally block
    WITH_ENTRY = auto()    # With context entry
    WITH_EXIT = auto()     # With context exit


@dataclass
class BasicBlock:
    """A basic block (state) in the CFG."""
    id: int
    block_type: BlockType
    lines: Set[int]                    # Source lines in this block
    statements: List[ast.stmt]         # AST nodes
    condition: Optional[ast.expr]      # Condition for branches
    is_initial: bool = False
    is_final: bool = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.block_type.name,
            'lines': sorted(self.lines),
            'is_initial': self.is_initial,
            'is_final': self.is_final,
        }


@dataclass
class CFGTransition:
    """A transition (edge) in the CFG."""
    source_id: int
    target_id: int
    label: str = ""                    # Edge label (e.g., "True", "False")
    guard: Optional[str] = None        # Guard condition as string
    event: str = ""                    # Triggering event (e.g., "ENTER_LOOP")

    def to_dict(self) -> Dict:
        return {
            'from': self.source_id,
            'to': self.target_id,
            'label': self.label,
            'guard': self.guard,
            'event': self.event,
        }


@dataclass
class ProgramStatechart:
    """
    A program represented as a statechart.

    States = basic blocks
    Transitions = control flow edges
    Guards = branch conditions
    """
    blocks: Dict[int, BasicBlock]
    transitions: List[CFGTransition]
    entry_block: int
    exit_blocks: Set[int]
    line_to_block: Dict[int, int]      # Map source line -> block ID
    source: str                         # Original source code

    def get_reachable_from(self, start_id: int) -> Set[int]:
        """Get all blocks reachable from start (BFS)."""
        reachable = set()
        queue = [start_id]

        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            for trans in self.transitions:
                if trans.source_id == current:
                    queue.append(trans.target_id)

        return reachable

    def get_covered_lines(self, visited_blocks: Set[int]) -> Set[int]:
        """Get source lines covered by visiting given blocks."""
        lines = set()
        for block_id in visited_blocks:
            if block_id in self.blocks:
                lines |= self.blocks[block_id].lines
        return lines

    def to_dict(self) -> Dict:
        """Convert to JSON-serializable format."""
        return {
            'blocks': [b.to_dict() for b in self.blocks.values()],
            'transitions': [t.to_dict() for t in self.transitions],
            'entry_block': self.entry_block,
            'exit_blocks': list(self.exit_blocks),
        }

    def to_sc_proto(self) -> Dict:
        """
        Convert to sc proto format (compatible with statecharts.proto).

        Returns a statechart definition with:
        - root_state containing all blocks as children
        - transitions with guards
        """
        # Build root state
        children = []
        for block in self.blocks.values():
            child = {
                'label': f"block_{block.id}",
                'type': 1,  # STATE_TYPE_BASIC
                'is_initial': block.is_initial,
            }
            if block.lines:
                child['lines'] = sorted(block.lines)
            children.append(child)

        root_state = {
            'label': '__root__',
            'type': 2,  # STATE_TYPE_NORMAL
            'children': children,
        }

        # Build transitions
        sc_transitions = []
        for trans in self.transitions:
            sc_trans = {
                'from': [f"block_{trans.source_id}"],
                'to': [f"block_{trans.target_id}"],
            }
            if trans.guard:
                sc_trans['guard'] = {'expression': trans.guard}
            if trans.event:
                sc_trans['event'] = trans.event
            sc_transitions.append(sc_trans)

        return {
            'root_state': root_state,
            'transitions': sc_transitions,
        }

    def to_mermaid(self) -> str:
        """Generate Mermaid stateDiagram."""
        lines = ['stateDiagram-v2']

        # Define states
        for block in self.blocks.values():
            label = f"block_{block.id}"
            desc = f"L{min(block.lines) if block.lines else '?'}"
            if block.is_initial:
                lines.append(f"    [*] --> {label}")
            if block.is_final:
                lines.append(f"    {label} --> [*]")
            lines.append(f"    {label}: {label} ({desc})")

        # Define transitions
        for trans in self.transitions:
            src = f"block_{trans.source_id}"
            dst = f"block_{trans.target_id}"
            label = trans.label or trans.guard or ""
            if label:
                lines.append(f"    {src} --> {dst}: {label[:20]}")
            else:
                lines.append(f"    {src} --> {dst}")

        return '\n'.join(lines)


class CFGBuilder(ast.NodeVisitor):
    """
    Build a Control Flow Graph (CFG) from Python AST.

    Each statement becomes part of a basic block.
    Control flow constructs (if/while/for/try) create edges.
    """

    def __init__(self, source: str):
        self.source = source
        self.blocks: Dict[int, BasicBlock] = {}
        self.transitions: List[CFGTransition] = []
        self.line_to_block: Dict[int, int] = {}

        self.next_block_id = 0
        self.current_block: Optional[BasicBlock] = None

        # For loop handling
        self.loop_headers: List[int] = []
        self.loop_exits: List[int] = []

    def new_block(
        self,
        block_type: BlockType = BlockType.BASIC,
        condition: Optional[ast.expr] = None,
    ) -> BasicBlock:
        """Create a new basic block."""
        block = BasicBlock(
            id=self.next_block_id,
            block_type=block_type,
            lines=set(),
            statements=[],
            condition=condition,
        )
        self.next_block_id += 1
        self.blocks[block.id] = block
        return block

    def add_transition(
        self,
        source: BasicBlock,
        target: BasicBlock,
        label: str = "",
        guard: Optional[str] = None,
        event: str = "",
    ):
        """Add a CFG edge."""
        trans = CFGTransition(
            source_id=source.id,
            target_id=target.id,
            label=label,
            guard=guard,
            event=event,
        )
        self.transitions.append(trans)

    def add_stmt_to_block(self, block: BasicBlock, stmt: ast.stmt):
        """Add a statement to a block."""
        block.statements.append(stmt)
        if hasattr(stmt, 'lineno'):
            block.lines.add(stmt.lineno)
            self.line_to_block[stmt.lineno] = block.id

    def build(self, tree: ast.Module) -> ProgramStatechart:
        """Build CFG from AST."""
        # Create entry and exit blocks
        entry = self.new_block(BlockType.ENTRY)
        entry.is_initial = True

        exit_block = self.new_block(BlockType.EXIT)
        exit_block.is_final = True

        # Process top-level statements
        self.current_block = entry
        last_block = self.visit_statements(tree.body)

        # Connect last block to exit
        if last_block and last_block.id != exit_block.id:
            self.add_transition(last_block, exit_block)

        return ProgramStatechart(
            blocks=self.blocks,
            transitions=self.transitions,
            entry_block=entry.id,
            exit_blocks={exit_block.id},
            line_to_block=self.line_to_block,
            source=self.source,
        )

    def visit_statements(self, stmts: List[ast.stmt]) -> Optional[BasicBlock]:
        """Visit a list of statements, return the final block."""
        for stmt in stmts:
            result = self.visit(stmt)
            if result is not None:
                self.current_block = result

        return self.current_block

    def visit_FunctionDef(self, node: ast.FunctionDef) -> BasicBlock:
        """Handle function definition."""
        # Function header
        header = self.new_block(BlockType.ENTRY)
        header.lines.add(node.lineno)
        self.line_to_block[node.lineno] = header.id

        if self.current_block:
            self.add_transition(self.current_block, header)

        # Function body
        self.current_block = header
        body_end = self.visit_statements(node.body)

        # Implicit return
        exit_block = self.new_block(BlockType.EXIT)
        exit_block.is_final = True
        if body_end:
            self.add_transition(body_end, exit_block)

        return exit_block

    def visit_If(self, node: ast.If) -> BasicBlock:
        """Handle if statement."""
        # Condition block
        cond_block = self.new_block(BlockType.BRANCH, condition=node.test)
        self.add_stmt_to_block(cond_block, node)

        if self.current_block:
            self.add_transition(self.current_block, cond_block)

        # True branch
        true_start = self.new_block()
        self.add_transition(
            cond_block, true_start,
            label="True",
            guard=ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition",
        )
        self.current_block = true_start
        true_end = self.visit_statements(node.body)

        # False/else branch
        if node.orelse:
            false_start = self.new_block()
            self.add_transition(
                cond_block, false_start,
                label="False",
                guard=f"not ({ast.unparse(node.test)})" if hasattr(ast, 'unparse') else "not condition",
            )
            self.current_block = false_start
            false_end = self.visit_statements(node.orelse)
        else:
            false_end = cond_block

        # Merge block
        merge = self.new_block()
        if true_end:
            self.add_transition(true_end, merge)
        if false_end and (node.orelse or false_end == cond_block):
            if not node.orelse:
                self.add_transition(cond_block, merge, label="False")
            else:
                self.add_transition(false_end, merge)

        return merge

    def visit_While(self, node: ast.While) -> BasicBlock:
        """Handle while loop."""
        # Loop header (condition)
        header = self.new_block(BlockType.LOOP_HEADER, condition=node.test)
        self.add_stmt_to_block(header, node)

        if self.current_block:
            self.add_transition(self.current_block, header)

        # Remember for break/continue
        exit_block = self.new_block()
        self.loop_headers.append(header.id)
        self.loop_exits.append(exit_block.id)

        # Loop body
        body_start = self.new_block(BlockType.LOOP_BODY)
        self.add_transition(
            header, body_start,
            label="True",
            guard=ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition",
        )
        self.current_block = body_start
        body_end = self.visit_statements(node.body)

        # Back edge to header
        if body_end:
            self.add_transition(body_end, header, label="loop back")

        # Exit edge
        self.add_transition(
            header, exit_block,
            label="False",
            guard=f"not ({ast.unparse(node.test)})" if hasattr(ast, 'unparse') else "not condition",
        )

        # Handle else clause
        if node.orelse:
            self.current_block = exit_block
            exit_block = self.visit_statements(node.orelse) or exit_block

        self.loop_headers.pop()
        self.loop_exits.pop()

        return exit_block

    def visit_For(self, node: ast.For) -> BasicBlock:
        """Handle for loop."""
        # Loop header
        header = self.new_block(BlockType.LOOP_HEADER)
        self.add_stmt_to_block(header, node)

        if self.current_block:
            self.add_transition(self.current_block, header, event="ENTER_FOR")

        # Exit block
        exit_block = self.new_block()
        self.loop_headers.append(header.id)
        self.loop_exits.append(exit_block.id)

        # Loop body
        body_start = self.new_block(BlockType.LOOP_BODY)
        self.add_transition(header, body_start, label="has_next")
        self.current_block = body_start
        body_end = self.visit_statements(node.body)

        # Back edge
        if body_end:
            self.add_transition(body_end, header, label="next_iter")

        # Exit
        self.add_transition(header, exit_block, label="exhausted")

        # Handle else
        if node.orelse:
            self.current_block = exit_block
            exit_block = self.visit_statements(node.orelse) or exit_block

        self.loop_headers.pop()
        self.loop_exits.pop()

        return exit_block

    def visit_Try(self, node: ast.Try) -> BasicBlock:
        """Handle try/except/finally."""
        # Try block
        try_start = self.new_block()
        if self.current_block:
            self.add_transition(self.current_block, try_start)

        self.current_block = try_start
        try_end = self.visit_statements(node.body)

        # Merge point for all paths
        merge = self.new_block()

        # Exception handlers
        for handler in node.handlers:
            handler_block = self.new_block(BlockType.EXCEPT)
            if handler.lineno:
                handler_block.lines.add(handler.lineno)
            # Any block in try can raise and go to handler
            self.add_transition(
                try_start, handler_block,
                label="except",
                event=f"EXCEPT_{handler.type.id if handler.type else 'BaseException'}",
            )
            self.current_block = handler_block
            handler_end = self.visit_statements(handler.body)
            if handler_end:
                self.add_transition(handler_end, merge)

        # Normal exit from try
        if try_end:
            self.add_transition(try_end, merge)

        # Finally
        if node.finalbody:
            finally_block = self.new_block(BlockType.FINALLY)
            self.add_transition(merge, finally_block, event="FINALLY")
            self.current_block = finally_block
            merge = self.visit_statements(node.finalbody) or finally_block

        return merge

    def visit_With(self, node: ast.With) -> BasicBlock:
        """Handle with statement."""
        # Context entry
        entry = self.new_block(BlockType.WITH_ENTRY)
        self.add_stmt_to_block(entry, node)

        if self.current_block:
            self.add_transition(self.current_block, entry, event="ENTER_CONTEXT")

        # Body
        self.current_block = entry
        body_end = self.visit_statements(node.body)

        # Context exit
        exit_block = self.new_block(BlockType.WITH_EXIT)
        if body_end:
            self.add_transition(body_end, exit_block, event="EXIT_CONTEXT")

        return exit_block

    def visit_Return(self, node: ast.Return) -> BasicBlock:
        """Handle return statement."""
        ret_block = self.new_block(BlockType.EXIT)
        ret_block.is_final = True
        self.add_stmt_to_block(ret_block, node)

        if self.current_block:
            self.add_transition(self.current_block, ret_block)

        return ret_block

    def visit_Break(self, node: ast.Break) -> Optional[BasicBlock]:
        """Handle break statement."""
        if self.loop_exits:
            exit_id = self.loop_exits[-1]
            if self.current_block:
                self.add_transition(
                    self.current_block,
                    self.blocks[exit_id],
                    event="BREAK",
                )
        return None  # Control flow leaves current block

    def visit_Continue(self, node: ast.Continue) -> Optional[BasicBlock]:
        """Handle continue statement."""
        if self.loop_headers:
            header_id = self.loop_headers[-1]
            if self.current_block:
                self.add_transition(
                    self.current_block,
                    self.blocks[header_id],
                    event="CONTINUE",
                )
        return None  # Control flow leaves current block

    def generic_visit(self, node: ast.stmt) -> BasicBlock:
        """Handle regular statements."""
        if self.current_block is None:
            self.current_block = self.new_block()

        self.add_stmt_to_block(self.current_block, node)
        return self.current_block


def build_cfg(source: str) -> ProgramStatechart:
    """Build a CFG from Python source code."""
    tree = ast.parse(source)
    builder = CFGBuilder(source)
    return builder.build(tree)


def build_cfg_for_function(source: str, func_name: str) -> Optional[ProgramStatechart]:
    """Build CFG for a specific function in source."""
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Build CFG for just this function
            func_source = ast.unparse(node) if hasattr(ast, 'unparse') else source
            func_tree = ast.Module(body=[node], type_ignores=[])
            builder = CFGBuilder(func_source)
            return builder.build(func_tree)

    return None


def demo():
    """Demonstrate program to statechart conversion."""
    print("=" * 60)
    print("PROGRAM TO STATECHART DEMO")
    print("=" * 60)

    source = '''
def factorial(n):
    if n <= 1:
        return 1
    else:
        result = n * factorial(n - 1)
        return result
'''

    print("\nSource code:")
    for i, line in enumerate(source.strip().split('\n'), 1):
        print(f"  {i:2d}: {line}")

    cfg = build_cfg(source.strip())

    print(f"\nCFG has {len(cfg.blocks)} blocks and {len(cfg.transitions)} transitions")

    print("\nBlocks:")
    for block in cfg.blocks.values():
        print(f"  Block {block.id} ({block.block_type.name}): lines {sorted(block.lines)}")

    print("\nTransitions:")
    for trans in cfg.transitions:
        label = f" [{trans.label}]" if trans.label else ""
        guard = f" (guard: {trans.guard[:30]}...)" if trans.guard else ""
        print(f"  {trans.source_id} -> {trans.target_id}{label}{guard}")

    print("\nMermaid diagram:")
    print(cfg.to_mermaid())

    # More complex example
    print("\n" + "=" * 60)
    print("LOOP EXAMPLE")
    print("=" * 60)

    loop_source = '''
def sum_positive(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
        else:
            continue
    return total
'''

    print("\nSource code:")
    for i, line in enumerate(loop_source.strip().split('\n'), 1):
        print(f"  {i:2d}: {line}")

    cfg2 = build_cfg(loop_source.strip())

    print(f"\nCFG has {len(cfg2.blocks)} blocks and {len(cfg2.transitions)} transitions")

    print("\nStatechart proto format:")
    proto = cfg2.to_sc_proto()
    print(json.dumps(proto, indent=2)[:500] + "...")

    return cfg, cfg2


if __name__ == "__main__":
    demo()
