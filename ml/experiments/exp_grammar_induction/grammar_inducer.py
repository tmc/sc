"""
Grammar Inducer - Main Pipeline

Combines token prediction + SAE states to build a statechart
representing the induced grammar.

Output: SC proto format statechart

Pipeline:
    1. Train sequence model on Go token sequences
    2. Apply SAE to discover syntax states
    3. Build transition graph from state sequences
    4. Output as Harel statechart
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from enum import Enum

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

from .token_collector import GoTokenCollector, TokenSequence, GoToken, TOKEN_VOCAB
from .sequence_model import GoSyntaxTransformer, TransformerConfig, SyntaxTrainer
from .sae_syntax import SyntaxSAE, SAEConfig, GoProductionRule


class StateType(Enum):
    """Types of states in the induced grammar."""
    BASIC = 1       # Leaf state
    NORMAL = 2      # OR-decomposition
    PARALLEL = 3    # AND-decomposition


@dataclass
class SyntaxState:
    """A state in the induced syntax statechart."""
    label: str
    type: StateType = StateType.BASIC
    children: List['SyntaxState'] = field(default_factory=list)
    is_initial: bool = False
    is_final: bool = False
    
    # Feature associations
    feature_ids: Set[int] = field(default_factory=set)
    token_types: Set[str] = field(default_factory=set)
    production_rule: Optional[GoProductionRule] = None
    
    # Statistics
    entry_count: int = 0
    exit_count: int = 0
    
    def to_proto_dict(self) -> Dict:
        """Convert to proto-compatible dict."""
        type_map = {
            StateType.BASIC: 'STATE_TYPE_BASIC',
            StateType.NORMAL: 'STATE_TYPE_OR',
            StateType.PARALLEL: 'STATE_TYPE_AND',
        }
        
        return {
            'label': self.label,
            'type': type_map[self.type],
            'children': [c.to_proto_dict() for c in self.children],
            'is_initial': self.is_initial,
            'is_final': self.is_final,
            'metadata': {
                'feature_ids': list(self.feature_ids),
                'token_types': list(self.token_types),
                'production_rule': self.production_rule.value if self.production_rule else None,
                'entry_count': self.entry_count,
                'exit_count': self.exit_count,
            },
        }


@dataclass
class SyntaxTransition:
    """A transition in the induced syntax statechart."""
    label: str
    from_states: List[str]
    to_states: List[str]
    event: str  # Token type that triggers transition
    guard: Optional[str] = None
    priority: int = 0
    
    # Statistics
    fire_count: int = 0
    
    def to_proto_dict(self) -> Dict:
        """Convert to proto-compatible dict."""
        return {
            'label': self.label,
            'from': self.from_states,
            'to': self.to_states,
            'event': self.event,
            'guard': {'expression': self.guard} if self.guard else None,
            'priority': self.priority,
            'metadata': {
                'fire_count': self.fire_count,
            },
        }


@dataclass
class InducedGrammar:
    """The induced grammar as a statechart."""
    name: str = "induced_go_grammar"
    description: str = "Go syntax grammar induced from code observation"
    
    root_state: Optional[SyntaxState] = None
    transitions: List[SyntaxTransition] = field(default_factory=list)
    events: Set[str] = field(default_factory=set)
    
    # Statistics
    total_sequences: int = 0
    total_tokens: int = 0
    
    def to_proto_dict(self) -> Dict:
        """Convert to SC proto format."""
        return {
            'name': self.name,
            'description': self.description,
            'root_state': self.root_state.to_proto_dict() if self.root_state else None,
            'transitions': [t.to_proto_dict() for t in self.transitions],
            'events': [{'label': e} for e in self.events],
            'variables': {
                'total_sequences': self.total_sequences,
                'total_tokens': self.total_tokens,
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_proto_dict(), indent=indent)
    
    def to_mermaid(self) -> str:
        """Generate Mermaid stateDiagram."""
        lines = ['stateDiagram-v2']
        
        # Add states
        def add_state(state: SyntaxState, indent: int = 4):
            prefix = ' ' * indent
            label = state.label
            
            if state.children:
                lines.append(f"{prefix}state {label} {{")
                for child in state.children:
                    add_state(child, indent + 4)
                lines.append(f"{prefix}}}")
            else:
                # Add token info
                tokens = ', '.join(list(state.token_types)[:3])
                if tokens:
                    lines.append(f"{prefix}{label}: {label}\\n({tokens})")
                else:
                    lines.append(f"{prefix}{label}")
                
                if state.is_initial:
                    lines.append(f"{prefix}[*] --> {label}")
                if state.is_final:
                    lines.append(f"{prefix}{label} --> [*]")
        
        if self.root_state:
            for child in self.root_state.children:
                add_state(child)
        
        # Add transitions
        for trans in self.transitions:
            src = trans.from_states[0] if trans.from_states else '?'
            dst = trans.to_states[0] if trans.to_states else '?'
            event = trans.event
            lines.append(f"    {src} --> {dst}: {event}")
        
        return '\n'.join(lines)


class GrammarInducer:
    """
    Main pipeline for grammar induction.
    
    Steps:
    1. Collect tokens from Go code
    2. Train transformer on token sequences
    3. Apply SAE to discover syntax states
    4. Build statechart from state transitions
    """
    
    def __init__(
        self,
        transformer_config: Optional[TransformerConfig] = None,
        sae_config: Optional[SAEConfig] = None,
    ):
        # Configs
        self.transformer_config = transformer_config or TransformerConfig(
            vocab_size=len(TOKEN_VOCAB),
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            ff_dim=512,
        )
        
        self.sae_config = sae_config or SAEConfig(
            input_dim=self.transformer_config.hidden_dim,
            expansion_factor=8,
            k_active=16,
        )
        
        # Components
        self.collector: Optional[GoTokenCollector] = None
        self.transformer: Optional[GoSyntaxTransformer] = None
        self.trainer: Optional[SyntaxTrainer] = None
        self.sae: Optional[SyntaxSAE] = None
        
        # Induced grammar
        self.grammar = InducedGrammar()
        
        # State tracking
        self.state_sequences: List[List[str]] = []
        self.transition_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    
    def collect_tokens(
        self,
        sources: List[str],
        include_stdlib: bool = True,
        max_files: int = 100,
    ) -> 'GrammarInducer':
        """
        Step 1: Collect tokens from Go code.
        
        Args:
            sources: Paths to Go files/directories
            include_stdlib: Include stdlib packages
            max_files: Maximum files to process
        
        Returns:
            self for chaining
        """
        self.collector = GoTokenCollector()
        self.collector.collect(
            sources=sources,
            include_stdlib=include_stdlib,
            max_total_files=max_files,
        )
        
        self.grammar.total_sequences = len(self.collector.sequences)
        self.grammar.total_tokens = sum(len(s) for s in self.collector.sequences)
        
        # Track all token types as events
        for seq in self.collector.sequences:
            for token in seq.tokens:
                self.grammar.events.add(token.type_name)
        
        return self
    
    def train_transformer(
        self,
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> 'GrammarInducer':
        """
        Step 2: Train transformer on token sequences.
        
        Args:
            num_epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            self for chaining
        """
        if not HAS_MLX:
            print("MLX not available - skipping transformer training")
            return self
        
        if not self.collector:
            raise RuntimeError("Must collect tokens first")
        
        # Create model
        self.transformer = GoSyntaxTransformer(self.transformer_config)
        self.trainer = SyntaxTrainer(self.transformer, learning_rate)
        
        # Prepare data
        train_seqs, val_seqs, _ = self.collector.split()
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for seq in train_seqs:
                # Convert to tensor
                token_ids = seq.to_type_ids()
                if len(token_ids) < 10:
                    continue
                
                # Create batch
                batch = mx.array([token_ids[:self.transformer_config.max_seq_len]])
                
                loss = self.trainer.train_step(batch)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch + 1}/{num_epochs}: loss = {avg_loss:.4f}")
        
        return self
    
    def discover_states(
        self,
        num_epochs: int = 5,
    ) -> 'GrammarInducer':
        """
        Step 3: Apply SAE to discover syntax states.
        
        Args:
            num_epochs: SAE training epochs
        
        Returns:
            self for chaining
        """
        self.sae = SyntaxSAE(self.sae_config)
        
        if not HAS_MLX or not self.transformer:
            print("Running SAE in simulation mode")
            self._discover_states_simulated()
            return self
        
        # Collect hidden states
        for seq in self.collector.sequences[:100]:
            token_ids = seq.to_type_ids()
            if len(token_ids) < 10:
                continue
            
            batch = mx.array([token_ids[:self.transformer_config.max_seq_len]])
            hidden = self.transformer.get_hidden_states(batch)
            
            token_types = [t.type_name for t in seq.tokens[:hidden.shape[1]]]
            self.sae.train_step(hidden, token_types)
        
        # Discover states from SAE
        self.sae.discover_states(min_activation_count=10)
        
        # Map to production rules
        token_rule_map = {
            'func': GoProductionRule.FUNC_DECL,
            'type': GoProductionRule.TYPE_DECL,
            'const': GoProductionRule.CONST_DECL,
            'var': GoProductionRule.VAR_DECL,
            'if': GoProductionRule.IF_STMT,
            'for': GoProductionRule.FOR_STMT,
            'switch': GoProductionRule.SWITCH_STMT,
            'select': GoProductionRule.SELECT_STMT,
            'return': GoProductionRule.RETURN_STMT,
            'go': GoProductionRule.GO_STMT,
            'defer': GoProductionRule.DEFER_STMT,
            '{': GoProductionRule.BLOCK,
            '}': GoProductionRule.BLOCK,
            'struct': GoProductionRule.STRUCT_TYPE,
            'interface': GoProductionRule.INTERFACE_TYPE,
            'map': GoProductionRule.MAP_TYPE,
            'chan': GoProductionRule.CHANNEL_TYPE,
        }
        self.sae.map_to_production_rules(token_rule_map)
        
        return self
    
    def _discover_states_simulated(self):
        """Simulated state discovery without MLX."""
        if not self.collector:
            return
        
        # Create states based on token patterns
        token_patterns = defaultdict(int)
        for seq in self.collector.sequences:
            for token in seq.tokens:
                token_patterns[token.type_name] += 1
        
        # Create a state for common token contexts
        for token_type, count in token_patterns.items():
            if count >= 10:
                feature_id = hash(token_type) % (self.sae_config.input_dim * self.sae_config.expansion_factor)
                self.sae.features[feature_id].activation_count = count
                self.sae.features[feature_id].associated_tokens.add(token_type)
        
        self.sae.discover_states(min_activation_count=0)
    
    def build_statechart(self) -> 'GrammarInducer':
        """
        Step 4: Build statechart from discovered states.
        
        Returns:
            self for chaining
        """
        # Create root state
        self.grammar.root_state = SyntaxState(
            label="__root__",
            type=StateType.NORMAL,
            is_initial=True,
        )
        
        # Create states from SAE discoveries
        state_objects: Dict[str, SyntaxState] = {}
        
        for state_name, feature_ids in self.sae.syntax_states.items():
            # Get associated tokens
            tokens = set()
            rule = None
            for fid in feature_ids:
                feature = self.sae.features.get(fid)
                if feature:
                    tokens.update(feature.associated_tokens)
                    if feature.associated_rules:
                        rule = list(feature.associated_rules)[0]
            
            state = SyntaxState(
                label=state_name,
                type=StateType.BASIC,
                feature_ids=feature_ids,
                token_types=tokens,
                production_rule=rule,
            )
            state_objects[state_name] = state
            self.grammar.root_state.children.append(state)
        
        # If no SAE states, create from token types
        if not state_objects:
            self._build_from_token_patterns(state_objects)
        
        # Set initial state
        if self.grammar.root_state.children:
            self.grammar.root_state.children[0].is_initial = True
        
        # Build transitions from sequence analysis
        self._build_transitions(state_objects)
        
        return self
    
    def _build_from_token_patterns(self, state_objects: Dict[str, SyntaxState]):
        """Build states from token type patterns."""
        if not self.collector:
            return
        
        # Get bigram patterns
        transition_counts = self.collector.get_transition_counts()
        
        # Create states for top token types
        token_stats = self.collector.get_token_stats()
        
        for token_type, count in list(token_stats.items())[:20]:
            if token_type in ('EOF', 'ILLEGAL', ';', ','):
                continue
            
            state = SyntaxState(
                label=f"state_{token_type}",
                type=StateType.BASIC,
                token_types={token_type},
            )
            state_objects[state.label] = state
            self.grammar.root_state.children.append(state)
    
    def _build_transitions(self, state_objects: Dict[str, SyntaxState]):
        """Build transitions from token sequence patterns."""
        if not self.collector:
            return
        
        # Count state-to-state transitions
        transition_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        
        for seq in self.collector.sequences:
            prev_state = None
            for token in seq.tokens:
                # Find matching state
                current_state = None
                for state_name, state in state_objects.items():
                    if token.type_name in state.token_types:
                        current_state = state_name
                        state.entry_count += 1
                        break
                
                if prev_state and current_state and prev_state != current_state:
                    key = (prev_state, current_state, token.type_name)
                    transition_counts[key] += 1
                    state_objects[prev_state].exit_count += 1
                
                prev_state = current_state
        
        # Create transitions for common patterns
        for (from_state, to_state, event), count in transition_counts.items():
            if count >= 5:  # Minimum threshold
                trans = SyntaxTransition(
                    label=f"{from_state}_to_{to_state}",
                    from_states=[from_state],
                    to_states=[to_state],
                    event=event,
                    fire_count=count,
                )
                self.grammar.transitions.append(trans)
    
    def induce(
        self,
        sources: List[str],
        include_stdlib: bool = True,
        max_files: int = 100,
        transformer_epochs: int = 10,
        sae_epochs: int = 5,
    ) -> InducedGrammar:
        """
        Run the full induction pipeline.
        
        Args:
            sources: Go file/directory paths
            include_stdlib: Include stdlib
            max_files: Maximum files
            transformer_epochs: Transformer training epochs
            sae_epochs: SAE training epochs
        
        Returns:
            InducedGrammar object
        """
        print("=" * 60)
        print("GRAMMAR INDUCTION PIPELINE")
        print("=" * 60)
        
        print("\n[1/4] Collecting tokens...")
        self.collect_tokens(sources, include_stdlib, max_files)
        print(f"      Collected {self.grammar.total_sequences} sequences, "
              f"{self.grammar.total_tokens} tokens")
        
        print("\n[2/4] Training transformer...")
        self.train_transformer(num_epochs=transformer_epochs)
        
        print("\n[3/4] Discovering syntax states...")
        self.discover_states(num_epochs=sae_epochs)
        print(f"      Found {len(self.sae.syntax_states)} states")
        
        print("\n[4/4] Building statechart...")
        self.build_statechart()
        print(f"      Created {len(self.grammar.root_state.children)} states, "
              f"{len(self.grammar.transitions)} transitions")
        
        print("\n" + "=" * 60)
        print("INDUCTION COMPLETE")
        print("=" * 60)
        
        return self.grammar


def demo():
    """Demonstrate grammar induction."""
    print("=" * 60)
    print("GRAMMAR INDUCER DEMO")
    print("=" * 60)
    
    # Create inducer
    inducer = GrammarInducer()
    
    # For demo, just show the structure
    print("\nPipeline steps:")
    print("  1. collect_tokens() - Tokenize Go source files")
    print("  2. train_transformer() - Train next-token prediction")
    print("  3. discover_states() - Apply SAE for state discovery")
    print("  4. build_statechart() - Build SC proto statechart")
    
    # Create a minimal grammar for demo
    inducer.sae = SyntaxSAE(inducer.sae_config)
    
    # Add some fake states
    inducer.sae.syntax_states = {
        'state_package': {0, 1, 2},
        'state_import': {3, 4, 5},
        'state_func': {6, 7, 8},
        'state_stmt': {9, 10, 11},
        'state_expr': {12, 13, 14},
    }
    
    # Add token associations
    for fid in [0, 1, 2]:
        inducer.sae.features[fid].associated_tokens.add('package')
    for fid in [3, 4, 5]:
        inducer.sae.features[fid].associated_tokens.add('import')
    for fid in [6, 7, 8]:
        inducer.sae.features[fid].associated_tokens.add('func')
        inducer.sae.features[fid].associated_rules.add(GoProductionRule.FUNC_DECL)
    
    # Build statechart
    inducer.build_statechart()
    
    # Add some transitions
    inducer.grammar.transitions = [
        SyntaxTransition(
            label='package_to_import',
            from_states=['state_package'],
            to_states=['state_import'],
            event='import',
            fire_count=100,
        ),
        SyntaxTransition(
            label='import_to_func',
            from_states=['state_import'],
            to_states=['state_func'],
            event='func',
            fire_count=90,
        ),
        SyntaxTransition(
            label='func_to_stmt',
            from_states=['state_func'],
            to_states=['state_stmt'],
            event='{',
            fire_count=85,
        ),
    ]
    
    print("\n=== INDUCED GRAMMAR ===")
    print(f"States: {len(inducer.grammar.root_state.children)}")
    print(f"Transitions: {len(inducer.grammar.transitions)}")
    
    print("\n=== MERMAID DIAGRAM ===")
    print(inducer.grammar.to_mermaid())
    
    print("\n=== JSON (truncated) ===")
    json_str = inducer.grammar.to_json()
    print(json_str[:500] + "...")


if __name__ == "__main__":
    demo()
