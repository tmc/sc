"""
Context Evolver

Evolves minimal context schemas to explain behavioral differences.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Set

class VarType:
    BOOL = "bool"
    INT = "int"

@dataclass
class ContextGenome:
    variables: Dict[str, str] = field(default_factory=dict) # name -> type
    fitness: float = 0.0

    def copy(self) -> 'ContextGenome':
        return copy.deepcopy(self)

    def __repr__(self):
        return str(self.variables)

class ContextEvolver:
    def __init__(self):
        self.var_names_pool = [f"v{i}" for i in range(10)]
        self.types = [VarType.BOOL, VarType.INT]

    def create_genome(self) -> ContextGenome:
        return ContextGenome()

    def mutate(self, genome: ContextGenome) -> ContextGenome:
        mutant = genome.copy()
        
        choice = random.choice(['add', 'remove'])
        
        if choice == 'add':
            name = random.choice(self.var_names_pool)
            if name not in mutant.variables:
                mutant.variables[name] = random.choice(self.types)
        elif choice == 'remove':
            if mutant.variables:
                name = random.choice(list(mutant.variables.keys()))
                del mutant.variables[name]
                
        return mutant

    def crossover(self, p1: ContextGenome, p2: ContextGenome) -> ContextGenome:
        # Union/Intersection crossover
        p1_vars = set(p1.variables.keys())
        p2_vars = set(p2.variables.keys())
        
        common = p1_vars & p2_vars
        diff = (p1_vars | p2_vars) - common
        
        child_vars = {}
        # Keep common
        for v in common:
            child_vars[v] = p1.variables[v] # Inherit type from p1
            
        # Randomly keep diff
        for v in diff:
            if random.random() < 0.5:
                # Get type from whichever parent had it
                tgt = p1 if v in p1.variables else p2
                child_vars[v] = tgt.variables[v]
                
        return ContextGenome(variables=child_vars)

    def evaluate(self, genome: ContextGenome, observations: List[Dict]) -> float:
        """
        Evaluate if the schema *could* explain the observations.
        Fitness = Ability to assign unique states + simplicity.
        
        observations: [{'input': ..., 'output': ...}]
        We group by input (state+event). If output differs, we need distinct context states.
        
        A schema with N variables can represent 2^N (if bool) distinct states.
        If a conflict group has K outcomes, we need >= ceil(log2(K)) bits.
        
        This is a simplified proxy: we just check if the schema has *capacity* to explain.
        Ideally we'd co-evolve guards, but for this experiment we assume 'capacity implies explainability'.
        """
        
        # Group observations
        groups = {} # (state, event) -> set(outcomes)
        for obs in observations:
            key = (obs['state'], obs['event'])
            out = obs['outcome']
            if key not in groups:
                groups[key] = set()
            groups[key].add(out)
            
        # Calculate max 'conflict size' (how many distinct outcomes for same input)
        max_conflict = 0
        for outcomes in groups.values():
            max_conflict = max(max_conflict, len(outcomes))
            
        # Calculate schema capacity
        # bool = 2 states, int = effectively infinite (let's say 10 for parsimony metrics)
        capacity_log2 = 0
        for t in genome.variables.values():
            if t == VarType.BOOL:
                capacity_log2 += 1
            else:
                capacity_log2 += 4 # equivalent to 16 states
                
        # We need 2^capacity >= max_conflict (so capacity_log2 >= log2(max_conflict))
        import math
        needed_bits = math.ceil(math.log2(max_conflict)) if max_conflict > 0 else 0
        
        if capacity_log2 >= needed_bits:
            explainability = 1.0
        else:
            explainability = capacity_log2 / (needed_bits + 1e-6)
            
        # Parsimony: Fewer variables is better, provided we explain data.
        n_vars = len(genome.variables)
        parsimony = 1.0 / (1 + n_vars)
        
        genome.fitness = explainability * 0.8 + parsimony * 0.2
        return genome.fitness
