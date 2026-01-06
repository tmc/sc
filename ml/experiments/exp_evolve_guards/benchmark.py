"""
Benchmark for Guard Evolution
"""

import random
import os
from experiments.exp_evolve_guards.guard_evolver import GuardEvolver, GuardGenome

def run_evolution(scenario: str):
    print(f"Running scenario: {scenario}")
    evolver = GuardEvolver(variables=['x', 'y'])
    
    traces = []
    for _ in range(50):
        ctx = {'x': random.randint(0, 20), 'y': random.randint(0, 20)}
        fired = False
        
        if scenario == "threshold":
            fired = ctx['x'] > 10
        elif scenario == "compound":
            fired = (ctx['x'] > 10) and (ctx['y'] < 5)
        elif scenario == "range":
            fired = (5 < ctx['x'] < 15)
            
        traces.append({'context': ctx, 'fired': fired})
        
    pop = [evolver.create_genome() for _ in range(50)]
    best_genome = None
    
    for gen in range(50):
        for g in pop:
            evolver.evaluate_fitness(g, traces)
            
        pop.sort(key=lambda g: g.fitness, reverse=True)
        best_genome = pop[0]
        
        if best_genome.fitness > 0.95:
            break
            
        next_pop = pop[:10]
        while len(next_pop) < 50:
            p1 = random.choice(pop[:20])
            p2 = random.choice(pop[:20])
            child = evolver.crossover(p1, p2)
            if random.random() < 0.3:
                child = evolver.mutate(child)
            next_pop.append(child)
        pop = next_pop

    print(f"Found Guard: {best_genome.root}")
    print(f"Fitness: {best_genome.fitness:.4f}")
    return best_genome

def main():
    scenarios = ["threshold", "compound", "range"]
    results = {}
    
    for s in scenarios:
        best = run_evolution(s)
        results[s] = best
        
    sid = os.environ.get("ITERM_SESSION_ID", "UNKNOWN")
    acc_strs = []
    for s, g in results.items():
        acc = "PASS" if g.fitness > 0.85 else "FAIL"
        acc_strs.append(f"{s}:{acc}")

    print(f"[{sid}]: EVOLVE_GUARDS scenarios={','.join(acc_strs)}")

if __name__ == "__main__":
    main()
