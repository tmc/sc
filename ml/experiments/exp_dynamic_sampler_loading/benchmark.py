
import os
import json
import time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.common import load_model_and_tokenizer
from experiments.exp_dynamic_sampler_loading.dynamic_sampler import DynamicSCSampler

def run_benchmark():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    sampler = DynamicSCSampler(model, tokenizer)
    
    # Pre-defined SCs (small vs large)
    sc_small = json.dumps({
        "root_state": {"label": "__root__", "children": [{"label": "A", "is_initial": True}, {"label": "B"}]},
        "transitions": [{"from": ["A"], "to": ["B"], "event": "E1"}],
        "events": ["E1"]
    })
    
    # "Large" SC (still simple but more states)
    children = [{"label": f"S{i}"} for i in range(50)]
    children[0]["is_initial"] = True
    transitions = [{"from": [f"S{i}"], "to": [f"S{i+1}"], "event": "1"} for i in range(49)]
    sc_large = json.dumps({
        "root_state": {"label": "__root__", "children": children},
        "transitions": transitions,
        "events": ["1"]
    })
    
    # 1. Benchmark Loading Time
    print("Benchmarking Loading Time...")
    iterations = 50 # Reduced iteration for real map building if slow
    
    start = time.time()
    for _ in range(iterations):
        sampler.parse_and_load_sc(sc_small)
    avg_small = (time.time() - start) / iterations * 1000
    
    start = time.time()
    for _ in range(iterations):
        sampler.parse_and_load_sc(sc_large)
    avg_large = (time.time() - start) / iterations * 1000
    
    print(f"Avg Load Time (Small): {avg_small:.4f} ms")
    print(f"Avg Load Time (Large): {avg_large:.4f} ms")
    
    # 2. Benchmark Mode Switching (Simulating rapid context switches)
    print("Benchmarking Mode Switching...")
    start = time.time()
    for i in range(10): # Toggle 10 pairs
        sampler.parse_and_load_sc(sc_small)
        # sampler.generate_constrained("Prompt") # Skip actual generation for pure switching speed
        sampler.parse_and_load_sc(sc_large)
    avg_switch = (time.time() - start) / 20 * 1000
    print(f"Avg Switch Time: {avg_switch:.4f} ms")

    results = {
        "load_time_small_ms": avg_small,
        "load_time_large_ms": avg_large,
        "switch_overhead_ms": avg_switch
    }
    
    with open("results_real.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Report: [0400]: REAL_DYNAMIC_LOADING small={avg_small:.2f}ms, large={avg_large:.2f}ms, switch={avg_switch:.2f}ms")
    
    cmd = f'it2 session send-text B90CCCD4 "[0400]: REAL_DYNAMIC_LOADING small={avg_small:.2f}ms, large={avg_large:.2f}ms, switch={avg_switch:.2f}ms"'
    os.system(cmd)

if __name__ == "__main__":
    run_benchmark()
