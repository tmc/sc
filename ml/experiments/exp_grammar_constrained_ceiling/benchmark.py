
import os
import json
import time
import sys

# Add parent directory to path to allow importing common
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.common import load_model_and_tokenizer
from experiments.exp_grammar_constrained_ceiling.constrained_sampler import RealConstrainedSampler

def run_benchmark():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Initializing sampler...")
    sampler = RealConstrainedSampler(model, tokenizer)
    
    prompts = [f"Generate a statechart" for i in range(10)] # Reduced to 10 for speed on real model
    modes = ["NO_CONSTRAINT", "FULL_SC_GRAMMAR"] # Skip JSON_ONLY simplification
    results = {}
    
    for mode in modes:
        print(f"Running mode: {mode}")
        sampler.set_mode(mode)
        mode_results = []
        
        start_total = time.time()
        for i, prompt in enumerate(prompts):
            print(f"  Generating {i+1}/{len(prompts)}...")
            res = sampler.generate(prompt, max_tokens=50) # Usage short max_tokens for speed
            mode_results.append(res)
        end_total = time.time()
        
        # Calculate stats
        valid_json_count = sum(1 for r in mode_results if r["valid_json"])
        valid_sc_count = sum(1 for r in mode_results if r["valid_sc"])
        avg_time = (end_total - start_total) / len(prompts)
        
        results[mode] = {
            "valid_json_rate": valid_json_count / len(prompts),
            "valid_sc_rate": valid_sc_count / len(prompts),
            "avg_generation_time": avg_time
        }
    
    # Save results
    with open("results_real.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Report
    print(f"Report: [91C6]: GRAMMAR_CEILING no_constraint={results['NO_CONSTRAINT']['valid_sc_rate']*100}%, full_grammar={results['FULL_SC_GRAMMAR']['valid_sc_rate']*100}%")
    
    # Send to iTerm 
    cmd = f'it2 session send-text B90CCCD4 "[91C6]: REAL_MODEL_GRAMMAR_CEILING no_constraint={results["NO_CONSTRAINT"]["valid_sc_rate"]*100}%, full_grammar={results["FULL_SC_GRAMMAR"]["valid_sc_rate"]*100}%"'
    os.system(cmd)

if __name__ == "__main__":
    run_benchmark()
