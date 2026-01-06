
import os
import json
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.common import load_model_and_tokenizer
from experiments.exp_fim_chart.fim_sampler import FIMSampler

def run_benchmark():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    sampler = FIMSampler(model, tokenizer)
    
    # Prefix: [ [A, C], [
    # We expect: [A, C, 1]
    prefix = '[ [A, C], ['
    suffix = ']]'
    
    print(f"Prefix: {prefix}")
    
    start = time.time()
    generated_gap = sampler.generate_transition_gap(prefix)
    duration = time.time() - start
    
    print(f"Generated Gap: {generated_gap}")
    
    # Stitch
    full_str = prefix + generated_gap + suffix
    print(f"Full: {full_str}")
    
    # Clean check
    gap_clean = generated_gap.strip()
    # It might be [A , C , 1 ] depending on tokenizer spaces
    
    is_valid_insertion = False
    if "A" in gap_clean and "C" in gap_clean and "1" in gap_clean:
        if gap_clean.startswith("[") and gap_clean.endswith("]"):
             is_valid_insertion = True
             
    print(f"Valid Insertion: {is_valid_insertion}")
    print(f"Duration: {duration:.4f}s")
    
    metrics = {
        "valid_insertion": is_valid_insertion,
        "duration": duration,
        "output": gap_clean
    }
    
    with open("results.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Report
    success_int = 1 if is_valid_insertion else 0
    print(f"Report: [FIM]: FIM_CHART valid_insertion={success_int*100}%, duration={duration:.4f}s")
    
    cmd = f'it2 session send-text B90CCCD4 "[FIM]: FIM_CHART valid_insertion={success_int*100}%, duration={duration:.4f}s"'
    os.system(cmd)

if __name__ == "__main__":
    run_benchmark()
