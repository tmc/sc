
import json
import time
import sys
import os
from typing import Dict, List, Set, Optional

# Ensure we can import from ml/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from grammars.dynamic_constrained_sampler import DynamicConstrainedSampler

class DynamicSCSampler:
    """
    Wrapper around real DynamicConstrainedSampler for benchmarking loading.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.real_sampler = DynamicConstrainedSampler(model, tokenizer)
        self.mode = "SC_GRAMMAR" 

    def switch_mode(self, new_mode: str):
        self.mode = new_mode

    def parse_and_load_sc(self, sc_text: str) -> bool:
        """
        Benchmark key: Parse JSON and build constraint map.
        """
        start = time.time()
        try:
            # We assume sc_text is JSON
            if isinstance(sc_text, str):
                sc_json = json.loads(sc_text)
            else:
                sc_json = sc_text
                
            # This triggers the expensive map building part
            self.real_sampler.load_constraint_sc(sc_json)
            
            load_time = (time.time() - start) * 1000
            # print(f"SC loaded in {load_time:.2f}ms")
            return True
        except Exception as e:
            print(f"Failed to load SC: {e}")
            return False

    def generate_constrained(self, prompt: str):
        """
        Generate using the loaded constraint.
        """
        return self.real_sampler.generate(prompt)
