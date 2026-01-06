
import sys
import os
import json
import time
import subprocess
import tempfile
from typing import Dict, List, Any, Optional

# Add ml/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from experiments.common import load_model_and_tokenizer

class ExecutionBenchmark:
    def __init__(self, model, tokenizer, sc_binary: str):
        self.model = model
        self.tokenizer = tokenizer
        self.sc_binary = sc_binary
        self.results = {}

    def run_suite(self):
        """Run all execution prediction experiments."""
        print("Starting Execution Benchmark Suite...")
        
        experiments = [
            ("Context Prediction", self.run_context_prediction),
            ("Terminal State", self.run_terminal_state),
            ("Guard Outcome", self.run_guard_outcome),
            ("Path Length", self.run_path_length),
            ("Trace Continuation", self.run_trace_continuation),
            ("Reachability", self.run_reachability),
        ]
        
        for name, func in experiments:
            print(f"\n[EXEC] Running {name}...")
            try:
                score = func()
                self.results[name] = score
                print(f"  -> Score: {score:.1f}%")
            except Exception as e:
                print(f"  -> Failed: {e}")
                self.results[name] = 0.0

        print(f"\nSuite Complete. Results: {self.results}")

    def _generate_answer(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate response from model."""
        input_ids = self.tokenizer.encode(prompt)
        tokens = []
        for _ in range(max_tokens):
            import mlx.core as mx
            full_seq = mx.array(input_ids + tokens)[None, :]
            # Model returns logits of shape (1, L, V)
            logits = self.model(full_seq)
            # Take last token's logits: [0, -1, :] -> (V,)
            next_token_logits = logits[0, -1, :]
            token_id = mx.argmax(next_token_logits).item()
            
            eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else self.tokenizer.char_to_idx.get("<EOS>", -1)
            if token_id == eos_id:
                break
            tokens.append(token_id)
            
        return self.tokenizer.decode(tokens).strip()

    def _get_ground_truth(self, sc_json: Dict, events: List[str]) -> Dict:
        """Run sc binary to get ground truth execution result."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(sc_json, tmp)
            tmp_path = tmp.name
            
        try:
            cmd = [self.sc_binary, "step", "-json"]
            for e in events:
                cmd.extend(["-e", str(e)])
            cmd.append(tmp_path)
            
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"sc error: {res.stderr}")
                return {}
            return json.loads(res.stdout)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # --- Experiments ---

    def run_context_prediction(self) -> float:
        """
        Experiment 1: Context Prediction
        Given SC + Trace, predict the final value of a context variable.
        """
        # SC with __root__ and child
        sc = {
            "name": "Counter",
            "context": {"count": 0},
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Active", "type": 1, "is_initial": True}
                ]
            },
            "events": [{"label": "INC"}],
            "transitions": [
                {
                    "from": ["Active"], "to": ["Active"], "event": "INC",
                    "actions": [{"expression": "count = count + 1"}]
                }
            ]
        }
        
        trace = ["INC", "INC", "INC"]
        prompt = f"Statechart: {json.dumps(sc)}\nTrace: {trace}\nPredict final count: "
        
        # Expected: {"configuration": ["Active"], "context": {"count": 3}}
        # We need to extract '3' from the model output.
        expected_val = "3" 
        
        generated = self._generate_answer(prompt)
        print(f"    Expected: {expected_val}, Got: {generated}")
        return 100.0 if expected_val in generated else 0.0

    def run_terminal_state(self) -> float:
        """Experiment 2: Predict final configuration."""
        sc = {
            "root_state": {
                "label": "__root__", "type": 2, "children": [
                    {"label": "A", "type": 1, "is_initial": True},
                    {"label": "B", "type": 1}
                ]
            },
            "transitions": [{"from": ["A"], "to": ["B"], "event": "NEXT"}]
        }
        trace = ["NEXT"]
        prompt = f"SC: {json.dumps(sc)}\nTrace: {trace}\nFinal State: "
        
        expected = "B"
        generated = self._generate_answer(prompt)
        print(f"    Expected: {expected}, Got: {generated}")
        return 100.0 if expected in generated else 0.0

    def run_guard_outcome(self) -> float:
        """Experiment 3: Predict if transition happens (Guard)."""
        sc = {
            "root_state": {
                "label": "__root__", "type": 2, "children": [
                     {"label": "A", "type": 1, "is_initial": True},
                     {"label": "B", "type": 1}
                ]
            },
            "context": {"x": 10},
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "TEST", "guard": {"expression": "x > 5"}}
            ]
        }
        prompt = f"SC: {json.dumps(sc)}\nContext: x=10\nEvent: TEST\nTransition taken? (Yes/No): "
        
        expected = "Yes"
        generated = self._generate_answer(prompt)
        print(f"    Expected: {expected}, Got: {generated}")
        return 100.0 if "Yes" in generated else 0.0

    def run_path_length(self) -> float:
        """Experiment 4: Predict length of trace to reach state."""
        sc = {
            "root_state": {"label": "__root__", "type": 2, "children": [
                 {"label": "A", "type": 1, "is_initial": True},
                 {"label": "B", "type": 1},
                 {"label": "C", "type": 1}
            ]},
            "transitions": [
                {"from": ["A"], "to": ["B"], "event": "E1"},
                {"from": ["B"], "to": ["C"], "event": "E2"}
            ]
        }
        prompt = f"SC: {json.dumps(sc)}\nInitial: A\nTarget: C\nMin path length: "
        expected = "2"
        generated = self._generate_answer(prompt)
        print(f"    Expected: {expected}, Got: {generated}")
        return 100.0 if expected in generated else 0.0

    def run_trace_continuation(self) -> float:
        """Experiment 5: Autocomplete trace."""
        sc = {
             "root_state": {"label": "__root__", "type": 2, "children": [
                 {"label": "A", "type": 1, "is_initial": True},
                 {"label": "B", "type": 1},
                 {"label": "C", "type": 1}
             ]},
             "transitions": [
                 {"from": ["A"], "to": ["B"], "event": "E1"},
                 {"from": ["B"], "to": ["C"], "event": "E2"}
             ]
        }
        prompt = f"SC: {json.dumps(sc)}\nPartial Trace: ['E1']\nNext event: "
        expected = "E2"
        generated = self._generate_answer(prompt)
        print(f"    Expected: {expected}, Got: {generated}")
        return 100.0 if expected in generated else 0.0

    def run_reachability(self) -> float:
        """Experiment 6: Is state reachable?"""
        sc = {
             "root_state": {"label": "__root__", "type": 2, "children": [
                 {"label": "A", "type": 1, "is_initial": True},
                 {"label": "B", "type": 1}
             ]},
             "transitions": [] # Disconnected
        }
        prompt = f"SC: {json.dumps(sc)}\nInitial: A\nTarget: B\nReachable? (Yes/No): "
        expected = "No"
        generated = self._generate_answer(prompt)
        print(f"    Expected: {expected}, Got: {generated}")
        return 100.0 if "No" in generated else 0.0


if __name__ == "__main__":
    # Locate SC binary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sc_bin = os.path.abspath(os.path.join(base_dir, "../../.venv/bin/sc"))
    if not os.path.exists(sc_bin):
        print(f"Error: sc binary not found at {sc_bin}")
        sys.exit(1)

    try:
        model, tokenizer = load_model_and_tokenizer()
        benchmark = ExecutionBenchmark(model, tokenizer, sc_binary=sc_bin)
        benchmark.run_suite()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
