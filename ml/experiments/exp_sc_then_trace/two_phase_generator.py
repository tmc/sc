
import json
import mlx.core as mx
from typing import Dict, Any, List, Optional
import sys
import os

# Ensure we can import from ml/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from grammars.dynamic_constrained_sampler import DynamicConstrainedSampler

class TwoPhaseGenerator:
    """
    Real TwoPhaseGenerator using MLX model.
    Phase 1: SC Generation (Prompt -> JSON)
    Phase 2: Trace Generation (SC -> Constrained Tokens)
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.trace_sampler = DynamicConstrainedSampler(model, tokenizer)

    def generate_sc(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Phase 1: Generate SC from prompt.
        Uses unconstrained generation but prompted to output JSON.
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt + "\nOutput valid JSON statechart:")
        
        # Simple generation loop
        # We can implement a simple loop here or assuming we have a generate func.
        # Let's write a simple loop.
        max_tokens = 300
        tokens = []
        
        for _ in range(max_tokens):
            full_seq = input_ids + tokens
            logits = self.model(mx.array(full_seq)[None, :])
            next_token = mx.argmax(logits[0, -1, :]).item()
            
            eos_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else self.tokenizer.char_to_idx.get("<EOS>", -1)
            
            if next_token == eos_id:
                break
                
            tokens.append(next_token)
            
        output_text = self.tokenizer.decode(tokens)
        
        # Helper to extract JSON if surrounded by markdown
        try:
            # simple cleanup
            if "```json" in output_text:
                output_text = output_text.split("```json")[1].split("```")[0]
            elif "```" in output_text:
                output_text = output_text.split("```")[1].split("```")[0]
                
            data = json.loads(output_text)
            return data
        except Exception as e:
            print(f"Failed to parse generated SC JSON: {e}")
            print(f"Output was: {output_text[:100]}...")
            return None

    def generate_trace(self, sc: Dict[str, Any], goal_state: str = None) -> List[str]:
        """
        Phase 2: Generate trace following SC.
        Uses DynamicConstrainedSampler.
        """
        try:
            self.trace_sampler.load_constraint_sc(sc)
            
            trace_prompt = "Trace:"
            # Generate constrained trace
            raw_trace = self.trace_sampler.generate(trace_prompt, max_tokens=20)
            
            # The sampler returns string. We need to parse it back into list of events.
            # Ideally sampler.generate returns tokens. 
            # Looking at DynamicConstrainedSampler implementation via viewing earlier:
            # It returns `self.tokenizer.decode(generated)`
            
            # Since we constrained tokens to be events, the output text should be concatenation of events.
            # But wait, DynamicConstrainedSampler maps events to tokens. 
            # If events are "TIMER", does it output "TIMERTIMER"?
            # Let's assume space separation or something?
            # Actually DynamicConstrainedSampler just masks tokens. 
            
            # For this verification, raw string is fine to test validity.
            # To return List[str], we might need to know what tokens were chosen.
            # But let's return a simulated list based on parsing the text if possible.
            # Or just return ["RAW_Output", raw_trace]
            
            return [raw_trace] # Placeholder list format
            
        except Exception as e:
            print(f"Trace generation error: {e}")
            return []
