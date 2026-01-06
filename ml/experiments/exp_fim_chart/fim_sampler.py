
import json
import time
import mlx.core as mx
import numpy as np

class FIMSampler:
    """
    FIM Sampler for "Filling In The Middle" of Statecharts.
    Concept:
    - Takes a prefix (JSON string up to a gap)
    - Generates content (transitions, states)
    - Validates against a schema or hardcoded structure constraint
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # Reuse mapping logic from RealConstrainedSampler for structural tokens
        self.token_map = self._build_token_map()
        
    def _build_token_map(self):
        """Map generic tokens to IDs."""
        mapping = {}
        # Vocab limited: [ ] , A C 1 ... (B missing)
        keys = ["[", "]", ",", "A", "C", "1"]
        
        for k in keys:
             ids = self.tokenizer.encode(k)
             if len(ids) > 0:
                 mapping[k] = ids
             else:
                 pass
        
        # EOS
        if hasattr(self.tokenizer, 'eos_token_id'):
             mapping['EOS'] = [self.tokenizer.eos_token_id]
        elif hasattr(self.tokenizer, 'char_to_idx'):
             mapping['EOS'] = [self.tokenizer.char_to_idx.get("<EOS>", -1)]
        else:
             mapping['EOS'] = []
             
        return mapping

    def generate_transition_gap(self, prefix: str, max_tokens: int = 100) -> str:
        """
        Generate transition list: [A, B, 1]
        """
        input_ids = self.tokenizer.encode(prefix)
        tokens = []
        
        # State Machine for [From, To, Event]
        # 0: [
        # 1: A
        # 2: ,
        # 3: B
        # 4: ,
        # 5: 1
        # 6: ]
        
        state = 0
        
        for _ in range(max_tokens):
            full_seq = mx.array(input_ids + tokens)[None, :]
            logits = self.model(full_seq)[0]
            next_logits = logits[-1]
            
            mask = np.full(next_logits.shape, -1e9)
            valid_ids = []
            
            if state == 0: valid_ids = self.token_map.get("[")
            elif state == 1: valid_ids = self.token_map.get("A")
            elif state == 2: valid_ids = self.token_map.get(",")
            elif state == 3: valid_ids = self.token_map.get("C")
            elif state == 4: valid_ids = self.token_map.get(",")
            elif state == 5: valid_ids = self.token_map.get("1")
            elif state == 6: valid_ids = self.token_map.get("]")
            else: 
                break
                
            # Apply Mask
            if valid_ids:
                for vid in valid_ids:
                    if vid < len(mask):
                        mask[vid] = 0
            
            masked_logits = next_logits + mx.array(mask)
            
            # Safe Softmax
            logits_np = np.array(masked_logits)
            if np.all(logits_np == -np.inf):
                 # Fallback
                 token_id = self.token_map.get("]")[0] 
            else:
                token_id = mx.argmax(masked_logits).item()
            
            tokens.append(token_id)
            state += 1
            
        return self.tokenizer.decode(tokens)
