"""
Bidirectional Evaluator: Measure both generation validity and explanation clarity.

Metrics:
- gen_validity: % of generated SCs that are valid JSON with required structure
- explain_clarity: % of explanations that coherently describe the SC
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import re


@dataclass
class EvaluationResult:
    """Results from bidirectional evaluation."""
    # Generation metrics
    gen_total: int
    gen_valid: int
    gen_validity: float
    
    # Explanation metrics  
    explain_total: int
    explain_clear: int
    explain_clarity: float
    
    # Details
    gen_errors: List[str]
    unclear_explanations: List[str]
    
    def summary(self) -> str:
        return (
            f"Bidirectional Evaluation Results\n"
            f"================================\n"
            f"Generation (text → SC):\n"
            f"  Valid: {self.gen_valid}/{self.gen_total} ({self.gen_validity:.1%})\n"
            f"Explanation (SC → text):\n"
            f"  Clear: {self.explain_clear}/{self.explain_total} ({self.explain_clarity:.1%})\n"
        )


class BidirectionalEvaluator:
    """Evaluate bidirectional SC generation/explanation."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Load model with mlx_lm."""
        if self.model is None:
            from mlx_lm import load
            print(f"Loading {self.model_name}...")
            self.model, self.tokenizer = load(self.model_name)
            print("Model loaded.")
    
    def _generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate text using the model."""
        from mlx_lm import generate
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        response = generate(
            self.model,
            self.tokenizer,
            prompt=text,
            max_tokens=max_tokens,
        )
        return response
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from generated text."""
        # Try direct parse
        try:
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON block
        patterns = [
            r'```json\s*([\s\S]*?)```',
            r'```\s*([\s\S]*?)```',
            r'(\{[\s\S]*\})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue
        
        return None
    
    def _check_sc_validity(self, sc: Optional[Dict]) -> Tuple[bool, str]:
        """Check if SC is valid with required structure."""
        if sc is None:
            return False, "Invalid JSON"
        
        # Check for root_state or states
        if "root_state" not in sc and "states" not in sc:
            return False, "Missing root_state or states"
        
        # Check root has children or is valid
        root = sc.get("root_state", {})
        if root and not root.get("children") and not root.get("label"):
            return False, "Root state malformed"
        
        return True, ""
    
    def _check_explanation_clarity(self, explanation: str, sc: Dict) -> Tuple[bool, str]:
        """Check if explanation clearly describes the SC."""
        explanation_lower = explanation.lower()
        
        # Must mention key concepts
        required_concepts = 0
        
        # Check for state mentions
        if "state" in explanation_lower:
            required_concepts += 1
        
        # Check for transition/event mentions
        if any(w in explanation_lower for w in ["transition", "event", "trigger", "when"]):
            required_concepts += 1
        
        # Check for flow description
        if any(w in explanation_lower for w in ["from", "to", "moves", "changes", "goes"]):
            required_concepts += 1
        
        # Check minimum length
        if len(explanation.split()) < 10:
            return False, "Too short"
        
        if required_concepts >= 2:
            return True, ""
        else:
            return False, f"Missing concepts ({required_concepts}/2)"
    
    def evaluate(
        self,
        test_pairs: List[Tuple[str, str]],  # (description, sc_json)
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate on test pairs.
        
        Args:
            test_pairs: List of (description, sc_json) pairs
            verbose: Print progress
            
        Returns:
            EvaluationResult with metrics
        """
        self._load_model()
        
        gen_valid = 0
        gen_total = 0
        gen_errors = []
        
        explain_clear = 0
        explain_total = 0
        unclear = []
        
        for i, (desc, sc_json) in enumerate(test_pairs):
            if verbose and i % 10 == 0:
                print(f"Evaluating {i+1}/{len(test_pairs)}...")
            
            # Test generation (text → SC)
            gen_prompt = f"Generate a statechart JSON for: {desc}\nOutput only valid JSON."
            generated = self._generate(gen_prompt, max_tokens=400)
            
            extracted = self._extract_json(generated)
            is_valid, error = self._check_sc_validity(extracted)
            
            gen_total += 1
            if is_valid:
                gen_valid += 1
            else:
                gen_errors.append(f"{desc[:50]}... -> {error}")
            
            # Test explanation (SC → text)
            sc_obj = json.loads(sc_json) if isinstance(sc_json, str) else sc_json
            explain_prompt = f"Explain this statechart concisely:\n{sc_json[:500]}"
            explanation = self._generate(explain_prompt, max_tokens=200)
            
            is_clear, reason = self._check_explanation_clarity(explanation, sc_obj)
            
            explain_total += 1
            if is_clear:
                explain_clear += 1
            else:
                unclear.append(f"{reason}: {explanation[:100]}...")
        
        return EvaluationResult(
            gen_total=gen_total,
            gen_valid=gen_valid,
            gen_validity=gen_valid / gen_total if gen_total > 0 else 0,
            explain_total=explain_total,
            explain_clear=explain_clear,
            explain_clarity=explain_clear / explain_total if explain_total > 0 else 0,
            gen_errors=gen_errors[:10],  # Keep first 10
            unclear_explanations=unclear[:10],
        )


def evaluate_bidirectional(
    test_pairs: List[Tuple[str, str]],
    verbose: bool = True,
) -> EvaluationResult:
    """Convenience function to run evaluation."""
    evaluator = BidirectionalEvaluator()
    return evaluator.evaluate(test_pairs, verbose)


def demo():
    """Run quick evaluation demo."""
    from .dataset import create_dataset
    
    print("=== Bidirectional Evaluation Demo ===")
    dataset = create_dataset()
    
    # Use first 5 pairs for demo
    pairs = [(p.description, p.statechart) for p in dataset.pairs[:5]]
    
    result = evaluate_bidirectional(pairs)
    print("\n" + result.summary())
    
    return result


if __name__ == "__main__":
    demo()
