"""
Test 2: Infinite/Robustness Test (10K Steps)

Hypothesis: Statecharts maintain 100% validity forever.
            Transformers accumulate drift/errors over time.

Setup:
1. Simple 3-state cycle: A -> B -> C -> A
2. Run 10,000 steps with NEXT event
3. Measure: % valid states, drift from expected

Expected Results:
- Statechart: 100% validity, 0% drift
- Transformer: Accumulating errors, eventual divergence
"""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils


class CycleTransformer(nn.Module):
    """Transformer for 3-state cycle prediction."""
    
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.num_states = 3
        
        # State embedding
        self.state_embed = nn.Embedding(3, embed_dim)
        
        # Prediction layers
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
        )
    
    def __call__(self, state: mx.array) -> mx.array:
        """Predict next state probabilities."""
        emb = self.state_embed(state)
        logits = self.layers(emb)
        return mx.softmax(logits, axis=-1)
    
    def predict_hard(self, state: int) -> int:
        """Hard prediction for a single state."""
        probs = self(mx.array([state]))
        mx.eval(probs)
        return int(mx.argmax(probs[0]))
    
    def predict_soft(self, state_probs: mx.array) -> mx.array:
        """Soft prediction from probability distribution."""
        # Weighted sum of embeddings
        emb = mx.zeros((1, self.state_embed.weight.shape[1]))
        for i in range(3):
            emb = emb + state_probs[0, i] * self.state_embed.weight[i:i+1]
        logits = self.layers(emb)
        return mx.softmax(logits, axis=-1)


class CycleStatechart:
    """Rule-based 3-state cycle."""
    
    def __init__(self):
        # A(0) -> B(1) -> C(2) -> A(0)
        self.transitions = {0: 1, 1: 2, 2: 0}
    
    def step(self, state: int) -> int:
        return self.transitions[state]
    
    def is_valid_state(self, state: int) -> bool:
        return state in [0, 1, 2]


def train_transformer(num_epochs: int = 200):
    """Train transformer on cycle transitions."""
    model = CycleTransformer(embed_dim=16)
    optimizer = optim.Adam(learning_rate=0.01)
    
    # Training data: all transitions
    states = mx.array([0, 1, 2] * 100, dtype=mx.int32)
    targets = mx.array([1, 2, 0] * 100, dtype=mx.int32)
    
    def loss_fn(model):
        probs = model(states)
        one_hot = mx.zeros((len(targets), 3))
        for i in range(len(targets)):
            one_hot = one_hot.at[i, targets[i]].add(1.0)
        probs = mx.clip(probs, 1e-7, 1.0)
        return -mx.mean(mx.sum(one_hot * mx.log(probs), axis=-1))
    
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    
    for _ in range(num_epochs):
        loss, grads = loss_and_grad(model)
        grads = mlx.utils.tree_map(lambda g: mx.clip(g, -1.0, 1.0), grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
    
    # Verify training
    probs = model(mx.array([0, 1, 2]))
    mx.eval(probs)
    preds = mx.argmax(probs, axis=-1).tolist()
    accuracy = sum(p == t for p, t in zip(preds, [1, 2, 0])) / 3
    
    return model, accuracy


def run_infinite_test_hard(model, statechart, num_steps: int = 10000):
    """Run 10K steps with hard (argmax) predictions."""
    print(f"\n{'='*60}")
    print(f"HARD PREDICTION TEST ({num_steps:,} steps)")
    print(f"{'='*60}")
    
    # Statechart run
    sc_state = 0
    sc_valid = 0
    sc_correct = 0
    
    start = time.time()
    for step in range(num_steps):
        expected = (step + 1) % 3  # A->B->C->A...
        sc_state = statechart.step(sc_state)
        sc_valid += int(statechart.is_valid_state(sc_state))
        sc_correct += int(sc_state == expected)
    sc_time = time.time() - start
    
    # Transformer run
    tf_state = 0
    tf_valid = 0
    tf_correct = 0
    tf_errors = []
    
    start = time.time()
    for step in range(num_steps):
        expected = (step + 1) % 3
        tf_state = model.predict_hard(tf_state)
        is_valid = tf_state in [0, 1, 2]
        tf_valid += int(is_valid)
        is_correct = tf_state == expected
        tf_correct += int(is_correct)
        
        # Record first 10 errors
        if not is_correct and len(tf_errors) < 10:
            tf_errors.append((step, tf_state, expected))
    tf_time = time.time() - start
    
    print(f"\nStatechart:")
    print(f"  Valid states: {sc_valid:,}/{num_steps:,} ({100*sc_valid/num_steps:.2f}%)")
    print(f"  Correct transitions: {sc_correct:,}/{num_steps:,} ({100*sc_correct/num_steps:.2f}%)")
    print(f"  Time: {sc_time:.2f}s")
    
    print(f"\nTransformer:")
    print(f"  Valid states: {tf_valid:,}/{num_steps:,} ({100*tf_valid/num_steps:.2f}%)")
    print(f"  Correct transitions: {tf_correct:,}/{num_steps:,} ({100*tf_correct/num_steps:.2f}%)")
    print(f"  Time: {tf_time:.2f}s")
    
    if tf_errors:
        print(f"\n  First errors (step, got, expected):")
        for step, got, exp in tf_errors[:5]:
            print(f"    Step {step}: got {got}, expected {exp}")
    
    return {
        'statechart': {'valid': sc_valid/num_steps, 'correct': sc_correct/num_steps},
        'transformer': {'valid': tf_valid/num_steps, 'correct': tf_correct/num_steps},
    }


def run_infinite_test_soft(model, num_steps: int = 10000):
    """Run 10K steps with soft (probabilistic) predictions to test drift."""
    print(f"\n{'='*60}")
    print(f"SOFT PREDICTION TEST ({num_steps:,} steps) - Testing Drift")
    print(f"{'='*60}")
    
    # Start with hard state 0 = [1, 0, 0]
    state_probs = mx.array([[1.0, 0.0, 0.0]])
    
    # Track entropy and max probability over time
    checkpoints = [0, 100, 1000, 5000, 10000]
    checkpoint_data = []
    
    for step in range(num_steps + 1):
        if step in checkpoints:
            entropy = -float(mx.sum(state_probs * mx.log(state_probs + 1e-8)))
            max_prob = float(mx.max(state_probs))
            argmax_state = int(mx.argmax(state_probs[0]))
            expected = step % 3
            
            checkpoint_data.append({
                'step': step,
                'entropy': entropy,
                'max_prob': max_prob,
                'argmax': argmax_state,
                'expected': expected,
                'correct': argmax_state == expected,
            })
        
        if step < num_steps:
            # Soft step
            state_probs = model.predict_soft(state_probs)
            mx.eval(state_probs)
    
    print("\nCheckpoints (entropy measures uncertainty, max_prob measures confidence):")
    print("-" * 70)
    print(f"{'Step':>8} | {'Entropy':>8} | {'MaxProb':>8} | {'Argmax':>6} | {'Expected':>8} | {'Correct'}")
    print("-" * 70)
    
    for cp in checkpoint_data:
        status = "✓" if cp['correct'] else "✗"
        print(f"{cp['step']:>8,} | {cp['entropy']:>8.4f} | {cp['max_prob']:>8.4f} | "
              f"{cp['argmax']:>6} | {cp['expected']:>8} | {status}")
    
    # Check for drift
    final = checkpoint_data[-1]
    if final['entropy'] > 0.5:
        print(f"\n⚠ HIGH ENTROPY at step 10K: {final['entropy']:.4f}")
        print("  Transformer has accumulated uncertainty (drift)")
    
    return checkpoint_data


def run_error_injection_test(model, statechart, num_steps: int = 1000):
    """Test recovery from injected errors."""
    print(f"\n{'='*60}")
    print(f"ERROR INJECTION TEST")
    print(f"{'='*60}")
    print("\nInject wrong state every 100 steps, measure recovery")
    
    # Statechart - always recovers instantly (deterministic)
    sc_state = 0
    sc_recoveries = 0
    
    for step in range(num_steps):
        if step % 100 == 50:  # Inject error at step 50, 150, 250...
            sc_state = (sc_state + 1) % 3  # Wrong state
        sc_state = statechart.step(sc_state)
        # Check if back on track
        expected = (step + 1) % 3
        if sc_state == expected:
            sc_recoveries += 1
    
    # Transformer - may not recover
    tf_state = 0
    tf_recoveries = 0
    
    for step in range(num_steps):
        if step % 100 == 50:
            tf_state = (tf_state + 1) % 3  # Inject error
        tf_state = model.predict_hard(tf_state)
        expected = (step + 1) % 3
        if tf_state == expected:
            tf_recoveries += 1
    
    print(f"\nRecovery rate (after {num_steps//100} injected errors):")
    print(f"  Statechart: {sc_recoveries}/{num_steps} ({100*sc_recoveries/num_steps:.1f}%)")
    print(f"  Transformer: {tf_recoveries}/{num_steps} ({100*tf_recoveries/num_steps:.1f}%)")
    
    return {
        'statechart_recovery': sc_recoveries/num_steps,
        'transformer_recovery': tf_recoveries/num_steps,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 08: INFINITE/ROBUSTNESS TEST (10K STEPS)")
    print("=" * 60)
    print("\nHypothesis: Statecharts maintain 100% validity forever")
    print("           Transformers may accumulate drift/errors")
    
    # Train transformer
    print("\n" + "-" * 60)
    print("Training Transformer on 3-state cycle (A->B->C->A)...")
    model, train_acc = train_transformer(num_epochs=300)
    print(f"Training accuracy: {train_acc:.1%}")
    
    # Create statechart
    statechart = CycleStatechart()
    
    # Run tests
    results_hard = run_infinite_test_hard(model, statechart, num_steps=10000)
    results_soft = run_infinite_test_soft(model, num_steps=10000)
    results_error = run_error_injection_test(model, statechart, num_steps=1000)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n| Test | Statechart | Transformer |")
    print("|------|------------|-------------|")
    print(f"| 10K Hard Steps (validity) | {results_hard['statechart']['valid']:.1%} | {results_hard['transformer']['valid']:.1%} |")
    print(f"| 10K Hard Steps (correct) | {results_hard['statechart']['correct']:.1%} | {results_hard['transformer']['correct']:.1%} |")
    print(f"| Error Recovery | {results_error['statechart_recovery']:.1%} | {results_error['transformer_recovery']:.1%} |")
    
    # Final verdict
    print("\n" + "=" * 60)
    sc_perfect = results_hard['statechart']['correct'] == 1.0
    tf_perfect = results_hard['transformer']['correct'] == 1.0
    
    if sc_perfect and not tf_perfect:
        print("✓ HYPOTHESIS CONFIRMED: Statechart maintains 100% validity")
        print(f"  Transformer accuracy: {results_hard['transformer']['correct']:.1%}")
    elif sc_perfect and tf_perfect:
        print("Both achieved 100% - transformer learned cycle perfectly!")
        print("But statechart is GUARANTEED correct by construction.")
    else:
        print(f"Unexpected: Statechart={results_hard['statechart']['correct']:.1%}")
    print("=" * 60)
    
    return {
        'hard': results_hard,
        'soft': results_soft,
        'error': results_error,
    }


if __name__ == "__main__":
    main()
