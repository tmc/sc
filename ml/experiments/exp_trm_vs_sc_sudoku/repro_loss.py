
import mlx.core as mx
import mlx.nn as nn

def test_loss_logic():
    B = 2
    L = 81
    V = 10
    
    logits = mx.zeros((B, L, V)) # Uniform logits
    
    # Solution indices 1-9
    solution = mx.random.randint(1, 10, (B, L))
    
    # Empty mask
    empty_mask = mx.ones((B, L))
    
    # Softmax
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-10)
    
    print(f"Logits range: {mx.min(logits).item()} to {mx.max(logits).item()}")
    print(f"Probs range: {mx.min(probs).item()} to {mx.max(probs).item()}")
    print(f"LogProbs range: {mx.min(log_probs).item()} to {mx.max(log_probs).item()}")
    
    target_indices = solution[:, :, None]
    correct_log_probs = mx.take_along_axis(log_probs, target_indices, axis=-1).squeeze(-1)
    
    print(f"Correct LogProbs mean: {mx.mean(correct_log_probs).item()}")
    
    loss = -mx.sum(correct_log_probs * empty_mask) / (mx.sum(empty_mask) + 1e-10)
    print(f"Loss: {loss.item()}")
    
    # Now try with random small logits
    logits = mx.random.normal((B, L, V)) * 0.02
    probs = mx.softmax(logits, axis=-1)
    log_probs = mx.log(probs + 1e-10)
    loss = -mx.sum(mx.take_along_axis(log_probs, target_indices, axis=-1).squeeze(-1) * empty_mask) / (mx.sum(empty_mask) + 1e-10)
    print(f"Loss (random small logits): {loss.item()}")

if __name__ == "__main__":
    test_loss_logic()
