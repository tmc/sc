import mlx.core as mx

def stablemax(logits, axis=-1, epsilon=1e-30):
    """
    StableMax: s(x) = 1/(1-x+eps) if x<0 else x+1
    Then normalize.
    """
    # Requires float64 for stability or float32 at least.
    # original uses float64.
    orig_dtype = logits.dtype
    logits = logits.astype(mx.float32) # or float64 if MLX works well with it, usually float32 is okay
    
    # s(x) logic
    # x < 0: 1 / (1 - x + eps)
    # x >= 0: x + 1
    
    s_x_neg = 1.0 / (1.0 - logits + epsilon)
    s_x_pos = logits + 1.0
    
    s_x = mx.where(logits < 0, s_x_neg, s_x_pos)
    
    sum_s_x = mx.sum(s_x, axis=axis, keepdims=True)
    probs = s_x / sum_s_x
    
    return probs.astype(orig_dtype)

def log_stablemax(logits, axis=-1, epsilon=1e-30):
    """
    Log StableMax for Loss computation.
    """
    orig_dtype = logits.dtype
    # Use float32 for computation
    logits = logits.astype(mx.float32)
    
    s_x_neg = 1.0 / (1.0 - logits + epsilon)
    s_x_pos = logits + 1.0
    
    s_x = mx.where(logits < 0, s_x_neg, s_x_pos)
    sum_s_x = mx.sum(s_x, axis=axis, keepdims=True)
    
    log_probs = mx.log(s_x) - mx.log(sum_s_x)
    return log_probs.astype(orig_dtype)
