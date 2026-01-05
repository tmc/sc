import mlx.core as mx
from .statechart import LeetSpeakStatechart


def constrained_generate(model, tokenizer, prompt, max_tokens=200):
    """Generate text with leetspeak constraints enforced by statechart.

    Args:
        model: The language model
        tokenizer: Tokenizer with char_to_idx and idx_to_char mappings
        prompt: Initial prompt string
        max_tokens: Maximum tokens to generate

    Returns:
        Generated string including prompt
    """
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)

    # Initialize Statechart
    sc = LeetSpeakStatechart(tokenizer)

    # Warmup statechart on prompt
    decoded_prompt = tokenizer.decode(tokens.tolist())
    for char in decoded_prompt:
        sc.step(char)

    for _ in range(max_tokens):
        x = tokens[None, :]
        logits = model(x)
        logits = logits[0, -1, :]  # [Vocab]

        # Get Statechart Constraints
        mask = sc.get_mask(logits)
        logits = logits + mask

        # Sample
        next_token = mx.argmax(logits).item()

        # Update State
        char = tokenizer.idx_to_char.get(next_token, "")
        sc.step(char)

        tokens = mx.concatenate([tokens, mx.array([next_token])], axis=0)

        if next_token == tokenizer.char_to_idx.get("<EOS>", 0):
            break

    return tokenizer.decode(tokens.tolist())
