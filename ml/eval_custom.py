
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sys
import os
import subprocess

# Fix path to find local modules
sys.path.append(os.path.dirname(__file__))

from tokenizer import CharTokenizer
from model import Transformer

def generate_text(model, tokenizer, prompt, max_tokens=500, temp=0.7):
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)
    
    # Generation Loop
    for _ in range(max_tokens):
        # Forward pass
        # Batch dim = 1
        x = tokens[None, :]
        logits = model(x)
        
        # Get last token logits
        last_logits = logits[0, -1, :]
        
        # Sampling
        if temp == 0:
            next_token = mx.argmax(last_logits).item()
        else:
            # mlx doesn't have categorical sample easily accessible in core yet?
            # We can use np for sampling for now if performance isn't critical
            p = mx.softmax(last_logits / temp)
            p = np.array(p)
            next_token = np.random.choice(len(p), p=p)
            
        # Append
        tokens = mx.concatenate([tokens, mx.array([next_token])], axis=0)
        
        # Stop check? (None for char level usually, or specific token)
        # If we had <EOS> we would stop.
        if next_token == tokenizer.char_to_idx.get("<EOS>"):
            break
            
    return tokenizer.decode(tokens.tolist())

def main():
    if not os.path.exists("ml/tokenizer.json") or not os.path.exists("ml/custom_model.npz"):
        print("Model or tokenizer not found. Run training first.")
        # Create dummy for testing if not exists?
        # verification step
        return

    # Load Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.load("ml/tokenizer.json")
    
    # Load Model structure
    # We need to know params. For now hardcode or save config.
    vocab_size = tokenizer.vocab_size
    model = Transformer(vocab_size)
    model.load_weights("ml/custom_model.npz")
    
    # Test Prompt for Recolor
    # Example Grid
    prompt = "User: Grid:\n[[1, 0], [0, 1]]\nReplace 1 with 2.\nAssistant:\n"
    
    print(f"Prompt: {prompt.strip()}")
    print("Generating...")
    
    output = generate_text(model, tokenizer, prompt)
    print("Generated Output:")
    print(output)
    
    # Extract code (simple heuristic: after "Assistant:\n")
    # In this prompt setup, output includes prompt.
    if "Assistant:\n" in output:
        code = output.split("Assistant:\n", 1)[1]
    else:
        code = output
        
    # Save code
    with open("generated.star", "w") as f:
        f.write(code)
        
    # Run verification with sc-run
    print("Verifying with sc-run...")
    res = subprocess.run(["sc-run", "generated.star"], capture_output=True, text=True)
    if res.returncode == 0:
        print("SUCCESS: Code executed successfully.")
        print(res.stdout)
    else:
        print("FAILURE: Code execution failed.")
        print(res.stderr)

if __name__ == "__main__":
    main()
