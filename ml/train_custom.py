
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json
import numpy as np
import sys
import os
import subprocess

sys.path.append(os.path.dirname(__file__))

from tokenizer import CharTokenizer
from model import Transformer
from sampling import constrained_generate

def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["train"], data["test"]

def format_example(example):
    # Format: Input -> Output
    # We will simply concatenate with a separator? 
    # Or just Input\nOutput.
    return f"User: {example['input']}\nAssistant:\n{example['output']}"

def train():
    # Hyperparams
    batch_size = 4
    context_size = 1024
    d_model = 256
    n_layers = 4
    n_heads = 4
    lr = 1e-3
    n_iters = 200

    # Load Data
    train_data, test_data = load_data("ml/data/solver/recolor_data.json")
    
    # Prepare Tokenizer
    # Collect all text to fit tokenizer
    all_text = ""
    for ex in train_data + test_data:
        all_text += format_example(ex)
        
    # DreamCoder: Load Library Data (Sleep Phase setup)
    lib_dir = "ml/data/solver/library"
    if os.path.exists(lib_dir):
        print("Dreaming: Loading valid solutions from library...")
        for f in os.listdir(lib_dir):
            if f.endswith(".txt"):
                with open(os.path.join(lib_dir, f), "r") as fd:
                     content = fd.read()
                     # Split into user/assistant roughly if needed, or just append as training text
                     # Our tokenizer fits on all text, so appending is fine for vocab.
                     # For training, we need to add to train_data.
                     # But content is raw text "User: ... Assistant: ...".
                     # We need to parse back into Pair or just use as raw text in batch if our batcher supports it.
                     # Let's simple format: assume content IS the formatted example.
                     all_text += content
    
    tokenizer = CharTokenizer()
    tokenizer.fit(all_text)
    tokenizer.save("ml/tokenizer.json")
    
    # Init Model
    model = Transformer(tokenizer.vocab_size, d_model, n_layers, n_heads, context_size)
    mx.eval(model.parameters())

    # Optimizer
    optimizer = optim.AdamW(learning_rate=lr)

    # Loss Function
    def loss_fn(model, x, y):
        logits = model(x)
        # Cross Entropy
        # mlx.nn.losses.cross_entropy expects (logits, targets)
        # targets should be same shape as logits keys or just indices?
        # nn.losses.cross_entropy(logits, targets, reduction='mean')
        # logits: [B, T, V], targets: [B, T]
        Loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        return Loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # RLHF / Validation function
    def validate_and_learn(generated_code):
        # Save to temp txtar
        temp_file = f"temp_gen_{np.random.randint(1000)}.txtar"
        # We need to wrap code in txtar format if it's not already?
        # The model output is raw code or text.
        # Let's extract code part if structured
        code = generated_code
        if "Code:" in generated_code:
            code = generated_code.split("Code:", 1)[1]
        
        # Ensure it has a main
        if "def main():" not in code:
             # Basic wrapper for generated solve function
             code += "\n\ndef main():\n    # Test with dummy grid\n    g = [[1,0],[0,1]]\n    print(solve(g))"
             
        # Create txtar content
        txtar_content = f"-- main.star --\nload('sc', 'sc')\n{code}"
        
        with open(temp_file, "w") as f:
            f.write(txtar_content)
            
        # Run sc-run
        res = subprocess.run(["sc-run", temp_file], capture_output=True, text=True)
        
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return res.returncode == 0

    # Data Loader (DreamCoder Mixed Batch)
    def batch_iter():
        library_files = []
        if os.path.exists(lib_dir):
            library_files = [os.path.join(lib_dir, f) for f in os.listdir(lib_dir) if f.endswith(".txt")]
            
        while True:
            # Mix Synthetic and Library Data
            # 80% synthetic, 20% library (if available)
            current_batch_size = batch_size
            lib_batch_count = 0
            if library_files and len(library_files) > 0:
                 lib_batch_count = max(1, int(batch_size * 0.2))
                 current_batch_size = batch_size - lib_batch_count
            
            batch_indices = np.random.randint(0, len(train_data), current_batch_size)
            batch_exs = [format_example(train_data[i]) for i in batch_indices]
            
            # Add library samples
            for _ in range(lib_batch_count):
                f = library_files[np.random.randint(len(library_files))]
                with open(f, "r") as fd:
                    batch_exs.append(fd.read())
            
            # Tokenize & Pad
            encoded = [tokenizer.encode(ex) for ex in batch_exs]
            max_len = max(len(e) for e in encoded)
            # Truncate if exceeds context
            if max_len > context_size + 1:
                max_len = context_size + 1
            
            # Setup X and Y (shifted)
            # Pad with 0 (<PAD>)
            X = np.zeros((batch_size, max_len - 1), dtype=np.int32)
            Y = np.zeros((batch_size, max_len - 1), dtype=np.int32)
            
            for i, e in enumerate(encoded):
                # crop
                if len(e) > max_len:
                    e = e[:max_len]
                
                # input: e[:-1], target: e[1:]
                l = len(e) - 1
                X[i, :l] = e[:-1]
                Y[i, :l] = e[1:]
            
            yield mx.array(X), mx.array(Y)

    data_iterator = batch_iter()

    # Loop
    print(f"Starting training for {n_iters} iterations with RLHF-lite...")
    for it in range(n_iters):
        X, Y = next(data_iterator)
        loss, grads = loss_and_grad(model, X, Y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        # DreamCoder Step: Wake Phase & Library Learning
        if it > 0 and it % 50 == 0:
             # Generate a sample using Constrained Sampling
             prompt = "User: Grid:\n[[1, 0], [0, 1]]\nTask: Replace 1 with 2.\nAssistant:\n"
             
             # Use the new sampling function
             gen_text = constrained_generate(model, tokenizer, prompt, max_tokens=200)
             
             is_valid = validate_and_learn(gen_text)
             status = "VALID" if is_valid else "INVALID"
             print(f"Iter {it} Validation: {status}")
             
             # Save to Library if valid
             if is_valid:
                 lib_dir = "ml/data/solver/library"
                 if not os.path.exists(lib_dir):
                     os.makedirs(lib_dir)
                 
                 # Save simple pair
                 # Ideally we parse out the input grid from prompt and full output
                 # For now, just save the raw text to be re-tokenized later (Simplified Dreaming)
                 timestamp = np.random.randint(100000)
                 with open(f"{lib_dir}/solution_{timestamp}.txt", "w") as f:
                     f.write(prompt + gen_text)

    print("Training Complete.")
    # Save weights
    model.save_weights("ml/custom_model.npz")

if __name__ == "__main__":
    train()
