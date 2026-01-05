import sys
import subprocess
from mlx_lm import load, generate

def main():
    model_path = "adapters" # Default mlx-lm output
    
    print(f"Loading model from {model_path}...")
    model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct", adapter_path=model_path)
    
    prompt = "User: Create a cyclical statechart with states A, B, C.\nAssistant:\n"
    
    print("Generating...")
    response = generate(model, tokenizer, prompt=prompt, max_tokens=500)
    
    print("Response:")
    print(response)
    
    # Save to file
    with open("generated.star", "w") as f:
        f.write(response)
    
    # Verify
    print("Verifying...")
    result = subprocess.run(["sc-run", "generated.star"], capture_output=True, text=True)
    if result.returncode == 0:
        print("Verification SUCCESS")
        print(result.stdout)
    else:
        print("Verification FAILED")
        print(result.stderr)

if __name__ == "__main__":
    main()
