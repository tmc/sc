from mlx_lm import load, generate

model, tokenizer = load("Qwen/Qwen2.5-0.5B-Instruct", adapter_path="ml/adapters")

prompts = [
    "Translate to Leetspeak: SIMPLICITY IS THE ULTIMATE SOPHISTICATION.",
    "Translate to Leetspeak: Hello world.",
]

for p in prompts:
    messages = [{"role": "user", "content": p}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(f"DEBUG PROMPT: {repr(prompt)}")
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
    print(f"Response: {response}\n")
