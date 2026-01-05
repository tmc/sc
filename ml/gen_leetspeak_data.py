import json
import random

def to_leetspeak(text):
    mapping = {
        'a': '4', 'A': '4',
        'e': '3', 'E': '3',
        'i': '1', 'I': '1',
        'o': '0', 'O': '0',
        't': '7', 'T': '7',
        'l': '1', 'L': '1',
        's': '5', 'S': '5',
        'g': '9', 'G': '9',
    }
    return "".join(mapping.get(c, c) for c in text)

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test.",
    "Machine learning is fascinating.",
    "Artificial intelligence will change the world.",
    "Coding is like magic without the spells.",
    "Never give up on your dreams.",
    "The journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Knowledge is power.",
    "Simplicity is the ultimate sophistication.",
    "Time flies like an arrow.",
    "Fruit flies like a banana.",
    "I love programming in Python.",
    "Go is a statically typed language.",
    "Rust guarantees memory safety.",
    "Starlark is a dialect of Python.",
    "Silicon Valley is the hub of tech.",
    "Keep calm and carry on.",
    "Live long and prosper.",
    "May the force be with you.",
    "Winter is coming.",
    "Just do it.",
    "Think different.",
    "Innovation distinguishes between a leader and a follower.",
    "Stay hungry, stay foolish.",
    "It does not matter how slowly you go as long as you do not stop.",
    "Everything you can imagine is real.",
    "What we think, we become.",
    "Do what you can, with what you have, where you are.",
]

def generate_dataset(output_path, num_examples=1000):
    data = []
    for _ in range(num_examples):
        sent = random.choice(sentences)
        # Add some variation? Maybe simple variation like uppercasing randomly
        if random.random() < 0.3:
            sent = sent.upper()
        elif random.random() < 0.3:
            sent = sent.lower()
            
        leet = to_leetspeak(sent)
        
        # Chat format for mlx-lm
        entry = {
            "messages": [
                {"role": "user", "content": f"Translate to Leetspeak: {sent}"},
                {"role": "assistant", "content": leet}
            ]
        }
        data.append(entry)
        
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Generated {num_examples} examples to {output_path}")

if __name__ == "__main__":
    generate_dataset("ml/data/leetspeak_train.jsonl")
    generate_dataset("ml/data/leetspeak_valid.jsonl", num_examples=100)
