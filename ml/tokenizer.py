
class CharTokenizer:
    def __init__(self, data=None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        if data:
            self.fit(data)

    def fit(self, text):
        chars = sorted(list(set(text)))
        # Reserve 0 for PAD? Or just standard.
        # Let's use 0 for PAD, 1 for SOS, 2 for EOS if needed.
        # For simplicity, 0 is just the first char.
        # Actually, let's explicit PAD/UNK.
        self.chars = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + chars
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char_to_idx.get(ch, self.char_to_idx["<UNK>"]) for ch in text]

    def decode(self, indices):
        return "".join([self.idx_to_char.get(idx, "") for idx in indices])
    
    def save(self, path):
        import json
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f)
            
    def load(self, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
            self.chars = data["chars"]
            self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
            self.vocab_size = len(self.chars)
