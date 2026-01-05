
import os
import mlx.core as mx
from mlx_lm.utils import load
from pathlib import Path
import sys

# Paths to model artifacts
# Assuming running from ml/experiments/experiment_name/
MODEL_PATH = Path("../../custom_model.npz").resolve()
TOKENIZER_PATH = Path("../../tokenizer.json").resolve()

# Fallback pointers if running from ml/ root
if not MODEL_PATH.exists():
    MODEL_PATH = Path("custom_model.npz").resolve()
    TOKENIZER_PATH = Path("tokenizer.json").resolve()

def load_model_and_tokenizer():
    """
    Load the custom MLX model and tokenizer.
    Returns: (model, tokenizer)
    """
    print(f"Loading model from {MODEL_PATH}...")
    
    # We need to construct a minimal directory structure that mlx_lm expects
    # or use the appropriate loading method for npz files if mlx_lm requires a dir.
    # Based on previous file explorations, custom_model.npz seems to be a weights file.
    # However, mlx_lm.load() typically expects a HuggingFace directory or repo id.
    
    # Let's check how eval_custom.py does it.
    # It appears eval_custom.py might assume a different structure or use a Model class.
    # Let's import the simple Model class from ml/model.py if possible, 
    # OR if this is an mlx-lm adapter, we load it differently.

    # Re-reading `ml/eval.py`:
    #   model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    #
    # Re-reading `ml/eval_custom.py`:
    #   vocab, tokenizer = load_vocab(...)
    #   model = Transformer(args, vocab)
    #   model.load_weights("custom_model.npz")
    
    # So we should follow the eval_custom.py pattern for "custom_model.npz".
    
    # 1. Load Tokenizer
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    
    try:
        # Try relative import first
        from ..tokenizer import CharTokenizer
        from ..model import Transformer
    except ImportError:
        # Fallback to direct import if ml/ is in sys.path
        try: 
            from tokenizer import CharTokenizer
            from model import Transformer
        except ImportError:
           # Fallback for deep relative path issues - try adding ../.. to path specifically for these
           sys.path.append(str(Path(__file__).parent.parent))
           from tokenizer import CharTokenizer
           from model import Transformer

    tokenizer = CharTokenizer()
    tokenizer.load(str(TOKENIZER_PATH))
    
    # 2. Initialize Model
    # We need to know the args used during training. 
    # For now, we'll assume default args or try to infer/hardcode for this specific custom model 
    # if config isn't saved. 
    # Looking at `metrics.json` or similar might help, but let's assume standard small config for the experiment.
    
    
    # NOTE: ModelArgs needs to match what was trained. 
    # Default matching eval_custom.py
    model = Transformer(tokenizer.vocab_size)
    
    print(f"Loading weights from {MODEL_PATH}...")
    model.load_weights(str(MODEL_PATH))
    
    return model, tokenizer

def load_mlx_lm_model(model_path):
    """(Alternative) Load a standard HF/MLX-LM model."""
    return load(model_path)
