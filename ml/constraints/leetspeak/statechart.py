import mlx.core as mx
import numpy as np


class LeetSpeakStatechart:
    """A simple statechart enforcing "Leetspeak" style constraints.

    Vocabulary: 4, 3, 1, 0, 7, !, s, p, k, n, m, space, newline

    Rules:
    1. Words must start with a number.
    2. Cannot have two spaces in a row.
    """

    STATE_START_WORD = 0
    STATE_IN_WORD = 1
    STATE_SPACE = 2

    ALLOWED_CHARS = "43107!spknm \n"
    NUMBERS = "43107"
    LETTERS = "!spknm"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = self.STATE_START_WORD
        self.last_char = '\n'

    def step(self, char):
        """Advance the statechart by one character."""
        if self.state == self.STATE_START_WORD:
            if char in self.NUMBERS:
                self.state = self.STATE_IN_WORD
            elif char == ' ' or char == '\n':
                self.state = self.STATE_SPACE
        elif self.state == self.STATE_IN_WORD:
            if char == ' ' or char == '\n':
                self.state = self.STATE_SPACE
            # else stay in word
        elif self.state == self.STATE_SPACE:
            if char in self.NUMBERS:
                self.state = self.STATE_IN_WORD
            # cannot stay in space (rule 2)

        self.last_char = char

    def get_mask(self, logits):
        """Get logit mask based on current state."""
        vocab_size = logits.shape[0]
        mask = np.zeros(vocab_size, dtype=np.float32)

        # Default: Forbid everything
        mask[:] = -float('inf')

        def allow(chars):
            for c in chars:
                idx = self.tokenizer.char_to_idx.get(c)
                if idx is not None:
                    mask[idx] = 0.0

        if self.state == self.STATE_START_WORD:
            # Must start with number
            allow(self.NUMBERS)
            # Also allow space/newline to skip
            allow(" \n")

        elif self.state == self.STATE_IN_WORD:
            # Can continue with anything allowed
            allow(self.ALLOWED_CHARS)

        elif self.state == self.STATE_SPACE:
            # Must start new word (Number)
            allow(self.NUMBERS)

        return mx.array(mask)

    def reset(self):
        """Reset statechart to initial state."""
        self.state = self.STATE_START_WORD
        self.last_char = '\n'
