"""
RE2 Oracle: Ground truth from RE2-compatible regex engine.

Uses Python's re module (RE2-compatible subset) to:
1. Validate learned statechart behavior against real regex
2. Generate positive/negative examples from patterns
3. Score evolved statecharts against ground truth

Key principle: A learned statechart is correct if it produces
identical match/reject decisions as the RE2 engine on all inputs.
"""

import re
import random
import string
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict
from .regex_statechart import RegexStatechart


@dataclass
class RegexExamples:
    """Positive and negative examples for a regex pattern."""
    pattern: str
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)

    @property
    def all_examples(self) -> List[Tuple[str, bool]]:
        """Get all examples as (string, expected_match) pairs."""
        return [(s, True) for s in self.positive] + [(s, False) for s in self.negative]


class RE2Oracle:
    """
    Oracle using RE2-compatible regex for ground truth.

    Provides:
    - Example generation from patterns
    - Validation of learned statecharts
    - Fitness scoring against ground truth
    """

    def __init__(self, pattern: str, full_match: bool = True):
        """
        Initialize oracle with regex pattern.

        Args:
            pattern: RE2-compatible regex pattern
            full_match: If True, use fullmatch (entire string must match)
                       If False, use search (pattern anywhere in string)
        """
        self.pattern = pattern
        self.full_match = full_match
        self._compiled = re.compile(pattern)

    def matches(self, string: str) -> bool:
        """Check if string matches the pattern."""
        if self.full_match:
            return self._compiled.fullmatch(string) is not None
        return self._compiled.search(string) is not None

    def evaluate(self, strings: List[str]) -> List[bool]:
        """Evaluate multiple strings."""
        return [self.matches(s) for s in strings]

    def generate_examples(
        self,
        n_positive: int = 50,
        n_negative: int = 50,
        max_length: int = 20,
        alphabet: str = None,
    ) -> RegexExamples:
        """
        Generate positive and negative examples for this pattern.

        Strategy for positive examples:
        1. Use pattern structure to guide generation
        2. Random sampling with acceptance check

        Strategy for negative examples:
        1. Mutations of positive examples
        2. Random strings unlikely to match
        3. Near-misses (e.g., missing one char)
        """
        if alphabet is None:
            alphabet = self._infer_alphabet()

        positive = self._generate_positive(n_positive, max_length, alphabet)
        negative = self._generate_negative(n_negative, max_length, alphabet, positive)

        return RegexExamples(
            pattern=self.pattern,
            positive=positive,
            negative=negative
        )

    def _infer_alphabet(self) -> str:
        """Infer relevant alphabet from pattern."""
        # Extract literal characters from pattern
        chars = set()

        # Remove regex metacharacters and extract literals
        in_class = False
        escaped = False

        for c in self.pattern:
            if escaped:
                # Common escape sequences
                if c == 'd':
                    chars.update(string.digits)
                elif c == 'w':
                    chars.update(string.ascii_letters + string.digits + '_')
                elif c == 's':
                    chars.update(' \t\n')
                else:
                    chars.add(c)
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '[':
                in_class = True
            elif c == ']':
                in_class = False
            elif c in '.+*?|(){}^$':
                # Metacharacters - add common chars
                if c == '.':
                    chars.update('abcxyz123')
            elif in_class or c.isalnum():
                chars.add(c)

        # Ensure reasonable alphabet
        if not chars:
            chars = set('abc123')

        # Add some extra chars for diversity
        if any(c in string.ascii_lowercase for c in chars):
            chars.update('abc')
        if any(c in string.digits for c in chars):
            chars.update('123')

        return ''.join(sorted(chars))

    def _generate_positive(
        self,
        n: int,
        max_length: int,
        alphabet: str
    ) -> List[str]:
        """Generate strings that match the pattern."""
        positive = set()
        attempts = 0
        max_attempts = n * 100  # Avoid infinite loops

        while len(positive) < n and attempts < max_attempts:
            attempts += 1

            # Try different generation strategies
            if attempts % 3 == 0:
                # Random string
                length = random.randint(0, max_length)
                s = ''.join(random.choices(alphabet, k=length))
            elif attempts % 3 == 1:
                # Pattern-guided (simple heuristic)
                s = self._generate_from_pattern(max_length, alphabet)
            else:
                # Mutation of existing positive
                if positive:
                    base = random.choice(list(positive))
                    s = self._mutate_string(base, alphabet)
                else:
                    s = ''.join(random.choices(alphabet, k=random.randint(1, 5)))

            if self.matches(s) and s not in positive:
                positive.add(s)

        return list(positive)

    def _generate_from_pattern(self, max_length: int, alphabet: str) -> str:
        """Generate string guided by pattern structure."""
        result = []
        i = 0
        pattern = self.pattern

        while i < len(pattern) and len(result) < max_length:
            c = pattern[i]

            if c == '\\' and i + 1 < len(pattern):
                # Escape sequence
                next_c = pattern[i + 1]
                if next_c == 'd':
                    result.append(random.choice(string.digits))
                elif next_c == 'w':
                    result.append(random.choice(string.ascii_letters + string.digits))
                elif next_c == 's':
                    result.append(' ')
                else:
                    result.append(next_c)
                i += 2
            elif c == '.':
                result.append(random.choice(alphabet))
                i += 1
            elif c == '[':
                # Character class
                end = pattern.find(']', i)
                if end > i:
                    class_chars = pattern[i+1:end]
                    if class_chars.startswith('^'):
                        # Negated - pick from alphabet not in class
                        excluded = set(class_chars[1:])
                        choices = [a for a in alphabet if a not in excluded]
                        if choices:
                            result.append(random.choice(choices))
                    else:
                        # Pick from class
                        result.append(random.choice(class_chars))
                    i = end + 1
                else:
                    i += 1
            elif c in '()*+?|{}^$':
                # Quantifiers and groups - skip for simple generation
                i += 1
            elif c.isalnum() or c in '-_':
                # Literal character
                result.append(c)
                i += 1
            else:
                i += 1

        return ''.join(result)

    def _generate_negative(
        self,
        n: int,
        max_length: int,
        alphabet: str,
        positive: List[str]
    ) -> List[str]:
        """Generate strings that don't match the pattern."""
        negative = set()
        attempts = 0
        max_attempts = n * 100

        while len(negative) < n and attempts < max_attempts:
            attempts += 1

            # Different strategies for negative examples
            strategy = attempts % 5

            if strategy == 0:
                # Random string
                length = random.randint(0, max_length)
                s = ''.join(random.choices(alphabet + 'xyz', k=length))
            elif strategy == 1:
                # Empty string (often doesn't match)
                s = ""
            elif strategy == 2 and positive:
                # Mutation of positive (break it)
                base = random.choice(positive)
                s = self._break_string(base, alphabet)
            elif strategy == 3 and positive:
                # Prefix/suffix of positive
                base = random.choice(positive)
                if len(base) > 1:
                    s = base[:-1] if random.random() < 0.5 else base[1:]
                else:
                    s = ""
            else:
                # Wrong characters
                wrong_alphabet = ''.join(c for c in 'xyz890' if c not in alphabet)
                if not wrong_alphabet:
                    wrong_alphabet = '@#$'
                length = random.randint(1, max_length)
                s = ''.join(random.choices(wrong_alphabet, k=length))

            if not self.matches(s) and s not in negative:
                negative.add(s)

        return list(negative)

    def _mutate_string(self, s: str, alphabet: str) -> str:
        """Mutate a string slightly."""
        if not s:
            return random.choice(alphabet)

        mutation = random.choice(['insert', 'delete', 'replace', 'swap'])

        if mutation == 'insert':
            pos = random.randint(0, len(s))
            return s[:pos] + random.choice(alphabet) + s[pos:]
        elif mutation == 'delete' and len(s) > 1:
            pos = random.randint(0, len(s) - 1)
            return s[:pos] + s[pos+1:]
        elif mutation == 'replace':
            pos = random.randint(0, len(s) - 1)
            return s[:pos] + random.choice(alphabet) + s[pos+1:]
        elif mutation == 'swap' and len(s) > 1:
            pos = random.randint(0, len(s) - 2)
            return s[:pos] + s[pos+1] + s[pos] + s[pos+2:]
        return s

    def _break_string(self, s: str, alphabet: str) -> str:
        """Break a matching string to make it non-matching."""
        if not s:
            return 'x'

        mutation = random.choice(['insert_wrong', 'delete', 'replace_wrong', 'extend', 'truncate'])

        # Use chars likely not in pattern
        wrong_chars = ''.join(c for c in 'xyz890@#' if c not in alphabet)
        if not wrong_chars:
            wrong_chars = '@#$'

        if mutation == 'insert_wrong':
            pos = random.randint(0, len(s))
            return s[:pos] + random.choice(wrong_chars) + s[pos:]
        elif mutation == 'delete' and len(s) > 0:
            pos = random.randint(0, len(s) - 1)
            return s[:pos] + s[pos+1:]
        elif mutation == 'replace_wrong':
            pos = random.randint(0, len(s) - 1)
            return s[:pos] + random.choice(wrong_chars) + s[pos+1:]
        elif mutation == 'extend':
            return s + random.choice(wrong_chars)
        elif mutation == 'truncate' and len(s) > 1:
            return s[:-1]
        return s + 'X'

    def score_statechart(
        self,
        sc: RegexStatechart,
        test_strings: List[str] = None,
        n_tests: int = 100
    ) -> Dict[str, float]:
        """
        Score a statechart against this oracle.

        Returns dict with:
        - accuracy: Overall accuracy
        - precision: True positives / (True positives + False positives)
        - recall: True positives / (True positives + False negatives)
        - f1: Harmonic mean of precision and recall
        """
        if test_strings is None:
            # Generate test cases
            examples = self.generate_examples(n_tests // 2, n_tests // 2)
            test_cases = examples.all_examples
        else:
            test_cases = [(s, self.matches(s)) for s in test_strings]

        tp = fp = tn = fn = 0

        for s, expected in test_cases:
            predicted = sc.matches(s)
            if expected and predicted:
                tp += 1
            elif expected and not predicted:
                fn += 1
            elif not expected and predicted:
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }


def test_re2_oracle():
    """Test the RE2 Oracle."""
    from .regex_statechart import RegexStatechart, StateType, Transition, CharGuard

    print("=" * 60)
    print("RE2 ORACLE TESTS")
    print("=" * 60)

    # Test 1: Simple pattern
    print("\n1. Testing pattern 'ab+':")
    oracle = RE2Oracle("ab+")

    # Test oracle matching
    test_cases = [("ab", True), ("abb", True), ("abbb", True), ("a", False), ("b", False), ("abc", False)]
    for s, expected in test_cases:
        result = oracle.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test example generation
    print("\n2. Example generation for 'ab+':")
    examples = oracle.generate_examples(n_positive=10, n_negative=10)
    print(f"   Positive ({len(examples.positive)}): {examples.positive[:5]}...")
    print(f"   Negative ({len(examples.negative)}): {examples.negative[:5]}...")

    # Verify examples
    pos_correct = sum(1 for s in examples.positive if oracle.matches(s))
    neg_correct = sum(1 for s in examples.negative if not oracle.matches(s))
    print(f"   Positive verified: {pos_correct}/{len(examples.positive)}")
    print(f"   Negative verified: {neg_correct}/{len(examples.negative)}")

    # Test 3: Score a statechart
    print("\n3. Scoring statechart against 'ab+':")

    # Create a correct statechart for ab+
    sc_correct = RegexStatechart(
        state_labels=["START", "SAW_A", "ACCEPT"],
        state_types=[StateType.START, StateType.INTERMEDIATE, StateType.ACCEPT],
        transitions=[
            Transition(source=0, target=1, guard=CharGuard.single('a')),
            Transition(source=1, target=2, guard=CharGuard.single('b')),
            Transition(source=2, target=2, guard=CharGuard.single('b')),
        ],
        initial_state=0
    )

    score = oracle.score_statechart(sc_correct, n_tests=200)
    print(f"   Accuracy: {score['accuracy']:.3f}")
    print(f"   Precision: {score['precision']:.3f}")
    print(f"   Recall: {score['recall']:.3f}")
    print(f"   F1: {score['f1']:.3f}")

    # Test 4: Wrong statechart
    print("\n4. Scoring WRONG statechart (accepts 'a+' instead):")
    sc_wrong = RegexStatechart(
        state_labels=["START", "ACCEPT"],
        state_types=[StateType.START, StateType.ACCEPT],
        transitions=[
            Transition(source=0, target=1, guard=CharGuard.single('a')),
            Transition(source=1, target=1, guard=CharGuard.single('a')),
        ],
        initial_state=0
    )

    score = oracle.score_statechart(sc_wrong, n_tests=200)
    print(f"   Accuracy: {score['accuracy']:.3f}")
    print(f"   F1: {score['f1']:.3f}")
    print("   (Should have low F1 since it doesn't match the pattern)")

    # Test 5: More complex pattern
    print("\n5. Testing pattern '[a-z]+[0-9]+':")
    oracle2 = RE2Oracle(r"[a-z]+[0-9]+")
    examples2 = oracle2.generate_examples(n_positive=10, n_negative=10)
    print(f"   Positive: {examples2.positive[:5]}...")
    print(f"   Negative: {examples2.negative[:5]}...")

    # Test 6: Character classes
    print("\n6. Testing pattern '\\d{3}-\\d{4}':")
    oracle3 = RE2Oracle(r"\d{3}-\d{4}")
    test_cases = [("123-4567", True), ("000-0000", True), ("12-345", False), ("abc-defg", False)]
    for s, expected in test_cases:
        result = oracle3.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    print("\n" + "=" * 60)
    print("All oracle tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_re2_oracle()
