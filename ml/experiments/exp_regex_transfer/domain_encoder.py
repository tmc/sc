"""
Domain Encoder - Domain-Agnostic Features for Regex Transfer

Extract structural features that transfer across regex domains:
- Email: local@domain.tld
- URL: protocol://domain/path
- Phone: +1-area-prefix-line

Transferable concepts:
1. SEGMENT: A run of similar characters (alphanumeric, digits)
2. DELIMITER: Special characters separating segments (@, /, ., -)
3. LENGTH: Segment length constraints (min, max, exact)
4. REPETITION: Patterns that repeat (a+, [a-z]+)
5. POSITION: Where in the pattern (start, middle, end)

Domain-agnostic encoding captures STRUCTURE, not specific characters.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
import re
import random


# =============================================================================
# Character Classes (Domain-Agnostic)
# =============================================================================

class CharType(Enum):
    """Abstract character types that transfer across domains."""
    ALPHA_LOWER = auto()    # a-z
    ALPHA_UPPER = auto()    # A-Z
    DIGIT = auto()          # 0-9
    DELIMITER_DOT = auto()  # .
    DELIMITER_AT = auto()   # @
    DELIMITER_SLASH = auto()  # /
    DELIMITER_DASH = auto()  # -
    DELIMITER_PLUS = auto()  # +
    DELIMITER_COLON = auto()  # :
    SPECIAL = auto()        # Other special chars
    WHITESPACE = auto()     # Space, tab
    UNKNOWN = auto()        # Unknown


def get_char_type(char: str) -> CharType:
    """Classify a character into abstract type."""
    if 'a' <= char <= 'z':
        return CharType.ALPHA_LOWER
    elif 'A' <= char <= 'Z':
        return CharType.ALPHA_UPPER
    elif '0' <= char <= '9':
        return CharType.DIGIT
    elif char == '.':
        return CharType.DELIMITER_DOT
    elif char == '@':
        return CharType.DELIMITER_AT
    elif char == '/':
        return CharType.DELIMITER_SLASH
    elif char == '-':
        return CharType.DELIMITER_DASH
    elif char == '+':
        return CharType.DELIMITER_PLUS
    elif char == ':':
        return CharType.DELIMITER_COLON
    elif char in ' \t\n':
        return CharType.WHITESPACE
    elif char in '_~!#$%^&*()[]{}|\\;\'",<>?':
        return CharType.SPECIAL
    else:
        return CharType.UNKNOWN


# =============================================================================
# Segment Representation
# =============================================================================

@dataclass
class Segment:
    """
    A segment is a structural unit in a pattern.

    Examples:
    - Email "user@domain.com":
      [Segment(ALPHA, 4), Segment(AT, 1), Segment(ALPHA, 6), Segment(DOT, 1), Segment(ALPHA, 3)]

    - URL "http://example.com/path":
      [Segment(ALPHA, 4), Segment(COLON, 1), Segment(SLASH, 2), Segment(ALPHA, 7), ...]
    """
    char_types: Set[CharType]  # Which character types appear
    min_length: int
    max_length: int
    position: int  # 0-indexed position in pattern
    is_delimiter: bool = False
    is_optional: bool = False
    can_repeat: bool = False

    def to_vector(self) -> List[float]:
        """Convert to fixed-size feature vector."""
        # 12 char type bits + 3 length features + 3 position features + 3 flags
        vec = [0.0] * 21

        # Char type one-hot (12 dims)
        for ct in self.char_types:
            vec[ct.value - 1] = 1.0

        # Length features (normalized)
        vec[12] = self.min_length / 20.0
        vec[13] = self.max_length / 20.0
        vec[14] = (self.max_length - self.min_length) / 10.0

        # Position features
        vec[15] = self.position / 10.0
        vec[16] = 1.0 if self.position == 0 else 0.0  # Is first
        vec[17] = 0.0  # Will be set later for "is last"

        # Flags
        vec[18] = 1.0 if self.is_delimiter else 0.0
        vec[19] = 1.0 if self.is_optional else 0.0
        vec[20] = 1.0 if self.can_repeat else 0.0

        return vec


@dataclass
class PatternStructure:
    """
    Abstract structural representation of a regex pattern.

    Domain-agnostic: focuses on structure, not specific characters.
    """
    segments: List[Segment] = field(default_factory=list)
    num_segments: int = 0
    has_repetition: bool = False
    has_optional: bool = False
    total_length_min: int = 0
    total_length_max: int = 0

    def to_vector(self) -> List[float]:
        """Convert to fixed-size feature vector."""
        # Global features (10) + segment vectors (up to 10 segments * 21 dims)
        vec = [0.0] * (10 + 10 * 21)

        # Global features
        vec[0] = self.num_segments / 10.0
        vec[1] = 1.0 if self.has_repetition else 0.0
        vec[2] = 1.0 if self.has_optional else 0.0
        vec[3] = self.total_length_min / 50.0
        vec[4] = self.total_length_max / 100.0

        # Count segment types
        n_delim = sum(1 for s in self.segments if s.is_delimiter)
        n_alpha = sum(1 for s in self.segments if CharType.ALPHA_LOWER in s.char_types)
        n_digit = sum(1 for s in self.segments if CharType.DIGIT in s.char_types)

        vec[5] = n_delim / 10.0
        vec[6] = n_alpha / 10.0
        vec[7] = n_digit / 10.0
        vec[8] = (n_alpha + n_digit) / 10.0  # Content segments
        vec[9] = n_delim / max(1, n_alpha + n_digit)  # Delimiter ratio

        # Segment vectors
        for i, seg in enumerate(self.segments[:10]):
            seg_vec = seg.to_vector()
            # Mark last segment
            if i == len(self.segments) - 1:
                seg_vec[17] = 1.0
            vec[10 + i * 21: 10 + (i + 1) * 21] = seg_vec

        return vec


# =============================================================================
# Pattern Analyzer
# =============================================================================

class PatternAnalyzer:
    """
    Analyze patterns to extract domain-agnostic structure.

    Works with raw strings or regex patterns.
    """

    def __init__(self):
        self.delimiter_types = {
            CharType.DELIMITER_DOT,
            CharType.DELIMITER_AT,
            CharType.DELIMITER_SLASH,
            CharType.DELIMITER_DASH,
            CharType.DELIMITER_PLUS,
            CharType.DELIMITER_COLON,
        }

    def analyze_string(self, s: str) -> PatternStructure:
        """
        Analyze a concrete string to extract structure.

        Segments are runs of similar character types.
        """
        if not s:
            return PatternStructure()

        segments = []
        current_types: Set[CharType] = set()
        current_start = 0

        for i, char in enumerate(s):
            char_type = get_char_type(char)
            is_delim = char_type in self.delimiter_types

            if i == 0:
                current_types.add(char_type)
            else:
                # Check if we should start a new segment
                prev_was_delim = any(t in self.delimiter_types for t in current_types)

                if is_delim != prev_was_delim or (is_delim and char_type not in current_types):
                    # End current segment
                    length = i - current_start
                    segments.append(Segment(
                        char_types=current_types,
                        min_length=length,
                        max_length=length,
                        position=len(segments),
                        is_delimiter=prev_was_delim,
                    ))
                    current_types = set()
                    current_start = i

                current_types.add(char_type)

        # Final segment
        length = len(s) - current_start
        is_delim = any(t in self.delimiter_types for t in current_types)
        segments.append(Segment(
            char_types=current_types,
            min_length=length,
            max_length=length,
            position=len(segments),
            is_delimiter=is_delim,
        ))

        return PatternStructure(
            segments=segments,
            num_segments=len(segments),
            total_length_min=len(s),
            total_length_max=len(s),
        )

    def analyze_examples(self, examples: List[str]) -> PatternStructure:
        """
        Analyze multiple examples to infer pattern structure.

        Generalizes from examples to find common structure.
        """
        if not examples:
            return PatternStructure()

        # Analyze each example
        structures = [self.analyze_string(s) for s in examples]

        # Find common structure (alignment by delimiter positions)
        # For simplicity, use first example as template and relax constraints
        template = structures[0]

        # Update length ranges from all examples
        for seg_idx in range(template.num_segments):
            if seg_idx >= len(template.segments):
                continue

            for struct in structures[1:]:
                if seg_idx < struct.num_segments:
                    other_seg = struct.segments[seg_idx]
                    template.segments[seg_idx].min_length = min(
                        template.segments[seg_idx].min_length,
                        other_seg.min_length
                    )
                    template.segments[seg_idx].max_length = max(
                        template.segments[seg_idx].max_length,
                        other_seg.max_length
                    )
                    template.segments[seg_idx].char_types.update(other_seg.char_types)

        # Update global stats
        template.total_length_min = min(s.total_length_min for s in structures)
        template.total_length_max = max(s.total_length_max for s in structures)
        template.has_repetition = any(s.has_repetition for s in structures)
        template.has_optional = any(len(s.segments) != template.num_segments for s in structures)

        return template


# =============================================================================
# Domain Encoder
# =============================================================================

class DomainEncoder:
    """
    Encode patterns in domain-agnostic representation.

    Key insight: Email, URL, and Phone share structural patterns:
    - Alternating content and delimiter segments
    - Length constraints on content segments
    - Fixed delimiter characters
    - Positional patterns (start, middle, end)

    The encoder produces vectors that capture STRUCTURE, not domain.
    """

    def __init__(self, feature_dim: int = 220):
        self.feature_dim = feature_dim
        self.analyzer = PatternAnalyzer()

        # Learned transformation (would be trained)
        self.W: Optional[mx.array] = None
        self.b: Optional[mx.array] = None

    def encode_pattern(self, examples: List[str]) -> mx.array:
        """
        Encode a pattern (given by examples) to domain-agnostic vector.
        """
        structure = self.analyzer.analyze_examples(examples)
        raw_vec = structure.to_vector()

        # Pad/truncate to feature_dim
        if len(raw_vec) < self.feature_dim:
            raw_vec = raw_vec + [0.0] * (self.feature_dim - len(raw_vec))
        else:
            raw_vec = raw_vec[:self.feature_dim]

        vec = mx.array(raw_vec)

        # Apply learned transformation if available
        if self.W is not None and self.b is not None:
            vec = mx.tanh(vec @ self.W + self.b)

        return vec

    def encode_single(self, s: str) -> mx.array:
        """Encode a single string."""
        return self.encode_pattern([s])

    def compute_similarity(self, vec1: mx.array, vec2: mx.array) -> float:
        """Compute structural similarity between two pattern vectors."""
        # Cosine similarity
        dot = float(mx.sum(vec1 * vec2))
        norm1 = float(mx.sqrt(mx.sum(vec1 * vec1))) + 1e-8
        norm2 = float(mx.sqrt(mx.sum(vec2 * vec2))) + 1e-8
        return dot / (norm1 * norm2)


# =============================================================================
# Domain Patterns
# =============================================================================

class DomainPatterns:
    """
    Generate examples from different domains.

    Used for training and evaluation.
    """

    @staticmethod
    def generate_email(n: int = 100) -> List[str]:
        """Generate email-like patterns."""
        emails = []
        local_parts = ['user', 'john', 'alice', 'bob', 'test', 'info', 'admin', 'sales']
        domains = ['example', 'gmail', 'yahoo', 'company', 'test', 'mail']
        tlds = ['com', 'org', 'net', 'io', 'edu']

        for _ in range(n):
            local = random.choice(local_parts)
            if random.random() < 0.3:
                local += str(random.randint(1, 999))
            domain = random.choice(domains)
            tld = random.choice(tlds)
            emails.append(f"{local}@{domain}.{tld}")

        return emails

    @staticmethod
    def generate_url(n: int = 100) -> List[str]:
        """Generate URL-like patterns."""
        urls = []
        protocols = ['http', 'https', 'ftp']
        domains = ['example', 'test', 'website', 'api', 'app', 'site']
        tlds = ['com', 'org', 'net', 'io']
        paths = ['', '/page', '/api/v1', '/user/profile', '/data', '/index']

        for _ in range(n):
            proto = random.choice(protocols)
            domain = random.choice(domains)
            tld = random.choice(tlds)
            path = random.choice(paths)
            urls.append(f"{proto}://{domain}.{tld}{path}")

        return urls

    @staticmethod
    def generate_phone(n: int = 100) -> List[str]:
        """Generate phone-like patterns."""
        phones = []
        formats = [
            lambda: f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
            lambda: f"({random.randint(200,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
            lambda: f"{random.randint(200,999)}.{random.randint(100,999)}.{random.randint(1000,9999)}",
            lambda: f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
        ]

        for _ in range(n):
            fmt = random.choice(formats)
            phones.append(fmt())

        return phones

    @staticmethod
    def generate_date(n: int = 100) -> List[str]:
        """Generate date-like patterns."""
        dates = []
        for _ in range(n):
            year = random.randint(1990, 2030)
            month = random.randint(1, 12)
            day = random.randint(1, 28)

            fmt = random.choice([
                f"{year}-{month:02d}-{day:02d}",
                f"{month:02d}/{day:02d}/{year}",
                f"{day:02d}.{month:02d}.{year}",
            ])
            dates.append(fmt)

        return dates

    @staticmethod
    def generate_ipv4(n: int = 100) -> List[str]:
        """Generate IPv4-like patterns."""
        ips = []
        for _ in range(n):
            octets = [str(random.randint(0, 255)) for _ in range(4)]
            ips.append('.'.join(octets))
        return ips


# =============================================================================
# Structural Similarity Matrix
# =============================================================================

def compute_domain_similarity() -> Dict[Tuple[str, str], float]:
    """
    Compute structural similarity between domains.

    This shows which domains have similar structure (good for transfer).
    """
    encoder = DomainEncoder()

    domains = {
        'email': DomainPatterns.generate_email(20),
        'url': DomainPatterns.generate_url(20),
        'phone': DomainPatterns.generate_phone(20),
        'date': DomainPatterns.generate_date(20),
        'ipv4': DomainPatterns.generate_ipv4(20),
    }

    # Encode each domain
    domain_vecs = {}
    for name, examples in domains.items():
        vec = encoder.encode_pattern(examples)
        domain_vecs[name] = vec

    # Compute pairwise similarity
    similarity = {}
    for name1, vec1 in domain_vecs.items():
        for name2, vec2 in domain_vecs.items():
            sim = encoder.compute_similarity(vec1, vec2)
            similarity[(name1, name2)] = sim

    return similarity


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate domain encoder."""
    print("=" * 60)
    print("Domain Encoder Demo")
    print("=" * 60)

    analyzer = PatternAnalyzer()
    encoder = DomainEncoder()

    # Analyze email
    print("\n--- Email Pattern Analysis ---")
    emails = DomainPatterns.generate_email(5)
    for email in emails[:3]:
        print(f"  {email}")
        struct = analyzer.analyze_string(email)
        print(f"    Segments: {struct.num_segments}")
        for i, seg in enumerate(struct.segments):
            types = [t.name for t in seg.char_types]
            print(f"      {i}: {types} len={seg.min_length}-{seg.max_length} delim={seg.is_delimiter}")

    # Analyze URL
    print("\n--- URL Pattern Analysis ---")
    urls = DomainPatterns.generate_url(5)
    for url in urls[:3]:
        print(f"  {url}")
        struct = analyzer.analyze_string(url)
        print(f"    Segments: {struct.num_segments}")

    # Compute domain similarity
    print("\n--- Domain Structural Similarity ---")
    similarity = compute_domain_similarity()

    domains = ['email', 'url', 'phone', 'date', 'ipv4']
    print(f"{'':>8}", end='')
    for d in domains:
        print(f"{d:>8}", end='')
    print()

    for d1 in domains:
        print(f"{d1:>8}", end='')
        for d2 in domains:
            sim = similarity[(d1, d2)]
            print(f"{sim:>8.3f}", end='')
        print()

    # Show which domains are most similar
    print("\n--- Transfer Candidates ---")
    pairs = [(d1, d2, sim) for (d1, d2), sim in similarity.items() if d1 < d2]
    pairs.sort(key=lambda x: -x[2])

    for d1, d2, sim in pairs[:5]:
        print(f"  {d1} <-> {d2}: {sim:.3f}")

    return encoder, similarity


if __name__ == "__main__":
    demo()
