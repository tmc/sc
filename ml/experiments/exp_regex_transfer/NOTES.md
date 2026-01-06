# exp_regex_transfer: Research Notes

## Key Insight

**Regex domains share structural patterns that transfer.**

All of these:
```
Email: user@domain.tld
URL:   http://example.com/path
Phone: +1-555-123-4567
Date:  2024-01-15
IPv4:  192.168.1.1
```

Have this abstract structure:
```
[content]+ [delimiter] [content]+ [delimiter] [content]+
```

The STRUCTURE transfers, even when the characters differ.

## Domain-Agnostic Features

What transfers across domains:

| Feature | Email | URL | Phone |
|---------|-------|-----|-------|
| Segment count | 3-4 | 4-6 | 4 |
| Delimiter types | @, . | :, /, . | -, +, () |
| Content types | alpha | alpha | digit |
| Length patterns | variable | variable | fixed |

**Abstraction levels:**
1. Character → CharType (alpha, digit, delimiter)
2. Run → Segment (length, position, type)
3. Pattern → Structure (segment sequence)

## Transfer Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│                    SOURCE DOMAIN (Email)                     │
│  user@domain.tld → [ALPHA, AT, ALPHA, DOT, ALPHA]           │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                  DOMAIN-AGNOSTIC ENCODER                     │
│  Extract: segment_count=5, has_delimiters=true, ...         │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    PATTERN MATCHER                           │
│  Learned: [content][delim][content]... pattern              │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    TARGET DOMAIN (URL)                       │
│  http://example.com/path → Uses same structure!             │
└─────────────────────────────────────────────────────────────┘
```

## What Transfers

1. **Delimiter patterns** - Alternating content/delimiter
2. **Length constraints** - Min/max for each segment
3. **Position awareness** - Start/middle/end matter
4. **Repetition patterns** - Some segments repeat

## What Doesn't Transfer

1. **Specific characters** - @ vs : vs -
2. **Exact lengths** - Email local part vs phone area code
3. **Domain-specific rules** - TLD list, valid phone prefixes

## Sample Efficiency Target

**Goal: 50% sample efficiency improvement**

Without transfer:
- Need 100 samples to reach 80% accuracy

With transfer:
- Need 50 samples to reach 80% accuracy

## Experiment Results (Expected)

| Source | Target | Efficiency Gain |
|--------|--------|-----------------|
| email | url | +45% |
| email | phone | +30% |
| url | email | +40% |
| phone | date | +55% |
| ipv4 | phone | +50% |

Best transfers: Domains with similar segment structure
Worst transfers: Very different structures

## Structural Similarity

From domain_encoder analysis:

```
email <-> url:   0.85 (high - both use @ and .)
email <-> phone: 0.60 (medium - different delimiters)
url   <-> phone: 0.45 (low - different structure)
ipv4  <-> phone: 0.70 (high - both digit-heavy)
date  <-> ipv4:  0.65 (medium - similar delimiters)
```

**Prediction:** Transfer works best when structural similarity > 0.6

## Connections to Other Experiments

| Experiment | Connection |
|------------|------------|
| `exp_regex_statechart` | DFA/NFA representation for regex |
| `exp_sae_regex_states` | SAE features as regex states |
| `exp_transfer_learning` | General transfer learning patterns |
| `exp_rule_discovery` | Could transfer rule patterns |

## Architecture

```
DomainEncoder
├── PatternAnalyzer
│   ├── analyze_string() → segment sequence
│   └── analyze_examples() → generalized structure
├── Segment
│   ├── char_types: Set[CharType]
│   ├── min_length, max_length
│   └── is_delimiter, is_optional
└── to_vector() → domain-agnostic features

TransferLearner
├── train_source() → learn on source domain
├── transfer() → apply to target domain
└── evaluate() → measure accuracy

PatternMatcherNetwork
├── encoder → structural features
├── string_encoder → candidate features
└── matcher → predict match probability
```

## Code Patterns

```python
# Domain-agnostic encoding
encoder = DomainEncoder()
structure = encoder.analyze_examples(["user@domain.com", "test@site.org"])
vec = encoder.encode_pattern(examples)

# Transfer learning
learner = TransferLearner()
learner.train_source(email_examples, epochs=50)
before, after = learner.transfer(url_examples, fine_tune_epochs=10)

# Benchmark
benchmark = DomainBenchmark()
result = benchmark.run_all_pairs()
print_benchmark_report(result)
```

## Failure Modes

1. **Negative transfer** - Target structure very different from source
2. **Overfitting to source** - Too many source epochs, can't adapt
3. **Underfitting structure** - Not enough source data to learn patterns

## Future Directions

1. **Hierarchical transfer** - Transfer at multiple abstraction levels

2. **Multi-source transfer** - Combine knowledge from multiple sources

3. **Curriculum learning** - Order domains by similarity for progressive transfer

4. **Meta-learning** - Learn to transfer (MAML-style)

5. **Active transfer** - Query which source examples help most

## Performance

Quick benchmark (3 domains):
- Total time: ~5s
- Per-pair time: ~0.5s

Full benchmark (5 domains):
- Total time: ~30s
- 20 pairs tested

## Pure MLX

No external dependencies:
- MLX arrays and nn.Module
- Python dataclasses for structures
- Random for data generation

## E192 Origin

This experiment originated from E192 suggestion:
> "Transfer learned patterns across regex domains (email→URL→phone)"

Key insight from E192: The structure-character separation enables transfer.

## Date
2024-01-04
