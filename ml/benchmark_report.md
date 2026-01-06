# SC Benchmark Suite Report

Generated: 2026-01-04 19:18:48

## Executive Summary

- **Total Test Cases**: 55
- **Methods Compared**: 5
- **Best Overall Method**: BASELINE (100.0% validity)

### Test Distribution
- SIMPLE: 15 tests
- MEDIUM: 20 tests
- COMPLEX: 20 tests

## Method Comparison

| Method   | Valid | Total | Validity % | Avg States | Avg Trans |
|----------|-------|-------|------------|------------|-----------|
| BASELINE | 55    | 55    | 100.0%     | 4.1        | 3.9       |
| STEERING | 55    | 55    | 100.0%     | 4.1        | 3.9       |
| CIRCUITS | 55    | 55    | 100.0%     | 4.1        | 3.9       |
| TRM      | 55    | 55    | 100.0%     | 4.1        | 4.9       |
| HYBRID   | 55    | 55    | 100.0%     | 4.1        | 4.6       |

## Validity by Complexity

| Complexity | BASELINE | STEERING | CIRCUITS | TRM    | HYBRID |
|------------|----------|----------|----------|--------|--------|
| SIMPLE     | 100.0%   | 100.0%   | 100.0%   | 100.0% | 100.0% |
| MEDIUM     | 100.0%   | 100.0%   | 100.0%   | 100.0% | 100.0% |
| COMPLEX    | 100.0%   | 100.0%   | 100.0%   | 100.0% | 100.0% |

## Performance Metrics

| Method   | Avg Latency (ms) | Min  | Max  | Tokens/sec |
|----------|------------------|------|------|------------|
| BASELINE | 0.08             | 0.03 | 0.14 | 1064310    |
| STEERING | 1.39             | 1.14 | 1.90 | 64656      |
| CIRCUITS | 2.70             | 2.10 | 3.50 | 33304      |
| TRM      | 3.89             | 3.15 | 4.60 | 23140      |
| HYBRID   | 2.99             | 0.06 | 4.87 | 30049      |

## Efficiency Metrics

| Method   | Avg Tokens | Tokens/State | Tokens/Transition |
|----------|------------|--------------|-------------------|
| BASELINE | 90         | 22.5         | 23.6              |
| STEERING | 90         | 22.5         | 23.6              |
| CIRCUITS | 90         | 22.5         | 23.6              |
| TRM      | 90         | 22.5         | 18.4              |
| HYBRID   | 90         | 22.5         | 20.6              |

## Feature Support

| Method   | Parallel | History | Deep Nesting |
|----------|----------|---------|--------------|
| BASELINE | 7        | 0       | 1.2 levels   |
| STEERING | 7        | 0       | 1.2 levels   |
| CIRCUITS | 7        | 0       | 1.2 levels   |
| TRM      | 7        | 0       | 1.2 levels   |
| HYBRID   | 7        | 0       | 1.2 levels   |

## Detailed Results

### SIMPLE

**simple_01**: Create a simple on/off switch...
  - BASELINE: VALID (S:2 T:2)
  - STEERING: VALID (S:2 T:2)
  - CIRCUITS: VALID (S:2 T:2)
  - TRM: VALID (S:2 T:3)
  - HYBRID: VALID (S:2 T:2)

**simple_02**: Create a light bulb state machine with on and off ...
  - BASELINE: VALID (S:2 T:2)
  - STEERING: VALID (S:2 T:2)
  - CIRCUITS: VALID (S:2 T:2)
  - TRM: VALID (S:2 T:3)
  - HYBRID: VALID (S:2 T:2)

**simple_03**: Create a door that can be open or closed...
  - BASELINE: VALID (S:2 T:2)
  - STEERING: VALID (S:2 T:2)
  - CIRCUITS: VALID (S:2 T:2)
  - TRM: VALID (S:2 T:3)
  - HYBRID: VALID (S:2 T:2)

**simple_04**: Create a pause/play button state machine...
  - BASELINE: VALID (S:2 T:2)
  - STEERING: VALID (S:2 T:2)
  - CIRCUITS: VALID (S:2 T:2)
  - TRM: VALID (S:2 T:3)
  - HYBRID: VALID (S:2 T:2)

**simple_05**: Create a mute toggle for audio...
  - BASELINE: VALID (S:2 T:2)
  - STEERING: VALID (S:2 T:2)
  - CIRCUITS: VALID (S:2 T:2)
  - TRM: VALID (S:2 T:3)
  - HYBRID: VALID (S:2 T:2)

*...and 10 more tests*

### MEDIUM

**medium_01**: Create a media player with idle, playing, paused, ...
  - BASELINE: VALID (S:4 T:4)
  - STEERING: VALID (S:4 T:4)
  - CIRCUITS: VALID (S:4 T:4)
  - TRM: VALID (S:4 T:5)
  - HYBRID: VALID (S:4 T:5)

**medium_02**: Create a door with locked, unlocked, open, and clo...
  - BASELINE: VALID (S:4 T:4)
  - STEERING: VALID (S:4 T:4)
  - CIRCUITS: VALID (S:4 T:4)
  - TRM: VALID (S:4 T:5)
  - HYBRID: VALID (S:4 T:5)

**medium_03**: Create a phone call state machine: idle, dialing, ...
  - BASELINE: VALID (S:4 T:4)
  - STEERING: VALID (S:4 T:4)
  - CIRCUITS: VALID (S:4 T:4)
  - TRM: VALID (S:4 T:5)
  - HYBRID: VALID (S:4 T:5)

**medium_04**: Create an order status: pending, processing, shipp...
  - BASELINE: VALID (S:5 T:5)
  - STEERING: VALID (S:5 T:5)
  - CIRCUITS: VALID (S:5 T:5)
  - TRM: VALID (S:5 T:6)
  - HYBRID: VALID (S:5 T:6)

**medium_05**: Create a user authentication flow: logged_out, log...
  - BASELINE: VALID (S:4 T:4)
  - STEERING: VALID (S:4 T:4)
  - CIRCUITS: VALID (S:4 T:4)
  - TRM: VALID (S:4 T:5)
  - HYBRID: VALID (S:4 T:5)

*...and 15 more tests*

### COMPLEX

**complex_01**: Create a parallel state machine for a phone with c...
  - BASELINE: VALID (S:6 T:4)
  - STEERING: VALID (S:6 T:4)
  - CIRCUITS: VALID (S:6 T:4)
  - TRM: VALID (S:6 T:5)
  - HYBRID: VALID (S:6 T:5)

**complex_02**: Create a video player with history state that reme...
  - BASELINE: VALID (S:5 T:5)
  - STEERING: VALID (S:5 T:5)
  - CIRCUITS: VALID (S:5 T:5)
  - TRM: VALID (S:5 T:6)
  - HYBRID: VALID (S:5 T:6)

**complex_03**: Create a hierarchical ATM machine with authenticat...
  - BASELINE: VALID (S:4 T:4)
  - STEERING: VALID (S:4 T:4)
  - CIRCUITS: VALID (S:4 T:4)
  - TRM: VALID (S:4 T:5)
  - HYBRID: VALID (S:4 T:5)

**complex_04**: Create a smart home controller with parallel heati...
  - BASELINE: VALID (S:6 T:4)
  - STEERING: VALID (S:6 T:4)
  - CIRCUITS: VALID (S:6 T:4)
  - TRM: VALID (S:6 T:5)
  - HYBRID: VALID (S:6 T:5)

**complex_05**: Create a workflow engine with nested approval stat...
  - BASELINE: VALID (S:4 T:4)
  - STEERING: VALID (S:4 T:4)
  - CIRCUITS: VALID (S:4 T:4)
  - TRM: VALID (S:4 T:5)
  - HYBRID: VALID (S:4 T:5)

*...and 15 more tests*


## Recommendations

- **SIMPLE tasks**: Use BASELINE (100.0% validity)
- **MEDIUM tasks**: Use BASELINE (100.0% validity)
- **COMPLEX tasks**: Use BASELINE (100.0% validity)

### Overall Recommendations
- **Best Validity**: BASELINE
- **Best Speed**: BASELINE
- **Best Efficiency**: BASELINE