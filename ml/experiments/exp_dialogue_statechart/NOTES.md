# exp_dialogue_statechart: Research Notes

## Milestone: First Non-Game Domain!

This experiment applies statechart learning to **dialogue systems** - the first
non-game domain in the ML experiments. It demonstrates that the SAE + statechart
approach generalizes beyond games.

---

## Core Insight

**Dialogue follows predictable state patterns that can be learned from data.**

| Phase | Dialogue Acts | Example |
|-------|---------------|---------|
| OPENING | GREETING | "Hello, how can I help?" |
| INFO_GATHERING | REQUEST, INFORM, QUESTION | "I need a restaurant" / "What cuisine?" |
| CONFIRMATION | CONFIRM, ACCEPT | "So Italian for 2 at 7pm?" / "Yes" |
| RESOLUTION | OFFER, ACCEPT | "I found 3 options" / "Book first one" |
| CLOSING | FAREWELL, THANKING | "Goodbye!" / "Thank you!" |

---

## Architecture

### 1. Utterance Encoder
- Bag-of-words text embedding (demo)
- Dialogue act embedding
- Speaker embedding (user vs system)
- Combined via MLP

### 2. SAE State Bottleneck
- TopK SAE on utterance encodings
- k=6 active features per utterance
- Feature patterns = dialogue states
- Feature transitions = dialogue flow

### 3. Transition Graph Builder
- Track state-to-state transitions
- Associate transitions with dialogue acts
- Export to SC proto JSON format

---

## What Worked

1. **Phase-aware data generation** - Generating dialogues with explicit phase labels
   helps validate that SAE discovers the expected structure.

2. **Dialogue act embeddings** - Adding explicit act embeddings speeds convergence
   vs. learning purely from text.

3. **Feature-Act correlation tracking** - Mapping SAE features to dialogue acts
   provides interpretability.

4. **Transition filtering** - Filtering rare transitions (count < 2) removes noise.

## What Didn't Work (Initially)

1. **Pure text encoding** - Without dialogue act hints, SAE features were noisy.
   Solution: Include act embeddings.

2. **Too many SAE features** - k=16 created too many unique states.
   Solution: k=6 for cleaner clustering.

3. **No phase labels** - Hard to validate discovered structure.
   Solution: Track ground truth phases for comparison.

---

## Dialogue Acts (ISO 24617-2 inspired)

### Social
- GREETING, FAREWELL, THANKING

### Task
- REQUEST: User requests service
- INFORM: Provide information
- CONFIRM: Confirm understanding
- OFFER: System offers options

### Response
- ACCEPT, REJECT, ACKNOWLEDGE

### Exchange
- QUESTION, ANSWER

### Meta
- REPEAT, CLARIFY, CANCEL

---

## MultiWOZ-Style Data

### Domains
- Restaurant booking (implemented)
- Hotel reservation (slots defined)
- Train tickets (future)
- Taxi booking (future)

### Slots (Restaurant)
| Slot | Values |
|------|--------|
| cuisine | italian, chinese, indian, mexican, thai |
| price_range | cheap, moderate, expensive |
| location | north, south, east, west, center |
| party_size | 1, 2, 3, 4, 5, 6+ |
| date | today, tomorrow, monday, tuesday, weekend |
| time | lunch, dinner, 7pm, 8pm, 9pm |

---

## Connections to Other Experiments

### exp_sae_statechart
- **This**: SAE for dialogue features
- **That**: SAE for game state features
- **Shared**: TopKSAE, feature clustering, state discovery

### exp_learnable_policies
- **This**: Dialogue policy as statechart
- **That**: Game policy as statechart
- **Shared**: Policy structure discovery

### exp_guard_synthesis
- **This**: Dialogue act guards ("if user said X")
- **That**: Game state guards
- **Shared**: Guard expression evolution

### exp_execution_replay
- **This**: Learn from dialogue transcripts
- **That**: Learn from game traces
- **Shared**: Trace-based learning

### exp_temporal_guards
- **This**: Timeout guards ("if user silent for 30s")
- **That**: Game temporal guards
- **Shared**: Temporal predicate synthesis

---

## Future Directions

### 1. Real MultiWOZ Data
- Parse actual MultiWOZ dataset
- Compare discovered vs annotated states
- Measure state coverage

### 2. Pretrained Encoders
- Use BERT/RoBERTa for utterance encoding
- Compare to bag-of-words baseline
- Measure feature quality

### 3. Multi-Domain Learning
- Learn across restaurant + hotel + train
- Discover shared dialogue patterns
- Transfer learning between domains

### 4. Dialogue Generation
- Use learned statechart to generate dialogues
- Constrain LLM outputs to valid states
- Ensure task completion

### 5. Error Recovery
- Learn error states (misunderstanding, rejection)
- Discover recovery transitions
- Build robust dialogue managers

### 6. Real-Time Adaptation
- Online state discovery during conversation
- Adapt to user preferences
- Personalized dialogue flow

---

## Applications

### Customer Service Bots
- Discover call center dialogue patterns
- Identify stuck states (escalation triggers)
- Optimize flow for resolution time

### Healthcare Intake
- Learn patient intake dialogue structure
- Ensure all required information collected
- Flag concerning responses

### Educational Tutoring
- Discover effective teaching dialogue patterns
- Identify confusion states
- Adapt to student knowledge level

### Voice Assistants
- Learn natural conversation flow
- Handle multi-turn interactions
- Graceful error handling

---

## Files

```
exp_dialogue_statechart/
├── __init__.py             # Package exports
├── dialogue_statechart.py  # Main implementation
└── NOTES.md                # This file
```

---

## Metrics Summary

| Metric | Value |
|--------|-------|
| Dialogues | 100 |
| Avg turns/dialogue | ~12 |
| States discovered | ~20-30 |
| Transitions | ~40-60 |
| Feature-Act mappings | ~15 dialogue acts |

---

## References

- MultiWOZ: A Large-Scale Multi-Domain Wizard-of-Oz Dataset (Budzianowski et al., 2018)
- ISO 24617-2: Dialogue act annotation
- Anthropic SAE papers (2023-2025)
- exp_sae_statechart patterns

---

Created: 2026-01-04
Milestone: First non-game domain!
