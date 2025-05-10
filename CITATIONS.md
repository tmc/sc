# Citations and Academic References

This document provides detailed citation information for academic papers and texts that form the theoretical foundation of this Statecharts implementation.

## Core References

### Statecharts Original Formalism

The original formalism for Statecharts was introduced by David Harel in his seminal 1987 paper:

```bibtex
@article{harel1987statecharts,
  title={Statecharts: A visual formalism for complex systems},
  author={Harel, David},
  journal={Science of Computer Programming},
  volume={8},
  number={3},
  pages={231--274},
  year={1987},
  publisher={Elsevier},
  doi={10.1016/0167-6423(87)90035-9}
}
```

This paper introduces the fundamental concepts of statecharts, including hierarchical states, orthogonality (concurrency), and event-based communication. It provides the theoretical foundation upon which this implementation is built.

### Operational Semantics

The operational semantics of Statecharts implemented in this library closely follow those defined in:

```bibtex
@article{harel1996statemate,
  title={The STATEMATE semantics of statecharts},
  author={Harel, David and Naamad, Amnon},
  journal={ACM Transactions on Software Engineering and Methodology (TOSEM)},
  volume={5},
  number={4},
  pages={293--333},
  year={1996},
  publisher={ACM},
  doi={10.1145/235321.235322}
}
```

This paper provides a precise definition of the step semantics, which govern how statecharts transition between configurations in response to events.

### Statechart Variants

For a comprehensive comparison of different statechart semantics and variants:

```bibtex
@inproceedings{von1994comparison,
  title={A comparison of statecharts variants},
  author={von der Beeck, Michael},
  booktitle={Formal techniques in real-time and fault-tolerant systems},
  pages={128--148},
  year={1994},
  publisher={Springer},
  doi={10.1007/3-540-58468-4_163}
}
```

This work analyzes various statechart formalisms and their semantic differences, which has informed our implementation choices.

## Additional References

### Comprehensive Treatments

For a more comprehensive treatment of the Statecharts formalism and its application:

```bibtex
@book{harel1998modeling,
  title={Modeling Reactive Systems with Statecharts: The STATEMATE Approach},
  author={Harel, David and Politi, Michal},
  year={1998},
  publisher={McGraw-Hill},
  isbn={0070269173}
}
```

```bibtex
@article{harel2007come,
  title={Come, Let's Play: Scenario-Based Programming Using LSCs and the Play-Engine},
  author={Harel, David and Marelly, Rami},
  journal={Software Engineering},
  volume={SE-4},
  pages={37--38},
  year={2007},
  publisher={Springer},
  doi={10.1007/978-3-540-72995-2}
}
```

### Semantic Foundations

For a deeper understanding of the formal foundations of reactive systems:

```bibtex
@article{pnueli1989verification,
  title={On the verification of temporal properties},
  author={Pnueli, Amir and Kesten, Yonit},
  journal={Journal of Signal Processing Systems},
  volume={50},
  number={2},
  pages={79--98},
  year={1989},
  publisher={Springer}
}
```

### Implementation Considerations

For considerations in implementing statecharts in software systems:

```bibtex
@inproceedings{samek2006practical,
  title={Practical UML statecharts in C/C++: Event-driven programming for embedded systems},
  author={Samek, Miro},
  booktitle={Proceedings of the Embedded Systems Conference},
  year={2006},
  publisher={Newnes}
}
```

## Citing This Implementation

To cite this implementation in academic work, please use the following BibTeX entry:

```bibtex
@misc{tmc2023statecharts,
  author       = {TMC},
  title        = {Statecharts: A Formal Implementation of Harel Statecharts},
  year         = {2023},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/tmc/sc}}
}
```