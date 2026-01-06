#!/usr/bin/env python3
"""
React State Extraction Benchmark

Tests extraction accuracy on 10 React/XState components.
Measures: states found, transitions found, events found, accuracy.

Target: 80%+ extraction accuracy
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from pathlib import Path

from .react_parser import ReactParser
from .state_extractor import StateExtractor
from .sc_builder import SCBuilder


@dataclass
class ExpectedStatechart:
    """Expected statechart for validation."""
    name: str
    states: Set[str]
    events: Set[str]
    transitions: List[tuple]  # (from, event, to)


@dataclass
class ExtractionResult:
    """Result of extracting a single component."""
    name: str
    states_found: int
    states_expected: int
    events_found: int
    events_expected: int
    transitions_found: int
    transitions_expected: int
    accuracy: float
    details: Dict = field(default_factory=dict)


# Test components with expected outputs
TEST_COMPONENTS = [
    # 1. Simple toggle
    {
        "code": '''
        function Toggle() {
            const [isOn, setIsOn] = useState(false);

            const handleToggle = () => {
                setIsOn(!isOn);
            };

            return <button onClick={handleToggle}>{isOn ? 'ON' : 'OFF'}</button>;
        }
        ''',
        "expected": ExpectedStatechart(
            name="Toggle",
            states={"isOn_true", "isOn_false"},
            events={"TOGGLE"},
            transitions=[
                ("isOn_false", "TOGGLE", "isOn_true"),
                ("isOn_true", "TOGGLE", "isOn_false")
            ]
        )
    },

    # 2. Modal component
    {
        "code": '''
        function Modal() {
            const [isOpen, setIsOpen] = useState(false);

            const handleOpen = () => {
                setIsOpen(true);
            };

            const handleClose = () => {
                setIsOpen(false);
            };

            return (
                <div>
                    <button onClick={handleOpen}>Open</button>
                    {isOpen && <div onClick={handleClose}>Modal Content</div>}
                </div>
            );
        }
        ''',
        "expected": ExpectedStatechart(
            name="Modal",
            states={"isOpen_true", "isOpen_false"},
            events={"OPEN", "CLOSE"},
            transitions=[
                ("isOpen_false", "OPEN", "isOpen_true"),
                ("isOpen_true", "CLOSE", "isOpen_false")
            ]
        )
    },

    # 3. Counter with reducer
    {
        "code": '''
        function counterReducer(state, action) {
            switch (action.type) {
                case 'INCREMENT':
                    return { count: state.count + 1 };
                case 'DECREMENT':
                    return { count: state.count - 1 };
                case 'RESET':
                    return { count: 0 };
                default:
                    return state;
            }
        }

        function Counter() {
            const [state, dispatch] = useReducer(counterReducer, { count: 0 });

            const handleIncrement = () => {
                dispatch({ type: 'INCREMENT' });
            };

            const handleDecrement = () => {
                dispatch({ type: 'DECREMENT' });
            };

            const handleReset = () => {
                dispatch({ type: 'RESET' });
            };

            return <div>{state.count}</div>;
        }
        ''',
        "expected": ExpectedStatechart(
            name="Counter",
            states={"idle", "after_increment", "after_decrement", "after_reset"},
            events={"INCREMENT", "DECREMENT", "RESET"},
            transitions=[
                ("idle", "INCREMENT", "after_increment"),
                ("idle", "DECREMENT", "after_decrement"),
                ("idle", "RESET", "after_reset")
            ]
        )
    },

    # 4. Loading state
    {
        "code": '''
        function DataFetcher() {
            const [isLoading, setIsLoading] = useState(false);

            const handleFetch = () => {
                setIsLoading(true);
            };

            const handleComplete = () => {
                setIsLoading(false);
            };

            return <div>{isLoading ? 'Loading...' : 'Ready'}</div>;
        }
        ''',
        "expected": ExpectedStatechart(
            name="DataFetcher",
            states={"isLoading_true", "isLoading_false"},
            events={"FETCH", "COMPLETE"},
            transitions=[
                ("isLoading_false", "FETCH", "isLoading_true"),
                ("isLoading_true", "COMPLETE", "isLoading_false")
            ]
        )
    },

    # 5. Form validation
    {
        "code": '''
        function Form() {
            const [isValid, setIsValid] = useState(false);
            const [isSubmitting, setIsSubmitting] = useState(false);

            const handleValidate = () => {
                setIsValid(true);
            };

            const handleSubmit = () => {
                setIsSubmitting(true);
            };

            const handleReset = () => {
                setIsValid(false);
                setIsSubmitting(false);
            };

            return <form />;
        }
        ''',
        "expected": ExpectedStatechart(
            name="Form",
            states={"isValid_true", "isValid_false", "isSubmitting_true", "isSubmitting_false"},
            events={"VALIDATE", "SUBMIT", "RESET"},
            transitions=[
                ("isValid_false", "VALIDATE", "isValid_true"),
                ("isSubmitting_false", "SUBMIT", "isSubmitting_true")
            ]
        )
    },

    # 6. Accordion
    {
        "code": '''
        function Accordion() {
            const [isExpanded, setIsExpanded] = useState(false);

            const handleToggle = () => {
                setIsExpanded(!isExpanded);
            };

            return (
                <div>
                    <button onClick={handleToggle}>Toggle</button>
                    {isExpanded && <div>Content</div>}
                </div>
            );
        }
        ''',
        "expected": ExpectedStatechart(
            name="Accordion",
            states={"isExpanded_true", "isExpanded_false"},
            events={"TOGGLE"},
            transitions=[
                ("isExpanded_false", "TOGGLE", "isExpanded_true"),
                ("isExpanded_true", "TOGGLE", "isExpanded_false")
            ]
        )
    },

    # 7. Tabs
    {
        "code": '''
        function tabsReducer(state, action) {
            switch (action.type) {
                case 'SELECT_TAB':
                    return { ...state, activeTab: action.payload };
                case 'NEXT_TAB':
                    return { ...state, activeTab: state.activeTab + 1 };
                case 'PREV_TAB':
                    return { ...state, activeTab: state.activeTab - 1 };
                default:
                    return state;
            }
        }

        function Tabs() {
            const [state, dispatch] = useReducer(tabsReducer, { activeTab: 0 });

            const handleSelectTab = (index) => {
                dispatch({ type: 'SELECT_TAB', payload: index });
            };

            const handleNextTab = () => {
                dispatch({ type: 'NEXT_TAB' });
            };

            return <div />;
        }
        ''',
        "expected": ExpectedStatechart(
            name="Tabs",
            states={"idle", "after_select_tab", "after_next_tab", "after_prev_tab"},
            events={"SELECT_TAB", "NEXT_TAB", "PREV_TAB"},
            transitions=[
                ("idle", "SELECT_TAB", "after_select_tab"),
                ("idle", "NEXT_TAB", "after_next_tab")
            ]
        )
    },

    # 8. Dropdown
    {
        "code": '''
        function Dropdown() {
            const [isOpen, setIsOpen] = useState(false);

            const handleOpen = () => {
                setIsOpen(true);
            };

            const handleClose = () => {
                setIsOpen(false);
            };

            const handleSelect = () => {
                setIsOpen(false);
            };

            return (
                <div onMouseEnter={handleOpen} onMouseLeave={handleClose}>
                    <ul>{isOpen && <li onClick={handleSelect}>Option</li>}</ul>
                </div>
            );
        }
        ''',
        "expected": ExpectedStatechart(
            name="Dropdown",
            states={"isOpen_true", "isOpen_false"},
            events={"OPEN", "CLOSE", "SELECT"},
            transitions=[
                ("isOpen_false", "OPEN", "isOpen_true"),
                ("isOpen_true", "CLOSE", "isOpen_false"),
                ("isOpen_true", "SELECT", "isOpen_false")
            ]
        )
    },

    # 9. Notification
    {
        "code": '''
        function Notification() {
            const [isVisible, setIsVisible] = useState(false);

            const handleShow = () => {
                setIsVisible(true);
            };

            const handleDismiss = () => {
                setIsVisible(false);
            };

            return (
                <div>
                    <button onClick={handleShow}>Show</button>
                    {isVisible && (
                        <div>
                            Notification
                            <button onClick={handleDismiss}>X</button>
                        </div>
                    )}
                </div>
            );
        }
        ''',
        "expected": ExpectedStatechart(
            name="Notification",
            states={"isVisible_true", "isVisible_false"},
            events={"SHOW", "DISMISS"},
            transitions=[
                ("isVisible_false", "SHOW", "isVisible_true"),
                ("isVisible_true", "DISMISS", "isVisible_false")
            ]
        )
    },

    # 10. Checkbox
    {
        "code": '''
        function Checkbox() {
            const [isChecked, setIsChecked] = useState(false);

            const handleChange = () => {
                setIsChecked(!isChecked);
            };

            return (
                <input type="checkbox" checked={isChecked} onChange={handleChange} />
            );
        }
        ''',
        "expected": ExpectedStatechart(
            name="Checkbox",
            states={"isChecked_true", "isChecked_false"},
            events={"CHANGE"},
            transitions=[
                ("isChecked_false", "CHANGE", "isChecked_true"),
                ("isChecked_true", "CHANGE", "isChecked_false")
            ]
        )
    },
]


def calculate_accuracy(found: Set, expected: Set) -> float:
    """Calculate accuracy as recall (how much of expected was found)."""
    if not expected:
        return 1.0 if not found else 0.5  # Finding extra is okay
    # Recall: what fraction of expected items did we find?
    intersection = len(found & expected)
    recall = intersection / len(expected)
    # Slight penalty for finding too much extra (but not harsh)
    extra = len(found - expected)
    if extra > 0:
        recall = recall * (len(expected) / (len(expected) + extra * 0.5))
    return min(recall, 1.0)


def extract_and_compare(code: str, expected: ExpectedStatechart) -> ExtractionResult:
    """Extract statechart and compare to expected."""
    parser = ReactParser()
    extractor = StateExtractor()

    # Parse and extract
    components = parser.parse(code)
    if not components:
        return ExtractionResult(
            name=expected.name,
            states_found=0,
            states_expected=len(expected.states),
            events_found=0,
            events_expected=len(expected.events),
            transitions_found=0,
            transitions_expected=len(expected.transitions),
            accuracy=0.0,
            details={"error": "No components parsed"}
        )

    machine = extractor.extract(components[0])

    # Compare states
    found_states = set(machine.states)
    states_accuracy = calculate_accuracy(found_states, expected.states)

    # Compare events
    found_events = set(machine.events)
    events_accuracy = calculate_accuracy(found_events, expected.events)

    # Compare transitions (simplified: just count)
    found_trans = len(machine.transitions)
    expected_trans = len(expected.transitions)
    trans_accuracy = min(found_trans, expected_trans) / max(found_trans, expected_trans, 1)

    # Overall accuracy
    overall = (states_accuracy + events_accuracy + trans_accuracy) / 3

    return ExtractionResult(
        name=machine.name,
        states_found=len(found_states),
        states_expected=len(expected.states),
        events_found=len(found_events),
        events_expected=len(expected.events),
        transitions_found=found_trans,
        transitions_expected=expected_trans,
        accuracy=overall,
        details={
            "states_accuracy": states_accuracy,
            "events_accuracy": events_accuracy,
            "trans_accuracy": trans_accuracy,
            "found_states": list(found_states),
            "found_events": list(found_events)
        }
    )


def run_benchmark() -> Dict:
    """Run benchmark on all test components."""
    results = []

    print("=" * 70)
    print("REACT STATE EXTRACTION BENCHMARK")
    print("=" * 70)
    print(f"Testing {len(TEST_COMPONENTS)} components\n")

    for i, test in enumerate(TEST_COMPONENTS, 1):
        result = extract_and_compare(test["code"], test["expected"])
        results.append(result)

        status = "✓" if result.accuracy >= 0.8 else "✗"
        print(f"{i}. {result.name}: {status} {result.accuracy:.0%}")
        print(f"   States: {result.states_found}/{result.states_expected}")
        print(f"   Events: {result.events_found}/{result.events_expected}")
        print(f"   Transitions: {result.transitions_found}/{result.transitions_expected}")

    # Summary
    total_accuracy = sum(r.accuracy for r in results) / len(results)
    passed = sum(1 for r in results if r.accuracy >= 0.8)
    total_states = sum(r.states_found for r in results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy: {total_accuracy:.0%}")
    print(f"Components Passed (≥80%): {passed}/{len(results)}")
    print(f"Total States Found: {total_states}")

    report = f"REACT_EXTRACT: accuracy={total_accuracy:.0%}, components={len(results)}, states_found={total_states}"
    print(f"\nReport: {report}")

    return {
        "accuracy": total_accuracy,
        "components": len(results),
        "states_found": total_states,
        "passed": passed,
        "results": [
            {
                "name": r.name,
                "accuracy": r.accuracy,
                "states": r.states_found,
                "events": r.events_found,
                "transitions": r.transitions_found
            }
            for r in results
        ],
        "report": report
    }


def demo_extraction(component_index: int = 0) -> None:
    """Demo extraction on a single component."""
    if component_index >= len(TEST_COMPONENTS):
        print(f"Invalid index. Max: {len(TEST_COMPONENTS) - 1}")
        return

    test = TEST_COMPONENTS[component_index]

    parser = ReactParser()
    extractor = StateExtractor()
    builder = SCBuilder()

    # Pipeline
    components = parser.parse(test["code"])
    if not components:
        print("Failed to parse component")
        return

    machine = extractor.extract(components[0])
    statechart = builder.build(machine)

    print("=" * 60)
    print(f"EXTRACTION DEMO: {machine.name}")
    print("=" * 60)

    print("\n--- Input Code ---")
    print(test["code"].strip()[:200] + "...")

    print("\n--- Extracted Machine ---")
    print(f"States: {machine.states}")
    print(f"Events: {machine.events}")
    print(f"Transitions: {len(machine.transitions)}")
    for t in machine.transitions:
        print(f"  {t.from_state} --{t.event}--> {t.to_state}")

    print("\n--- SC Proto JSON ---")
    print(builder.to_json(statechart))

    print("\n--- Mermaid Diagram ---")
    print(builder.to_mermaid(statechart))


def main():
    """Run the benchmark."""
    summary = run_benchmark()
    return summary


if __name__ == "__main__":
    main()
