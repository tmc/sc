#!/usr/bin/env python3
"""
Validate induced statechart against real Go code sequences.
"""

import json
import sys
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional

from .token_collector import GoTokenCollector


def get_state(token_type: str, syntax_categories: Dict[str, Set[str]]) -> Optional[str]:
    """Map token type to syntax state."""
    for state, tokens in syntax_categories.items():
        if token_type in tokens:
            return state
    return None


def validate_statechart(run_dir: str):
    """Run validation on the induced statechart."""

    print('=' * 60)
    print('STATECHART VALIDATION AGAINST REAL GO CODE')
    print('=' * 60)

    # Load statechart
    with open(f'{run_dir}/induced_statechart.json') as f:
        statechart = json.load(f)

    print(f'\nLoaded statechart: {statechart["name"]}')
    print(f'  States: {len(statechart["states"])}')
    print(f'  Transitions: {len(statechart["transitions"])}')

    # Build transition lookup
    valid_transitions: Dict[str, Set[str]] = defaultdict(set)
    transition_probs: Dict[Tuple[str, str], float] = {}

    for t in statechart['transitions']:
        valid_transitions[t['source']].add(t['target'])
        transition_probs[(t['source'], t['target'])] = t['probability']

    # Build token-to-state mapping
    syntax_categories = {
        'PackageDecl': {'package', 'import'},
        'FuncDecl': {'func', 'return'},
        'TypeDecl': {'type', 'struct', 'interface'},
        'VarDecl': {'var', 'const'},
        'ControlFlow': {'if', 'else', 'for', 'switch', 'case', 'default',
                        'break', 'continue', 'select', 'go', 'defer',
                        'goto', 'fallthrough'},
        'Expression': {'IDENT', '.', '(', ')', '[', ']', '...'},
        'Block': {'{', '}'},
        'Statement': {';'},
        'Literal': {'STRING', 'INT', 'FLOAT', 'CHAR', 'IMAG'},
        'Operator': {'+', '-', '*', '/', '%', '=', '==', '!=', '<', '>',
                     '<=', '>=', '&&', '||', '!', '&', '|', '^', '<<', '>>',
                     '+=', '-=', '*=', '/=', ':=', '<-', ',', ':'},
    }

    # Load test data
    collector = GoTokenCollector()
    collector.load(f'{run_dir}/sequences.json')
    _, _, test_seqs = collector.split()

    print(f'\nValidation set: {len(test_seqs)} sequences')

    # Validation metrics
    total_transitions = 0
    valid_trans = 0
    invalid_trans = 0
    unknown_tokens = 0
    state_visits: Dict[str, int] = defaultdict(int)
    invalid_pairs: Dict[Tuple[str, str], int] = defaultdict(int)

    print('\n' + '=' * 60)
    print('RUNNING VALIDATION')
    print('=' * 60)

    for seq_idx, seq in enumerate(test_seqs):
        prev_state = None

        for token in seq.tokens[:200]:
            curr_state = get_state(token.type_name, syntax_categories)

            if curr_state is None:
                unknown_tokens += 1
                continue

            state_visits[curr_state] += 1

            if prev_state is not None and prev_state != curr_state:
                total_transitions += 1

                if curr_state in valid_transitions[prev_state]:
                    valid_trans += 1
                else:
                    invalid_trans += 1
                    invalid_pairs[(prev_state, curr_state)] += 1

            prev_state = curr_state

        if (seq_idx + 1) % 5 == 0:
            print(f'  Processed {seq_idx + 1}/{len(test_seqs)} sequences...')

    print('\n' + '=' * 60)
    print('VALIDATION RESULTS')
    print('=' * 60)

    accuracy = valid_trans / total_transitions if total_transitions > 0 else 0
    coverage = len([v for v in state_visits.values() if v > 0]) / len(syntax_categories)

    print(f'\nTransition Accuracy: {accuracy:.1%}')
    print(f'  Valid: {valid_trans:,}')
    print(f'  Invalid: {invalid_trans:,}')
    print(f'  Total: {total_transitions:,}')

    print(f'\nState Coverage: {coverage:.1%}')
    print(f'  States visited: {len([v for v in state_visits.values() if v > 0])}/{len(syntax_categories)}')

    print(f'\nUnknown tokens: {unknown_tokens:,}')

    print('\nState Visit Distribution:')
    total_visits = sum(state_visits.values())
    for state, count in sorted(state_visits.items(), key=lambda x: -x[1]):
        pct = count / total_visits * 100
        bar = '#' * int(pct / 2)
        print(f'  {state:15} {count:6,} ({pct:5.1f}%) {bar}')

    print('\nTop Invalid Transitions (missing from statechart):')
    for (src, dst), count in sorted(invalid_pairs.items(), key=lambda x: -x[1])[:10]:
        print(f'  {src:15} -> {dst:15}: {count:4} occurrences')

    # Save validation report
    report = {
        'accuracy': accuracy,
        'coverage': coverage,
        'total_transitions': total_transitions,
        'valid_transitions': valid_trans,
        'invalid_transitions': invalid_trans,
        'unknown_tokens': unknown_tokens,
        'state_visits': dict(state_visits),
        'top_invalid_pairs': [
            {'source': src, 'target': dst, 'count': count}
            for (src, dst), count in sorted(invalid_pairs.items(), key=lambda x: -x[1])[:20]
        ]
    }

    report_path = f'{run_dir}/validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nReport saved to: {report_path}')

    # Suggest improvements
    print('\n' + '=' * 60)
    print('SUGGESTED IMPROVEMENTS')
    print('=' * 60)

    if invalid_pairs:
        print('\nAdd these transitions to improve accuracy:')
        cumulative = 0
        for (src, dst), count in sorted(invalid_pairs.items(), key=lambda x: -x[1])[:5]:
            improvement = count / total_transitions * 100
            cumulative += improvement
            print(f'  {src} --> {dst}  (+{improvement:.1f}%, cumulative: {cumulative:.1f}%)')

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate induced statechart')
    parser.add_argument('--run-dir', type=str,
                        default='experiments/exp_grammar_induction/runs/run_20260104_154119',
                        help='Run directory')
    args = parser.parse_args()
    validate_statechart(args.run_dir)
