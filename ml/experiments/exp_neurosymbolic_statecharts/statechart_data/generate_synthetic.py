#!/usr/bin/env python3
"""Generate synthetic statecharts in sc proto JSON format.

Produces charts with controlled structural properties for training and
evaluation. Charts are validated with sc-trace-gen to ensure they are
well-formed and executable.

Usage:
    python generate_synthetic.py --tier medium --validate --seed 42
    python generate_synthetic.py --tier large --out /tmp/synthetic
    python generate_synthetic.py --family guard_probe --count 50
"""

import argparse
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# State types in sc proto
BASIC = 1
OR = 2
AND = 3


@dataclass
class State:
    label: str
    type: int = BASIC
    children: List["State"] = field(default_factory=list)
    is_initial: bool = False
    is_history: bool = False
    history_type: int = 0  # 1=shallow, 2=deep

    def to_dict(self) -> dict:
        d: dict = {"label": self.label, "type": self.type}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.is_initial:
            d["is_initial"] = True
        if self.is_history:
            d["is_history"] = True
            d["history_type"] = self.history_type
        return d


@dataclass
class Transition:
    frm: List[str]
    to: List[str]
    event: str
    guard: str = ""

    def to_dict(self) -> dict:
        d: dict = {"from": self.frm, "to": self.to, "event": self.event}
        if self.guard:
            d["guard"] = self.guard
        return d


@dataclass
class Chart:
    name: str
    root: State
    transitions: List[Transition]
    family: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root_state": self.root.to_dict(),
            "transitions": [t.to_dict() for t in self.transitions],
        }


def _labels(prefix: str, n: int) -> List[str]:
    """Generate n state labels with a prefix."""
    return [f"{prefix}{i}" for i in range(n)]


def _chain_transitions(states: List[str], prefix: str = "e") -> List[Transition]:
    """Create a chain: s0 -> s1 -> ... -> sN."""
    return [
        Transition([states[i]], [states[i + 1]], f"{prefix}_{states[i]}_{states[i+1]}")
        for i in range(len(states) - 1)
    ]


def _cycle_transitions(states: List[str], prefix: str = "e") -> List[Transition]:
    """Create a cycle: s0 -> s1 -> ... -> sN -> s0."""
    ts = _chain_transitions(states, prefix)
    ts.append(Transition([states[-1]], [states[0]], f"{prefix}_{states[-1]}_{states[0]}"))
    return ts


# ── Family generators ──────────────────────────────────────────────


def gen_flat_chain(rng: random.Random, n_states: int = 0) -> Chart:
    """Flat chain: all basic states under one OR root."""
    n = n_states or rng.randint(3, 8)
    labels = _labels("S", n)
    children = [State(l, BASIC, is_initial=(i == 0)) for i, l in enumerate(labels)]
    root = State("__root__", OR, children)
    return Chart(f"flat_chain_{n}", root, _chain_transitions(labels), "flat")


def gen_flat_cycle(rng: random.Random, n_states: int = 0) -> Chart:
    """Flat cycle: all basic states under one OR root, last→first."""
    n = n_states or rng.randint(3, 8)
    labels = _labels("S", n)
    children = [State(l, BASIC, is_initial=(i == 0)) for i, l in enumerate(labels)]
    root = State("__root__", OR, children)
    return Chart(f"flat_cycle_{n}", root, _cycle_transitions(labels), "flat")


def gen_flat_star(rng: random.Random, n_states: int = 0) -> Chart:
    """Flat star: hub state with spoke transitions to/from each leaf."""
    n = n_states or rng.randint(4, 8)
    labels = _labels("S", n)
    children = [State(l, BASIC, is_initial=(i == 0)) for i, l in enumerate(labels)]
    root = State("__root__", OR, children)
    hub = labels[0]
    ts = []
    for l in labels[1:]:
        ts.append(Transition([hub], [l], f"go_{l}"))
        ts.append(Transition([l], [hub], f"back_{l}"))
    return Chart(f"flat_star_{n}", root, ts, "flat")


def gen_shallow_nested(rng: random.Random, n_groups: int = 0) -> Chart:
    """Depth-2 hierarchy: OR root with OR children containing basic leaves."""
    n = n_groups or rng.randint(2, 4)
    group_sizes = [rng.randint(2, 4) for _ in range(n)]
    groups = []
    all_leaves = []
    for gi in range(n):
        leaves = [State(f"G{gi}_S{si}", BASIC, is_initial=(si == 0))
                  for si in range(group_sizes[gi])]
        groups.append(State(f"G{gi}", OR, leaves, is_initial=(gi == 0)))
        all_leaves.extend([(f"G{gi}_S{si}", gi) for si in range(group_sizes[gi])])

    root = State("__root__", OR, groups)
    # Intra-group chains + inter-group transitions
    ts = []
    for gi in range(n):
        gl = [f"G{gi}_S{si}" for si in range(group_sizes[gi])]
        ts.extend(_chain_transitions(gl, f"g{gi}"))
    # Link last leaf of each group to first of next
    for gi in range(n - 1):
        src = f"G{gi}_S{group_sizes[gi]-1}"
        dst = f"G{gi+1}_S0"
        ts.append(Transition([src], [dst], f"cross_{gi}_{gi+1}"))
    return Chart(f"shallow_{n}g", root, ts, "shallow_nested")


def gen_deep_nested(rng: random.Random, depth: int = 0) -> Chart:
    """Depth-3+ hierarchy: nested OR states."""
    d = depth or rng.randint(3, 5)
    # Build a single deep nesting chain with siblings at each level
    def _build(prefix: str, level: int) -> Tuple[State, List[str]]:
        if level >= d:
            n = rng.randint(2, 4)
            leaves = [State(f"{prefix}_L{i}", BASIC, is_initial=(i == 0))
                      for i in range(n)]
            return State(prefix, OR, leaves, is_initial=True), [l.label for l in leaves]
        # Create one nested child and 1-2 sibling leaves
        nested, nested_leaves = _build(f"{prefix}_D{level}", level + 1)
        siblings = [State(f"{prefix}_Sib{i}", BASIC) for i in range(rng.randint(1, 2))]
        nested.is_initial = True
        return State(prefix, OR, [nested] + siblings, is_initial=True), nested_leaves

    inner, leaves = _build("N", 1)
    root = State("__root__", OR, [inner])
    ts = _cycle_transitions(leaves)
    return Chart(f"deep_{d}d", root, ts, "deep_nested")


def gen_orthogonal(rng: random.Random, n_regions: int = 0) -> Chart:
    """AND (parallel) state with independent regions."""
    n = n_regions or rng.randint(2, 3)
    regions = []
    all_ts = []
    for ri in range(n):
        n_states = rng.randint(2, 4)
        leaves = [State(f"R{ri}_S{si}", BASIC, is_initial=(si == 0))
                  for si in range(n_states)]
        regions.append(State(f"Region{ri}", OR, leaves))
        labels = [l.label for l in leaves]
        all_ts.extend(_cycle_transitions(labels, f"r{ri}"))

    par = State("Parallel", AND, regions, is_initial=True)
    root = State("__root__", OR, [par])
    return Chart(f"ortho_{n}r", root, all_ts, "orthogonal")


def gen_history_shallow(rng: random.Random) -> Chart:
    """Chart with a shallow history pseudostate."""
    n_states = rng.randint(3, 5)
    labels = _labels("S", n_states)
    children = [State(l, BASIC, is_initial=(i == 0)) for i, l in enumerate(labels)]
    # Add a history pseudostate
    hist = State("H", BASIC, is_history=True, history_type=1)
    children.append(hist)

    group = State("Main", OR, children, is_initial=True)
    pause = State("Paused", BASIC)
    root = State("__root__", OR, [group, pause])

    ts = _cycle_transitions(labels)
    # Pause and resume via history
    ts.append(Transition([labels[-1]], ["Paused"], "pause"))
    ts.append(Transition(["Paused"], ["H"], "resume"))
    return Chart(f"hist_shallow_{n_states}", root, ts, "history")


def gen_history_deep(rng: random.Random) -> Chart:
    """Chart with a deep history pseudostate in a nested structure."""
    inner_labels = _labels("Inner", 3)
    inner_children = [State(l, BASIC, is_initial=(i == 0))
                      for i, l in enumerate(inner_labels)]
    inner = State("InnerGroup", OR, inner_children, is_initial=True)

    hist = State("H_deep", BASIC, is_history=True, history_type=2)
    outer_children = [inner, State("Sibling", BASIC), hist]
    outer = State("OuterGroup", OR, outer_children, is_initial=True)

    idle = State("Idle", BASIC)
    root = State("__root__", OR, [outer, idle])

    ts = _cycle_transitions(inner_labels)
    ts.append(Transition(["Sibling"], ["InnerGroup"], "enter_inner"))
    ts.append(Transition([inner_labels[-1]], ["Idle"], "suspend"))
    ts.append(Transition(["Idle"], ["H_deep"], "restore"))
    return Chart("hist_deep", root, ts, "history")


def gen_inter_level(rng: random.Random) -> Chart:
    """Chart with transitions that cross hierarchy levels."""
    inner_labels = _labels("Inner", rng.randint(2, 4))
    inner_children = [State(l, BASIC, is_initial=(i == 0))
                      for i, l in enumerate(inner_labels)]
    inner = State("Nested", OR, inner_children, is_initial=True)

    outer_sibling = State("OuterSib", BASIC)
    root = State("__root__", OR, [inner, outer_sibling])

    ts = _chain_transitions(inner_labels)
    # Inter-level: deep leaf → outer sibling, outer → deep leaf
    ts.append(Transition([inner_labels[-1]], ["OuterSib"], "escape"))
    ts.append(Transition(["OuterSib"], [inner_labels[0]], "dive"))
    return Chart(f"inter_{len(inner_labels)}", root, ts, "inter_level")


def gen_guard_probe(rng: random.Random) -> Chart:
    """Chart with branching paths for testing guard learning.

    S0 branches to S1 (via branch_high) or S2 (via branch_low).
    During training, the model must learn which branch to take based on
    context — simulating guard predicate behavior without proto guards.
    """
    n_branches = rng.randint(2, 4)
    hub = State("Hub", BASIC, is_initial=True)
    branches = [State(f"B{i}", BASIC) for i in range(n_branches)]
    root = State("__root__", OR, [hub] + branches)
    ts = []
    for i, b in enumerate(branches):
        ts.append(Transition(["Hub"], [b.label], f"branch_{i}"))
        ts.append(Transition([b.label], ["Hub"], f"return_{i}"))
    return Chart(f"guard_probe_{n_branches}b", root, ts, "guard_probe")


def gen_scale_ladder(rng: random.Random, n_states: int = 0) -> Chart:
    """Same flat cycle pattern at controlled sizes (4/8/16/32/64).

    Used to measure how model performance scales with state count.
    """
    n = n_states or rng.choice([4, 8, 16, 32, 64])
    labels = _labels("S", n)
    children = [State(l, BASIC, is_initial=(i == 0)) for i, l in enumerate(labels)]
    root = State("__root__", OR, children)
    return Chart(f"scale_{n}", root, _cycle_transitions(labels), "scale_ladder")


def gen_adversarial(rng: random.Random) -> Chart:
    """Near-isomorphic chart with symmetric decoy paths.

    Two parallel branches from hub with identical structure but different
    events. Tests whether the model can distinguish structurally similar
    but semantically different topologies.
    """
    hub = State("Hub", BASIC, is_initial=True)
    a_states = [State(f"A{i}", BASIC) for i in range(3)]
    b_states = [State(f"B{i}", BASIC) for i in range(3)]
    root = State("__root__", OR, [hub] + a_states + b_states)
    ts = [Transition(["Hub"], ["A0"], "go_a"), Transition(["Hub"], ["B0"], "go_b")]
    for i in range(2):
        ts.append(Transition([f"A{i}"], [f"A{i+1}"], f"step_a{i}"))
        ts.append(Transition([f"B{i}"], [f"B{i+1}"], f"step_b{i}"))
    ts.append(Transition(["A2"], ["Hub"], "return_a"))
    ts.append(Transition(["B2"], ["Hub"], "return_b"))
    return Chart("adversarial_symmetric", root, ts, "adversarial")


def gen_composition(rng: random.Random) -> Chart:
    """Composition: AND state with history inside one region.

    Tests interaction between parallel execution and history restore.
    """
    # Region 0: normal cycle
    r0_labels = _labels("R0_S", 3)
    r0_children = [State(l, BASIC, is_initial=(i == 0))
                   for i, l in enumerate(r0_labels)]
    r0 = State("Region0", OR, r0_children)

    # Region 1: has history
    r1_labels = _labels("R1_S", 3)
    r1_children = [State(l, BASIC, is_initial=(i == 0))
                   for i, l in enumerate(r1_labels)]
    hist = State("R1_H", BASIC, is_history=True, history_type=1)
    r1_children.append(hist)
    r1 = State("Region1", OR, r1_children)

    par = State("Active", AND, [r0, r1], is_initial=True)
    idle = State("Idle", BASIC)
    root = State("__root__", OR, [par, idle])

    ts = _cycle_transitions(r0_labels, "r0")
    ts.extend(_cycle_transitions(r1_labels, "r1"))
    ts.append(Transition([r0_labels[-1]], ["Idle"], "pause"))
    ts.append(Transition(["Idle"], ["R1_H"], "resume"))
    return Chart("composition_and_hist", root, ts, "composition")


def gen_noise_variant(rng: random.Random, base_gen=None) -> Chart:
    """Generate a clean chart then add noise: extra unreachable states or
    redundant transitions. Produces paired (clean, noisy) charts.
    """
    if base_gen is None:
        base_gen = gen_flat_cycle
    chart = base_gen(rng)
    # Add 1-3 unreachable states
    n_noise = rng.randint(1, 3)
    for i in range(n_noise):
        noise = State(f"Noise{i}", BASIC)
        chart.root.children.append(noise)
    chart.name = f"noisy_{chart.name}"
    chart.family = "noise_variant"
    return chart


# ── Generator registry ─────────────────────────────────────────────

GENERATORS = {
    "flat": [gen_flat_chain, gen_flat_cycle, gen_flat_star],
    "shallow_nested": [gen_shallow_nested],
    "deep_nested": [gen_deep_nested],
    "orthogonal": [gen_orthogonal],
    "history": [gen_history_shallow, gen_history_deep],
    "inter_level": [gen_inter_level],
    "guard_probe": [gen_guard_probe],
    "scale_ladder": [gen_scale_ladder],
    "adversarial": [gen_adversarial],
    "composition": [gen_composition],
    "noise_variant": [gen_noise_variant],
}

# ── Tier definitions ───────────────────────────────────────────────

TIER_COUNTS = {
    "small": {
        "flat": 4, "shallow_nested": 3, "deep_nested": 2,
        "orthogonal": 2, "history": 2,
    },
    "medium": {
        "flat": 80, "shallow_nested": 80, "deep_nested": 60,
        "orthogonal": 60, "history": 40, "inter_level": 40,
        "guard_probe": 40, "scale_ladder": 40, "adversarial": 30,
        "composition": 30, "noise_variant": 30,
    },
    "large": {
        "flat": 300, "shallow_nested": 300, "deep_nested": 200,
        "orthogonal": 200, "history": 150, "inter_level": 150,
        "guard_probe": 150, "scale_ladder": 150, "adversarial": 100,
        "composition": 100, "noise_variant": 100,
    },
}


# ── Validation ─────────────────────────────────────────────────────


def validate_chart(chart_path: str, sc_trace_gen: str = "sc-trace-gen") -> bool:
    """Run sc-trace-gen on a chart file to check validity."""
    try:
        result = subprocess.run(
            [sc_trace_gen, "-chart", chart_path, "-traces", "1",
             "-steps", "5", "-detail", "minimal"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ── Main generation loop ──────────────────────────────────────────


def generate_tier(
    tier: str,
    seed: int,
    out_dir: str,
    validate: bool = False,
    sc_trace_gen: str = "sc-trace-gen",
    family_filter: Optional[str] = None,
    count_override: Optional[int] = None,
) -> Dict[str, int]:
    """Generate charts for a tier, write to out_dir/{family}/*.scjson."""
    rng = random.Random(seed)
    counts = TIER_COUNTS.get(tier, {})
    if family_filter:
        counts = {family_filter: count_override or counts.get(family_filter, 50)}
    elif count_override:
        counts = {f: count_override for f in counts}

    stats: Dict[str, int] = {}
    manifest = []
    os.makedirs(out_dir, exist_ok=True)

    for family, target in counts.items():
        gens = GENERATORS.get(family)
        if not gens:
            print(f"  skip unknown family: {family}", file=sys.stderr)
            continue

        family_dir = os.path.join(out_dir, family)
        os.makedirs(family_dir, exist_ok=True)
        generated = 0
        attempts = 0
        max_attempts = target * 3

        while generated < target and attempts < max_attempts:
            attempts += 1
            gen = rng.choice(gens)
            chart = gen(rng)
            fname = f"{generated:04d}_{chart.name}.scjson"
            fpath = os.path.join(family_dir, fname)

            with open(fpath, "w") as f:
                json.dump(chart.to_dict(), f, indent=2)

            if validate:
                if not validate_chart(fpath, sc_trace_gen):
                    os.remove(fpath)
                    continue

            manifest.append({
                "path": fpath,
                "name": chart.name,
                "families": [chart.family],
                "generator": gen.__name__,
            })
            generated += 1

        stats[family] = generated
        print(f"  {family}: {generated}/{target}"
              + (f" ({attempts} attempts)" if validate else ""),
              file=sys.stderr)

    # Write manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "tier": tier,
            "seed": seed,
            "validated": validate,
            "counts": stats,
            "charts": manifest,
        }, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic statecharts for training/evaluation")
    parser.add_argument("--tier", choices=["small", "medium", "large"],
                        default="medium", help="Generation tier (default: medium)")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: ./synthetic_{tier})")
    parser.add_argument("--validate", action="store_true",
                        help="Validate each chart with sc-trace-gen")
    parser.add_argument("--sc-trace-gen", default="sc-trace-gen",
                        help="Path to sc-trace-gen binary")
    parser.add_argument("--family", type=str, default=None,
                        help="Generate only this family")
    parser.add_argument("--count", type=int, default=None,
                        help="Override count for family/tier")
    args = parser.parse_args()

    out_dir = args.out or f"synthetic_{args.tier}"
    print(f"Generating {args.tier} tier (seed={args.seed}) → {out_dir}",
          file=sys.stderr)

    stats = generate_tier(
        tier=args.tier,
        seed=args.seed,
        out_dir=out_dir,
        validate=args.validate,
        sc_trace_gen=args.sc_trace_gen,
        family_filter=args.family,
        count_override=args.count,
    )

    total = sum(stats.values())
    print(f"\nGenerated {total} charts across {len(stats)} families",
          file=sys.stderr)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
