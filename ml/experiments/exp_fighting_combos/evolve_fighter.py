"""
Evolve Fighting Game Strategy from Self-Play.

The genome encodes a statechart-like decision table:
- State: (spacing, frame_advantage, health_diff, meter)
- Action: Which move to do

Evolution should discover:
1. Punish patterns (jab when opponent is -8)
2. Combo routes (jab -> jab -> heavy when +4)
3. Spacing control (back dash when close and unsafe)
4. Meter management (save for wake-up special)
"""

import random
import math
import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fighting_game import (
    FightingGame, MoveType, MOVES, PlayerState,
    get_game_state, MAX_HEALTH, MAX_METER
)


def randn():
    """Box-Muller transform for normal random."""
    u1 = random.random() + 1e-10
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


# Available combat actions
ACTIONS = [
    MoveType.JAB,
    MoveType.HEAVY,
    MoveType.SWEEP,
    MoveType.OVERHEAD,
    MoveType.GRAB,
    MoveType.BLOCK,
    MoveType.BACK_DASH,
    MoveType.FORWARD_DASH,
    MoveType.SPECIAL,
    MoveType.NONE,  # Do nothing (wait)
]

NUM_ACTIONS = len(ACTIONS)


class FighterGenome:
    """
    Genome encoding fighting strategy.

    Encodes weights for:
    - Spacing preferences (close/mid/far)
    - Frame advantage thresholds (when to punish)
    - Health-based aggression
    - Meter usage patterns
    """

    def __init__(self):
        # Spacing mode action preferences (3 modes x N actions)
        self.spacing_weights = {
            "close": [randn() * 0.5 for _ in range(NUM_ACTIONS)],
            "mid": [randn() * 0.5 for _ in range(NUM_ACTIONS)],
            "far": [randn() * 0.5 for _ in range(NUM_ACTIONS)],
        }

        # Frame advantage modifiers
        # [can_punish, neutral, disadvantage]
        self.advantage_weights = {
            "can_punish": [randn() * 0.3 for _ in range(NUM_ACTIONS)],
            "neutral": [0.0] * NUM_ACTIONS,
            "disadvantage": [randn() * 0.3 for _ in range(NUM_ACTIONS)],
        }

        # Health-based modifiers (low_health, even, high_health)
        self.health_weights = {
            "low": [randn() * 0.2 for _ in range(NUM_ACTIONS)],
            "even": [0.0] * NUM_ACTIONS,
            "high": [randn() * 0.2 for _ in range(NUM_ACTIONS)],
        }

        # Meter thresholds
        self.meter_use_threshold = 50 + randn() * 20  # When to use special
        self.meter_save_threshold = 25 + randn() * 10  # Min meter to keep

        # Opponent state reaction
        self.punish_bonus = 2.0 + randn() * 0.5  # Extra weight for punishing
        self.pressure_bonus = 1.0 + randn() * 0.3  # Extra weight for pressure

        self.fitness = 0.0

    def copy(self):
        g = FighterGenome.__new__(FighterGenome)
        g.spacing_weights = {k: v.copy() for k, v in self.spacing_weights.items()}
        g.advantage_weights = {k: v.copy() for k, v in self.advantage_weights.items()}
        g.health_weights = {k: v.copy() for k, v in self.health_weights.items()}
        g.meter_use_threshold = self.meter_use_threshold
        g.meter_save_threshold = self.meter_save_threshold
        g.punish_bonus = self.punish_bonus
        g.pressure_bonus = self.pressure_bonus
        g.fitness = 0.0
        return g

    def mutate(self, rate=0.15, strength=0.3):
        g = self.copy()

        for mode in ["close", "mid", "far"]:
            for i in range(NUM_ACTIONS):
                if random.random() < rate:
                    g.spacing_weights[mode][i] += randn() * strength

        for adv in ["can_punish", "neutral", "disadvantage"]:
            for i in range(NUM_ACTIONS):
                if random.random() < rate:
                    g.advantage_weights[adv][i] += randn() * strength * 0.5

        for hp in ["low", "high"]:
            for i in range(NUM_ACTIONS):
                if random.random() < rate:
                    g.health_weights[hp][i] += randn() * strength * 0.3

        if random.random() < rate:
            g.meter_use_threshold = max(20, min(80, g.meter_use_threshold + randn() * 10))
        if random.random() < rate:
            g.meter_save_threshold = max(0, min(50, g.meter_save_threshold + randn() * 5))
        if random.random() < rate:
            g.punish_bonus = max(0.5, g.punish_bonus + randn() * 0.2)
        if random.random() < rate:
            g.pressure_bonus = max(0.5, g.pressure_bonus + randn() * 0.1)

        return g

    def crossover(self, other):
        g = FighterGenome.__new__(FighterGenome)

        g.spacing_weights = {}
        for mode in ["close", "mid", "far"]:
            g.spacing_weights[mode] = [
                a if random.random() > 0.5 else b
                for a, b in zip(self.spacing_weights[mode], other.spacing_weights[mode])
            ]

        g.advantage_weights = {}
        for adv in ["can_punish", "neutral", "disadvantage"]:
            g.advantage_weights[adv] = [
                a if random.random() > 0.5 else b
                for a, b in zip(self.advantage_weights[adv], other.advantage_weights[adv])
            ]

        g.health_weights = {}
        for hp in ["low", "even", "high"]:
            g.health_weights[hp] = [
                a if random.random() > 0.5 else b
                for a, b in zip(self.health_weights[hp], other.health_weights[hp])
            ]

        alpha = random.random()
        g.meter_use_threshold = alpha * self.meter_use_threshold + (1-alpha) * other.meter_use_threshold
        g.meter_save_threshold = alpha * self.meter_save_threshold + (1-alpha) * other.meter_save_threshold
        g.punish_bonus = self.punish_bonus if random.random() > 0.5 else other.punish_bonus
        g.pressure_bonus = self.pressure_bonus if random.random() > 0.5 else other.pressure_bonus

        g.fitness = 0.0
        return g


class FighterPlayer:
    """Player controlled by a genome."""

    def __init__(self, genome: FighterGenome):
        self.g = genome

    def get_action(self, state: dict) -> MoveType:
        if not state["i_can_act"]:
            return MoveType.NONE

        # Compute action weights
        weights = [0.0] * NUM_ACTIONS

        # 1. Spacing weights
        spacing = state["spacing"]
        for i in range(NUM_ACTIONS):
            weights[i] += self.g.spacing_weights[spacing][i]

        # 2. Frame advantage
        fa = state["frame_advantage"]
        if fa >= 6:  # Can punish
            adv_mode = "can_punish"
        elif fa <= -4:  # Disadvantage
            adv_mode = "disadvantage"
        else:
            adv_mode = "neutral"

        for i in range(NUM_ACTIONS):
            weights[i] += self.g.advantage_weights[adv_mode][i]

        # 3. Health differential
        health_diff = state["my_health"] - state["opp_health"]
        if health_diff < -0.2:
            hp_mode = "low"
        elif health_diff > 0.2:
            hp_mode = "high"
        else:
            hp_mode = "even"

        for i in range(NUM_ACTIONS):
            weights[i] += self.g.health_weights[hp_mode][i]

        # 4. Punish opportunity bonus
        if state["opp_recovering"]:
            # Big bonus for attacking moves
            for i, action in enumerate(ACTIONS):
                if action in [MoveType.JAB, MoveType.HEAVY, MoveType.SPECIAL]:
                    weights[i] += self.g.punish_bonus

        # 5. Pressure bonus when opponent blocking
        if state["opp_state"] == "BLOCK_STUN":
            for i, action in enumerate(ACTIONS):
                if action == MoveType.GRAB:
                    weights[i] += self.g.pressure_bonus

        # 6. Meter management
        my_meter = state["my_meter"] * MAX_METER
        special_idx = ACTIONS.index(MoveType.SPECIAL)

        if my_meter >= self.g.meter_use_threshold:
            weights[special_idx] += 1.0
        elif my_meter < self.g.meter_save_threshold:
            weights[special_idx] -= 2.0

        # Softmax selection
        max_w = max(weights)
        exp_weights = [math.exp(w - max_w) for w in weights]
        total = sum(exp_weights)
        probs = [w / total for w in exp_weights]

        # Sample action
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return ACTIONS[i]

        return ACTIONS[-1]


def play_match(genome1: FighterGenome, genome2: FighterGenome, num_rounds=3) -> Tuple:
    """Play a match between two genomes."""
    p1_wins = 0
    p2_wins = 0
    total_p1_damage = 0
    total_p2_damage = 0

    player1 = FighterPlayer(genome1)
    player2 = FighterPlayer(genome2)

    for _ in range(num_rounds):
        game = FightingGame()

        while not game.round_over:
            state1 = get_game_state(game, 1)
            state2 = get_game_state(game, 2)

            action1 = player1.get_action(state1)
            action2 = player2.get_action(state2)

            game.step(action1, action2)

        if game.winner == 1:
            p1_wins += 1
        elif game.winner == 2:
            p2_wins += 1

        total_p1_damage += game.p1.damage_dealt
        total_p2_damage += game.p2.damage_dealt

    return p1_wins, p2_wins, total_p1_damage, total_p2_damage


def evaluate(genome: FighterGenome, opponents: list, num_matches=10) -> float:
    """Evaluate a genome against opponents."""
    total_wins = 0
    total_matches = 0
    total_damage = 0

    for opp in opponents:
        for _ in range(num_matches // len(opponents)):
            wins, losses, my_dmg, _ = play_match(genome, opp, num_rounds=3)
            total_wins += wins
            total_matches += wins + losses
            total_damage += my_dmg

    win_rate = total_wins / max(1, total_matches)
    dmg_bonus = total_damage / (max(1, total_matches) * MAX_HEALTH) * 0.1

    return win_rate + dmg_bonus


def evaluate_vs_random(genome: FighterGenome, num_games=20) -> float:
    """Evaluate against random opponent."""
    wins = 0
    damage = 0

    player = FighterPlayer(genome)

    for _ in range(num_games):
        game = FightingGame()

        while not game.round_over:
            state = get_game_state(game, 1)
            action = player.get_action(state)

            # Random opponent
            opp_action = random.choice(ACTIONS)

            game.step(action, opp_action)

        if game.winner == 1:
            wins += 1
        damage += game.p1.damage_dealt

    return wins / num_games + damage / (num_games * MAX_HEALTH) * 0.1


def run_evolution():
    """Run evolutionary algorithm."""
    print("=" * 70)
    print("FIGHTING GAME EVOLUTION - DISCOVER OPTIMAL STRATEGY")
    print("=" * 70)

    pop_size = 30
    generations = 3  # Reduced for quick demo
    elite = 4

    print(f"Pop: {pop_size}, Gens: {generations}, Elite: {elite}")
    print()

    # Initialize population
    pop = [FighterGenome() for _ in range(pop_size)]

    print("Evaluating initial population...")
    for g in pop:
        g.fitness = evaluate_vs_random(g, 20)

    pop.sort(key=lambda x: x.fitness, reverse=True)
    print(f"Gen  0: best={pop[0].fitness:.3f}")

    start = time.time()

    for gen in range(1, generations + 1):
        # Adaptive rates
        rate = 0.15 * (1 - gen/generations * 0.3)
        strength = 0.3 * (1 - gen/generations * 0.2)

        # Create next generation
        new_pop = [g.copy() for g in pop[:elite]]

        while len(new_pop) < pop_size:
            # Tournament selection
            t1 = random.sample(pop, 3)
            t2 = random.sample(pop, 3)
            p1 = max(t1, key=lambda x: x.fitness)
            p2 = max(t2, key=lambda x: x.fitness)

            child = p1.crossover(p2)
            child = child.mutate(rate, strength)
            new_pop.append(child)

        pop = new_pop[:pop_size]

        # Evaluate non-elite
        for g in pop[elite:]:
            g.fitness = evaluate_vs_random(g, 20)

        pop.sort(key=lambda x: x.fitness, reverse=True)

        if gen % 1 == 0:
            elapsed = time.time() - start
            avg = sum(g.fitness for g in pop) / len(pop)
            print(f"Gen {gen:2d}: best={pop[0].fitness:.3f}, avg={avg:.3f}, time={elapsed:.0f}s")

    print()
    return pop[0]


def analyze_strategy(genome: FighterGenome):
    """Analyze what strategy evolved."""
    print("=" * 70)
    print("EVOLVED STRATEGY ANALYSIS")
    print("=" * 70)

    print("\nSpacing Preferences (higher = more likely):")
    print("-" * 50)

    for mode in ["close", "mid", "far"]:
        weights = genome.spacing_weights[mode]
        print(f"\n{mode.upper()} range:")

        # Get top 3 actions
        sorted_actions = sorted(enumerate(weights), key=lambda x: -x[1])[:3]
        for i, (idx, w) in enumerate(sorted_actions):
            print(f"  {i+1}. {ACTIONS[idx].name:<15} {w:+.2f}")

    print("\n\nFrame Advantage Reactions:")
    print("-" * 50)

    for adv, label in [("can_punish", "CAN PUNISH (+6 or more)"),
                       ("disadvantage", "DISADVANTAGE (-4 or less)")]:
        weights = genome.advantage_weights[adv]
        print(f"\n{label}:")
        sorted_actions = sorted(enumerate(weights), key=lambda x: -x[1])[:3]
        for i, (idx, w) in enumerate(sorted_actions):
            print(f"  {i+1}. {ACTIONS[idx].name:<15} {w:+.2f}")

    print("\n\nMeter Management:")
    print("-" * 50)
    print(f"Use special when meter >= {genome.meter_use_threshold:.0f}")
    print(f"Save at least {genome.meter_save_threshold:.0f} meter")

    print("\n\nBonuses:")
    print("-" * 50)
    print(f"Punish bonus: {genome.punish_bonus:.2f}")
    print(f"Pressure bonus: {genome.pressure_bonus:.2f}")


def showcase_match(genome: FighterGenome):
    """Run a showcase match and display actions."""
    print("\n" + "=" * 70)
    print("SHOWCASE MATCH")
    print("=" * 70)

    game = FightingGame()
    player = FighterPlayer(genome)

    frame_log = []

    while not game.round_over and game.frame < 500:
        state = get_game_state(game, 1)
        action = player.get_action(state)

        # Random opponent with some intelligence
        opp_action = random.choice([MoveType.JAB, MoveType.BLOCK, MoveType.NONE])

        if state["i_can_act"] and action != MoveType.NONE:
            frame_log.append({
                "frame": game.frame,
                "spacing": state["spacing"],
                "fa": state["frame_advantage"],
                "action": action.name,
                "health": f"{int(state['my_health']*100)}v{int(state['opp_health']*100)}"
            })

        game.step(action, opp_action)

    print(f"\nResult: {'WIN' if game.winner == 1 else 'LOSS' if game.winner == 2 else 'DRAW'}")
    print(f"Final health: P1={game.p1.health} P2={game.p2.health}")
    print(f"Damage dealt: {game.p1.damage_dealt}")

    print("\nKey moments:")
    for entry in frame_log[:20]:
        print(f"  Frame {entry['frame']:4d} | {entry['spacing']:<5} | "
              f"FA:{entry['fa']:+3d} | {entry['action']:<12} | HP:{entry['health']}")


if __name__ == "__main__":
    random.seed(42)
    best = run_evolution()
    analyze_strategy(best)
    showcase_match(best)

    # Save evolved genome
    genome_data = {
        "spacing_weights": best.spacing_weights,
        "advantage_weights": best.advantage_weights,
        "health_weights": best.health_weights,
        "meter_use_threshold": best.meter_use_threshold,
        "meter_save_threshold": best.meter_save_threshold,
        "punish_bonus": best.punish_bonus,
        "pressure_bonus": best.pressure_bonus,
        "fitness": best.fitness,
    }

    with open("evolved_fighter.json", "w") as f:
        json.dump(genome_data, f, indent=2)

    print("\nSaved to evolved_fighter.json")
