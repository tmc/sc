"""
Simple 1D Fighting Game with Frame Data.

Core concepts:
- Frame advantage: Who can act first after an interaction
- Hit confirm: Recognizing a hit to follow up with combos
- Spacing: Distance affects what moves are available
- Meter: Resource for special moves

Moves have:
- Startup frames (before hitbox)
- Active frames (hitbox out)
- Recovery frames (vulnerable after)
- Damage
- Block stun vs hit stun

Frame advantage = (opponent's stun) - (my recovery)
Positive = I can act first (safe on block)
Negative = I'm punishable (opponent can hit me)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple
import random

# === GAME CONSTANTS ===

MAX_HEALTH = 100
ROUND_TIME = 3000  # frames (50 seconds at 60fps)
STARTING_DISTANCE = 5  # units apart
MAX_METER = 100

class MoveType(Enum):
    JAB = auto()       # Fast, low damage, safe
    HEAVY = auto()     # Slow, high damage, punishable
    SWEEP = auto()     # Low attack, must block low
    OVERHEAD = auto()  # Must block standing
    GRAB = auto()      # Beats block, loses to attack
    BLOCK = auto()     # Defensive
    BACK_DASH = auto() # Create space
    FORWARD_DASH = auto() # Close distance
    SPECIAL = auto()   # Uses meter, invincible startup
    NONE = auto()      # No input


@dataclass
class MoveData:
    """Frame data for a move."""
    name: str
    startup: int       # Frames before active
    active: int        # Frames hitbox is out
    recovery: int      # Frames after active
    damage: int
    hit_stun: int      # Frames opponent is stuck on hit
    block_stun: int    # Frames opponent is stuck on block
    range: int         # Distance this move reaches
    meter_gain: int    # Meter gained on hit
    meter_cost: int    # Meter required
    invincible: int    # Invincible startup frames

    @property
    def total_frames(self) -> int:
        return self.startup + self.active + self.recovery

    @property
    def frame_advantage_on_hit(self) -> int:
        return self.hit_stun - self.recovery

    @property
    def frame_advantage_on_block(self) -> int:
        return self.block_stun - self.recovery


# Frame data table - realistic fighting game values
MOVES = {
    MoveType.JAB: MoveData(
        name="Jab",
        startup=4, active=2, recovery=8,
        damage=5, hit_stun=12, block_stun=6,
        range=1, meter_gain=5, meter_cost=0, invincible=0
    ),
    MoveType.HEAVY: MoveData(
        name="Heavy",
        startup=10, active=4, recovery=20,
        damage=25, hit_stun=25, block_stun=12,
        range=2, meter_gain=15, meter_cost=0, invincible=0
    ),
    MoveType.SWEEP: MoveData(
        name="Sweep",
        startup=8, active=3, recovery=22,
        damage=12, hit_stun=0, block_stun=0,  # Knockdown
        range=2, meter_gain=10, meter_cost=0, invincible=0
    ),
    MoveType.OVERHEAD: MoveData(
        name="Overhead",
        startup=20, active=4, recovery=15,
        damage=18, hit_stun=18, block_stun=8,
        range=1, meter_gain=8, meter_cost=0, invincible=0
    ),
    MoveType.GRAB: MoveData(
        name="Grab",
        startup=5, active=2, recovery=30,  # Whiff punishable
        damage=30, hit_stun=0, block_stun=0,  # Unblockable
        range=0, meter_gain=0, meter_cost=0, invincible=0  # Point blank
    ),
    MoveType.BLOCK: MoveData(
        name="Block",
        startup=1, active=999, recovery=5,
        damage=0, hit_stun=0, block_stun=0,
        range=0, meter_gain=0, meter_cost=0, invincible=0
    ),
    MoveType.BACK_DASH: MoveData(
        name="Back Dash",
        startup=3, active=10, recovery=10,
        damage=0, hit_stun=0, block_stun=0,
        range=-3, meter_gain=0, meter_cost=0, invincible=3  # Some invuln
    ),
    MoveType.FORWARD_DASH: MoveData(
        name="Forward Dash",
        startup=3, active=10, recovery=8,
        damage=0, hit_stun=0, block_stun=0,
        range=3, meter_gain=0, meter_cost=0, invincible=0
    ),
    MoveType.SPECIAL: MoveData(
        name="Special",
        startup=6, active=6, recovery=25,
        damage=35, hit_stun=30, block_stun=15,
        range=3, meter_gain=0, meter_cost=50, invincible=6
    ),
}


class PlayerState(Enum):
    NEUTRAL = auto()    # Can act freely
    STARTUP = auto()    # Committed to move startup
    ACTIVE = auto()     # Move is attacking
    RECOVERY = auto()   # Vulnerable after move
    HIT_STUN = auto()   # Got hit, can't act
    BLOCK_STUN = auto() # Blocked, can't act
    KNOCKDOWN = auto()  # On ground


@dataclass
class Player:
    """Player state in the fighting game."""
    health: int = MAX_HEALTH
    meter: int = 0
    position: float = 0.0  # Position on 1D stage

    state: PlayerState = PlayerState.NEUTRAL
    current_move: Optional[MoveType] = None
    frame_counter: int = 0  # Frames into current state
    stun_remaining: int = 0

    blocking: bool = False
    blocking_low: bool = False

    # Stats tracking
    damage_dealt: int = 0
    combos_landed: int = 0
    punishes_landed: int = 0


class FightingGame:
    """1D Fighting game with frame data."""

    def __init__(self):
        self.p1 = Player(position=-STARTING_DISTANCE/2)
        self.p2 = Player(position=STARTING_DISTANCE/2)
        self.frame = 0
        self.round_over = False
        self.winner: Optional[int] = None  # 1, 2, or None for draw

        # Combat log for analysis
        self.log: List[str] = []

    def distance(self) -> float:
        """Distance between players."""
        return abs(self.p2.position - self.p1.position)

    def get_spacing_mode(self) -> str:
        """Classify current spacing."""
        d = self.distance()
        if d <= 1:
            return "close"
        elif d <= 3:
            return "mid"
        else:
            return "far"

    def can_hit(self, attacker: Player, defender: Player, move: MoveData) -> bool:
        """Check if attacker's move can reach defender."""
        d = abs(defender.position - attacker.position)
        return d <= move.range

    def process_input(self, player: Player, move: MoveType) -> bool:
        """Process player input if they can act."""
        if player.state != PlayerState.NEUTRAL:
            return False

        if move == MoveType.NONE:
            return False

        move_data = MOVES[move]

        # Check meter cost
        if move_data.meter_cost > player.meter:
            return False

        # Start the move
        player.current_move = move
        player.state = PlayerState.STARTUP
        player.frame_counter = 0
        player.meter -= move_data.meter_cost

        # Handle blocking special case
        if move == MoveType.BLOCK:
            player.blocking = True

        return True

    def resolve_combat(self, attacker: Player, defender: Player) -> Tuple[bool, bool]:
        """
        Resolve combat when attacker is in active frames.
        Returns (hit, blocked).
        """
        if attacker.state != PlayerState.ACTIVE:
            return False, False

        if attacker.current_move is None:
            return False, False

        move = MOVES[attacker.current_move]

        # Check range
        if not self.can_hit(attacker, defender, move):
            return False, False

        # Grab special case - beats block, loses to everything else
        if attacker.current_move == MoveType.GRAB:
            if defender.blocking:
                # Grab beats block!
                defender.health -= move.damage
                defender.state = PlayerState.HIT_STUN
                defender.stun_remaining = 20
                attacker.damage_dealt += move.damage
                self.log.append(f"Grab landed! {move.damage} damage")
                return True, False
            else:
                # Whiff or get hit
                return False, False

        # Normal moves
        if defender.state in [PlayerState.STARTUP, PlayerState.RECOVERY,
                              PlayerState.HIT_STUN, PlayerState.KNOCKDOWN]:
            # Defender is vulnerable - combo/punish opportunity
            defender.health -= move.damage
            defender.state = PlayerState.HIT_STUN
            defender.stun_remaining = move.hit_stun
            attacker.damage_dealt += move.damage
            attacker.meter = min(MAX_METER, attacker.meter + move.meter_gain)

            # Track punish
            if defender.state == PlayerState.RECOVERY:
                attacker.punishes_landed += 1
                self.log.append(f"Punish! {move.name} for {move.damage}")

            return True, False

        if defender.blocking:
            # Blocked
            defender.state = PlayerState.BLOCK_STUN
            defender.stun_remaining = move.block_stun
            defender.blocking = False
            self.log.append(f"{move.name} blocked, advantage: {move.frame_advantage_on_block}")
            return False, True

        # Defender in neutral, not blocking - clean hit
        if defender.state == PlayerState.NEUTRAL:
            defender.health -= move.damage
            defender.state = PlayerState.HIT_STUN
            defender.stun_remaining = move.hit_stun
            attacker.damage_dealt += move.damage
            attacker.meter = min(MAX_METER, attacker.meter + move.meter_gain)
            self.log.append(f"{move.name} hit! {move.damage} damage")

            # Sweep causes knockdown
            if attacker.current_move == MoveType.SWEEP:
                defender.state = PlayerState.KNOCKDOWN
                defender.stun_remaining = 30

            return True, False

        return False, False

    def update_player(self, player: Player):
        """Update player state machine."""
        if player.state == PlayerState.NEUTRAL:
            player.blocking = False
            return

        player.frame_counter += 1

        if player.current_move is not None:
            move = MOVES[player.current_move]

            if player.state == PlayerState.STARTUP:
                if player.frame_counter >= move.startup:
                    player.state = PlayerState.ACTIVE
                    player.frame_counter = 0

            elif player.state == PlayerState.ACTIVE:
                # Handle movement moves
                if player.current_move == MoveType.BACK_DASH:
                    player.position -= 0.3
                elif player.current_move == MoveType.FORWARD_DASH:
                    player.position += 0.3  # Will be adjusted for direction

                if player.frame_counter >= move.active:
                    player.state = PlayerState.RECOVERY
                    player.frame_counter = 0

            elif player.state == PlayerState.RECOVERY:
                if player.frame_counter >= move.recovery:
                    player.state = PlayerState.NEUTRAL
                    player.current_move = None
                    player.frame_counter = 0

        if player.state in [PlayerState.HIT_STUN, PlayerState.BLOCK_STUN, PlayerState.KNOCKDOWN]:
            player.stun_remaining -= 1
            if player.stun_remaining <= 0:
                player.state = PlayerState.NEUTRAL
                player.current_move = None

    def step(self, p1_input: MoveType, p2_input: MoveType) -> bool:
        """
        Advance game by one frame.
        Returns True if round is over.
        """
        if self.round_over:
            return True

        self.frame += 1

        # Process inputs
        self.process_input(self.p1, p1_input)
        self.process_input(self.p2, p2_input)

        # Resolve combat (check both players attacking)
        self.resolve_combat(self.p1, self.p2)
        self.resolve_combat(self.p2, self.p1)

        # Update states
        self.update_player(self.p1)
        self.update_player(self.p2)

        # Check win conditions
        if self.p1.health <= 0 or self.p2.health <= 0:
            self.round_over = True
            if self.p1.health <= 0 and self.p2.health <= 0:
                self.winner = None  # Draw
            elif self.p1.health <= 0:
                self.winner = 2
            else:
                self.winner = 1

        if self.frame >= ROUND_TIME:
            self.round_over = True
            if self.p1.health > self.p2.health:
                self.winner = 1
            elif self.p2.health > self.p1.health:
                self.winner = 2
            else:
                self.winner = None

        return self.round_over


def get_game_state(game: FightingGame, player_num: int) -> dict:
    """Get observable state for a player."""
    me = game.p1 if player_num == 1 else game.p2
    opp = game.p2 if player_num == 1 else game.p1

    return {
        "my_health": me.health / MAX_HEALTH,
        "opp_health": opp.health / MAX_HEALTH,
        "my_meter": me.meter / MAX_METER,
        "opp_meter": opp.meter / MAX_METER,
        "distance": game.distance() / 10,
        "spacing": game.get_spacing_mode(),
        "my_state": me.state.name,
        "opp_state": opp.state.name,
        "i_can_act": me.state == PlayerState.NEUTRAL,
        "opp_recovering": opp.state == PlayerState.RECOVERY,
        "opp_in_startup": opp.state == PlayerState.STARTUP,
        "frame_advantage": (opp.stun_remaining if opp.state in
                          [PlayerState.HIT_STUN, PlayerState.BLOCK_STUN, PlayerState.RECOVERY]
                          else 0) - (me.stun_remaining if me.state in
                          [PlayerState.HIT_STUN, PlayerState.BLOCK_STUN, PlayerState.RECOVERY]
                          else 0),
    }


if __name__ == "__main__":
    # Quick test
    game = FightingGame()

    print("Frame Data Analysis:")
    print("-" * 60)
    print(f"{'Move':<12} {'Total':<6} {'Start':<6} {'Act':<5} {'Rec':<5} {'+Hit':<5} {'+Blk':<5}")
    print("-" * 60)

    for move_type, data in MOVES.items():
        if move_type in [MoveType.NONE, MoveType.BLOCK]:
            continue
        print(f"{data.name:<12} {data.total_frames:<6} {data.startup:<6} {data.active:<5} "
              f"{data.recovery:<5} {data.frame_advantage_on_hit:+<5} {data.frame_advantage_on_block:+<5}")

    print("\nKey Insights:")
    print("- Jab is +4 on hit (can link into another jab)")
    print("- Heavy is -8 on block (PUNISHABLE by jab)")
    print("- Sweep causes knockdown")
    print("- Grab beats block but loses to attacks")
    print("- Special has invincible startup")
