
import sys
import os
import time
from typing import List

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fighting_game import FightingGame, MoveType, PlayerState, Player, STARTING_DISTANCE, MAX_METER, MAX_HEALTH

def render_health_bar(current: int, max_val: int, width: int = 20, color_code: str = "") -> str:
    filled = int((current / max_val) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{color_code}[{bar}] {current}/{max_val}\033[0m"

def render_meter_bar(current: int, max_val: int, width: int = 10) -> str:
    filled = int((current / max_val) * width)
    bar = "▒" * filled + "-" * (width - filled)
    return f"[{bar}] {current}"

def render_state(player: Player) -> str:
    state_str = player.state.name
    if player.state == PlayerState.NEUTRAL:
        return ""
    if player.state == PlayerState.STARTUP:
        return f"(Start: {player.frame_counter})"
    if player.state == PlayerState.ACTIVE:
        return f"** {player.current_move.name} **"
    if player.state == PlayerState.RECOVERY:
        return f"(Rec: {player.frame_counter})"
    if player.state in [PlayerState.HIT_STUN, PlayerState.BLOCK_STUN, PlayerState.KNOCKDOWN]:
        return f"STUN: {player.stun_remaining}"
    return state_str

def render_ascii_scene(game: FightingGame, p1_name="P1", p2_name="P2"):
    # Clear screen (optional based on mode, simpler to just print for log)
    # print("\033[H\033[J") 
    
    # 1. Health Bars
    p1_health = render_health_bar(game.p1.health, MAX_HEALTH, color_code="\033[92m") # Green
    p2_health = render_health_bar(game.p2.health, MAX_HEALTH, color_code="\033[91m") # Red
    print(f"{p1_name:<30} VS {p2_name:>30}")
    print(f"{p1_health:<30}    {p2_health:>30}")
    
    # 2. Meter
    p1_meter = render_meter_bar(game.p1.meter, MAX_METER)
    p2_meter = render_meter_bar(game.p2.meter, MAX_METER)
    print(f"{p1_meter:<30}    {p2_meter:>30}")
    
    # 3. The Stage (Spacing)
    # Total stage width ~60 chars.
    # Positions are roughly -5 to +5. Map to 0-60.
    stage_width = 60
    # Normalize positions (centering around 0)
    # min dist is usually 0, max is maybe 15? 
    # Let's fix the "camera" on separation distance or just absolute positions?
    # fighting_game.py uses p1 at -2.5, p2 at 2.5.
    
    # Map -10 to +10 range to 0-60
    def pos_to_char(pos):
        return int((pos + 10) / 20 * stage_width)
    
    p1_idx = max(0, min(stage_width-1, pos_to_char(game.p1.position)))
    p2_idx = max(0, min(stage_width-1, pos_to_char(game.p2.position)))
    
    stage = ["."] * stage_width
    stage[p1_idx] = "P1" if stage[p1_idx] == "." else "X" # Collision overlap
    # P1 is usually left, P2 right. If crossed up?
    
    # Actually, simpler: just draw dot-bar based on distance
    dist = game.distance()
    dist_dots = int(dist * 4) # Scale factor
    
    # 4. Avatar / Action Display
    p1_state = render_state(game.p1)
    p2_state = render_state(game.p2)
    
    # Line 1: Actions (overhead)
    print(f"{p1_state:^30}      {p2_state:^30}")
    
    # Line 2: Avatars
    # Using emojis or simple chars
    p1_char = "🥷" if not game.p1.blocking else "🛡️"
    if game.p1.state == PlayerState.KNOCKDOWN: p1_char = "🛌"
    if game.p1.state == PlayerState.HIT_STUN: p1_char = "💥"
    
    p2_char = "🤖" if not game.p2.blocking else "🛡️"
    if game.p2.state == PlayerState.KNOCKDOWN: p2_char = "🛌"
    if game.p2.state == PlayerState.HIT_STUN: p2_char = "💥"
    
    # Draw stage with players at relative distance
    # Center is the midpoint between players
    # We draw them separated by 'dist_dots' spaces
    spacing_display = " " * 20 + p1_char + ("." * dist_dots) + p2_char
    
    print(f"\n{spacing_display}\n")
    
    # 5. Combat Log (Last event)
    if game.log:
        print(f"> {game.log[-1]}")
    print("-" * 70)



import json
import time
import random
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fighting_game import FightingGame, MoveType, PlayerState, Player, STARTING_DISTANCE, MAX_METER, MAX_HEALTH, get_game_state

# Import classes from evolve_fighter (assuming it is in same dir)
from evolve_fighter import FighterGenome, FighterPlayer, ACTIONS

def load_genome(filepath):
    """Load genome from JSON."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
    with open(filepath, "r") as f:
        data = json.load(f)
        
    g = FighterGenome()
    # Manual hydration because strict typing not used in original
    g.spacing_weights = data["spacing_weights"]
    g.advantage_weights = data["advantage_weights"]
    g.health_weights = data["health_weights"]
    g.meter_use_threshold = data["meter_use_threshold"]
    g.meter_save_threshold = data["meter_save_threshold"]
    g.punish_bonus = data["punish_bonus"]
    g.pressure_bonus = data["pressure_bonus"]
    g.fitness = data.get("fitness", 0.0)
    
    return g

def visualize_match(genome1, genome2, delay=0.1):
    """Run and visualize a match."""
    game = FightingGame()
    p1 = FighterPlayer(genome1)
    
    # If genome2 is None, use random opponent logic
    p2 = FighterPlayer(genome2) if genome2 else None
    
    print("\033[2J") # Clear screen
    
    while not game.round_over:
        # 1. Get State & Actions
        s1 = get_game_state(game, 1)
        s2 = get_game_state(game, 2)
        
        a1 = p1.get_action(s1)
        
        if p2:
            a2 = p2.get_action(s2)
        else:
            # Random baseline opponent
            a2 = random.choice(ACTIONS)
            
        # 2. Step Game
        game.step(a1, a2)
        
        # 3. Render
        print("\033[H") # Home cursor
        render_ascii_scene(game, "Evolved Agent", "Random Opponent" if not p2 else "Evolved Clone")
        
        # 4. Delay
        time.sleep(delay)

    print(f"\nWinner: {game.winner}")

if __name__ == "__main__":
    json_path = "evolved_fighter.json"
    
    # Try to load evolved agent
    if os.path.exists(json_path):
        print(f"Loading best agent from {json_path}...")
        g = load_genome(json_path)
        visualize_match(g, None) # Playing vs Random for demo
    else:
        print("No evolved_fighter.json found. Run evolve_fighter.py first.")
        # Create a random genome for testing visualizer
        g = FighterGenome()
        visualize_match(g, None)

