"""
Game Trace Export for 9x9 Go

Export games in SGF (Smart Game Format) and JSON for analysis/visualization.
"""

import json
import random
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from go_statechart import (
    Go9x9Statechart, KoState, BOARD_SIZE, TOTAL_POINTS,
    BLACK, WHITE, EMPTY, idx_to_xy
)


@dataclass
class MoveRecord:
    """Record of a single move."""
    move_num: int
    player: str  # "B" or "W"
    x: int
    y: int
    is_pass: bool
    captures: List[Tuple[int, int]]
    ko_state: str
    ko_point: Optional[Tuple[int, int]]


@dataclass
class GameTrace:
    """Complete game trace."""
    game_id: str
    timestamp: str
    board_size: int
    moves: List[MoveRecord]
    result: str
    black_captures: int
    white_captures: int
    ko_situations: int
    total_moves: int


def coords_to_sgf(x: int, y: int) -> str:
    """Convert board coordinates to SGF format (a-i for 9x9)."""
    return chr(ord('a') + x) + chr(ord('a') + y)


def play_and_record_game(seed: Optional[int] = None) -> GameTrace:
    """Play a random game and record all moves."""
    if seed is not None:
        random.seed(seed)

    game = Go9x9Statechart()
    moves = []
    ko_situations = 0

    game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(0, 9999):04d}"

    move_num = 0
    consecutive_passes = 0

    while not game.is_game_over() and move_num < 200:
        player = "B" if game.current_player() == BLACK else "W"

        # Track Ko situation
        if game.ko_state == KoState.KO_FORBIDDEN:
            ko_situations += 1
            ko_state = "forbidden"
            ko_point = game.ko_point
        else:
            ko_state = "none"
            ko_point = None

        # Get legal moves
        legal = game.get_legal_moves()

        if not legal or random.random() < 0.05:  # 5% pass chance
            # Pass
            game.play_pass()
            moves.append(MoveRecord(
                move_num=move_num,
                player=player,
                x=-1,
                y=-1,
                is_pass=True,
                captures=[],
                ko_state=ko_state,
                ko_point=ko_point,
            ))
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
        else:
            consecutive_passes = 0
            x, y = random.choice(legal)

            # Track captures before move
            prev_stones = game.board.stones.copy()

            game.play_move(x, y)

            # Find captures
            captures = []
            for idx in range(TOTAL_POINTS):
                if prev_stones[idx] != EMPTY and game.board.stones[idx] == EMPTY:
                    cx, cy = idx_to_xy(idx)
                    captures.append((cx, cy))

            moves.append(MoveRecord(
                move_num=move_num,
                player=player,
                x=x,
                y=y,
                is_pass=False,
                captures=captures,
                ko_state=ko_state,
                ko_point=ko_point,
            ))

        move_num += 1

    # Determine result (simplified - just count territory roughly)
    result = f"B+{game.black_captures - game.white_captures}" if game.black_captures > game.white_captures else f"W+{game.white_captures - game.black_captures}"

    return GameTrace(
        game_id=game_id,
        timestamp=datetime.now().isoformat(),
        board_size=BOARD_SIZE,
        moves=moves,
        result=result,
        black_captures=game.black_captures,
        white_captures=game.white_captures,
        ko_situations=ko_situations,
        total_moves=len(moves),
    )


def trace_to_sgf(trace: GameTrace) -> str:
    """Convert game trace to SGF format."""
    lines = [
        f"(;GM[1]FF[4]CA[UTF-8]",
        f"AP[GoStatechart:1.0]",
        f"SZ[{trace.board_size}]",
        f"DT[{trace.timestamp[:10]}]",
        f"RE[{trace.result}]",
        f"GC[Ko situations: {trace.ko_situations}]",
    ]

    # Add moves
    for move in trace.moves:
        if move.is_pass:
            lines.append(f";{move.player}[]")
        else:
            coord = coords_to_sgf(move.x, move.y)
            comment = ""
            if move.captures:
                comment = f"C[Captures: {move.captures}]"
            if move.ko_state == "forbidden":
                comment = f"C[Ko at {move.ko_point}]"
            lines.append(f";{move.player}[{coord}]{comment}")

    lines.append(")")
    return "\n".join(lines)


def trace_to_json(trace: GameTrace) -> str:
    """Convert game trace to JSON format."""
    # Convert to dict, handling tuples
    data = asdict(trace)

    # Convert tuple lists to list of lists for JSON
    for move in data['moves']:
        move['captures'] = [list(c) for c in move['captures']]
        if move['ko_point']:
            move['ko_point'] = list(move['ko_point'])

    return json.dumps(data, indent=2)


def generate_traces(num_games: int = 10, output_dir: str = "traces") -> List[str]:
    """Generate and save game traces."""
    os.makedirs(output_dir, exist_ok=True)

    trace_files = []

    print(f"Generating {num_games} game traces...")

    for i in range(num_games):
        trace = play_and_record_game(seed=i)

        # Save SGF
        sgf_path = os.path.join(output_dir, f"{trace.game_id}.sgf")
        with open(sgf_path, 'w') as f:
            f.write(trace_to_sgf(trace))

        # Save JSON
        json_path = os.path.join(output_dir, f"{trace.game_id}.json")
        with open(json_path, 'w') as f:
            f.write(trace_to_json(trace))

        trace_files.append(trace.game_id)

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_games} games...")

    print(f"\nSaved {num_games} traces to {output_dir}/")
    print(f"  - SGF files for Go viewers")
    print(f"  - JSON files for analysis")

    return trace_files


def main():
    """Generate sample game traces."""
    print("=" * 60)
    print("GENERATING GO GAME TRACES")
    print("=" * 60)

    traces = generate_traces(num_games=20, output_dir="traces")

    # Show sample trace
    print("\n" + "-" * 40)
    print("SAMPLE TRACE (first game)")
    print("-" * 40)

    trace = play_and_record_game(seed=0)

    print(f"\nGame ID: {trace.game_id}")
    print(f"Total moves: {trace.total_moves}")
    print(f"Ko situations: {trace.ko_situations}")
    print(f"Result: {trace.result}")

    print("\nFirst 10 moves:")
    for move in trace.moves[:10]:
        if move.is_pass:
            print(f"  {move.move_num}: {move.player} PASS")
        else:
            caps = f" (captures {move.captures})" if move.captures else ""
            ko = f" [Ko at {move.ko_point}]" if move.ko_state == "forbidden" else ""
            print(f"  {move.move_num}: {move.player} ({move.x},{move.y}){caps}{ko}")

    print("\n" + "-" * 40)
    print("SGF FORMAT (first 5 moves):")
    print("-" * 40)
    sgf = trace_to_sgf(trace)
    print(sgf[:500] + "..." if len(sgf) > 500 else sgf)

    print("\n" + "=" * 60)
    print(f"Generated {len(traces)} game traces in traces/")
    print("=" * 60)


if __name__ == "__main__":
    main()
