#!/usr/bin/env python3
"""
StatechartAlphaZero Training Script

Train a Go 9x9 agent with statechart-guaranteed legal moves.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from datetime import datetime

from statechart_nnet import NNetWrapper
from statechart_coach import StatechartCoach


def main():
    parser = argparse.ArgumentParser(
        description='Train StatechartAlphaZero for Go 9x9'
    )

    # Network architecture
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='Embedding dimension (default: 256)')
    parser.add_argument('--num-res-blocks', type=int, default=4,
                       help='Number of residual blocks (default: 4)')

    # Training parameters
    parser.add_argument('--num-iters', type=int, default=100,
                       help='Number of training iterations (default: 100)')
    parser.add_argument('--num-episodes', type=int, default=100,
                       help='Self-play episodes per iteration (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs per iteration (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')

    # MCTS parameters
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS simulations per move (default: 100)')
    parser.add_argument('--cpuct', type=float, default=1.0,
                       help='PUCT exploration constant (default: 1.0)')
    parser.add_argument('--temp-threshold', type=int, default=30,
                       help='Move number to switch to temp=0 (default: 30)')

    # Other
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--history-size', type=int, default=100000,
                       help='Max training examples to keep (default: 100000)')
    parser.add_argument('--no-symmetries', action='store_true',
                       help='Disable 8-fold symmetry augmentation')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("StatechartAlphaZero Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Network: {args.embed_dim}D embedding, {args.num_res_blocks} res blocks")
    print(f"  Training: {args.num_iters} iters, {args.num_episodes} episodes/iter")
    print(f"  MCTS: {args.mcts_sims} sims, cpuct={args.cpuct}")
    print(f"  Symmetries: {'disabled' if args.no_symmetries else 'enabled (8x)'}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    # Create network
    print("\nInitializing neural network...")
    nnet = NNetWrapper(
        embed_dim=args.embed_dim,
        num_res_blocks=args.num_res_blocks,
        lr=args.lr
    )

    # Resume if requested
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        nnet.load_checkpoint(args.checkpoint_dir, args.resume)

    # Create coach
    print("Initializing coach...")
    coach = StatechartCoach(
        nnet=nnet,
        mcts_sims=args.mcts_sims,
        cpuct=args.cpuct,
        temp_threshold=args.temp_threshold,
        history_size=args.history_size,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save config
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    config_path = Path(args.checkpoint_dir) / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Train
    print("\nStarting training...")
    coach.learn(
        num_iters=args.num_iters,
        num_episodes=args.num_episodes,
        epochs_per_iter=args.epochs,
        batch_size=args.batch_size,
        use_symmetries=not args.no_symmetries,
        verbose=args.verbose
    )

    # Final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Metrics:")
    print(f"  Total games: {coach.total_games}")
    print(f"  Total moves: {coach.total_moves}")
    print(f"  Illegal moves: {coach.illegal_moves}")
    print(f"  Illegal rate: {coach.illegal_moves / max(1, coach.total_moves):.6f}")

    # This should always be true!
    if coach.illegal_moves == 0:
        print("\n*** VERIFIED: 100% legal moves across ALL training! ***")
    else:
        print("\n*** ERROR: Illegal moves detected! This should never happen. ***")


if __name__ == '__main__':
    main()
