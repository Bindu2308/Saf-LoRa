"""
receiver/ml_advanced/baselines/dqn_baseline.py

DQN baseline, using the SAME offline replay environment as PPO
(ppo_env.py) -- an apples-to-apples RL comparison. Per the doc's
reasoning (section 21), DQN is a legitimate simpler baseline; PPO is the
proposed method because it generalizes better if you scale beyond 2
gateways later.

Usage (Colab):
    python3 baselines/dqn_baseline.py --csv graph_observations.csv \\
        --tgat-checkpoint tgat_encoder.pt --timesteps 50000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from ppo_env import ISACLoRaReplayEnv


def main():
    parser = argparse.ArgumentParser(description="DQN baseline for ISAC-LoRa gateway selection")
    parser.add_argument("--csv", default="graph_observations.csv")
    parser.add_argument("--tgat-checkpoint", default="tgat_encoder.pt")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--out", default="dqn_policy.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = Monitor(ISACLoRaReplayEnv(csv_path=args.csv, tgat_checkpoint=args.tgat_checkpoint, seed=args.seed))

    print(f"Training DQN for {args.timesteps} timesteps...")
    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.3,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"\nSaved trained DQN policy to {args.out}")


if __name__ == "__main__":
    main()
