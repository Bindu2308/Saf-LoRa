"""
receiver/ml_advanced/ppo_train.py

Stage 2 (per the doc): PPO fine-tuning on top of the pretrained TGAT
encoder, using the offline replay environment (ppo_env.py) so training
doesn't need live hardware.

Usage (in Colab, after tgat_pretrain.py has produced tgat_encoder.pt):
    python3 ppo_train.py --csv graph_observations.csv \\
        --tgat-checkpoint tgat_encoder.pt --timesteps 50000
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from ppo_env import ISACLoRaReplayEnv


def main():
    parser = argparse.ArgumentParser(description="PPO fine-tuning for ISAC-LoRa gateway selection")
    parser.add_argument("--csv", default="graph_observations.csv")
    parser.add_argument("--tgat-checkpoint", default="tgat_encoder.pt")
    parser.add_argument("--timesteps", type=int, default=50000,
                         help="total environment steps to train for -- start smaller "
                              "(e.g. 5000) to confirm the reward curve is trending up "
                              "before committing to a long run")
    parser.add_argument("--out", default="ppo_policy.zip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = Monitor(ISACLoRaReplayEnv(csv_path=args.csv, tgat_checkpoint=args.tgat_checkpoint, seed=args.seed))

    print(f"Training PPO for {args.timesteps} timesteps...")
    print("Watch 'ep_rew_mean' in the logged output -- it should trend upward over time. "
          "If it stays flat or decreases, something is likely wrong (check reward shaping "
          "in ppo_env.py, or that the TGAT checkpoint actually loaded).")

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f"\nSaved trained PPO policy to {args.out}")
    print("Download both ppo_policy.zip AND tgat_encoder.pt -- inference.py on the RPi4 needs both.")


if __name__ == "__main__":
    main()
