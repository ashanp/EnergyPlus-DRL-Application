"""
V3 SAC Training for the ETS Gymnasium Environment
=================================================

This script trains a Soft Actor-Critic (SAC) agent on ETSGymV2.

Key design choices
------------------
- Training and test periods are separated chronologically.
- Episodes remain 24 hours (96 x 15-minute steps).
- The action is one continuous CHW flow command in [0, 1].
- The EnergyPlus baseline CSV remains the exogenous data source.
- V2 remains offline: unmet cooling is the action-dependent service proxy.

Usage
-----
python train_sac_v3.py /path/to/eplusout.csv

Optional:
python train_sac_v3.py /path/to/eplusout.csv --timesteps 100000
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ets_gym_v2 import ETSGymV2


def make_train_env(csv_path: str, split_index: int):
    def _init():
        env = ETSGymV2(csv_path, random_start=True)
        # Restrict random starts to training region.
        original_reset = env.reset

        def reset_train(*, seed=None, options=None):
            if options is None:
                options = {}
            max_start = max(0, split_index - env.episode_steps - 1)
            if "start_index" not in options:
                if max_start > 0:
                    start = int(env.np_random.integers(0, max_start + 1))
                else:
                    start = 0
                options["start_index"] = start
            return original_reset(seed=seed, options=options)

        env.reset = reset_train
        return Monitor(env)

    return _init


def make_eval_env(csv_path: str, split_index: int):
    def _init():
        env = ETSGymV2(csv_path, random_start=False)
        original_reset = env.reset

        def reset_eval(*, seed=None, options=None):
            if options is None:
                options = {}
            max_start = len(env.df) - env.episode_steps - 1
            start = min(max(split_index, 0), max_start)
            options["start_index"] = start
            return original_reset(seed=seed, options=options)

        env.reset = reset_eval
        return Monitor(env)

    return _init


def evaluate_policy_full_test(model, csv_path: str, split_index: int, output_csv: Path):
    env = ETSGymV2(csv_path, random_start=False)

    rows = []
    start = split_index
    end = len(env.df) - 1

    obs, info = env.reset(options={"start_index": start})
    env.episode_steps = end - start

    total_reward = 0.0
    required_wh = 0.0
    delivered_wh = 0.0
    unmet_wh = 0.0
    pump_wh = 0.0
    delta_ts = []
    actions = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)

        required_wh += step_info["q_required_w"] * 0.25
        delivered_wh += step_info["q_delivered_w"] * 0.25
        unmet_wh += step_info["q_unmet_w"] * 0.25
        pump_wh += step_info["pump_power_w"] * 0.25
        delta_ts.append(step_info["delta_t_c"])
        actions.append(step_info["flow_command"])
        total_reward += reward

        rows.append({
            "dataset_index": step_info["dataset_index"],
            "reward": reward,
            "flow_command": step_info["flow_command"],
            "flow_kg_s": step_info["flow_kg_s"],
            "q_required_w": step_info["q_required_w"],
            "q_delivered_w": step_info["q_delivered_w"],
            "q_unmet_w": step_info["q_unmet_w"],
            "delta_t_c": step_info["delta_t_c"],
            "pump_power_w": step_info["pump_power_w"],
            "low_delta_t": step_info["low_delta_t"],
            "low_delta_t_severity": step_info["low_delta_t_severity"],
        })

        done = terminated or truncated

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_csv, index=False)

    active = result_df[result_df["q_required_w"] > 10.0]

    metrics = {
        "test_start_index": int(split_index),
        "test_steps": int(len(result_df)),
        "total_reward": float(total_reward),
        "required_cooling_kwh_th": float(required_wh / 1000.0),
        "delivered_cooling_kwh_th": float(delivered_wh / 1000.0),
        "unmet_cooling_kwh_th": float(unmet_wh / 1000.0),
        "delivery_fraction_pct": float(
            100.0 * delivered_wh / required_wh if required_wh > 0 else 100.0
        ),
        "pump_energy_kwh": float(pump_wh / 1000.0),
        "mean_delta_t_c_active": float(active["delta_t_c"].mean()) if len(active) else 0.0,
        "median_delta_t_c_active": float(active["delta_t_c"].median()) if len(active) else 0.0,
        "mean_flow_command_active": float(active["flow_command"].mean()) if len(active) else 0.0,
        "low_delta_t_fraction_pct_active": float(
            100.0 * active["low_delta_t"].mean()
        ) if len(active) else 0.0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = str(Path(args.input_csv).resolve())
    df = pd.read_csv(csv_path)

    split_index = int(len(df) * args.train_fraction)

    # Align split to whole 24-hour episodes.
    split_index = (split_index // 96) * 96

    output_dir = Path("v3_sac_output")
    output_dir.mkdir(exist_ok=True)

    train_env = DummyVecEnv([make_train_env(csv_path, split_index)])
    eval_env = DummyVecEnv([make_eval_env(csv_path, split_index)])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_ets"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=5_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(output_dir / "tensorboard"),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    print("\nV3 SAC training")
    print("=" * 55)
    print(f"Dataset rows          : {len(df)}")
    print(f"Training rows         : {split_index}")
    print(f"Testing rows          : {len(df) - split_index}")
    print(f"Training fraction     : {args.train_fraction:.2f}")
    print(f"Training timesteps    : {args.timesteps}")
    print(f"Episode length        : 96 steps (24 h)")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_model_path = output_dir / "sac_ets_v3_final"
    model.save(str(final_model_path))

    evaluation_csv = output_dir / "sac_test_results.csv"
    metrics = evaluate_policy_full_test(
        model,
        csv_path,
        split_index,
        evaluation_csv
    )

    metrics["training_rows"] = split_index
    metrics["testing_rows"] = len(df) - split_index
    metrics["training_timesteps"] = args.timesteps
    metrics["seed"] = args.seed

    metrics_path = output_dir / "sac_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("\nV3 training complete")
    print("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:38s}: {v:.4f}")
        else:
            print(f"{k:38s}: {v}")

    print(f"\nFinal model     : {final_model_path}.zip")
    print(f"Test results    : {evaluation_csv}")
    print(f"Test metrics    : {metrics_path}")
    print(f"Best model dir  : {output_dir / 'best_model'}")


if __name__ == "__main__":
    main()
