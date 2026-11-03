"""
ETS Gymnasium Environment - V2
==============================

Offline reinforcement-learning environment for the district-cooling/ETS study.

Important scope:
- EnergyPlus eplusout.csv remains the source of exogenous building/weather data.
- The V1 ETS physics model supplies the controllable water-side response.
- EnergyPlus itself is NOT yet stepped interactively in V2.
- Therefore unmet cooling is used as an action-dependent proxy for service/
  comfort degradation. Baseline zone temperature is included as context but
  is not modified by the action in this version.

Action
------
1 continuous value in [0, 1]:
    secondary chilled-water flow command

Observation (16 values)
-----------------------
0  cooling requirement [kW]
1  CHW supply temperature [C]
2  outdoor dry-bulb [C]
3  outdoor relative humidity [%]
4  direct solar [W/m2]
5  diffuse solar [W/m2]
6  occupancy [people]
7  zone temperature [C]
8  cooling setpoint [C]
9  previous flow command [0..1]
10 previous Delta-T [C]
11 previous pump power fraction [0..1]
12 hour sin
13 hour cos
14 day-of-year sin
15 day-of-year cos

Reward
------
reward = -(
    w_pump   * pump_power_fraction
  + w_unmet  * unmet_cooling_fraction
  + w_dt     * low_delta_t_severity
  + w_smooth * abs(action - previous_action)
)

The default weights strongly discourage unmet cooling while still rewarding
lower pump energy and improved Delta-T.

Run a smoke test:
    python ets_gym_v2.py "eplusout.csv"

Use with Stable-Baselines3 SAC:
    from stable_baselines3 import SAC
    env = ETSGymV2("eplusout.csv")
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ets_model_v1 import ETSModelV1, calibrate_from_energyplus


# EnergyPlus columns
COL_Q_COIL = "MAIN ZONE FCU COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)"
COL_T_SUPPLY = "MAIN ZONE FCU COOLING COIL WATER INLET:System Node Temperature [C](TimeStep)"
COL_OUT_TEMP = "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"
COL_OUT_RH = "Environment:Site Outdoor Air Relative Humidity [%](TimeStep)"
COL_DIRECT_SOLAR = "Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)"
COL_DIFFUSE_SOLAR = "Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)"
COL_OCCUPANCY = "MAIN ZONE:Zone People Occupant Count [](TimeStep)"
COL_ZONE_TEMP = "MAIN ZONE:Zone Mean Air Temperature [C](TimeStep)"
COL_ZONE_SETPOINT = "MAIN ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"


@dataclass
class RewardWeights:
    pump: float = 1.0
    unmet: float = 8.0
    low_delta_t: float = 2.0
    smoothness: float = 0.10


class ETSGymV2(gym.Env):
    """
    Offline Gymnasium environment backed by EnergyPlus baseline data and V1 ETS physics.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        csv_path: str,
        episode_steps: int = 96,       # 24 hours at 15-minute timestep
        design_delta_t_c: float = 4.0,
        reward_weights: RewardWeights | None = None,
        random_start: bool = True,
    ):
        super().__init__()

        self.csv_path = str(csv_path)
        self.df = pd.read_csv(self.csv_path).reset_index(drop=True)
        self.episode_steps = int(episode_steps)
        self.random_start = bool(random_start)
        self.weights = reward_weights or RewardWeights()

        self._validate_columns()
        self._add_time_features()

        cfg = calibrate_from_energyplus(self.df, design_delta_t_c)
        self.ets = ETSModelV1(cfg)

        # SAC-friendly continuous action.
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 16 observations, each with explicit engineering bounds.
        low = np.array([
            0.0,       # load kW
            0.0,       # supply C
            -10.0,     # outdoor C
            0.0,       # RH
            0.0,       # direct solar
            0.0,       # diffuse solar
            0.0,       # occupancy
            10.0,      # zone temp
            10.0,      # setpoint
            0.0,       # previous action
            0.0,       # previous Delta-T
            0.0,       # previous pump power fraction
            -1.0, -1.0, -1.0, -1.0
        ], dtype=np.float32)

        high = np.array([
            50.0,      # load kW
            25.0,      # supply C
            60.0,      # outdoor C
            100.0,     # RH
            1500.0,    # direct solar
            1000.0,    # diffuse solar
            100.0,     # occupancy
            40.0,      # zone temp
            35.0,      # setpoint
            1.0,       # previous action
            20.0,      # previous Delta-T
            1.5,       # pump fraction; headroom
            1.0, 1.0, 1.0, 1.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.start_index = 0
        self.index = 0
        self.steps_taken = 0
        self.previous_action = 0.0
        self.previous_delta_t = 0.0
        self.previous_pump_fraction = 0.0

    def _validate_columns(self):
        needed = [
            COL_Q_COIL, COL_T_SUPPLY, COL_OUT_TEMP, COL_OUT_RH,
            COL_DIRECT_SOLAR, COL_DIFFUSE_SOLAR, COL_OCCUPANCY,
            COL_ZONE_TEMP, COL_ZONE_SETPOINT,
        ]
        missing = [c for c in needed if c not in self.df.columns]
        if missing:
            raise ValueError("Missing required columns:\n  - " + "\n  - ".join(missing))

        if len(self.df) <= self.episode_steps:
            raise ValueError("Dataset is shorter than one episode.")

    def _add_time_features(self):
        # EnergyPlus output is exactly 15-min in this project.
        idx = np.arange(len(self.df))
        quarter_of_day = idx % 96
        day_index = idx // 96

        angle_h = 2.0 * np.pi * quarter_of_day / 96.0
        angle_d = 2.0 * np.pi * day_index / 365.0

        self.df["_hour_sin"] = np.sin(angle_h)
        self.df["_hour_cos"] = np.cos(angle_h)
        self.df["_doy_sin"] = np.sin(angle_d)
        self.df["_doy_cos"] = np.cos(angle_d)

    def _observation(self):
        row = self.df.iloc[self.index]
        obs = np.array([
            float(row[COL_Q_COIL]) / 1000.0,
            float(row[COL_T_SUPPLY]),
            float(row[COL_OUT_TEMP]),
            float(row[COL_OUT_RH]),
            float(row[COL_DIRECT_SOLAR]),
            float(row[COL_DIFFUSE_SOLAR]),
            float(row[COL_OCCUPANCY]),
            float(row[COL_ZONE_TEMP]),
            float(row[COL_ZONE_SETPOINT]),
            float(self.previous_action),
            float(self.previous_delta_t),
            float(self.previous_pump_fraction),
            float(row["_hour_sin"]),
            float(row["_hour_cos"]),
            float(row["_doy_sin"]),
            float(row["_doy_cos"]),
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        if "start_index" in options:
            start = int(options["start_index"])
            max_start = len(self.df) - self.episode_steps - 1
            self.start_index = int(np.clip(start, 0, max_start))
        elif self.random_start:
            max_start = len(self.df) - self.episode_steps - 1
            self.start_index = int(self.np_random.integers(0, max_start + 1))
        else:
            self.start_index = 0

        self.index = self.start_index
        self.steps_taken = 0

        # Start from a neutral controller state.
        self.previous_action = 0.0
        self.previous_delta_t = 0.0
        self.previous_pump_fraction = 0.0

        return self._observation(), {
            "start_index": self.start_index,
            "max_flow_kg_s": self.ets.cfg.max_flow_kg_s,
            "design_delta_t_c": self.ets.cfg.design_delta_t_c,
        }

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))
        row = self.df.iloc[self.index]

        result = self.ets.step(
            q_required_w=float(row[COL_Q_COIL]),
            supply_temp_c=float(row[COL_T_SUPPLY]),
            flow_command=a,
        )

        required = max(result.q_required_w, 0.0)

        unmet_fraction = (
            result.q_unmet_w / required if required > 10.0 else 0.0
        )

        max_pump = max(
            self.ets.cfg.pump_w_per_kg_s * self.ets.cfg.max_flow_kg_s,
            1e-6,
        )
        pump_fraction = result.pump_power_w / max_pump
        action_change = abs(a - self.previous_action)

        components = {
            "pump": self.weights.pump * pump_fraction,
            "unmet": self.weights.unmet * unmet_fraction,
            "low_delta_t": self.weights.low_delta_t * result.low_delta_t_severity,
            "smoothness": self.weights.smoothness * action_change,
        }

        reward = -float(sum(components.values()))

        self.previous_action = a
        self.previous_delta_t = result.delta_t_c
        self.previous_pump_fraction = pump_fraction

        self.index += 1
        self.steps_taken += 1

        terminated = False
        truncated = self.steps_taken >= self.episode_steps

        obs = self._observation()

        info = {
            "flow_command": result.flow_command,
            "flow_kg_s": result.flow_kg_s,
            "q_required_w": result.q_required_w,
            "q_capacity_w": result.q_capacity_w,
            "q_delivered_w": result.q_delivered_w,
            "q_unmet_w": result.q_unmet_w,
            "delivery_fraction": result.delivery_fraction,
            "delta_t_c": result.delta_t_c,
            "supply_temp_c": result.supply_temp_c,
            "return_temp_c": result.return_temp_c,
            "pump_power_w": result.pump_power_w,
            "pump_power_fraction": pump_fraction,
            "low_delta_t": result.low_delta_t,
            "low_delta_t_severity": result.low_delta_t_severity,
            "reward_components": components,
            "dataset_index": self.index - 1,
        }

        return obs, reward, terminated, truncated, info


def run_smoke_test(csv_path: str):
    from gymnasium.utils.env_checker import check_env

    env = ETSGymV2(csv_path, random_start=False)

    # Gymnasium API/space validation.
    check_env(env, skip_render_check=True)

    obs, info = env.reset(seed=42, options={"start_index": 96})
    print("Gymnasium environment check: PASS")
    print(f"Observation shape       : {obs.shape}")
    print(f"Action space            : {env.action_space}")
    print(f"Observation space       : {env.observation_space.shape}")
    print(f"Episode length          : {env.episode_steps} steps")
    print(f"Calibrated max flow     : {env.ets.cfg.max_flow_kg_s:.6f} kg/s")
    print(f"Design Delta-T          : {env.ets.cfg.design_delta_t_c:.3f} C")

    total_reward = 0.0
    delivered = 0.0
    required = 0.0
    pump_wh = 0.0

    # Use minimum-feasible heuristic as a deterministic smoke-test policy.
    for _ in range(env.episode_steps):
        row = env.df.iloc[env.index]
        q = float(row[COL_Q_COIL])
        action = np.array(
            [env.ets.minimum_feasible_command(q)],
            dtype=np.float32
        )
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        delivered += step_info["q_delivered_w"] * 0.25
        required += step_info["q_required_w"] * 0.25
        pump_wh += step_info["pump_power_w"] * 0.25
        if terminated or truncated:
            break

    print("\n24-hour heuristic smoke test")
    print(f"Total reward            : {total_reward:.4f}")
    print(f"Required cooling        : {required/1000.0:.4f} kWh_th")
    print(f"Delivered cooling       : {delivered/1000.0:.4f} kWh_th")
    print(f"Pump electricity        : {pump_wh/1000.0:.4f} kWh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="EnergyPlus eplusout.csv")
    args = parser.parse_args()
    run_smoke_test(args.input_csv)
