"""
ETS Physics Module - V1 Controllable Offline Model
==================================================

Purpose
-------
Create a controllable, physics-constrained ETS model using the validated
EnergyPlus chilled-water baseline as its engineering reference.

V1 does NOT control EnergyPlus yet. It performs counterfactual offline
experiments:

    EnergyPlus cooling requirement -> flow command -> ETS physics response

Control action
--------------
    flow_command in [0, 1]

where:
    0.0 = zero secondary CHW flow
    1.0 = calibrated maximum secondary CHW flow observed in EnergyPlus

Physics
-------
Maximum useful water-side heat transfer is constrained by:

    Q_capacity = m_dot * Cp * DeltaT_design

Then:

    Q_delivered = min(Q_required, Q_capacity)
    DeltaT       = Q_delivered / (m_dot * Cp)
    T_return     = T_supply + DeltaT
    Q_unmet      = Q_required - Q_delivered

This prevents the controller from obtaining unrealistically large Delta-T by
simply reducing flow. If flow is too low, cooling becomes unmet.

Pump power
----------
The current EnergyPlus Pump:VariableSpeed baseline was observed to behave
linearly with mass flow. V1 therefore calibrates:

    PumpPower = k_flow * m_dot

directly from the supplied EnergyPlus CSV rather than inventing a synthetic
pump curve.

Usage
-----
    python ets_model_v1.py "eplusout(3).csv"

Optional:
    python ets_model_v1.py "eplusout(3).csv" --action 0.65
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


CP_WATER = 4180.0  # J/(kg.K)

COL_Q_COIL = "MAIN ZONE FCU COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)"
COL_T_SUPPLY = "MAIN ZONE FCU COOLING COIL WATER INLET:System Node Temperature [C](TimeStep)"
COL_T_RETURN = "MAIN ZONE FCU COOLING COIL WATER OUTLET:System Node Temperature [C](TimeStep)"
COL_COIL_FLOW = "MAIN ZONE FCU COOLING COIL WATER INLET:System Node Mass Flow Rate [kg/s](TimeStep)"
COL_PUMP_FLOW = "CHW CIRC PUMP:Pump Mass Flow Rate [kg/s](TimeStep)"
COL_PUMP_POWER = "CHW CIRC PUMP:Pump Electricity Rate [W](TimeStep)"
COL_ZONE_TEMP = "MAIN ZONE:Zone Mean Air Temperature [C](TimeStep)"
COL_ZONE_SETPOINT = "MAIN ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"


@dataclass
class ETSConfig:
    cp_water: float = CP_WATER
    design_delta_t_c: float = 4.0
    max_flow_kg_s: float = 0.0
    pump_w_per_kg_s: float = 0.0
    low_delta_t_threshold_c: float = 4.0
    low_delta_t_tolerance_c: float = 0.01


@dataclass
class ETSResult:
    flow_command: float
    flow_kg_s: float
    q_required_w: float
    q_capacity_w: float
    q_delivered_w: float
    q_unmet_w: float
    delivery_fraction: float
    supply_temp_c: float
    return_temp_c: float
    delta_t_c: float
    pump_power_w: float
    low_delta_t: bool
    low_delta_t_severity: float


class ETSModelV1:
    def __init__(self, config: ETSConfig):
        self.cfg = config

    def minimum_feasible_flow(self, q_required_w: float) -> float:
        """Minimum flow required to meet load at the design Delta-T."""
        if q_required_w <= 0:
            return 0.0
        return q_required_w / (
            self.cfg.cp_water * self.cfg.design_delta_t_c
        )

    def minimum_feasible_command(self, q_required_w: float) -> float:
        if self.cfg.max_flow_kg_s <= 0:
            return 0.0
        return float(np.clip(
            self.minimum_feasible_flow(q_required_w) /
            self.cfg.max_flow_kg_s,
            0.0,
            1.0,
        ))

    def step(
        self,
        q_required_w: float,
        supply_temp_c: float,
        flow_command: float,
    ) -> ETSResult:

        action = float(np.clip(flow_command, 0.0, 1.0))
        q_required = max(float(q_required_w), 0.0)
        flow = action * self.cfg.max_flow_kg_s

        if flow <= 1e-9 or q_required <= 1e-9:
            q_capacity = 0.0 if flow <= 1e-9 else (
                flow * self.cfg.cp_water * self.cfg.design_delta_t_c
            )
            q_delivered = 0.0 if q_required <= 1e-9 else min(q_required, q_capacity)
            delta_t = 0.0 if flow <= 1e-9 else (
                q_delivered / (flow * self.cfg.cp_water)
            )
        else:
            q_capacity = (
                flow * self.cfg.cp_water * self.cfg.design_delta_t_c
            )
            q_delivered = min(q_required, q_capacity)
            delta_t = q_delivered / (flow * self.cfg.cp_water)

        q_unmet = max(q_required - q_delivered, 0.0)
        delivery_fraction = (
            1.0 if q_required <= 1e-9 else q_delivered / q_required
        )

        return_temp = supply_temp_c + delta_t
        pump_power = self.cfg.pump_w_per_kg_s * flow

        active = q_required > 10.0 and flow > 1e-9
        low_dt = active and delta_t < (self.cfg.low_delta_t_threshold_c - self.cfg.low_delta_t_tolerance_c)

        severity = 0.0
        if active and self.cfg.low_delta_t_threshold_c > 0:
            severity = max(
                0.0,
                (self.cfg.low_delta_t_threshold_c - delta_t)
                / self.cfg.low_delta_t_threshold_c,
            )

        return ETSResult(
            flow_command=action,
            flow_kg_s=flow,
            q_required_w=q_required,
            q_capacity_w=q_capacity,
            q_delivered_w=q_delivered,
            q_unmet_w=q_unmet,
            delivery_fraction=delivery_fraction,
            supply_temp_c=float(supply_temp_c),
            return_temp_c=float(return_temp),
            delta_t_c=float(delta_t),
            pump_power_w=float(pump_power),
            low_delta_t=bool(low_dt),
            low_delta_t_severity=float(severity),
        )


def calibrate_from_energyplus(df: pd.DataFrame, design_delta_t_c: float) -> ETSConfig:
    required = [
        COL_Q_COIL, COL_T_SUPPLY, COL_T_RETURN, COL_COIL_FLOW,
        COL_PUMP_FLOW, COL_PUMP_POWER
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing EnergyPlus columns:\n  - " + "\n  - ".join(missing))

    active = df[
        (df[COL_PUMP_FLOW] > 1e-6)
        & (df[COL_PUMP_POWER] > 1e-6)
        & (df[COL_Q_COIL] > 10.0)
    ].copy()

    max_flow = float(active[COL_PUMP_FLOW].max())

    # Least-squares fit through origin: P = k*m_dot
    flow = active[COL_PUMP_FLOW].to_numpy(float)
    power = active[COL_PUMP_POWER].to_numpy(float)
    k_flow = float(np.dot(flow, power) / np.dot(flow, flow))

    return ETSConfig(
        cp_water=CP_WATER,
        design_delta_t_c=design_delta_t_c,
        max_flow_kg_s=max_flow,
        pump_w_per_kg_s=k_flow,
        low_delta_t_threshold_c=design_delta_t_c,
        low_delta_t_tolerance_c=0.01,
    )


def replay_with_action(
    df: pd.DataFrame,
    model: ETSModelV1,
    action_mode: str,
    fixed_action: float,
) -> pd.DataFrame:

    rows = []

    for i, row in df.iterrows():
        q_req = float(row[COL_Q_COIL])
        supply = float(row[COL_T_SUPPLY])

        if action_mode == "baseline":
            action = (
                float(row[COL_COIL_FLOW]) / model.cfg.max_flow_kg_s
                if model.cfg.max_flow_kg_s > 0 else 0.0
            )
        elif action_mode == "minimum_feasible":
            action = model.minimum_feasible_command(q_req)
        else:
            action = fixed_action

        result = model.step(q_req, supply, action)

        rows.append({
            "Date/Time": row["Date/Time"],
            "action_mode": action_mode,
            "flow_command": result.flow_command,
            "q_required_w": result.q_required_w,
            "q_capacity_w": result.q_capacity_w,
            "q_delivered_w": result.q_delivered_w,
            "q_unmet_w": result.q_unmet_w,
            "delivery_fraction": result.delivery_fraction,
            "supply_temp_c": result.supply_temp_c,
            "return_temp_c": result.return_temp_c,
            "delta_t_c": result.delta_t_c,
            "flow_kg_s": result.flow_kg_s,
            "pump_power_w": result.pump_power_w,
            "low_delta_t": result.low_delta_t,
            "low_delta_t_severity": result.low_delta_t_severity,
            "baseline_flow_kg_s": float(row[COL_COIL_FLOW]),
            "baseline_pump_power_w": float(row[COL_PUMP_POWER]),
            "baseline_return_temp_c": float(row[COL_T_RETURN]),
            "zone_temp_c": float(row[COL_ZONE_TEMP]) if COL_ZONE_TEMP in df else np.nan,
            "zone_setpoint_c": float(row[COL_ZONE_SETPOINT]) if COL_ZONE_SETPOINT in df else np.nan,
        })

    return pd.DataFrame(rows)


def summarise(result_df: pd.DataFrame) -> dict:
    active = result_df[result_df["q_required_w"] > 10.0]
    total_required_kwh = active["q_required_w"].sum() * 0.25 / 1000.0
    total_delivered_kwh = active["q_delivered_w"].sum() * 0.25 / 1000.0
    total_unmet_kwh = active["q_unmet_w"].sum() * 0.25 / 1000.0
    pump_kwh = active["pump_power_w"].sum() * 0.25 / 1000.0

    return {
        "active_timesteps": len(active),
        "mean_flow_command": active["flow_command"].mean(),
        "mean_flow_kg_s": active["flow_kg_s"].mean(),
        "mean_delta_t_c": active["delta_t_c"].mean(),
        "median_delta_t_c": active["delta_t_c"].median(),
        "mean_pump_power_w": active["pump_power_w"].mean(),
        "pump_energy_kwh": pump_kwh,
        "required_cooling_kwh_th": total_required_kwh,
        "delivered_cooling_kwh_th": total_delivered_kwh,
        "unmet_cooling_kwh_th": total_unmet_kwh,
        "delivery_fraction_pct": (
            100.0 * total_delivered_kwh / total_required_kwh
            if total_required_kwh > 0 else 100.0
        ),
        "low_delta_t_fraction_pct": 100.0 * active["low_delta_t"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument(
        "--design-delta-t", type=float, default=4.0,
        help="Current EnergyPlus design Delta-T in degC"
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "minimum_feasible", "fixed"],
        default="baseline",
    )
    parser.add_argument(
        "--action", type=float, default=0.65,
        help="Normalised flow command used with --mode fixed"
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    df = pd.read_csv(input_path)

    config = calibrate_from_energyplus(df, args.design_delta_t)
    model = ETSModelV1(config)

    result = replay_with_action(
        df, model, action_mode=args.mode, fixed_action=args.action
    )
    summary = summarise(result)

    suffix = args.mode
    if args.mode == "fixed":
        suffix += f"_{args.action:.2f}"

    output_csv = input_path.with_name(f"ets_v1_{suffix}.csv")
    summary_txt = input_path.with_name(f"ets_v1_{suffix}_summary.txt")

    result.to_csv(output_csv, index=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("ETS Physics Module V1\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Design Delta-T: {config.design_delta_t_c:.4f} C\n")
        f.write(f"Calibrated max flow: {config.max_flow_kg_s:.6f} kg/s\n")
        f.write(f"Calibrated pump coefficient: {config.pump_w_per_kg_s:.6f} W/(kg/s)\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value:.6f}\n" if isinstance(value, float)
                    else f"{key}: {value}\n")

    print("\nETS Physics Module V1")
    print("=" * 50)
    print(f"Mode                     : {args.mode}")
    print(f"Design Delta-T           : {config.design_delta_t_c:.3f} C")
    print(f"Calibrated max flow      : {config.max_flow_kg_s:.6f} kg/s")
    print(f"Pump coefficient         : {config.pump_w_per_kg_s:.3f} W/(kg/s)")
    print()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")

    print(f"\nOutput: {output_csv}")
    print(f"Summary: {summary_txt}")


if __name__ == "__main__":
    main()
