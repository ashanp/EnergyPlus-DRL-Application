"""
ETS Physics Module - V0 Validation
==================================
Purpose:
    Validate the chilled-water-side physics exported by EnergyPlus before
    introducing Gymnasium or reinforcement learning.

Checks:
    1. Delta-T = T_return - T_supply
    2. Q_water = m_dot * Cp * Delta-T
    3. Compare Q_water against EnergyPlus Cooling Coil Total Cooling Rate
    4. Flag low-Delta-T operation relative to the current IDF design Delta-T
    5. Preserve pump-flow and pump-power variables for later control work

Usage:
    python ets_model_v0.py "eplusout(3).csv"

Optional:
    python ets_model_v0.py "eplusout(3).csv" --design-delta-t 4.0
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

CP_WATER = 4180.0  # J/(kg.K)

COL_Q_COIL = (
    "MAIN ZONE FCU COOLING COIL:Cooling Coil Total Cooling Rate [W](TimeStep)"
)
COL_T_SUPPLY = (
    "MAIN ZONE FCU COOLING COIL WATER INLET:System Node Temperature [C](TimeStep)"
)
COL_T_RETURN = (
    "MAIN ZONE FCU COOLING COIL WATER OUTLET:System Node Temperature [C](TimeStep)"
)
COL_COIL_FLOW = (
    "MAIN ZONE FCU COOLING COIL WATER INLET:System Node Mass Flow Rate [kg/s](TimeStep)"
)
COL_PUMP_FLOW = "CHW CIRC PUMP:Pump Mass Flow Rate [kg/s](TimeStep)"
COL_PUMP_POWER = "CHW CIRC PUMP:Pump Electricity Rate [W](TimeStep)"
COL_PLANT_SUPPLY = "CHW SUPPLY OUTLET NODE:System Node Temperature [C](TimeStep)"
COL_PLANT_RETURN = "CHW DEMAND OUTLET NODE:System Node Temperature [C](TimeStep)"
COL_ZONE_TEMP = "MAIN ZONE:Zone Mean Air Temperature [C](TimeStep)"
COL_ZONE_SETPOINT = (
    "MAIN ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)"
)

REQUIRED_COLUMNS = [
    "Date/Time",
    COL_Q_COIL,
    COL_T_SUPPLY,
    COL_T_RETURN,
    COL_COIL_FLOW,
    COL_PUMP_FLOW,
    COL_PUMP_POWER,
    COL_PLANT_SUPPLY,
    COL_PLANT_RETURN,
    COL_ZONE_TEMP,
    COL_ZONE_SETPOINT,
]


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Required EnergyPlus columns are missing:\n  - "
            + "\n  - ".join(missing)
        )


def build_ets_v0(
    input_csv: str,
    design_delta_t: float = 4.0,
    flow_epsilon: float = 1e-6,
    cooling_epsilon_w: float = 10.0,
):
    input_path = Path(input_csv)
    df = pd.read_csv(input_path)
    validate_columns(df)

    out = pd.DataFrame()
    out["Date/Time"] = df["Date/Time"]

    # Raw EnergyPlus states
    out["q_coil_w"] = df[COL_Q_COIL]
    out["chw_supply_temp_c"] = df[COL_T_SUPPLY]
    out["chw_return_temp_c"] = df[COL_T_RETURN]
    out["coil_flow_kg_s"] = df[COL_COIL_FLOW]
    out["pump_flow_kg_s"] = df[COL_PUMP_FLOW]
    out["pump_power_w"] = df[COL_PUMP_POWER]
    out["plant_supply_temp_c"] = df[COL_PLANT_SUPPLY]
    out["plant_return_temp_c"] = df[COL_PLANT_RETURN]
    out["zone_temp_c"] = df[COL_ZONE_TEMP]
    out["zone_cooling_setpoint_c"] = df[COL_ZONE_SETPOINT]

    # ETS physics
    out["delta_t_c"] = (
        out["chw_return_temp_c"] - out["chw_supply_temp_c"]
    )

    out["q_water_w"] = (
        out["coil_flow_kg_s"] * CP_WATER * out["delta_t_c"]
    )

    out["q_balance_abs_error_w"] = (
        out["q_water_w"] - out["q_coil_w"]
    ).abs()

    out["q_balance_rel_error_pct"] = np.where(
        out["q_coil_w"].abs() > cooling_epsilon_w,
        100.0
        * out["q_balance_abs_error_w"]
        / out["q_coil_w"].abs(),
        np.nan,
    )

    # Active cooling: do not classify off-cycle/no-flow periods as low-Delta-T
    out["active_cooling"] = (
        (out["coil_flow_kg_s"] > flow_epsilon)
        & (out["q_coil_w"] > cooling_epsilon_w)
    )

    out["low_delta_t"] = (
        out["active_cooling"] & (out["delta_t_c"] < design_delta_t)
    )

    # Smooth severity for later RL reward design
    out["low_delta_t_severity"] = np.where(
        out["active_cooling"],
        np.maximum(
            0.0, (design_delta_t - out["delta_t_c"]) / design_delta_t
        ),
        0.0,
    )

    # Helpful engineering diagnostics
    out["pump_minus_coil_flow_kg_s"] = (
        out["pump_flow_kg_s"] - out["coil_flow_kg_s"]
    )

    out["zone_temp_error_c"] = (
        out["zone_temp_c"] - out["zone_cooling_setpoint_c"]
    )

    active = out[out["active_cooling"]].copy()

    summary = {
        "input_rows": len(out),
        "active_cooling_rows": len(active),
        "design_delta_t_c": design_delta_t,
        "mean_q_coil_kw": active["q_coil_w"].mean() / 1000.0,
        "max_q_coil_kw": active["q_coil_w"].max() / 1000.0,
        "mean_delta_t_c": active["delta_t_c"].mean(),
        "median_delta_t_c": active["delta_t_c"].median(),
        "min_delta_t_c": active["delta_t_c"].min(),
        "max_delta_t_c": active["delta_t_c"].max(),
        "mean_coil_flow_kg_s": active["coil_flow_kg_s"].mean(),
        "max_coil_flow_kg_s": active["coil_flow_kg_s"].max(),
        "mean_pump_power_w": active["pump_power_w"].mean(),
        "max_pump_power_w": active["pump_power_w"].max(),
        "water_side_q_mae_w": active["q_balance_abs_error_w"].mean(),
        "median_water_side_q_error_pct": active[
            "q_balance_rel_error_pct"
        ].median(),
        "mean_water_side_q_error_pct": active[
            "q_balance_rel_error_pct"
        ].mean(),
        "q_water_vs_q_coil_correlation": active[
            ["q_coil_w", "q_water_w"]
        ].corr().iloc[0, 1],
        "low_delta_t_fraction_pct": 100.0 * active["low_delta_t"].mean(),
    }

    output_csv = input_path.with_name("ets_v0_validated.csv")
    summary_txt = input_path.with_name("ets_v0_summary.txt")

    out.to_csv(output_csv, index=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("ETS Physics Module V0 - Validation Summary\n")
        f.write("=" * 44 + "\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")

    return out, summary, output_csv, summary_txt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="EnergyPlus eplusout.csv")
    parser.add_argument(
        "--design-delta-t",
        type=float,
        default=4.0,
        help="Current model design chilled-water Delta-T in degC",
    )
    args = parser.parse_args()

    _, summary, output_csv, summary_txt = build_ets_v0(
        args.input_csv, design_delta_t=args.design_delta_t
    )

    print("\nETS Physics Module V0 - PASS\n")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:38s}: {value:.4f}")
        else:
            print(f"{key:38s}: {value}")

    print(f"\nValidated timestep dataset: {output_csv}")
    print(f"Summary report:             {summary_txt}")


if __name__ == "__main__":
    main()
