"""
Phase 2 - Prepare Reinforcement Learning Dataset

MSc Project:
Reinforcement Learning for HVAC Optimisation using EnergyPlus

Purpose:
    Convert the raw EnergyPlus eplusout.csv output into a clean,
    validated, RL-ready dataset.

Input:
    data/raw/eplusout.csv

Outputs:
    data/processed/rl_dataset.csv
    results/phase2/data_quality_report.txt
    results/phase2/*.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "eplusout.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2"

PROCESSED_FILE = PROCESSED_DIR / "rl_dataset.csv"
REPORT_FILE = RESULTS_DIR / "data_quality_report.txt"


# Create directories if they do not exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXPECTED ENERGYPLUS COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "Date/Time",
    "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
    "Environment:Site Outdoor Air Relative Humidity [%](TimeStep)",
    "Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)",
    "Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)",
    "Environment:Site Day Type Index [](TimeStep)",
    "MAIN ZONE:Zone People Occupant Count [](TimeStep)",
    "MAIN ZONE:Zone Mean Air Temperature [C](TimeStep)",
    "MAIN ZONE:Zone Air Relative Humidity [%](TimeStep)",
    "MAIN ZONE:Zone Air System Sensible Heating Energy [J](TimeStep)",
    "MAIN ZONE:Zone Air System Sensible Cooling Energy [J](TimeStep)",
    "MAIN ZONE:Zone Air System Sensible Heating Rate [W](TimeStep)",
    "MAIN ZONE:Zone Air System Sensible Cooling Rate [W](TimeStep)",
    "MAIN ZONE:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)",
    "MAIN ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)",
    "OCCUPANCY SCHEDULE:Schedule Value [](TimeStep)",
    "Electricity:HVAC [J](TimeStep)",
    "Electricity:Facility [J](TimeStep)",
    "Cooling:Electricity [J](TimeStep)",
]


# ============================================================
# 3. COLUMN RENAMING
# ============================================================

COLUMN_MAP = {
    "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)":
        "outdoor_temp",

    "Environment:Site Outdoor Air Relative Humidity [%](TimeStep)":
        "outdoor_rh",

    "Environment:Site Diffuse Solar Radiation Rate per Area [W/m2](TimeStep)":
        "solar_diffuse",

    "Environment:Site Direct Solar Radiation Rate per Area [W/m2](TimeStep)":
        "solar_direct",

    "Environment:Site Day Type Index [](TimeStep)":
        "day_type",

    "MAIN ZONE:Zone People Occupant Count [](TimeStep)":
        "occupancy",

    "MAIN ZONE:Zone Mean Air Temperature [C](TimeStep)":
        "indoor_temp",

    "MAIN ZONE:Zone Air Relative Humidity [%](TimeStep)":
        "indoor_rh",

    "MAIN ZONE:Zone Air System Sensible Heating Energy [J](TimeStep)":
        "heating_energy_j",

    "MAIN ZONE:Zone Air System Sensible Cooling Energy [J](TimeStep)":
        "cooling_energy_j",

    "MAIN ZONE:Zone Air System Sensible Heating Rate [W](TimeStep)":
        "heating_power_w",

    "MAIN ZONE:Zone Air System Sensible Cooling Rate [W](TimeStep)":
        "cooling_power_w",

    "MAIN ZONE:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)":
        "heating_setpoint",

    "MAIN ZONE:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)":
        "cooling_setpoint",

    "OCCUPANCY SCHEDULE:Schedule Value [](TimeStep)":
        "occupancy_schedule",

    "Electricity:HVAC [J](TimeStep)":
        "hvac_energy_j",

    "Electricity:Facility [J](TimeStep)":
        "facility_energy_j",

    "Cooling:Electricity [J](TimeStep)":
        "cooling_electricity_j",
}


# ============================================================
# 4. LOAD RAW ENERGYPLUS DATA
# ============================================================

print("=" * 70)
print("PHASE 2 - PREPARE RL DATASET")
print("=" * 70)

print("\n[1/10] Loading EnergyPlus CSV...")

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"\nRaw EnergyPlus file not found:\n{RAW_FILE}\n\n"
        "Please copy eplusout.csv to data/raw/"
    )

df = pd.read_csv(RAW_FILE)

# EnergyPlus headers can include accidental leading/trailing whitespace.
# Normalise once so strict column checks remain reliable.
df.columns = df.columns.str.strip()

print(f"Loaded rows    : {len(df):,}")
print(f"Loaded columns : {len(df.columns)}")


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[2/10] Checking EnergyPlus columns...")

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    print("\nERROR: The following expected columns are missing:")

    for col in missing_columns:
        print(f"  - {col}")

    raise ValueError(
        "\nEnergyPlus output does not contain all expected variables."
    )

print("All expected EnergyPlus columns are present.")


# ============================================================
# 6. REMOVE DESIGN-DAY DATA
# ============================================================

print("\n[3/10] Removing EnergyPlus design-day data...")

original_rows = len(df)

# The EnergyPlus CSV contains two design-day periods:
# January 21 and July 21.
#
# These are not part of the actual annual RunPeriod.
#
# The actual annual simulation begins at the first 01/01 record.

date_strings = df["Date/Time"].astype(str).str.strip()

annual_start_index = None

for i, value in enumerate(date_strings):

    if value.startswith("01/01"):
        annual_start_index = i
        break

if annual_start_index is None:
    raise ValueError(
        "Could not locate the beginning of the annual simulation (01/01)."
    )

df = df.iloc[annual_start_index:].copy()

removed_rows = original_rows - len(df)

print(f"Original rows        : {original_rows:,}")
print(f"Design-day rows      : {removed_rows:,}")
print(f"Annual rows remaining: {len(df):,}")

EXPECTED_ANNUAL_ROWS = 365 * 24 * 4

if len(df) != EXPECTED_ANNUAL_ROWS:
    print(
        f"WARNING: Expected {EXPECTED_ANNUAL_ROWS:,} annual rows "
        f"but found {len(df):,}."
    )
else:
    print("Annual timestep count: PASS")


# ============================================================
# 7. PARSE ENERGYPLUS DATE/TIME
# ============================================================

print("\n[4/10] Processing EnergyPlus Date/Time...")

time_values = (
    df["Date/Time"]
    .astype(str)
    .str.strip()
)

# EnergyPlus represents midnight at the end of a day as 24:00.
# Python datetime does not accept 24:00.
#
# Example:
#
# 01/21 24:00
#
# becomes:
#
# 01/22 00:00
#
# We handle this explicitly.

date_parts = time_values.str.split()

dates = date_parts.str[0]
times = date_parts.str[1]

hours = times.str[:2].astype(int)
minutes = times.str[3:5].astype(int)
seconds = times.str[6:8].astype(int)

# Identify EnergyPlus 24:00
midnight_mask = hours == 24

# Convert 24:00 to 00:00 temporarily
normalised_hours = hours.copy()
normalised_hours.loc[midnight_mask] = 0

datetime_strings = (
    dates
    + " "
    + normalised_hours.astype(str).str.zfill(2)
    + ":"
    + minutes.astype(str).str.zfill(2)
    + ":"
    + seconds.astype(str).str.zfill(2)
)

df["datetime"] = pd.to_datetime(
    datetime_strings,
    format="%m/%d %H:%M:%S"
)

# 24:00 represents the end of the current day.
# Add one day to those records.
df.loc[midnight_mask, "datetime"] += pd.Timedelta(days=1)


# ============================================================
# 8. CREATE TIME FEATURES
# ============================================================

print("\n[5/10] Creating time features...")

df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute

df["day_of_year"] = df["datetime"].dt.dayofyear

# Decimal hour
df["hour_decimal"] = (
    df["hour"]
    + df["minute"] / 60.0
)

# Cyclic daily encoding
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour_decimal"] / 24.0
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour_decimal"] / 24.0
)

# Cyclic annual encoding
df["day_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365.0
)

df["day_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365.0
)


# ============================================================
# 9. RENAME ENERGYPLUS COLUMNS
# ============================================================

print("\n[6/10] Renaming EnergyPlus variables...")

df = df.rename(columns=COLUMN_MAP)


# ============================================================
# 10. ENERGY UNIT CONVERSION
# ============================================================

print("\n[7/10] Converting energy from Joules to kWh...")

JOULES_PER_KWH = 3_600_000.0

energy_columns = [
    "heating_energy_j",
    "cooling_energy_j",
    "hvac_energy_j",
    "facility_energy_j",
    "cooling_electricity_j",
]

for column in energy_columns:

    kwh_column = column.replace("_j", "_kwh")

    df[kwh_column] = (
        df[column] / JOULES_PER_KWH
    )


# ============================================================
# 11. CALCULATE ELECTRICAL POWER
# ============================================================

print("\n[8/10] Calculating electrical power...")

# 15-minute timestep = 0.25 hours

TIMESTEP_HOURS = 0.25

df["hvac_power_kw"] = (
    df["hvac_energy_kwh"] / TIMESTEP_HOURS
)

df["facility_power_kw"] = (
    df["facility_energy_kwh"] / TIMESTEP_HOURS
)

df["cooling_electricity_power_kw"] = (
    df["cooling_electricity_kwh"] / TIMESTEP_HOURS
)


# ============================================================
# 12. DATA QUALITY VALIDATION
# ============================================================

print("\n[9/10] Performing data-quality validation...")

report_lines = []

def report(message):
    print(message)
    report_lines.append(message)


report("=" * 70)
report("PHASE 2 - RL DATASET QUALITY REPORT")
report("=" * 70)

report("")
report(f"Source file: {RAW_FILE}")
report(f"Processed file: {PROCESSED_FILE}")

report("")
report("DATASET SIZE")
report("-" * 70)
report(f"Rows: {len(df):,}")
report(f"Columns: {len(df.columns)}")

# Missing values
missing_values = df.isna().sum()
total_missing = missing_values.sum()

report("")
report("MISSING VALUES")
report("-" * 70)
report(f"Total missing values: {total_missing:,}")

if total_missing > 0:

    for column, count in missing_values.items():

        if count > 0:
            report(f"  {column}: {count}")

# Duplicate timestamps
duplicate_timestamps = df["datetime"].duplicated().sum()

report("")
report("DUPLICATE TIMESTAMPS")
report("-" * 70)
report(f"Duplicate timestamps: {duplicate_timestamps}")

# Timestep differences
time_differences = (
    df["datetime"]
    .diff()
    .dropna()
    .dt.total_seconds()
    / 60
)

unexpected_timesteps = (
    time_differences != 15
).sum()

report("")
report("TIMESTEP VALIDATION")
report("-" * 70)
report(f"Expected timestep: 15 minutes")
report(
    f"Unexpected timestep intervals: {unexpected_timesteps}"
)

# Temperature
report("")
report("TEMPERATURE")
report("-" * 70)
report(
    f"Outdoor minimum: {df['outdoor_temp'].min():.2f} °C"
)
report(
    f"Outdoor maximum: {df['outdoor_temp'].max():.2f} °C"
)
report(
    f"Outdoor mean:    {df['outdoor_temp'].mean():.2f} °C"
)

report(
    f"Indoor minimum:  {df['indoor_temp'].min():.2f} °C"
)
report(
    f"Indoor maximum:  {df['indoor_temp'].max():.2f} °C"
)
report(
    f"Indoor mean:     {df['indoor_temp'].mean():.2f} °C"
)

# Humidity
report("")
report("HUMIDITY")
report("-" * 70)

report(
    f"Outdoor RH minimum: {df['outdoor_rh'].min():.2f}%"
)
report(
    f"Outdoor RH maximum: {df['outdoor_rh'].max():.2f}%"
)

report(
    f"Indoor RH minimum:  {df['indoor_rh'].min():.2f}%"
)
report(
    f"Indoor RH maximum:  {df['indoor_rh'].max():.2f}%"
)

# Solar
report("")
report("SOLAR RADIATION")
report("-" * 70)

report(
    f"Diffuse maximum: {df['solar_diffuse'].max():.2f} W/m²"
)
report(
    f"Direct maximum:  {df['solar_direct'].max():.2f} W/m²"
)

# Occupancy
report("")
report("OCCUPANCY")
report("-" * 70)

report(
    f"Maximum occupancy: {df['occupancy'].max():.2f}"
)

report(
    f"Mean occupancy:    {df['occupancy'].mean():.2f}"
)

occupied_timesteps = (
    df["occupancy"] > 0
).sum()

report(
    f"Occupied timesteps: {occupied_timesteps:,}"
)

report(
    f"Occupied percentage: "
    f"{occupied_timesteps / len(df) * 100:.2f}%"
)

# Baseline setpoints
report("")
report("BASELINE SETPOINTS")
report("-" * 70)

report(
    f"Heating setpoint range: "
    f"{df['heating_setpoint'].min():.2f} – "
    f"{df['heating_setpoint'].max():.2f} °C"
)

report(
    f"Cooling setpoint range: "
    f"{df['cooling_setpoint'].min():.2f} – "
    f"{df['cooling_setpoint'].max():.2f} °C"
)

# Annual energy
report("")
report("ANNUAL ENERGY")
report("-" * 70)

report(
    f"Sensible heating energy: "
    f"{df['heating_energy_kwh'].sum():,.2f} kWh"
)

report(
    f"Sensible cooling energy: "
    f"{df['cooling_energy_kwh'].sum():,.2f} kWh"
)

report(
    f"HVAC electricity: "
    f"{df['hvac_energy_kwh'].sum():,.2f} kWh"
)

report(
    f"Facility electricity: "
    f"{df['facility_energy_kwh'].sum():,.2f} kWh"
)

report(
    f"Cooling electricity: "
    f"{df['cooling_electricity_kwh'].sum():,.2f} kWh"
)

# Validation status
validation_passed = True

if len(df) != EXPECTED_ANNUAL_ROWS:
    validation_passed = False

if total_missing > 0:
    validation_passed = False

if duplicate_timestamps > 0:
    validation_passed = False

if unexpected_timesteps > 0:
    validation_passed = False

report("")
report("=" * 70)

if validation_passed:
    report("OVERALL DATA QUALITY: PASS")
else:
    report("OVERALL DATA QUALITY: REVIEW REQUIRED")

report("=" * 70)


# ============================================================
# 13. SAVE QUALITY REPORT
# ============================================================

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("\n".join(report_lines))

print(f"\nQuality report saved to:")
print(REPORT_FILE)


# ============================================================
# 14. CREATE RL DATASET
# ============================================================

print("\n[10/10] Creating final RL dataset...")

# Arrange columns in a logical order.

final_columns = [
    # Time
    "datetime",
    "month",
    "day",
    "hour",
    "minute",
    "day_of_year",
    "hour_decimal",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",

    # Weather
    "outdoor_temp",
    "outdoor_rh",
    "solar_diffuse",
    "solar_direct",
    "day_type",

    # Building
    "occupancy",
    "occupancy_schedule",
    "indoor_temp",
    "indoor_rh",

    # HVAC
    "heating_setpoint",
    "cooling_setpoint",
    "heating_power_w",
    "cooling_power_w",

    # Energy - original Joules
    "heating_energy_j",
    "cooling_energy_j",
    "hvac_energy_j",
    "facility_energy_j",
    "cooling_electricity_j",

    # Energy - kWh
    "heating_energy_kwh",
    "cooling_energy_kwh",
    "hvac_energy_kwh",
    "facility_energy_kwh",
    "cooling_electricity_kwh",

    # Electrical power
    "hvac_power_kw",
    "facility_power_kw",
    "cooling_electricity_power_kw",
]

df_final = df[final_columns].copy()

# Save
df_final.to_csv(
    PROCESSED_FILE,
    index=False
)

print("\nRL dataset saved to:")
print(PROCESSED_FILE)

print("\nFinal dataset shape:")
print(df_final.shape)


# ============================================================
# 15. GENERATE PLOTS
# ============================================================

print("\nGenerating Phase 2 plots...")


# ------------------------------------------------------------
# Plot 1 - Outdoor and Indoor Temperature
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df_final["datetime"],
    df_final["outdoor_temp"],
    label="Outdoor Temperature",
    linewidth=0.8
)

plt.plot(
    df_final["datetime"],
    df_final["indoor_temp"],
    label="Indoor Temperature",
    linewidth=0.8
)

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Outdoor vs Indoor Temperature - Annual Baseline")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "01_temperature_profile.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Plot 2 - Solar Radiation
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df_final["datetime"],
    df_final["solar_direct"],
    label="Direct Solar"
)

plt.plot(
    df_final["datetime"],
    df_final["solar_diffuse"],
    label="Diffuse Solar"
)

plt.xlabel("Date")
plt.ylabel("Solar Radiation (W/m²)")
plt.title("Annual Solar Radiation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "02_solar_radiation.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Plot 3 - Occupancy
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df_final["datetime"],
    df_final["occupancy"],
    linewidth=0.8
)

plt.xlabel("Date")
plt.ylabel("Occupants")
plt.title("Annual Occupancy Profile")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "03_occupancy.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Plot 4 - HVAC Power
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df_final["datetime"],
    df_final["hvac_power_kw"],
    label="HVAC Electrical Power"
)

plt.plot(
    df_final["datetime"],
    df_final["cooling_electricity_power_kw"],
    label="Cooling Electrical Power"
)

plt.xlabel("Date")
plt.ylabel("Power (kW)")
plt.title("Annual HVAC Electrical Power")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "04_hvac_power.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Plot 5 - Indoor Temperature vs Setpoint
# ------------------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df_final["datetime"],
    df_final["indoor_temp"],
    label="Indoor Temperature",
    linewidth=0.8
)

plt.plot(
    df_final["datetime"],
    df_final["cooling_setpoint"],
    label="Cooling Setpoint",
    linestyle="--"
)

plt.plot(
    df_final["datetime"],
    df_final["heating_setpoint"],
    label="Heating Setpoint",
    linestyle="--"
)

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Indoor Temperature and Baseline Thermostat Setpoints")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "05_temperature_vs_setpoints.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# Plot 6 - Monthly HVAC Energy
# ------------------------------------------------------------

monthly_energy = (
    df_final
    .groupby("month")["hvac_energy_kwh"]
    .sum()
)

plt.figure(figsize=(10, 6))

plt.bar(
    monthly_energy.index,
    monthly_energy.values
)

plt.xlabel("Month")
plt.ylabel("HVAC Electricity (kWh)")
plt.title("Monthly HVAC Electricity Consumption - Baseline")
plt.xticks(range(1, 13))
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "06_monthly_hvac_energy.png",
    dpi=150
)

plt.close()


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 COMPLETE")
print("=" * 70)

print(f"""
Processed dataset:
    {PROCESSED_FILE}

Quality report:
    {REPORT_FILE}

Rows:
    {len(df_final):,}

Columns:
    {len(df_final.columns)}

Expected annual rows:
    {EXPECTED_ANNUAL_ROWS:,}

HVAC electricity:
    {df_final["hvac_energy_kwh"].sum():,.2f} kWh

Facility electricity:
    {df_final["facility_energy_kwh"].sum():,.2f} kWh

Sensible cooling:
    {df_final["cooling_energy_kwh"].sum():,.2f} kWh

Sensible heating:
    {df_final["heating_energy_kwh"].sum():,.2f} kWh
""")

print("=" * 70)
print("Ready for Phase 3 - Gymnasium RL Environment")
print("=" * 70)