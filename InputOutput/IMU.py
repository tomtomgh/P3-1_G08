# io/imu.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import re
import pandas as pd


# Regex pattern based on your IMU_read script
_IMU_PATTERN = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+).*?"
    r"Gyroscope = SensorInfo { Timestamp = (?P<gyro_ts>\d+), Accuracy = (?P<gyro_acc>\d+), "
    r"Data = <(?P<gyro_x>[-\d\.]+), (?P<gyro_y>[-\d\.]+), (?P<gyro_z>[-\d\.]+)> }"
    r".*?Accelerometer = SensorInfo { Timestamp = (?P<acc_ts>\d+), Accuracy = (?P<acc_acc>\d+), "
    r"Data = <(?P<acc_x>[-\d\.]+), (?P<acc_y>[-\d\.]+), (?P<acc_z>[-\d\.]+)> }"
    r".*?MagneticField = SensorInfo { Timestamp = (?P<mag_ts>\d+), Accuracy = (?P<mag_acc>\d+), "
    r"Data = <(?P<mag_x>[-\d\.]+), (?P<mag_y>[-\d\.]+), (?P<mag_z>[-\d\.]+)> }"
    r".*?Gravity = SensorInfo { Timestamp = (?P<grav_ts>\d+), Accuracy = (?P<grav_acc>\d+), "
    r"Data = <(?P<grav_x>[-\d\.]+), (?P<grav_y>[-\d\.]+), (?P<grav_z>[-\d\.]+)> }"
    r".*?Rotation = SensorInfo { Timestamp = (?P<rot_ts>\d+), Accuracy = (?P<rot_acc>\d+), "
    r"Data = {X:(?P<rot_x>[-\d\.]+) Y:(?P<rot_y>[-\d\.]+) Z:(?P<rot_z>[-\d\.]+) W:(?P<rot_w>[-\d\.]+)} }",
    re.DOTALL,
)


def parse_imu_log(
    input_file: str | Path,
    output_file: str | Path | None = None,
    *,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Parse an IMU log file into a DataFrame.

    This is the modular replacement for IMU_read.py.
    If output_file is given, the parsed data is also written to CSV.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"IMU log file not found: {input_path}")

    text = input_path.read_text(encoding=encoding)

    records = [m.groupdict() for m in _IMU_PATTERN.finditer(text)]
    if not records:
        raise ValueError("No IMU records found. Check the file or regex pattern.")

    df = pd.DataFrame(records)

    # Convert all non-time columns to numeric where possible
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if output_file is not None:
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)

    return df
