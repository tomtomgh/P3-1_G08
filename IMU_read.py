import re
import pandas as pd

# --- CONFIG ---
input_file = "imu_log.txt"   # your log file name
output_file = "imu_data.csv" # name of the csv to create

# --- REGEX PATTERN ---
sensor_pattern = re.compile(
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+).*?"
    r"Gyroscope = SensorInfo { Timestamp = (?P<gyro_ts>\d+), Accuracy = (?P<gyro_acc>\d+), Data = <(?P<gyro_x>[-\d\.]+), (?P<gyro_y>[-\d\.]+), (?P<gyro_z>[-\d\.]+)> }"
    r".*?Accelerometer = SensorInfo { Timestamp = (?P<acc_ts>\d+), Accuracy = (?P<acc_acc>\d+), Data = <(?P<acc_x>[-\d\.]+), (?P<acc_y>[-\d\.]+), (?P<acc_z>[-\d\.]+)> }"
    r".*?MagneticField = SensorInfo { Timestamp = (?P<mag_ts>\d+), Accuracy = (?P<mag_acc>\d+), Data = <(?P<mag_x>[-\d\.]+), (?P<mag_y>[-\d\.]+), (?P<mag_z>[-\d\.]+)> }"
    r".*?Gravity = SensorInfo { Timestamp = (?P<grav_ts>\d+), Accuracy = (?P<grav_acc>\d+), Data = <(?P<grav_x>[-\d\.]+), (?P<grav_y>[-\d\.]+), (?P<grav_z>[-\d\.]+)> }"
    r".*?Rotation = SensorInfo { Timestamp = (?P<rot_ts>\d+), Accuracy = (?P<rot_acc>\d+), Data = {X:(?P<rot_x>[-\d\.]+) Y:(?P<rot_y>[-\d\.]+) Z:(?P<rot_z>[-\d\.]+) W:(?P<rot_w>[-\d\.]+)} }",
    re.DOTALL
)

# --- READ FILE ---
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# --- PARSE ---
records = [m.groupdict() for m in sensor_pattern.finditer(text)]

if not records:
    print("⚠️ No data found — check your file formatting or regex pattern.")
    exit()

# --- CONVERT TO DATAFRAME ---
df = pd.DataFrame(records)

# Convert numeric columns
for col in df.columns:
    if col != "time":
        df[col] = pd.to_numeric(df[col], errors="ignore")

# --- SAVE TO CSV ---
df.to_csv(output_file, index=False)
print(f"✅ Done! Parsed {len(df)} entries and saved to '{output_file}'.")