import re
import glob
import pandas as pd

# Step 1: Load all user log files
files = sorted(glob.glob("User*.log"))

# Step 2: Regex for parsing log lines
pattern = re.compile(
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d+).*?(?:User\s*(?P<user_id>\d+))?\s*sets\s+(?P<param>[a-zA-Z ]+)\s+to\s+(?P<value>[-\d\.]+)',
    re.IGNORECASE
)

records = []

# Step 3: Extract from all logs
for file in files:
    user_hint = re.search(r'User(\d+)', file)
    default_user = user_hint.group(1) if user_hint else None
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                user = m.group('user_id') or default_user
                records.append({
                    'time': m.group('time'),
                    'user': int(user),
                    'param': m.group('param').strip(),
                    'value': float(m.group('value'))
                })

# Step 4: Organize in DataFrame
df = pd.DataFrame(records)
df = df.sort_values(by=['user', 'param', 'time']).reset_index(drop=True)

# Step 5: Detect parameter change segments
results = []
for (user, param), group in df.groupby(['user', 'param']):
    group = group.reset_index(drop=True)
    for i in range(1, len(group)):
        prev = group.loc[i-1]
        curr = group.loc[i]
        if prev['value'] != curr['value']:
            results.append({
                'user': user,
                'param': param,
                'start_time': prev['time'],
                'end_time': curr['time'],
                'from_value': prev['value'],
                'to_value': curr['value']
            })

summary = pd.DataFrame(results)

# Step 6: Save as CSV
summary.to_csv("parameter_changes_summary.csv", index=False)
print("✅ Saved parameter change summary to parameter_changes_summary.csv")

# Optional: preview few rows
print(summary.head(10))
