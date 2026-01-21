# Strategy Analysis Pipeline

This repo turns robot video plus GUI logs into per-student strategy predictions and interactive plots.

Pipeline overview:
1) speed/interp.py: track robot markers in video and write a speed time series.
2) speed/plot_three_states.py: convert speed series into trend segments (dull/increasing/decreasing) and save to speed/trends.csv.
3) run_full_strategy_analysis.py: parse logs and assign strategies per segment.
4) plot_strategy.py: final timeline GUI and dashboard plots for students.

## Requirements
- Python 3.10+ (tested)
- Packages: opencv-python, numpy, pandas, matplotlib, scikit-learn

Install example:
```
pip install opencv-python numpy pandas matplotlib scikit-learn
```

## Inputs
- Video file (MP4) with ArUco markers.
- Logs in `logs/` (one or more .log files).

## Step-by-step run
1) Generate speed CSV from video:
```
python speed/interp.py path\to\video.mp4 --center-marker 5 --speed speed/speed.csv --no-preview
```
Optional flags for `interp.py`:
- `--robot-ids 4 5 24 47` to track specific markers
- `--mat-size 1100 1700` for mat dimensions (mm)
- `--threads 8 --frame-skip 2` for faster processing

2) Create speed trends (required by the classifier):
```
python speed/plot_three_states.py --file speed/speed.csv --summary-csv speed/trends.csv --out graphs/three_states.png --no-show
```

3) Run the full strategy analysis:
```
python run_full_strategy_analysis.py
```

4) Show the final GUI for students 
```
python plot_strategy.py
```
- Use the Open Dashboard button inside the timeline GUI.
- Or open the dashboard directly:
```
python plot_strategy.py --dashboard --show-dashboard
```

## One-command pipeline
Use the helper script below to run all steps in one command:
```
python run_all_pipeline.py --video path\to\video.mp4 --center-marker 5 --no-preview
```
Common options:
- `--robot-ids 4 5 24 47`
- `--frame-skip 2 --threads 8`
- `--dashboard` (open the dashboard instead of the timeline)

## Outputs
- `speed/speed.csv` (speed time series)
- `speed/trends.csv` (trend segments for segmentation)
- `segment_strategy_with_global_label.csv` (main predictions file)
- `segment_strategy_predictions.json` (JSON export)
- `graphs/strategy_usage_report.png` (bar plots)
- `graphs/strategy_usage_radar.png` (radar plots)
- `graphs/three_states.png` (speed trend visualization)

## Troubleshooting
- If `run_full_strategy_analysis.py` fails with "speed/trends.csv not found", run step 2 first.
- If `plot_strategy.py` opens with no data, check that `segment_strategy_with_global_label.csv` exists and is non-empty.
- If `interp.py` fails to detect markers, confirm marker IDs and lighting and try `--no-preview` for speed.

## Current dominance calculation (notes)
- For shared parameters (like frequency), the system tracks which user "owns" each time point in user_at_time.
- When a user changes the frequency, they become the owner from that moment until another user changes it.
- Dominance is calculated by counting how many timeline points (at TIME_RESOLUTION = 0.1s intervals) belong to each user.
- This is converted to seconds: time_controlled = control_points * TIME_RESOLUTION.

## How to test classifier accuracy
1) Evaluate existing labeled data:
```
python tests/evaluate_classifier.py --labeled-dir logs/Labelled
```
2) Run synthetic test:
```
python tests/evaluate_classifier.py --synthetic
```
3) Create labels for new data:
```
python tests/ground_truth_format.py --create-template path\to\ground_truth_labels.json
```
