# Strategy Analysis Pipeline

Analyzes robot video and GUI logs to generate per-student strategy predictions and visualizations.

## Project Structure
```
├── run_all_pipeline.py          # One-command full pipeline
├── run_full_strategy_analysis.py # Strategy classification
├── plot_strategy.py             # Timeline GUI and dashboard
├── log_parser.py                # Log file parsing
├── logs/                        # Input log files (User0.log, etc.)
├── speed/                       # Video processing & speed analysis
├── strategy_classifier/         # Strategy classification module
├── scripts/                     # Utility/analysis scripts
├── tests/                       # Evaluation tests
├── outputs/                     # Generated CSV outputs
└── graphs/                      # Generated plots
```

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start (One Command)
```bash
python run_all_pipeline.py --video path/to/video.mp4 --center-marker 5 --no-preview
```

## Step-by-Step Usage

### 1. Extract speed from video
```bash
python speed/interp.py path/to/video.mp4 --center-marker 5 --speed speed/speed.csv --no-preview
```

### 2. Generate speed trends
```bash
python speed/plot_three_states.py --file speed/speed.csv --summary-csv speed/trends.csv --out graphs/three_states.png --no-show
```

### 3. Run strategy analysis
```bash
python run_full_strategy_analysis.py
```

### 4. View results
```bash
python plot_strategy.py              # Timeline GUI
python plot_strategy.py --dashboard  # Dashboard view
```

## Inputs
- Video file (MP4) with ArUco markers
- Log files in `logs/` directory

## Outputs
- `segment_strategy_with_global_label.csv` - Main predictions (generated in root)
- `graphs/` - Visualization plots

## Adding New Data

### 1. Add log files
Place your log files (e.g., `User0.log`, `User1.log`, `User2.log`, `User3.log`) in the `logs/` folder:
```
logs/
├── User0.log
├── User1.log
├── User2.log
└── User3.log
```

### 2. Process video (if you have one)
Run video processing to generate speed data:
```bash
python speed/interp.py path/to/your_video.mp4 --center-marker 5 --speed speed/speed.csv --no-preview
python speed/plot_three_states.py --file speed/speed.csv --summary-csv speed/trends.csv --no-show
```

### 3. Run analysis
```bash
python run_full_strategy_analysis.py
```

### 4. View results
```bash
python plot_strategy.py
```
Click "Open Dashboard" in the GUI for summary statistics.

## Testing
```bash
python tests/evaluate_classifier.py --labeled-dir logs/Labelled
python tests/evaluate_classifier.py --synthetic
```
