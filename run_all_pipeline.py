#!/usr/bin/env python3
"""Run the full pipeline: video -> speed.csv -> trends.csv -> strategy predictions -> plots."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    print("[RUN] " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd))


def ensure_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run full strategy pipeline end to end.")
    parser.add_argument("--video", required=True, help="Path to input MP4 video.")
    parser.add_argument("--speed-csv", default="speed/speed.csv", help="Output speed CSV path.")
    parser.add_argument("--trends-csv", default="speed/trends.csv", help="Output trends CSV path.")
    parser.add_argument("--three-states-plot", default="graphs/three_states.png", help="Output plot from trend detection.")

    # interp.py options
    parser.add_argument("--center-marker", type=int, default=None, help="Marker ID for center tracking.")
    parser.add_argument("--robot-ids", nargs="+", type=int, default=None, help="Marker IDs on the robot.")
    parser.add_argument("--mat-size", nargs=2, type=float, default=[1100, 1700],
                        metavar=("WIDTH_MM", "HEIGHT_MM"), help="Mat size in mm.")
    parser.add_argument("--threads", type=int, default=4, help="Threads for video processing.")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--no-preview", action="store_true", help="Disable video preview window.")

    # plot_three_states options
    parser.add_argument("--window", type=int, default=200, help="Rolling window size in samples.")
    parser.add_argument("--delta-rel", type=float, default=0.02, help="Relative delta threshold.")
    parser.add_argument("--delta-abs", type=float, default=None, help="Absolute delta threshold.")
    parser.add_argument("--min-duration", type=float, default=5.0, help="Minimum segment duration (sec).")

    # plot_strategy options
    parser.add_argument("--dashboard", action="store_true", help="Open the dashboard instead of the timeline.")
    parser.add_argument("--language", default="EN", choices=["EN", "NL"], help="Timeline UI language.")

    # skip flags
    parser.add_argument("--skip-speed", action="store_true", help="Skip video processing step.")
    parser.add_argument("--skip-trends", action="store_true", help="Skip trend detection step.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip strategy analysis step.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plotting step.")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    video_path = Path(args.video)
    speed_csv = repo_root / args.speed_csv
    trends_csv = repo_root / args.trends_csv
    three_states_plot = repo_root / args.three_states_plot

    if not args.skip_speed:
        ensure_exists(video_path, "Input video")
        speed_csv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(repo_root / "speed" / "interp.py"),
            str(video_path),
            "--speed", str(speed_csv),
            "--threads", str(args.threads),
            "--frame-skip", str(args.frame_skip),
            "--mat-size", str(args.mat_size[0]), str(args.mat_size[1]),
        ]
        if args.center_marker is not None:
            cmd += ["--center-marker", str(args.center_marker)]
        if args.robot_ids:
            cmd += ["--robot-ids"] + [str(x) for x in args.robot_ids]
        if args.no_preview:
            cmd += ["--no-preview"]

        run_cmd(cmd, repo_root)

    if not args.skip_trends:
        ensure_exists(speed_csv, "Speed CSV")
        three_states_plot.parent.mkdir(parents=True, exist_ok=True)
        trends_csv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(repo_root / "speed" / "plot_three_states.py"),
            "--file", str(speed_csv),
            "--summary-csv", str(trends_csv),
            "--out", str(three_states_plot),
            "--window", str(args.window),
            "--delta-rel", str(args.delta_rel),
            "--min-duration", str(args.min_duration),
            "--no-show",
        ]
        if args.delta_abs is not None:
            cmd += ["--delta-abs", str(args.delta_abs)]

        run_cmd(cmd, repo_root)

    if not args.skip_analysis:
        logs_dir = repo_root / "logdata"
        if not logs_dir.exists() or not list(logs_dir.glob("*.log")):
            raise FileNotFoundError("No .log files found in logdata/. Place logs before running analysis.")
        ensure_exists(trends_csv, "Trends CSV")
        run_cmd([sys.executable, str(repo_root / "run_full_strategy_analysis.py")], repo_root)

    if not args.skip_plots:
        cmd = [sys.executable, str(repo_root / "plot_strategy.py"), "--language", args.language]
        if args.dashboard:
            cmd += ["--dashboard", "--show-dashboard"]
        run_cmd(cmd, repo_root)


if __name__ == "__main__":
    main()