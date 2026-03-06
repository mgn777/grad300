"""Command‑line interface for the GRAD‑300 pipeline."""

import argparse
from .pipeline import GradPipeline


def main():
    parser = argparse.ArgumentParser(description="Run GRAD-300 processing pipeline")
    parser.add_argument('obs_date', help="Observation date folder (e.g. 20250204)")
    parser.add_argument('--target', help="Restrict to a single target", default=None)
    parser.add_argument('--base', help="Base path to GRAD-300 data", default="./GRAD-300/GRAD-300")
    parser.add_argument('--window', type=int, default=7, help="Savgol window for TPI")
    parser.add_argument('--polyorder', type=int, default=2, help="Savgol polynomial order")
    parser.add_argument('--start-ignore', type=float, default=1, help="Seconds to ignore at start")
    parser.add_argument('--end-ignore', type=float, default=0, help="Seconds to ignore at end")
    args = parser.parse_args()

    pipe = GradPipeline(base_path=args.base)
    pipe.run(args.obs_date, target=args.target,
             window=args.window, polyorder=args.polyorder,
             start_ignore=args.start_ignore, end_ignore=args.end_ignore)


if __name__ == '__main__':
    main()
