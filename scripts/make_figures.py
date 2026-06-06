#!/usr/bin/env python
from __future__ import annotations

import argparse

from ads_dinov3.config import load_config
from ads_dinov3.runlog import append_run_log, utc_now
from ads_dinov3.visualize import make_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report figures from retrieval outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["sanity", "full"], default="full")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    start = utc_now()
    status = "success"
    outputs = []
    try:
        outputs = make_all_figures(cfg, args.mode)
    except Exception as exc:
        status = "failed"
        append_run_log(cfg, start=start, end=utc_now(), run_name="make_figures", mode=args.mode, status=status, output_path=";".join(map(str, outputs)), notes=str(exc))
        raise
    append_run_log(cfg, start=start, end=utc_now(), run_name="make_figures", mode=args.mode, status=status, output_path=";".join(map(str, outputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
