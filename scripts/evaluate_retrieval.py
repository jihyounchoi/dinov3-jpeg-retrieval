#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ads_dinov3.config import ensure_dir, feature_path, load_config, model_keys, qualities_for_mode, resolve_path
from ads_dinov3.features import load_feature_npz
from ads_dinov3.retrieval import evaluate_pair, mine_failure_cases, write_metric_rows
from ads_dinov3.runlog import append_run_log, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics from extracted features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["sanity", "full"], default="full")
    parser.add_argument("--model", default="all")
    parser.add_argument("--qualities", nargs="*", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    start = utc_now()
    status = "success"
    tables_dir = resolve_path(cfg, cfg["retrieval"]["tables_dir"])
    ensure_dir(tables_dir)
    metrics_path = tables_dir / f"retrieval_metrics_{args.mode}.csv"
    stability_path = tables_dir / f"feature_stability_{args.mode}.csv"
    failure_path = tables_dir / f"failure_cases_{args.mode}.csv"
    topk_dir = tables_dir / "topk" / args.mode
    ensure_dir(topk_dir)

    if not args.append:
        for path in [metrics_path, stability_path, failure_path]:
            if path.exists():
                path.unlink()

    all_topk_rows: list[dict[str, object]] = []
    try:
        models = model_keys(cfg) if args.model == "all" else [args.model]
        qualities = args.qualities if args.qualities is not None else qualities_for_mode(cfg, args.mode)
        bootstrap_B = int(cfg["retrieval"].get("bootstrap_B", 0))
        bootstrap_alpha = float(cfg["retrieval"].get("bootstrap_alpha", 0.05))
        bootstrap_seed = int(cfg["runtime"]["seed"])
        rank_shift_threshold = int(cfg["retrieval"].get("failure_rank_shift_threshold", 5))
        for model_key in models:
            gallery = load_feature_npz(feature_path(cfg, args.mode, model_key, "clean"))
            for quality in qualities:
                query = load_feature_npz(feature_path(cfg, args.mode, model_key, "query", quality))
                metric_rows, stability_rows, topk_rows = evaluate_pair(
                    model_key=model_key,
                    quality=quality,
                    gallery=gallery,
                    query=query,
                    ks=[int(k) for k in cfg["retrieval"]["ks"]],
                    topk_for_export=int(cfg["retrieval"]["topk_for_export"]),
                    topk_path=topk_dir / f"{model_key}_q{quality}.csv",
                    bootstrap_B=bootstrap_B,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_alpha=bootstrap_alpha,
                )
                write_metric_rows(metric_rows, metrics_path)
                write_metric_rows(stability_rows, stability_path)
                all_topk_rows.extend(topk_rows)
        mine_failure_cases(all_topk_rows, failure_path, rank_shift_threshold=rank_shift_threshold)
        if args.mode == "full":
            shutil.copyfile(metrics_path, tables_dir / "retrieval_metrics.csv")
            shutil.copyfile(stability_path, tables_dir / "feature_stability.csv")
            shutil.copyfile(failure_path, tables_dir / "failure_cases.csv")
    except Exception as exc:
        status = "failed"
        append_run_log(cfg, start=start, end=utc_now(), run_name="evaluate_retrieval", mode=args.mode, status=status, output_path=tables_dir, notes=str(exc))
        raise
    append_run_log(cfg, start=start, end=utc_now(), run_name="evaluate_retrieval", mode=args.mode, status=status, output_path=tables_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
