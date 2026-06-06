#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

from ads_dinov3.config import feature_path, load_config, manifest_path, model_keys, qualities_for_mode, query_manifest_path
from ads_dinov3.runlog import append_run_log, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen image features.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["sanity", "full"], default="full")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    parser.add_argument("--image-set", choices=["clean", "query", "all"], default="all")
    parser.add_argument("--qualities", nargs="*", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None, help="Original GPU id to expose to this process.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)

    from ads_dinov3.features import extract_features, filter_quality, read_table

    start = utc_now()
    status = "success"
    outputs: list[str] = []
    try:
        models = model_keys(cfg) if args.model == "all" else [args.model]
        qualities = args.qualities if args.qualities is not None else qualities_for_mode(cfg, args.mode)
        batch_size = args.batch_size or int(cfg["features"]["sanity_batch_size" if args.mode == "sanity" else "batch_size"])
        for model_key in models:
            if args.image_set in {"clean", "all"}:
                rows = read_table(manifest_path(cfg, args.mode))
                out = feature_path(cfg, args.mode, model_key, "clean")
                extract_features(cfg, model_key=model_key, rows=rows, path_field="image_path", output_path=out, device=args.device, batch_size=batch_size)
                outputs.append(str(out))
            if args.image_set in {"query", "all"}:
                query_rows = read_table(query_manifest_path(cfg, args.mode))
                for quality in qualities:
                    rows = filter_quality(query_rows, quality)
                    out = feature_path(cfg, args.mode, model_key, "query", quality)
                    extract_features(cfg, model_key=model_key, rows=rows, path_field="query_path", output_path=out, device=args.device, batch_size=batch_size)
                    outputs.append(str(out))
    except Exception as exc:
        status = "failed"
        append_run_log(cfg, start=start, end=utc_now(), run_name="extract_features", mode=args.mode, status=status, output_path=";".join(outputs), notes=str(exc))
        raise
    append_run_log(cfg, start=start, end=utc_now(), run_name="extract_features", mode=args.mode, status=status, output_path=";".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
