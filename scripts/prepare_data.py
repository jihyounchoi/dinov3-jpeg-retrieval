#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys

from ads_dinov3.compression import create_query_manifest
from ads_dinov3.config import load_config, manifest_path, qualities_for_mode, query_manifest_path
from ads_dinov3.data import build_full_manifest, download_oxford_pet, make_sanity_subset, write_manifest
from ads_dinov3.runlog import append_run_log, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pets manifests and JPEG queries.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["sanity", "full"], default="full")
    parser.add_argument("--download", action="store_true", help="Download Oxford-IIIT Pets archives if needed.")
    parser.add_argument("--compress", action="store_true", help="Create JPEG query images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--qualities", nargs="*", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    start = utc_now()
    status = "success"
    output_path = ""
    try:
        if args.download:
            download_oxford_pet(cfg, overwrite=args.overwrite)
        full_records = build_full_manifest(cfg)
        if args.mode == "sanity":
            records = make_sanity_subset(
                full_records,
                images_per_class=int(cfg["data"]["sanity_images_per_class"]),
                seed=int(cfg["runtime"]["seed"]),
            )
        else:
            records = full_records
        mpath = manifest_path(cfg, args.mode)
        write_manifest(records, mpath)
        output_path = str(mpath)
        if args.compress:
            qualities = args.qualities if args.qualities is not None else qualities_for_mode(cfg, args.mode)
            qpath = query_manifest_path(cfg, args.mode)
            create_query_manifest(cfg, mpath, qpath, qualities=qualities, overwrite=args.overwrite)
            output_path = str(qpath)
    except Exception as exc:
        status = "failed"
        append_run_log(cfg, start=start, end=utc_now(), run_name="prepare_data", mode=args.mode, status=status, output_path=output_path, notes=str(exc))
        raise
    append_run_log(cfg, start=start, end=utc_now(), run_name="prepare_data", mode=args.mode, status=status, output_path=output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
