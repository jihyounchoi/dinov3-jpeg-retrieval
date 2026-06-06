from __future__ import annotations

import csv
import datetime as dt
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from .config import ensure_dir, resolve_path


HEADER = [
    "timestamp_start",
    "timestamp_end",
    "duration_seconds",
    "run_name",
    "mode",
    "status",
    "gpu",
    "command",
    "config",
    "seed",
    "output_path",
    "notes",
]


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def command_string() -> str:
    return " ".join(shlex.quote(x) for x in sys.argv)


def append_run_log(
    cfg: dict[str, Any],
    *,
    start: dt.datetime,
    end: dt.datetime,
    run_name: str,
    mode: str,
    status: str,
    output_path: str | Path = "",
    notes: str = "",
) -> None:
    log_path = resolve_path(cfg, cfg.get("runtime", {}).get("run_log", "outputs/run_log.csv"))
    ensure_dir(log_path.parent)
    exists = log_path.exists()
    row = {
        "timestamp_start": start.isoformat(),
        "timestamp_end": end.isoformat(),
        "duration_seconds": f"{(end - start).total_seconds():.3f}",
        "run_name": run_name,
        "mode": mode,
        "status": status,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "command": command_string(),
        "config": cfg.get("_config_path", ""),
        "seed": cfg.get("runtime", {}).get("seed", ""),
        "output_path": str(output_path),
        "notes": notes,
    }
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
