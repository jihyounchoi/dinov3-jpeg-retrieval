from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(resolve_project_root(cfg))
    return cfg


def resolve_project_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("project", {}).get("root", ".")).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(cfg["_project_root"]) / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_keys(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["models"].keys())


def qualities_for_mode(cfg: dict[str, Any], mode: str) -> list[int]:
    key = "sanity_qualities" if mode == "sanity" else "qualities"
    return [int(q) for q in cfg["compression"][key]]


def manifest_path(cfg: dict[str, Any], mode: str) -> Path:
    return resolve_path(cfg, cfg["data"][f"manifest_{mode}"])


def query_manifest_path(cfg: dict[str, Any], mode: str) -> Path:
    return resolve_path(cfg, cfg["data"][f"query_manifest_{mode}"])


def feature_path(cfg: dict[str, Any], mode: str, model_key: str, image_set: str, quality: int | None = None) -> Path:
    base = resolve_path(cfg, cfg["features"]["output_dir"]) / mode / model_key
    if image_set == "clean":
        return base / "clean.npz"
    if quality is None:
        raise ValueError("quality is required for query feature paths")
    return base / f"q{int(quality)}.npz"
