from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import ensure_dir, resolve_path
from .data import read_rows, write_rows


JPEG_SUBSAMPLING = 2
JPEG_OPTIMIZE = False
JPEG_PROGRESSIVE = False


def compressed_path(cfg: dict, image_id: str, quality: int) -> Path:
    root = resolve_path(cfg, cfg["compression"]["jpeg_root"])
    return root / f"jpeg_q{int(quality)}" / f"{image_id}.jpg"


def save_jpeg(src: Path, dst: Path, quality: int, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        rgb.save(
            dst,
            format="JPEG",
            quality=int(quality),
            subsampling=JPEG_SUBSAMPLING,
            optimize=JPEG_OPTIMIZE,
            progressive=JPEG_PROGRESSIVE,
        )


def create_query_manifest(
    cfg: dict,
    manifest_path: Path,
    output_path: Path,
    qualities: list[int],
    overwrite: bool = False,
) -> list[dict[str, object]]:
    rows = read_rows(manifest_path)
    query_rows: list[dict[str, object]] = []
    for row in rows:
        for quality in qualities:
            dst = compressed_path(cfg, row["image_id"], quality)
            save_jpeg(Path(row["image_path"]), dst, quality, overwrite=overwrite)
            query_rows.append(
                {
                    **row,
                    "quality": int(quality),
                    "query_path": str(dst),
                    "clean_path": row["image_path"],
                }
            )
    write_rows(query_rows, output_path)
    return query_rows
