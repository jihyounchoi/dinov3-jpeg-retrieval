from __future__ import annotations

import csv
import random
import re
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import ensure_dir, resolve_path


@dataclass(frozen=True)
class PetRecord:
    image_id: str
    split: str
    class_idx: int
    species: int
    breed_id: int
    breed_name: str
    image_path: str


def download_file(url: str, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    urllib.request.urlretrieve(url, dst)


def extract_tarball(tar_path: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(dst_dir)


def download_oxford_pet(cfg: dict, overwrite: bool = False) -> None:
    raw_dir = resolve_path(cfg, cfg["data"]["raw_dir"])
    ensure_dir(raw_dir)
    archives = [
        (cfg["data"]["images_url"], raw_dir / "images.tar.gz"),
        (cfg["data"]["annotations_url"], raw_dir / "annotations.tar.gz"),
    ]
    for url, tar_path in archives:
        if overwrite and tar_path.exists():
            tar_path.unlink()
        download_file(url, tar_path)
        extract_tarball(tar_path, raw_dir)


def breed_name_from_image_id(image_id: str) -> str:
    return re.sub(r"_\d+$", "", image_id)


def parse_annotation_file(path: Path, split: str, images_dir: Path) -> list[PetRecord]:
    records: list[PetRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            image_id, class_id, species, breed_id = parts[:4]
            image_path = images_dir / f"{image_id}.jpg"
            records.append(
                PetRecord(
                    image_id=image_id,
                    split=split,
                    class_idx=int(class_id) - 1,
                    species=int(species),
                    breed_id=int(breed_id),
                    breed_name=breed_name_from_image_id(image_id),
                    image_path=str(image_path),
                )
            )
    return records


def build_full_manifest(cfg: dict) -> list[PetRecord]:
    images_dir = resolve_path(cfg, cfg["data"]["images_dir"])
    annotations_dir = resolve_path(cfg, cfg["data"]["annotations_dir"])
    trainval = parse_annotation_file(annotations_dir / "trainval.txt", "trainval", images_dir)
    test = parse_annotation_file(annotations_dir / "test.txt", "test", images_dir)
    records = trainval + test
    missing = [r.image_path for r in records if not Path(r.image_path).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images referenced by annotations are missing. First missing: {missing[0]}")
    return records


def write_manifest(records: Iterable[PetRecord], path: Path) -> None:
    ensure_dir(path.parent)
    rows = [asdict(r) for r in records]
    if not rows:
        raise ValueError("cannot write an empty manifest")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError("cannot write empty rows")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_sanity_subset(records: list[PetRecord], images_per_class: int, seed: int) -> list[PetRecord]:
    rng = random.Random(seed)
    by_class: dict[int, list[PetRecord]] = {}
    for record in records:
        by_class.setdefault(record.class_idx, []).append(record)
    subset: list[PetRecord] = []
    for class_idx in sorted(by_class):
        candidates = list(by_class[class_idx])
        rng.shuffle(candidates)
        subset.extend(sorted(candidates[:images_per_class], key=lambda r: r.image_id))
    return sorted(subset, key=lambda r: (r.class_idx, r.image_id))
