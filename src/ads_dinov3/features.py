from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import ensure_dir
from .metrics import l2_normalize
from .models import import_torch_stack, load_model


class ImagePathDataset:
    def __init__(self, rows: list[dict[str, str]], path_field: str, transform):
        self.rows = rows
        self.path_field = path_field
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = row[self.path_field]
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, row["image_id"], int(row["class_idx"]), path


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_quality(rows: list[dict[str, str]], quality: int) -> list[dict[str, str]]:
    return [row for row in rows if int(row["quality"]) == int(quality)]


def extract_features(
    cfg: dict[str, Any],
    *,
    model_key: str,
    rows: list[dict[str, str]],
    path_field: str,
    output_path: Path,
    device: str,
    batch_size: int,
) -> Path:
    torch, *_ = import_torch_stack()
    from torch.utils.data import DataLoader

    ensure_dir(output_path.parent)
    model, transform = load_model(cfg, model_key, device)
    dataset = ImagePathDataset(rows, path_field, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg["features"]["num_workers"]),
        pin_memory=device.startswith("cuda"),
    )

    features = []
    image_ids: list[str] = []
    class_indices: list[int] = []
    paths: list[str] = []

    for images, batch_ids, batch_classes, batch_paths in tqdm(loader, desc=f"{model_key}:{path_field}"):
        images = images.to(device, non_blocking=True)
        batch_features = model.extract(images).detach().float().cpu().numpy()
        features.append(batch_features)
        image_ids.extend(list(batch_ids))
        class_indices.extend([int(x) for x in batch_classes])
        paths.extend(list(batch_paths))

    matrix = l2_normalize(np.concatenate(features, axis=0).astype(np.float32))
    np.savez_compressed(
        output_path,
        features=matrix,
        image_ids=np.asarray(image_ids),
        class_indices=np.asarray(class_indices, dtype=np.int64),
        paths=np.asarray(paths),
        model_key=np.asarray(model_key),
    )
    return output_path


def load_feature_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "features": data["features"].astype(np.float32),
        "image_ids": data["image_ids"].astype(str),
        "class_indices": data["class_indices"].astype(np.int64),
        "paths": data["paths"].astype(str),
    }
