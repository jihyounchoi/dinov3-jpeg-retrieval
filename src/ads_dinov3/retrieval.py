from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import ensure_dir
from .metrics import RetrievalResult, clean_compressed_similarity, summarize_retrieval


def write_metric_rows(rows: list[dict[str, object]], path: Path) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def export_topk_rows(
    *,
    model_key: str,
    quality: int,
    query_ids: np.ndarray,
    query_classes: np.ndarray,
    query_paths: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_classes: np.ndarray,
    gallery_paths: np.ndarray,
    topk: RetrievalResult,
    path: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_idx, image_id in enumerate(query_ids):
        retrieved = topk.topk_indices[row_idx]
        retrieved_ids = gallery_ids[retrieved]
        instance_hits = retrieved_ids == image_id
        instance_rank = int(np.where(instance_hits)[0][0] + 1) if np.any(instance_hits) else -1
        item: dict[str, object] = {
            "model": model_key,
            "quality": int(quality),
            "image_id": image_id,
            "query_class": int(query_classes[row_idx]),
            "query_path": query_paths[row_idx],
            "instance_rank_within_exported_topk": instance_rank,
            "instance_hit_at_1": bool(instance_rank == 1),
            "instance_hit_at_5": bool(1 <= instance_rank <= 5),
            "instance_hit_at_10": bool(1 <= instance_rank <= 10),
            "same_class_at_10": int(np.sum(gallery_classes[retrieved[:10]] == query_classes[row_idx])),
        }
        for rank, gallery_idx in enumerate(retrieved, start=1):
            item[f"top{rank}_id"] = gallery_ids[gallery_idx]
            item[f"top{rank}_class"] = int(gallery_classes[gallery_idx])
            item[f"top{rank}_path"] = gallery_paths[gallery_idx]
            item[f"top{rank}_similarity"] = float(topk.similarities[row_idx, rank - 1])
        rows.append(item)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def evaluate_pair(
    *,
    model_key: str,
    quality: int,
    gallery: dict[str, np.ndarray],
    query: dict[str, np.ndarray],
    ks: list[int],
    topk_for_export: int,
    topk_path: Path,
    bootstrap_B: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_alpha: float = 0.05,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows, include, _ = summarize_retrieval(
        query["features"],
        query["image_ids"],
        query["class_indices"],
        gallery["features"],
        gallery["image_ids"],
        gallery["class_indices"],
        ks,
        topk_for_export,
        bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed,
        bootstrap_alpha=bootstrap_alpha,
    )
    metric_rows = [
        {"model": model_key, "quality": int(quality), **row}
        for row in metric_rows
    ]

    stability = clean_compressed_similarity(
        query["features"],
        query["image_ids"],
        gallery["features"],
        gallery["image_ids"],
    )
    stability_rows = [
        {
            "model": model_key,
            "quality": int(quality),
            "image_id": image_id,
            "class_idx": int(query["class_indices"][idx]),
            "clean_compressed_cosine": float(value),
        }
        for idx, (image_id, value) in enumerate(zip(query["image_ids"], stability))
    ]

    topk_rows = export_topk_rows(
        model_key=model_key,
        quality=quality,
        query_ids=query["image_ids"],
        query_classes=query["class_indices"],
        query_paths=query["paths"],
        gallery_ids=gallery["image_ids"],
        gallery_classes=gallery["class_indices"],
        gallery_paths=gallery["paths"],
        topk=include,
        path=topk_path,
    )
    return metric_rows, stability_rows, topk_rows


DEFAULT_RANK_SHIFT_THRESHOLD = 5
SEVERE_QUALITY = 10
MISSING_RANK_SENTINEL = 999


def mine_failure_cases(
    topk_rows: list[dict[str, object]],
    output_path: Path,
    rank_shift_threshold: int = DEFAULT_RANK_SHIFT_THRESHOLD,
) -> None:
    cases: list[dict[str, object]] = []
    for row in topk_rows:
        hit10 = str(row["instance_hit_at_10"]).lower() == "true" if isinstance(row["instance_hit_at_10"], str) else bool(row["instance_hit_at_10"])
        same_class_at_10 = int(row["same_class_at_10"])
        if not hit10 and same_class_at_10 > 0:
            case_type = "instance_fail_class_success"
        elif not hit10 and same_class_at_10 == 0:
            case_type = "instance_fail_class_fail"
        else:
            continue
        cases.append({"case_type": case_type, "rank_shift_threshold": int(rank_shift_threshold), **row})

    severe_by_image: dict[str, list[dict[str, object]]] = {}
    for row in topk_rows:
        if int(row["quality"]) == SEVERE_QUALITY:
            severe_by_image.setdefault(str(row["image_id"]), []).append(row)
    for image_id, rows in severe_by_image.items():
        if len(rows) < 2:
            continue
        ranks = [int(row["instance_rank_within_exported_topk"]) for row in rows]
        comparable = [rank if rank > 0 else MISSING_RANK_SENTINEL for rank in ranks]
        if max(comparable) - min(comparable) < int(rank_shift_threshold):
            continue
        worst_idx = int(np.argmax(comparable))
        worst_row = rows[worst_idx]
        cases.append(
            {
                "case_type": "severe_compression_rank_shift",
                "rank_shift_threshold": int(rank_shift_threshold),
                "rank_shift_image_id": image_id,
                "rank_shift_min_rank": min(comparable),
                "rank_shift_max_rank": max(comparable),
                "rank_shift_models": ";".join(f"{row['model']}:{rank}" for row, rank in zip(rows, ranks)),
                **worst_row,
            }
        )

    ensure_dir(output_path.parent)
    if not cases:
        output_path.write_text("case_type,rank_shift_threshold\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for case in cases:
        for key in case.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
