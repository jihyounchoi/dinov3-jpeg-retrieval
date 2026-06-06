from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    topk_indices: np.ndarray
    similarities: np.ndarray


def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(denom, eps)


def topk_search(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k: int,
    exclude_mask: np.ndarray | None = None,
) -> RetrievalResult:
    if k <= 0:
        raise ValueError("k must be positive")
    if gallery_features.shape[0] < k:
        raise ValueError(f"gallery has {gallery_features.shape[0]} items, smaller than k={k}")
    sims = query_features @ gallery_features.T
    if exclude_mask is not None:
        sims = sims.copy()
        sims[exclude_mask] = -np.inf
    unsorted = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    unsorted_sims = np.take_along_axis(sims, unsorted, axis=1)
    order = np.argsort(-unsorted_sims, axis=1)
    topk_indices = np.take_along_axis(unsorted, order, axis=1)
    topk_sims = np.take_along_axis(sims, topk_indices, axis=1)
    return RetrievalResult(topk_indices=topk_indices, similarities=topk_sims)


def same_id_exclude_mask(query_ids: np.ndarray, gallery_ids: np.ndarray) -> np.ndarray:
    return query_ids[:, None] == gallery_ids[None, :]


def instance_recall_per_query(topk_indices: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(query_ids), dtype=np.float64)
    for row_idx, qid in enumerate(query_ids):
        retrieved_ids = gallery_ids[topk_indices[row_idx, :k]]
        out[row_idx] = float(np.any(retrieved_ids == qid))
    return out


def class_precision_per_query(topk_indices: np.ndarray, query_classes: np.ndarray, gallery_classes: np.ndarray, k: int) -> np.ndarray:
    out = np.empty(len(query_classes), dtype=np.float64)
    for row_idx, qclass in enumerate(query_classes):
        retrieved_classes = gallery_classes[topk_indices[row_idx, :k]]
        out[row_idx] = float(np.mean(retrieved_classes == qclass))
    return out


def instance_recall_at_k(topk_indices: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray, k: int) -> float:
    return float(np.mean(instance_recall_per_query(topk_indices, query_ids, gallery_ids, k)))


def class_precision_at_k(topk_indices: np.ndarray, query_classes: np.ndarray, gallery_classes: np.ndarray, k: int) -> float:
    return float(np.mean(class_precision_per_query(topk_indices, query_classes, gallery_classes, k)))


def bootstrap_ci(per_query: np.ndarray, B: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    if B <= 0 or per_query.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    n = per_query.size
    idx = rng.integers(0, n, size=(int(B), n))
    means = per_query[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def summarize_retrieval(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    query_classes: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    gallery_classes: np.ndarray,
    ks: list[int],
    topk_for_export: int,
    bootstrap_B: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_alpha: float = 0.05,
) -> tuple[list[dict[str, float | int | str]], RetrievalResult, RetrievalResult]:
    max_k = max(max(ks), topk_for_export)
    include = topk_search(query_features, gallery_features, max_k)
    exclude = topk_search(query_features, gallery_features, max_k, same_id_exclude_mask(query_ids, gallery_ids))
    rows: list[dict[str, float | int | str]] = []

    def _add(metric: str, policy: str, k: int, per_query: np.ndarray) -> None:
        value = float(np.mean(per_query))
        ci_lo, ci_hi = bootstrap_ci(per_query, bootstrap_B, bootstrap_seed, bootstrap_alpha) if bootstrap_B > 0 else (float("nan"), float("nan"))
        rows.append({
            "metric": metric,
            "self_policy": policy,
            "k": k,
            "value": value,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "bootstrap_B": int(bootstrap_B),
            "n_queries": int(per_query.size),
        })

    for k in ks:
        _add("instance_recall", "include_original", k, instance_recall_per_query(include.topk_indices, query_ids, gallery_ids, k))
        _add("class_precision", "include_original", k, class_precision_per_query(include.topk_indices, query_classes, gallery_classes, k))
        _add("class_precision", "exclude_original", k, class_precision_per_query(exclude.topk_indices, query_classes, gallery_classes, k))
    return rows, include, exclude


def clean_compressed_similarity(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
) -> np.ndarray:
    gallery_lookup = {image_id: idx for idx, image_id in enumerate(gallery_ids)}
    values = []
    missing = []
    for idx, image_id in enumerate(query_ids):
        gallery_idx = gallery_lookup.get(image_id)
        if gallery_idx is None:
            missing.append(image_id)
            continue
        values.append(float(query_features[idx] @ gallery_features[gallery_idx]))
    if missing:
        raise KeyError(f"{len(missing)} query ids are missing from gallery; first missing: {missing[0]}")
    return np.asarray(values, dtype=np.float32)
