from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import ensure_dir, feature_path, resolve_path
from .features import load_feature_npz


def plot_metric_curves(metrics_csv: Path, output_dir: Path) -> list[Path]:
    ensure_dir(output_dir)
    df = pd.read_csv(metrics_csv)
    has_ci = {"ci_low", "ci_high"}.issubset(df.columns)
    outputs: list[Path] = []
    for metric in sorted(df["metric"].unique()):
        metric_df = df[df["metric"] == metric]
        for policy in sorted(metric_df["self_policy"].unique()):
            view = metric_df[metric_df["self_policy"] == policy]
            fig, ax = plt.subplots(figsize=(8, 5))
            for (model, k), group in view.groupby(["model", "k"]):
                group = group.sort_values("quality")
                if has_ci and group["ci_low"].notna().all() and group["ci_high"].notna().all():
                    yerr_low = (group["value"] - group["ci_low"]).clip(lower=0).to_numpy()
                    yerr_high = (group["ci_high"] - group["value"]).clip(lower=0).to_numpy()
                    ax.errorbar(group["quality"], group["value"], yerr=[yerr_low, yerr_high], marker="o", capsize=3, label=f"{model} @ {k}")
                else:
                    ax.plot(group["quality"], group["value"], marker="o", label=f"{model} @ {k}")
            ax.set_xlabel("JPEG quality")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} ({policy})")
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2)
            path = output_dir / f"{metric}_{policy}.png"
            fig.tight_layout()
            fig.savefig(path, dpi=180)
            plt.close(fig)
            outputs.append(path)
    return outputs


def plot_similarity(stability_csv: Path, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    df = pd.read_csv(stability_csv)
    summary = df.groupby(["model", "quality"])["clean_compressed_cosine"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in summary.groupby("model"):
        group = group.sort_values("quality")
        ax.errorbar(group["quality"], group["mean"], yerr=group["std"], marker="o", capsize=3, label=model)
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("Clean-compressed cosine similarity")
    ax.set_title("Feature stability under JPEG compression")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "feature_stability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _embed(
    features: np.ndarray,
    method: str,
    pca_components: int,
    seed: int,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_random_state: int | None = None,
    tsne_perplexity: int | None = None,
) -> tuple[np.ndarray, str]:
    pca_components = max(2, min(int(pca_components), features.shape[0] - 1, features.shape[1]))
    reduced = PCA(n_components=pca_components, random_state=seed).fit_transform(features)
    if method == "umap":
        try:
            import umap

            return umap.UMAP(
                n_components=2,
                n_neighbors=int(umap_n_neighbors),
                min_dist=float(umap_min_dist),
                random_state=int(umap_random_state if umap_random_state is not None else seed),
                metric="cosine",
            ).fit_transform(reduced), "umap"
        except Exception:
            pass
    if method in {"umap", "tsne"}:
        if tsne_perplexity is None:
            perplexity = min(30, max(5, (features.shape[0] - 1) // 3))
        else:
            perplexity = int(tsne_perplexity)
        return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(reduced), "tsne"
    return reduced[:, :2], "pca"


def plot_embedding(cfg: dict, mode: str, model_key: str, quality: int, output_dir: Path) -> Path:
    ensure_dir(output_dir)
    clean = load_feature_npz(feature_path(cfg, mode, model_key, "clean"))
    query = load_feature_npz(feature_path(cfg, mode, model_key, "query", quality))
    features = np.concatenate([clean["features"], query["features"]], axis=0)
    domains = np.asarray(["clean"] * len(clean["image_ids"]) + [f"jpeg_q{quality}"] * len(query["image_ids"]))
    classes = np.concatenate([clean["class_indices"], query["class_indices"]], axis=0)

    max_points = int(cfg["visualization"].get("max_embedding_points", 0))
    if max_points > 0 and features.shape[0] > max_points:
        rng = np.random.default_rng(int(cfg["runtime"]["seed"]))
        idx = np.sort(rng.choice(features.shape[0], size=max_points, replace=False))
        features = features[idx]
        domains = domains[idx]
        classes = classes[idx]

    vis_cfg = cfg["visualization"]
    xy, used_method = _embed(
        features,
        method=str(vis_cfg["embedding_method"]),
        pca_components=int(vis_cfg["pca_components_before_embedding"]),
        seed=int(cfg["runtime"]["seed"]),
        umap_n_neighbors=int(vis_cfg.get("umap_n_neighbors", 15)),
        umap_min_dist=float(vis_cfg.get("umap_min_dist", 0.1)),
        umap_random_state=int(vis_cfg.get("umap_random_state", cfg["runtime"]["seed"])),
        tsne_perplexity=int(vis_cfg["tsne_perplexity"]) if vis_cfg.get("tsne_perplexity") else None,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, marker in [("clean", "o"), (f"jpeg_q{quality}", "x")]:
        mask = domains == domain
        ax.scatter(xy[mask, 0], xy[mask, 1], s=8, alpha=0.45, marker=marker, label=domain, c=classes[mask], cmap="tab20")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{model_key} embedding: clean vs JPEG Q={quality} ({used_method})")
    ax.legend()
    path = output_dir / f"embedding_{model_key}_q{quality}_{used_method}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_retrieval_grid(failure_csv: Path, output_dir: Path, grid_k: int) -> list[Path]:
    ensure_dir(output_dir)
    rows = _load_csv_rows(failure_csv)
    if not rows:
        return []
    outputs: list[Path] = []
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        selected.setdefault(row["case_type"], row)
    for case_type, row in selected.items():
        cols = grid_k + 1
        fig, axes = plt.subplots(1, cols, figsize=(2.2 * cols, 2.6))
        paths = [row["query_path"]] + [row.get(f"top{i}_path", "") for i in range(1, grid_k + 1)]
        titles = ["query"] + [f"top{i}" for i in range(1, grid_k + 1)]
        for ax, path_str, title in zip(axes, paths, titles):
            ax.axis("off")
            if path_str and Path(path_str).exists():
                ax.imshow(Image.open(path_str).convert("RGB"))
            ax.set_title(title, fontsize=9)
        fig.suptitle(f"{case_type}: {row['model']} Q={row['quality']} image={row['image_id']}", fontsize=10)
        path = output_dir / f"retrieval_grid_{case_type}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def make_all_figures(cfg: dict, mode: str) -> list[Path]:
    tables_dir = resolve_path(cfg, cfg["retrieval"]["tables_dir"])
    figures_dir = resolve_path(cfg, cfg["visualization"]["figures_dir"]) / mode
    grid_dir = resolve_path(cfg, cfg["visualization"]["retrieval_grid_dir"]) / mode
    outputs: list[Path] = []
    outputs.extend(plot_metric_curves(tables_dir / f"retrieval_metrics_{mode}.csv", figures_dir))
    outputs.append(plot_similarity(tables_dir / f"feature_stability_{mode}.csv", figures_dir))
    outputs.append(
        plot_embedding(
            cfg,
            mode=mode,
            model_key=cfg["visualization"]["embedding_model"],
            quality=int(cfg["visualization"]["embedding_quality"]),
            output_dir=figures_dir,
        )
    )
    outputs.extend(make_retrieval_grid(tables_dir / f"failure_cases_{mode}.csv", grid_dir, int(cfg["visualization"]["retrieval_grid_k"])))
    return outputs
