#!/usr/bin/env python
"""Generate supplementary figures referenced by the v2 final report.

Outputs (under outputs/figures/full/):
  - instance_recall_at1_only.png
  - class_precision_at1_exclude_only.png
  - feature_stability_paper.png
  - ppt_main_recall_precision.png
  - ppt_main_cosine_mean.png
  - ppt_cosine_mean_iqr.png
  - ppt_selected_breed_umap_q10.png
  - ppt_complexity_recall_q10.png
  - ppt_failure_examples_q10.png
  - ppt_same_query_rank_shift_q10.png
  - ppt_training_objective_alignment.png
  - q1_stress_test.png
  - failure_category_counts.png
  - severe_rank_shift_by_model.png
  - failure_summary_paper.png
  - retrieval_case_rank_shift_paper.png
  - embedding_dinov3_q10_domain_umap.png
  - embedding_resnet50_q10_{umap|tsne}.png
  - embedding_vit_b_16_q10_{umap|tsne}.png
  - embedding_dinov3_vits16_q10_{umap|tsne}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ads_dinov3.config import ensure_dir, feature_path, load_config, resolve_path
from ads_dinov3.features import load_feature_npz
from ads_dinov3.visualize import _embed, plot_embedding


MODEL_ORDER = ["dinov3", "dinov3_vits16", "vit_b_16", "resnet50"]
MODEL_COLORS = {
    "dinov3": "#1D4E89",
    "dinov3_vits16": "#E07A1F",
    "resnet50": "#C0392B",
    "vit_b_16": "#16A085",
}
MODEL_LABELS = {
    "dinov3": "DINOv3 ViT-B/16",
    "dinov3_vits16": "DINOv3 ViT-S/16",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16 (sup.)",
}
MODEL_MARKERS = {
    "dinov3": "o",
    "dinov3_vits16": "s",
    "vit_b_16": "^",
    "resnet50": "D",
}
MODEL_PARAM_M = {
    "dinov3": 86.0,
    "dinov3_vits16": 22.0,
    "vit_b_16": 86.0,
    "resnet50": 25.6,
}
MODEL_PARAM_LABELS = {
    "dinov3": "86M",
    "dinov3_vits16": "22M",
    "vit_b_16": "86M",
    "resnet50": "26M",
}
MODEL_COMPLEXITY_LABELS = {
    "dinov3": "DINOv3-B",
    "dinov3_vits16": "DINOv3-S",
    "vit_b_16": "ViT-B/16",
    "resnet50": "ResNet-50",
}
MODEL_LINESTYLES = {
    "dinov3": "-",
    "dinov3_vits16": "-",
    "vit_b_16": "--",
    "resnet50": "--",
}
EMBEDDING_TARGETS = ["resnet50", "vit_b_16", "dinov3_vits16"]
MAIN_QUALITIES = [90, 70, 50, 30, 10]
SELECTED_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Persian",
    "Siamese",
    "British_Shorthair",
    "Egyptian_Mau",
    "Samoyed",
    "Yorkshire_Terrier",
]
BREED_COLORS = {
    "Abyssinian": "#1f77b4",
    "Bengal": "#ff7f0e",
    "Persian": "#2ca02c",
    "Siamese": "#d62728",
    "British_Shorthair": "#9467bd",
    "Egyptian_Mau": "#8c564b",
    "Samoyed": "#17becf",
    "Yorkshire_Terrier": "#bcbd22",
}


def _blend_with_white(color: str, amount: float = 0.55) -> tuple[float, float, float]:
    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(rgb * (1.0 - amount) + amount)


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def _short_label(model: str) -> str:
    return _model_label(model).replace("DINOv3 ", "DINOv3-").replace(" (sup.)", "")


def _plot_metric_line(ax, group: pd.DataFrame, model: str, ylabel: str, direct_label: bool = True) -> None:
    color = MODEL_COLORS.get(model, "#444444")
    yerr_low = (group["value"] - group["ci_low"]).clip(lower=0).to_numpy()
    yerr_high = (group["ci_high"] - group["value"]).clip(lower=0).to_numpy()
    ax.errorbar(
        group["quality"],
        group["value"],
        yerr=[yerr_low, yerr_high],
        marker=MODEL_MARKERS.get(model, "o"),
        capsize=3,
        linewidth=2.1,
        markersize=5.5,
        linestyle=MODEL_LINESTYLES.get(model, "-"),
        color=color,
        label=_model_label(model),
    )
    q10 = group[group["quality"] == 10]
    if direct_label and not q10.empty:
        offsets = {"dinov3": 0.002, "dinov3_vits16": -0.002, "vit_b_16": 0.0, "resnet50": 0.0}
        ax.text(
            8.0,
            float(q10["value"].iloc[0]) + offsets.get(model, 0.0),
            _short_label(model),
            color=color,
            fontsize=8.5,
            va="center",
        )
    ax.set_ylabel(ylabel)


def plot_instance_recall_at1(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    view = df[(df["metric"] == "instance_recall") & (df["k"] == 1) & (df["self_policy"] == "include_original")].copy()
    view = view[view["quality"].isin(MAIN_QUALITIES)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model in MODEL_ORDER:
        group = view[view["model"] == model].sort_values("quality")
        if not group.empty:
            _plot_metric_line(ax, group, model, "Instance Recall@1", direct_label=False)
    ax.set_xlabel("JPEG quality")
    ax.set_title("Instance Recall@1 vs JPEG quality")
    ax.invert_xaxis()
    ax.set_xlim(96, 4)
    ax.set_xticks([90, 70, 50, 30, 10])
    ax.set_ylim(0.65, 1.01)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_class_precision_at1_exclude(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    view = df[
        (df["metric"] == "class_precision")
        & (df["k"] == 1)
        & (df["self_policy"] == "exclude_original")
    ].copy()
    view = view[view["quality"].isin(MAIN_QUALITIES)]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model in MODEL_ORDER:
        group = view[view["model"] == model].sort_values("quality")
        if group.empty:
            continue
        _plot_metric_line(ax, group, model, "Class Precision@1 (exclude original)")
    ax.set_xlabel("JPEG quality")
    ax.set_title("Class Precision@1 vs JPEG quality")
    ax.invert_xaxis()
    ax.set_xlim(96, 4)
    ax.set_xticks([90, 70, 50, 30, 10])
    ax.set_ylim(0.78, 0.97)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_stability_paper(stability_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(stability_csv)
    df = df[df["quality"].isin(MAIN_QUALITIES)]
    summary = df.groupby(["model", "quality"])["clean_compressed_cosine"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model in MODEL_ORDER:
        group = summary[summary["model"] == model].sort_values("quality")
        if group.empty:
            continue
        color = MODEL_COLORS.get(model, "#444444")
        quality = group["quality"].to_numpy()
        mean = group["mean"].to_numpy()
        std = group["std"].to_numpy()
        ax.plot(
            quality,
            mean,
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=2.1,
            markersize=5.5,
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            color=color,
            label=_model_label(model),
        )
        ax.fill_between(quality, mean - std, mean + std, color=color, alpha=0.10, linewidth=0)
        q10 = group[group["quality"] == 10]
        if not q10.empty:
            ax.text(8.0, float(q10["mean"].iloc[0]), _short_label(model), color=color, fontsize=8.5, va="center")
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("Clean-compressed cosine similarity")
    ax.set_title("Feature stability vs JPEG quality")
    ax.invert_xaxis()
    ax.set_xlim(96, 4)
    ax.set_xticks([90, 70, 50, 30, 10])
    ax.set_ylim(0.66, 1.01)
    ax.grid(True, alpha=0.28)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_ppt_metric_line(ax, group: pd.DataFrame, model: str, ylabel: str) -> None:
    color = MODEL_COLORS.get(model, "#444444")
    yerr_low = (group["value"] - group["ci_low"]).clip(lower=0).to_numpy()
    yerr_high = (group["ci_high"] - group["value"]).clip(lower=0).to_numpy()
    ax.errorbar(
        group["quality"],
        group["value"],
        yerr=[yerr_low, yerr_high],
        marker=MODEL_MARKERS.get(model, "o"),
        capsize=4,
        linewidth=3.0,
        markersize=8.0,
        linestyle=MODEL_LINESTYLES.get(model, "-"),
        color=color,
        label=_model_label(model),
    )
    ax.set_ylabel(ylabel, fontsize=15)


def _format_ppt_quality_axis(ax, title: str, ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=18, pad=12)
    ax.set_xlabel("JPEG quality", fontsize=15)
    ax.set_xlim(96, 4)
    ax.set_ylim(*ylim)
    ax.set_xticks(MAIN_QUALITIES)
    ax.grid(True, alpha=0.24)
    ax.tick_params(axis="both", labelsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ppt_recall_precision(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    recall = df[
        (df["metric"] == "instance_recall")
        & (df["k"] == 1)
        & (df["self_policy"] == "include_original")
        & (df["quality"].isin(MAIN_QUALITIES))
    ].copy()
    precision = df[
        (df["metric"] == "class_precision")
        & (df["k"] == 1)
        & (df["self_policy"] == "exclude_original")
        & (df["quality"].isin(MAIN_QUALITIES))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.55))
    for model in MODEL_ORDER:
        rec_group = recall[recall["model"] == model].sort_values("quality")
        prec_group = precision[precision["model"] == model].sort_values("quality")
        if not rec_group.empty:
            _plot_ppt_metric_line(axes[0], rec_group, model, "Recall@1")
        if not prec_group.empty:
            _plot_ppt_metric_line(axes[1], prec_group, model, "Precision@1")

    _format_ppt_quality_axis(axes[0], "Instance Recall@1", (0.65, 1.01))
    _format_ppt_quality_axis(axes[1], "Class Precision@1\n(exclude original)", (0.78, 0.97))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ppt_cosine_mean(stability_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(stability_csv)
    df = df[df["quality"].isin(MAIN_QUALITIES)]
    summary = df.groupby(["model", "quality"])["clean_compressed_cosine"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for model in MODEL_ORDER:
        group = summary[summary["model"] == model].sort_values("quality")
        if group.empty:
            continue
        color = MODEL_COLORS.get(model, "#444444")
        ax.plot(
            group["quality"],
            group["clean_compressed_cosine"],
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=3.2,
            markersize=8.5,
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            color=color,
            label=_model_label(model),
        )
    _format_ppt_quality_axis(ax, "Clean-compressed cosine mean", (0.74, 1.01))
    ax.set_ylabel("Cosine mean", fontsize=15)
    ax.legend(loc="lower left", fontsize=12, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ppt_cosine_mean_iqr(stability_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(stability_csv)
    df = df[df["quality"].isin(MAIN_QUALITIES)]
    summary = (
        df.groupby(["model", "quality"])["clean_compressed_cosine"]
        .agg(
            mean="mean",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for model in MODEL_ORDER:
        group = summary[summary["model"] == model].sort_values("quality")
        if group.empty:
            continue
        color = MODEL_COLORS.get(model, "#444444")
        alpha = 0.18 if model.startswith("dinov3") else 0.13
        zorder = 3 if model.startswith("dinov3") else 2
        ax.fill_between(
            group["quality"].to_numpy(),
            group["q25"].to_numpy(),
            group["q75"].to_numpy(),
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )
        ax.plot(
            group["quality"],
            group["mean"],
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=3.2,
            markersize=8.5,
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            color=color,
            label=_model_label(model),
            zorder=zorder + 2,
        )

    _format_ppt_quality_axis(ax, "Clean-compressed cosine similarity", (0.70, 1.01))
    ax.set_ylabel("Cosine similarity", fontsize=15)
    ax.text(
        0.03,
        0.05,
        "Line = mean, shaded band = IQR (25–75%) over 7,349 queries",
        transform=ax.transAxes,
        fontsize=11.5,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.92, "boxstyle": "round,pad=0.3"},
    )
    ax.legend(loc="lower left", fontsize=11.5, framealpha=0.95, bbox_to_anchor=(0.02, 0.15))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _breed_name_from_id(image_id: str) -> str:
    return "_".join(str(image_id).split("_")[:-1])


def plot_selected_breed_umap_comparison(cfg: dict, mode: str, output_path: Path, quality: int = 10) -> None:
    vis_cfg = cfg["visualization"]
    selected = set(SELECTED_BREEDS)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    for ax, model_key in zip(axes.ravel(), MODEL_ORDER):
        clean = load_feature_npz(feature_path(cfg, mode, model_key, "clean"))
        query = load_feature_npz(feature_path(cfg, mode, model_key, "query", quality))
        clean_breeds = np.asarray([_breed_name_from_id(image_id) for image_id in clean["image_ids"]])
        query_breeds = np.asarray([_breed_name_from_id(image_id) for image_id in query["image_ids"]])
        clean_mask = np.isin(clean_breeds, list(selected))
        query_mask = np.isin(query_breeds, list(selected))
        features = np.concatenate([clean["features"][clean_mask], query["features"][query_mask]], axis=0)
        breeds = np.concatenate([clean_breeds[clean_mask], query_breeds[query_mask]], axis=0)
        domains = np.asarray(["clean"] * int(clean_mask.sum()) + [f"JPEG Q={quality}"] * int(query_mask.sum()))
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
        for breed in SELECTED_BREEDS:
            color = BREED_COLORS[breed]
            breed_mask = breeds == breed
            clean_domain = breed_mask & (domains == "clean")
            jpeg_domain = breed_mask & (domains != "clean")
            ax.scatter(
                xy[clean_domain, 0],
                xy[clean_domain, 1],
                s=13,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=0.65,
                alpha=0.55,
            )
            ax.scatter(
                xy[jpeg_domain, 0],
                xy[jpeg_domain, 1],
                s=13,
                marker="x",
                c=color,
                linewidths=0.75,
                alpha=0.58,
            )
        ax.set_title(_model_label(model_key), fontsize=14.5, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")

    breed_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BREED_COLORS[breed], markeredgecolor=BREED_COLORS[breed], markersize=7, label=breed.replace("_", " "))
        for breed in SELECTED_BREEDS
    ]
    fig.legend(handles=breed_handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Selected-breed feature UMAP: clean vs JPEG Q={quality}", fontsize=18, y=0.985)
    fig.text(
        0.5,
        0.925,
        f"Selected breeds only · color = breed · ○ clean, × JPEG Q={quality} · method = {used_method.upper()} after PCA-50",
        ha="center",
        fontsize=10.8,
        color="#444444",
    )
    fig.tight_layout(rect=[0.03, 0.115, 0.97, 0.895])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_complexity_recall_q10(metrics_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    view = df[
        (df["metric"] == "instance_recall")
        & (df["self_policy"] == "include_original")
        & (df["k"] == 1)
        & (df["quality"] == 10)
    ].set_index("model")

    fig, ax = plt.subplots(figsize=(7.4, 4.35))
    for model in MODEL_ORDER:
        if model not in view.index:
            continue
        x = MODEL_PARAM_M[model]
        y = float(view.loc[model, "value"])
        yerr = np.array(
            [
                [max(0.0, y - float(view.loc[model, "ci_low"]))],
                [max(0.0, float(view.loc[model, "ci_high"]) - y)],
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=MODEL_COLORS[model],
            elinewidth=1.8,
            capsize=4,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=155,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )

    label_offsets = {
        "dinov3": (-18, 22),
        "vit_b_16": (-18, -34),
        "dinov3_vits16": (10, -28),
        "resnet50": (10, -18),
    }
    for model in MODEL_ORDER:
        if model not in view.index:
            continue
        x = MODEL_PARAM_M[model]
        y = float(view.loc[model, "value"])
        dx, dy = label_offsets[model]
        ax.annotate(
            f"{MODEL_COMPLEXITY_LABELS[model]}\n{MODEL_PARAM_LABELS[model]}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.3,
            color=MODEL_COLORS[model],
            fontweight="bold",
            ha="left" if dx >= 0 else "right",
            va="center",
        )

    y_dino_b = float(view.loc["dinov3", "value"])
    y_vit_b = float(view.loc["vit_b_16", "value"])
    y_dino_s = float(view.loc["dinov3_vits16", "value"])
    y_resnet = float(view.loc["resnet50", "value"])
    ax.annotate(
        "",
        xy=(88.8, y_dino_b),
        xytext=(88.8, y_vit_b),
        arrowprops={"arrowstyle": "<->", "color": "#555555", "lw": 1.5},
    )
    ax.text(
        90.4,
        (y_dino_b + y_vit_b) / 2,
        "+4.8 pts\nsame ViT-B size",
        fontsize=10.0,
        color="#333333",
        va="center",
    )
    ax.annotate(
        "",
        xy=(MODEL_PARAM_M["dinov3_vits16"], y_dino_s - 0.006),
        xytext=(MODEL_PARAM_M["resnet50"], y_resnet + 0.006),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.6},
    )
    ax.text(
        31.0,
        0.875,
        "+27.1 pts\nsimilar params",
        fontsize=10.0,
        color="#333333",
        va="center",
    )

    ax.set_title("Q=10 Recall@1 vs. Parameters", fontsize=15, pad=12)
    ax.set_xlabel("Parameters (millions, approx.)", fontsize=12.5)
    ax.set_ylabel("Instance Recall@1", fontsize=12.5)
    ax.set_xlim(15, 96)
    ax.set_ylim(0.68, 1.035)
    ax.set_xticks([20, 40, 60, 80])
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rounded_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    title: str,
    body: str,
    title_color: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=2.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width * 0.06,
        y + height * 0.78,
        title,
        ha="left",
        va="top",
        fontsize=13.8,
        fontweight="bold",
        color=title_color,
        linespacing=1.02,
    )
    ax.text(
        x + width * 0.06,
        y + height * (0.52 if "\n" not in title else 0.36),
        body,
        ha="left",
        va="top",
        fontsize=10.8,
        color="#222222",
        linespacing=1.32,
    )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str, lw: float, label: str) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    ax.text(
        mid_x,
        mid_y + 0.04,
        label,
        ha="center",
        va="center",
        fontsize=11.2,
        color=color,
        fontweight="bold",
    )


def plot_training_objective_alignment(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Training Objective vs. Retrieval Task",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
        color="#1F2A60",
    )

    _rounded_box(
        ax,
        (0.075, 0.53),
        0.34,
        0.25,
        facecolor="#F4FBF9",
        edgecolor=MODEL_COLORS["vit_b_16"],
        title="ImageNet\nclassification",
        title_color=MODEL_COLORS["vit_b_16"],
        body=(
            "class label"
        ),
    )
    _rounded_box(
        ax,
        (0.585, 0.53),
        0.34,
        0.25,
        facecolor="#F4F8FD",
        edgecolor=MODEL_COLORS["dinov3"],
        title="DINOv3\nself-supervised",
        title_color=MODEL_COLORS["dinov3"],
        body=(
            "view alignment"
        ),
    )
    _rounded_box(
        ax,
        (0.27, 0.10),
        0.56,
        0.22,
        facecolor="#FFF9F3",
        edgecolor=MODEL_COLORS["dinov3_vits16"],
        title="JPEG retrieval",
        title_color=MODEL_COLORS["dinov3_vits16"],
        body=(
            "query -> clean original"
        ),
    )

    _arrow(ax, (0.245, 0.52), (0.43, 0.33), "#777777", 2.0, "partial match")
    _arrow(ax, (0.755, 0.52), (0.67, 0.33), MODEL_COLORS["dinov3"], 3.0, "better match")
    fig.tight_layout()
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_q1_stress_test(metrics_csv: Path, stability_csv: Path, output_path: Path) -> None:
    metrics = pd.read_csv(metrics_csv)
    stability = pd.read_csv(stability_csv)
    rec = metrics[
        (metrics["metric"] == "instance_recall")
        & (metrics["self_policy"] == "include_original")
        & (metrics["k"] == 1)
        & (metrics["quality"].isin([10, 1]))
    ].copy()
    cp = metrics[
        (metrics["metric"] == "class_precision")
        & (metrics["self_policy"] == "exclude_original")
        & (metrics["k"] == 1)
        & (metrics["quality"].isin([10, 1]))
    ].copy()
    stab = stability[stability["quality"].isin([10, 1])].groupby(["model", "quality"])["clean_compressed_cosine"].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.65), sharex=True)
    panels = [
        (axes[0], rec, "Instance Recall@1", "value"),
        (axes[1], cp, "Class Precision@1\n(exclude original)", "value"),
        (axes[2], stab, "Clean-compressed\ncosine mean", "clean_compressed_cosine"),
    ]
    x_base = range(len(MODEL_ORDER))
    width = 0.34
    for ax, data, title, value_col in panels:
        q10_vals = []
        q1_vals = []
        for model in MODEL_ORDER:
            model_data = data[data["model"] == model]
            q10_vals.append(float(model_data[model_data["quality"] == 10][value_col].iloc[0]))
            q1_vals.append(float(model_data[model_data["quality"] == 1][value_col].iloc[0]))

        for idx, model in enumerate(MODEL_ORDER):
            color = MODEL_COLORS[model]
            ax.bar(
                idx - width / 2,
                q10_vals[idx],
                width=width,
                color=_blend_with_white(color, 0.60),
                edgecolor=color,
                linewidth=1.2,
                label="Q=10" if idx == 0 else None,
            )
            ax.bar(
                idx + width / 2,
                q1_vals[idx],
                width=width,
                color=color,
                edgecolor=color,
                linewidth=1.2,
                alpha=0.86,
                hatch="//",
                label="Q=1" if idx == 0 else None,
            )
        for x, y in zip([x + width / 2 for x in x_base], q1_vals):
            ax.text(x, y + 0.018, f"{y:.2f}", ha="center", va="bottom", fontsize=8.2, color="#222222")
        ax.set_title(title, fontsize=11.4, pad=8)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(list(x_base))
        ax.set_xticklabels(["DINOv3-B", "DINOv3-S", "ViT-B", "ResNet"], rotation=25, ha="right", fontsize=9.4)
        for tick, model in zip(ax.get_xticklabels(), MODEL_ORDER):
            tick.set_color(MODEL_COLORS[model])
            tick.set_fontweight("bold")
        ax.tick_params(axis="y", labelsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(facecolor="#EAEAEA", edgecolor="#555555", linewidth=1.2, label="Q=10"),
        Patch(facecolor="#888888", edgecolor="#555555", linewidth=1.2, hatch="//", label="Q=1"),
    ]
    axes[0].legend(handles=legend_handles, loc="lower left", fontsize=9.0, framealpha=0.95)
    fig.suptitle("Extreme JPEG Q=1 stress test", fontsize=14.5, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_failure_category_counts(failure_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(failure_csv)
    df = df[df["quality"].isin(MAIN_QUALITIES)]
    counts = df["case_type"].value_counts().reindex(
        ["instance_fail_class_success", "instance_fail_class_fail", "severe_compression_rank_shift"]
    )
    labels = [
        "instance fail / class success",
        "instance fail / class fail",
        "severe rank shift (Q=10)",
    ]
    colors = ["#1D4E89", "#C0392B", "#777777"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.barh(labels, counts.values, color=colors)
    for bar, value in zip(bars, counts.values):
        ax.text(value + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}", va="center", fontsize=10)
    ax.set_xlabel("number of mined cases")
    ax.set_title(f"Failure mining: {int(counts.sum()):,} cases across 4 models, 5 qualities")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts.values) * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_failure_summary_paper(failure_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(failure_csv)
    df = df[df["quality"].isin(MAIN_QUALITIES)]
    category_order = ["severe_compression_rank_shift", "instance_fail_class_success", "instance_fail_class_fail"]
    category_counts = df["case_type"].value_counts().reindex(category_order).fillna(0).astype(int)
    category_labels = ["severe rank shift\n(Q=10)", "instance fail\nclass success", "instance fail\nclass fail"]

    view = df[df["case_type"] == "severe_compression_rank_shift"].copy()
    model_counts = view["model"].value_counts().reindex(MODEL_ORDER).fillna(0).astype(int)
    model_labels = [_short_label(model) for model in MODEL_ORDER]
    total_shift = int(model_counts.sum())

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.05, 1.0]})

    ax = axes[0]
    category_colors = ["#777777", "#1D4E89", "#C0392B"]
    bars = ax.barh(category_labels, category_counts.values, color=category_colors)
    for bar, value in zip(bars, category_counts.values):
        ax.text(value + max(category_counts.values) * 0.02, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=9)
    ax.set_title("(a) Failure categories", fontsize=11)
    ax.set_xlabel("number of mined cases")
    ax.invert_yaxis()
    ax.set_xlim(0, max(category_counts.values) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    bars = ax.barh(model_labels, model_counts.values, color=[MODEL_COLORS[m] for m in MODEL_ORDER])
    for bar, value in zip(bars, model_counts.values):
        pct = 100.0 * value / total_shift if total_shift else 0.0
        x = value + max(model_counts.values) * 0.02 if value > 0 else max(model_counts.values) * 0.02
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{int(value):,} ({pct:.0f}%)", va="center", fontsize=9)
    ax.set_title("(b) Worst model in rank-shift cases", fontsize=11)
    ax.set_xlabel("number of cases")
    ax.invert_yaxis()
    ax.set_xlim(0, max(model_counts.values) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Failure mining summary", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_severe_rank_shift_by_model(failure_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(failure_csv)
    view = df[df["case_type"] == "severe_compression_rank_shift"].copy()
    counts = view["model"].value_counts().reindex(
        ["dinov3", "dinov3_vits16", "vit_b_16", "resnet50"]
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    colors = [MODEL_COLORS[m] for m in counts.index]
    labels = [MODEL_LABELS[m] for m in counts.index]
    bars = ax.barh(labels, counts.values, color=colors)
    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        pct = 100.0 * value / total if total else 0.0
        ax.text(value + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(value):,} ({pct:.0f}%)", va="center", fontsize=10)
    ax.set_xlabel("number of cases (worst-rank model)")
    ax.set_title(f"Severe rank shift (Q=10): worst model among {total:,} cases")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts.values) * 1.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _show_image(ax, path: str, title: str, border: str = "#444444", title_color: str = "#222222") -> None:
    image = Image.open(path).convert("RGB")
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    canvas_size = 512
    image = ImageOps.contain(image, (canvas_size, canvas_size), method=resampling)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (252, 252, 252))
    left = (canvas_size - image.width) // 2
    top = (canvas_size - image.height) // 2
    canvas.paste(image, (left, top))
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.7, color=title_color, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.2)
        spine.set_edgecolor(border)


def _show_image_slide(
    ax,
    path: str | Path,
    title: str,
    border: str = "#444444",
    title_color: str = "#222222",
    title_size: float = 9.5,
    border_width: float = 2.8,
) -> None:
    image = Image.open(path).convert("RGB")
    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    canvas_size = 560
    image = ImageOps.contain(image, (canvas_size, canvas_size), method=resampling)
    canvas = Image.new("RGB", (canvas_size, canvas_size), (252, 252, 252))
    left = (canvas_size - image.width) // 2
    top = (canvas_size - image.height) // 2
    canvas.paste(image, (left, top))
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=title_size, color=title_color, pad=3)
    ax.set_anchor("C")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(border_width)
        spine.set_edgecolor(border)


def _add_image_slide(
    fig,
    left_in: float,
    bottom_in: float,
    size_in: float,
    path: str | Path,
    title: str,
    border: str = "#444444",
    title_color: str = "#222222",
    title_size: float = 9.5,
    border_width: float = 2.8,
):
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes([left_in / fig_w, bottom_in / fig_h, size_in / fig_w, size_in / fig_h])
    _show_image_slide(ax, path, title, border=border, title_color=title_color, title_size=title_size, border_width=border_width)
    return ax


def _clean_original_path_from_row(row: pd.Series) -> Path:
    data_root = Path(str(row["query_path"])).parents[2]
    return data_root / "raw" / "oxford_iiit_pet" / "images" / f"{row['image_id']}.jpg"


def _relation_style(query_id: str, retrieved_id: str) -> tuple[str, str]:
    if retrieved_id == query_id:
        return "#2E7D32", "clean original"
    if _breed_name_from_id(retrieved_id) == _breed_name_from_id(query_id):
        return "#E07A1F", "same breed"
    return "#C0392B", "different breed"


def _rank_text(rank: int) -> str:
    return f"rank {rank}" if rank > 0 else ">10"


def _slide_image_id_label(image_id: str) -> str:
    if "_" not in image_id:
        return image_id
    breed, number = image_id.rsplit("_", 1)
    return f"{breed.replace('_', ' ')}\n{number}"


def _ranked_slide_title(prefix: str, image_id: str) -> str:
    return f"{prefix}: {_slide_image_id_label(image_id)}"


def plot_model_failure_examples_q10(topk_dir: Path, output_path: Path, quality: int = 10) -> None:
    examples = {
        "dinov3": "Bombay_114",
        "dinov3_vits16": "Bombay_107",
        "vit_b_16": "Abyssinian_141",
        "resnet50": "Abyssinian_108",
    }
    rows = {}
    for model, image_id in examples.items():
        df = pd.read_csv(topk_dir / f"{model}_q{quality}.csv")
        match = df[df["image_id"] == image_id]
        if match.empty:
            raise ValueError(f"{image_id} not found for {model}")
        rows[model] = match.iloc[0]

    fig_w, fig_h = 10.4, 8.1
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle("Representative top-1 errors at JPEG Q=10", fontsize=20.0, y=0.985)

    image_size = 1.08
    model_x = 0.15
    query_x = 1.62
    original_x = 3.32
    top1_x = 5.02
    failure_x = 6.9
    row_bottoms = [5.2, 3.68, 2.16, 0.64]
    header_y = 6.85

    headers = [
        (model_x, "Model", "left"),
        (query_x + image_size / 2, f"JPEG Q={quality}\nquery", "center"),
        (original_x + image_size / 2, "Clean\noriginal", "center"),
        (top1_x + image_size / 2, "Top-1\nretrieval", "center"),
        (failure_x + 1.15, "What failed?", "center"),
    ]
    for x_in, header, ha in headers:
        fig.text(x_in / fig_w, header_y / fig_h, header, ha=ha, va="center", fontsize=15.5, fontweight="bold")

    for bottom, model in zip(row_bottoms, MODEL_ORDER):
        row = rows[model]
        model_color = MODEL_COLORS[model]
        query_id = str(row["image_id"])
        top1_id = str(row["top1_id"])
        top1_border, top1_relation = _relation_style(query_id, top1_id)
        rank = int(row["instance_rank_within_exported_topk"])
        error_label = "same-breed near miss" if top1_relation == "same breed" else "wrong-breed retrieval"

        fig.text(
            model_x / fig_w,
            (bottom + 0.64) / fig_h,
            MODEL_COMPLEXITY_LABELS.get(model, _short_label(model)),
            ha="left",
            va="center",
            fontsize=15.6,
            fontweight="bold",
            color=model_color,
        )
        fig.text(
            model_x / fig_w,
            (bottom + 0.38) / fig_h,
            f"{MODEL_PARAM_LABELS.get(model, '')}",
            ha="left",
            va="center",
            fontsize=12.8,
            color=model_color,
        )

        _add_image_slide(
            fig,
            query_x,
            bottom,
            image_size,
            row["query_path"],
            _slide_image_id_label(query_id),
            border="#555555",
            title_color="#333333",
            title_size=9.3,
        )
        _add_image_slide(
            fig,
            original_x,
            bottom,
            image_size,
            _clean_original_path_from_row(row),
            "clean original",
            border="#2E7D32",
            title_color="#2E7D32",
            title_size=9.5,
        )
        _add_image_slide(
            fig,
            top1_x,
            bottom,
            image_size,
            row["top1_path"],
            _ranked_slide_title("top-1", top1_id),
            border=top1_border,
            title_color=top1_border,
            title_size=9.0,
        )
        fig.text(
            failure_x / fig_w,
            (bottom + 0.66) / fig_h,
            error_label,
            ha="left",
            va="center",
            fontsize=14.0,
            fontweight="bold",
            color=top1_border,
        )
        fig.text(
            failure_x / fig_w,
            (bottom + 0.42) / fig_h,
            f"clean original: {_rank_text(rank)}",
            ha="left",
            va="center",
            fontsize=12.4,
            color="#333333",
        )

    legend_handles = [
        Patch(facecolor="white", edgecolor="#2E7D32", linewidth=2.4, label="clean original"),
        Patch(facecolor="white", edgecolor="#E07A1F", linewidth=2.4, label="same breed, different image"),
        Patch(facecolor="white", edgecolor="#C0392B", linewidth=2.4, label="different breed"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11.8, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_same_query_rank_shift_ppt(topk_dir: Path, output_path: Path, image_id: str = "Abyssinian_102", quality: int = 10) -> None:
    rows = {}
    for model in MODEL_ORDER:
        df = pd.read_csv(topk_dir / f"{model}_q{quality}.csv")
        match = df[df["image_id"] == image_id]
        if match.empty:
            raise ValueError(f"{image_id} not found for {model}")
        rows[model] = match.iloc[0]

    fig_w, fig_h = 10.1, 8.35
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle("Same Q=10 query, different retrieval behavior", fontsize=20.0, y=0.985)

    image_size = 0.9
    top_image_size = 0.72
    model_x = 0.15
    top1_x = 1.58
    top2_x = 3.18
    top3_x = 4.78
    rank_x = 6.62
    rank_w = 1.38
    rank_h = 0.72
    row_bottoms = [4.8, 3.45, 2.1, 0.75]
    header_y = 6.1

    ref = rows[MODEL_ORDER[0]]
    _add_image_slide(fig, top1_x + 0.12, 6.85, top_image_size, ref["query_path"], f"JPEG query\nQ={quality}", border="#555555", title_size=10.4)
    _add_image_slide(
        fig,
        top2_x + 0.12,
        6.85,
        top_image_size,
        _clean_original_path_from_row(ref),
        "clean original",
        border="#2E7D32",
        title_color="#2E7D32",
        title_size=10.4,
    )
    fig.text(
        (top3_x + 0.25) / fig_w,
        7.2 / fig_h,
        f"Case: {image_id}\nGoal: retrieve the exact clean original",
        fontsize=13.2,
        ha="left",
        va="center",
        color="#222222",
    )

    headers = [
        (model_x, "Model", "left"),
        (top1_x + image_size / 2, "Top-1", "center"),
        (top2_x + image_size / 2, "Top-2", "center"),
        (top3_x + image_size / 2, "Top-3", "center"),
        (rank_x + rank_w / 2, "Original\nrank", "center"),
    ]
    for x_in, header, ha in headers:
        fig.text(x_in / fig_w, header_y / fig_h, header, ha=ha, va="center", fontsize=15.2, fontweight="bold")

    for bottom, model in zip(row_bottoms, MODEL_ORDER):
        row = rows[model]
        model_color = MODEL_COLORS[model]
        fig.text(
            model_x / fig_w,
            (bottom + 0.47) / fig_h,
            MODEL_COMPLEXITY_LABELS.get(model, _short_label(model)),
            color=model_color,
            fontsize=15.2,
            fontweight="bold",
            ha="left",
            va="center",
        )
        for left, top_i in zip([top1_x, top2_x, top3_x], [1, 2, 3]):
            top_id = str(row[f"top{top_i}_id"])
            border, relation = _relation_style(image_id, top_id)
            _add_image_slide(
                fig,
                left,
                bottom,
                image_size,
                row[f"top{top_i}_path"],
                _ranked_slide_title(f"r{top_i}", top_id),
                border=border,
                title_color=border,
                title_size=8.7,
            )
        rank = int(row["instance_rank_within_exported_topk"])
        rank_color = "#2E7D32" if rank == 1 else ("#E07A1F" if 1 < rank <= 3 else "#C0392B")
        rank_ax = fig.add_axes([rank_x / fig_w, (bottom + 0.1) / fig_h, rank_w / fig_w, rank_h / fig_h])
        rank_ax.set_xticks([])
        rank_ax.set_yticks([])
        rank_ax.text(
            0.5,
            0.5,
            _rank_text(rank),
            color=rank_color,
            fontsize=16.0,
            fontweight="bold",
            ha="center",
            va="center",
        )
        for spine in rank_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.1)
            spine.set_edgecolor(rank_color)

    legend_handles = [
        Patch(facecolor="white", edgecolor="#2E7D32", linewidth=2.4, label="clean original"),
        Patch(facecolor="white", edgecolor="#E07A1F", linewidth=2.4, label="same breed, different image"),
        Patch(facecolor="white", edgecolor="#C0392B", linewidth=2.4, label="different breed"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11.8, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_case_rank_shift(topk_dir: Path, output_path: Path, image_id: str = "Abyssinian_102", quality: int = 10) -> None:
    rows = {}
    for model in MODEL_ORDER:
        csv_path = topk_dir / f"{model}_q{quality}.csv"
        df = pd.read_csv(csv_path)
        match = df[df["image_id"] == image_id]
        if match.empty:
            raise ValueError(f"{image_id} not found in {csv_path}")
        rows[model] = match.iloc[0]

    fig, axes = plt.subplots(
        len(MODEL_ORDER) + 1,
        5,
        figsize=(10.4, 7.2),
        gridspec_kw={"height_ratios": [0.82, 1, 1, 1, 1], "width_ratios": [0.95, 1, 1, 1, 0.90]},
    )
    for ax in axes.ravel():
        ax.axis("off")

    ref = next(iter(rows.values()))
    _show_image(axes[0, 1], ref["query_path"], f"JPEG query\nQ={quality}", border="#555555")
    original_path = None
    for i in range(1, 11):
        if ref.get(f"top{i}_id") == image_id:
            original_path = ref.get(f"top{i}_path")
            break
    if original_path is None:
        original_path = str(Path(ref["query_path"]).parents[1] / "raw" / "oxford_iiit_pet" / "images" / f"{image_id}.jpg")
    _show_image(axes[0, 2], original_path, "clean original", border="#2E7D32")
    axes[0, 3].text(0.0, 0.55, f"Case: {image_id}\nGoal: retrieve clean original", fontsize=10, va="center")

    axes[1, 1].set_title("Top-1", fontsize=9, pad=6)
    axes[1, 2].set_title("Top-2", fontsize=9, pad=6)
    axes[1, 3].set_title("Top-3", fontsize=9, pad=6)
    axes[1, 4].set_title("Original rank", fontsize=9, pad=6)

    for row_idx, model in enumerate(MODEL_ORDER, start=1):
        row = rows[model]
        rank = int(row["instance_rank_within_exported_topk"])
        color = MODEL_COLORS[model]
        axes[row_idx, 0].text(
            0.98,
            0.5,
            _model_label(model),
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="center",
        )
        for col, top_i in enumerate([1, 2, 3], start=1):
            top_id = str(row[f"top{top_i}_id"])
            is_original = top_id == image_id
            same_class = int(row[f"top{top_i}_class"]) == int(row["query_class"])
            border = "#2E7D32" if is_original else ("#E07A1F" if same_class else "#C0392B")
            title_color = border
            title = f"r{top_i}: {top_id}"
            _show_image(axes[row_idx, col], row[f"top{top_i}_path"], title, border=border, title_color=title_color)
        rank_color = "#2E7D32" if rank == 1 else ("#E07A1F" if 1 < rank <= 3 else "#C0392B")
        axes[row_idx, 4].text(0.5, 0.5, f"rank {rank}" if rank > 0 else ">10", color=rank_color, fontsize=13, fontweight="bold", ha="center", va="center")
        axes[row_idx, 4].axis("on")
        axes[row_idx, 4].set_xticks([])
        axes[row_idx, 4].set_yticks([])
        for spine in axes[row_idx, 4].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
            spine.set_edgecolor(rank_color)

    legend_handles = [
        Patch(facecolor="white", edgecolor="#2E7D32", linewidth=2.0, label="clean original"),
        Patch(facecolor="white", edgecolor="#E07A1F", linewidth=2.0, label="same breed, different image"),
        Patch(facecolor="white", edgecolor="#C0392B", linewidth=2.0, label="different breed"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Representative severe rank-shift case", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_embeddings(cfg: dict, mode: str, figures_dir: Path) -> None:
    quality = int(cfg["visualization"]["embedding_quality"])
    for model_key in EMBEDDING_TARGETS:
        path = plot_embedding(cfg, mode=mode, model_key=model_key, quality=quality, output_dir=figures_dir)
        print(f"  wrote {path}")


def plot_domain_embedding(cfg: dict, mode: str, model_key: str, output_path: Path) -> None:
    quality = int(cfg["visualization"]["embedding_quality"])
    clean = load_feature_npz(feature_path(cfg, mode, model_key, "clean"))
    query = load_feature_npz(feature_path(cfg, mode, model_key, "query", quality))
    features = np.concatenate([clean["features"], query["features"]], axis=0)
    domains = ["clean"] * len(clean["image_ids"]) + [f"JPEG Q={quality}"] * len(query["image_ids"])
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
    domain_series = pd.Series(domains)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for domain, color, marker, alpha, size in [
        ("clean", "#222222", "o", 0.28, 10),
        (f"JPEG Q={quality}", "#D81B60", "x", 0.42, 11),
    ]:
        mask = domain_series.eq(domain).to_numpy()
        if domain == "clean":
            ax.scatter(xy[mask, 0], xy[mask, 1], facecolors="none", edgecolors=color, marker=marker, s=size, alpha=alpha, label=domain, linewidths=0.7)
        else:
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, marker=marker, s=size, alpha=alpha, label=domain, linewidths=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{_model_label(model_key)} embedding: clean vs JPEG Q={quality} ({used_method.upper()})")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate supplementary figures for the final report.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip per-model embedding plots (slow).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    tables_dir = resolve_path(cfg, cfg["retrieval"]["tables_dir"])
    figures_dir = resolve_path(cfg, cfg["visualization"]["figures_dir"]) / args.mode
    ensure_dir(figures_dir)

    metrics_csv = tables_dir / f"retrieval_metrics_{args.mode}.csv"
    failure_csv = tables_dir / f"failure_cases_{args.mode}.csv"

    print(f"[1/12] Instance Recall@1 figure")
    plot_instance_recall_at1(metrics_csv, figures_dir / "instance_recall_at1_only.png")

    print(f"[2/12] Class Precision@1 exclude-original figure")
    plot_class_precision_at1_exclude(metrics_csv, figures_dir / "class_precision_at1_exclude_only.png")

    print(f"[3/12] Feature stability figure")
    plot_feature_stability_paper(tables_dir / f"feature_stability_{args.mode}.csv", figures_dir / "feature_stability_paper.png")

    print(f"[4/12] Q=1 stress test figure")
    plot_q1_stress_test(metrics_csv, tables_dir / f"feature_stability_{args.mode}.csv", figures_dir / "q1_stress_test.png")

    print(f"[5/12] PPT recall + precision figure")
    plot_ppt_recall_precision(metrics_csv, figures_dir / "ppt_main_recall_precision.png")

    print(f"[6/12] PPT cosine mean figure")
    plot_ppt_cosine_mean(tables_dir / f"feature_stability_{args.mode}.csv", figures_dir / "ppt_main_cosine_mean.png")

    print(f"[7/13] PPT cosine mean + IQR figure")
    plot_ppt_cosine_mean_iqr(tables_dir / f"feature_stability_{args.mode}.csv", figures_dir / "ppt_cosine_mean_iqr.png")

    print(f"[8/14] Selected-breed UMAP comparison")
    plot_selected_breed_umap_comparison(cfg, args.mode, figures_dir / "ppt_selected_breed_umap_q10.png", quality=10)

    print(f"[9/15] Complexity vs Q=10 Recall figure")
    plot_complexity_recall_q10(metrics_csv, figures_dir / "ppt_complexity_recall_q10.png")

    print(f"[10/18] Training objective alignment figure")
    plot_training_objective_alignment(figures_dir / "ppt_training_objective_alignment.png")

    print(f"[11/18] Model-wise top-1 failure examples")
    plot_model_failure_examples_q10(tables_dir / "topk" / args.mode, figures_dir / "ppt_failure_examples_q10.png", quality=10)

    print(f"[12/18] Same-query rank-shift failure example")
    plot_same_query_rank_shift_ppt(tables_dir / "topk" / args.mode, figures_dir / "ppt_same_query_rank_shift_q10.png", image_id="Abyssinian_102", quality=10)

    print(f"[13/18] Failure category counts")
    plot_failure_category_counts(failure_csv, figures_dir / "failure_category_counts.png")

    print(f"[14/18] Severe rank shift by model")
    plot_severe_rank_shift_by_model(failure_csv, figures_dir / "severe_rank_shift_by_model.png")

    print(f"[15/18] Combined failure summary")
    plot_failure_summary_paper(failure_csv, figures_dir / "failure_summary_paper.png")

    print(f"[16/18] Retrieval case study")
    plot_retrieval_case_rank_shift(tables_dir / "topk" / args.mode, figures_dir / "retrieval_case_rank_shift_paper.png")

    print(f"[17/18] Domain-only embedding")
    plot_domain_embedding(cfg, args.mode, "dinov3", figures_dir / "embedding_dinov3_q10_domain_umap.png")

    if args.skip_embeddings:
        print("[18/18] Per-model embeddings: skipped")
    else:
        print(f"[18/18] Per-model embeddings (Q=10)")
        generate_embeddings(cfg, args.mode, figures_dir)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
