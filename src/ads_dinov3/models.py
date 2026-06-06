from __future__ import annotations

from pathlib import Path
from typing import Any


def import_torch_stack():
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

    return torch, nn, transforms, ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16


def make_transform(cfg: dict[str, Any]):
    _, _, transforms, *_ = import_torch_stack()
    pp = cfg["preprocess"]
    return transforms.Compose(
        [
            transforms.Resize(int(pp["resize_size"]), antialias=True),
            transforms.CenterCrop(int(pp["crop_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ]
    )


class ImageFeatureModel:
    def __init__(self, model_key: str, model, kind: str):
        self.model_key = model_key
        self.model = model
        self.kind = kind

    def eval(self):
        self.model.eval()
        return self

    def to(self, device: str):
        self.model.to(device)
        return self

    def extract(self, images):
        torch, *_ = import_torch_stack()
        with torch.inference_mode():
            if self.kind == "dinov3_hub" and hasattr(self.model, "forward_features"):
                out = self.model.forward_features(images)
                if isinstance(out, dict):
                    for key in ("x_norm_clstoken", "x_norm_cls_token", "cls_token", "pooler_output"):
                        if key in out:
                            return out[key]
                    if "x_norm_patchtokens" in out:
                        return out["x_norm_patchtokens"].mean(dim=1)
                    raise KeyError(f"Unsupported DINOv3 forward_features keys: {sorted(out.keys())}")
                if getattr(out, "ndim", 0) == 3:
                    return out[:, 0]
                return out

            out = self.model(images)
            if hasattr(out, "pooler_output"):
                return out.pooler_output
            if isinstance(out, dict) and "pooler_output" in out:
                return out["pooler_output"]
            if getattr(out, "ndim", 0) == 3:
                return out[:, 0]
            return out


def _enum_value(enum_cls, value: str):
    if value == "DEFAULT":
        return enum_cls.DEFAULT
    return getattr(enum_cls, value)


def load_model(cfg: dict[str, Any], model_key: str, device: str) -> tuple[ImageFeatureModel, Any]:
    torch, nn, _, ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16 = import_torch_stack()
    model_cfg = cfg["models"][model_key]
    kind = model_cfg["kind"]

    if kind == "dinov3_hub":
        repo_dir = Path(cfg["_project_root"]) / model_cfg["repo_dir"]
        weights = Path(cfg["_project_root"]) / model_cfg["weights"]
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {repo_dir}")
        if not weights.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        hub_dir = Path(cfg["_project_root"]) / "outputs" / ".cache" / "torch" / "hub"
        hub_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(hub_dir))
        model = torch.hub.load(str(repo_dir), model_cfg["hub_name"], source="local", weights=str(weights))
    elif kind == "torchvision_resnet50":
        weights = _enum_value(ResNet50_Weights, model_cfg["weights"])
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
    elif kind == "torchvision_vit_b_16":
        weights = _enum_value(ViT_B_16_Weights, model_cfg["weights"])
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Unsupported model kind: {kind}")

    wrapper = ImageFeatureModel(model_key=model_key, model=model, kind=kind).eval().to(device)
    return wrapper, make_transform(cfg)
