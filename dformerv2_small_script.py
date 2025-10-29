"""Assemble a DFormerV2 Small network from its individual building blocks."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_mmengine_stubs() -> None:
    """Provide light-weight fallbacks for mmengine/mmcv/mmseg if missing.

    The decoding heads shipped with the project rely on helpers from the
    OpenMMLab stack.  When those heavy dependencies are not installed we create
    small stand-ins that mimic the tiny parts of the API exercised by the
    DFormer modules so that the network can still be instantiated.
    """

    if importlib.util.find_spec("mmengine") is None:
        mmengine_module = types.ModuleType("mmengine")
        model_module = types.ModuleType("mmengine.model")
        base_module = types.ModuleType("mmengine.model.base_module")

        class BaseModule(nn.Module):
            def __init__(self, init_cfg: Optional[dict] = None):
                super().__init__()
                self.init_cfg = init_cfg

            def init_weights(self) -> None:  # pragma: no cover - simple stub
                if self.init_cfg is None:
                    return
                if isinstance(self.init_cfg, dict) and self.init_cfg.get("type") == "Normal":
                    std = self.init_cfg.get("std", 1.0)
                    name = self.init_cfg.get("override", {}).get("name")
                    if name is not None and hasattr(self, name):
                        module = getattr(self, name)
                        if isinstance(module, nn.Conv2d):
                            nn.init.normal_(module.weight, std=std)
                            if module.bias is not None:
                                nn.init.constant_(module.bias, 0)

        base_module.BaseModule = BaseModule
        model_module.base_module = base_module
        mmengine_module.model = model_module

        sys.modules["mmengine"] = mmengine_module
        sys.modules["mmengine.model"] = model_module
        sys.modules["mmengine.model.base_module"] = base_module

    if importlib.util.find_spec("mmcv") is None:
        mmcv_module = types.ModuleType("mmcv")
        cnn_module = types.ModuleType("mmcv.cnn")

        class ConvModule(nn.Module):
            def __init__(
                self,
                in_channels: int,
                out_channels: int,
                kernel_size: int,
                stride: int = 1,
                padding: int = 0,
                bias: Optional[bool] = None,
                conv_cfg: Optional[dict] = None,
                norm_cfg: Optional[dict] = None,
                act_cfg: Optional[dict] = dict(type="ReLU"),
            ) -> None:
                super().__init__()
                if conv_cfg is not None:
                    raise ValueError("This lightweight ConvModule stub only supports conv_cfg=None")

                if bias is None:
                    bias = norm_cfg is None

                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )

                self.norm: Optional[nn.Module]
                if norm_cfg is None:
                    self.norm = None
                else:
                    norm_type = norm_cfg.get("type", "BN")
                    requires_grad = norm_cfg.get("requires_grad", True)
                    if norm_type in ("BN", "SyncBN"):
                        self.norm = nn.BatchNorm2d(out_channels)
                    else:
                        raise ValueError(f"Unsupported norm type: {norm_type}")
                    if not requires_grad:
                        for param in self.norm.parameters():
                            param.requires_grad = False

                if act_cfg is None:
                    self.activate = None
                else:
                    act_type = act_cfg.get("type", "ReLU")
                    if act_type == "ReLU":
                        inplace = act_cfg.get("inplace", True)
                        self.activate = nn.ReLU(inplace=inplace)
                    elif act_type == "GELU":
                        self.activate = nn.GELU()
                    else:
                        raise ValueError(f"Unsupported activation type: {act_type}")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                if self.norm is not None:
                    x = self.norm(x)
                if self.activate is not None:
                    x = self.activate(x)
                return x

        cnn_module.ConvModule = ConvModule
        mmcv_module.cnn = cnn_module
        sys.modules["mmcv"] = mmcv_module
        sys.modules["mmcv.cnn"] = cnn_module

    if importlib.util.find_spec("mmseg.ops") is None:
        mmseg_module = sys.modules.get("mmseg")
        if mmseg_module is None:
            mmseg_module = types.ModuleType("mmseg")
        ops_module = types.ModuleType("mmseg.ops")

        def resize(input: torch.Tensor, size=None, scale_factor=None, mode="bilinear", align_corners=False):
            return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        ops_module.resize = resize
        mmseg_module.ops = ops_module
        sys.modules["mmseg"] = mmseg_module
        sys.modules["mmseg.ops"] = ops_module


@dataclass
class DFormerV2SmallConfig:
    """Configuration for assembling the DFormerV2 Small network."""

    num_classes: int = 40
    drop_path_rate: float = 0.25
    decoder_embed_dim: int = 512
    bn_eps: float = 1e-3
    bn_momentum: float = 0.1
    use_syncbn: bool = False
    pretrained_path: Optional[str] = None

    @property
    def channels(self) -> Sequence[int]:
        return (64, 128, 256, 512)

    @property
    def norm_cfg(self) -> dict:
        return dict(type="SyncBN" if self.use_syncbn else "BN", requires_grad=True)


class DFormerV2_Small(nn.Module):
    """Explicit composition of the DFormerV2 Small RGB-D segmentation model."""

    def __init__(self, cfg: Optional[DFormerV2SmallConfig] = None) -> None:
        super().__init__()
        _ensure_mmengine_stubs()

        if cfg is None:
            cfg = DFormerV2SmallConfig()
        self.cfg = cfg

        from models.encoders.DFormerv2 import DFormerv2_S
        from models.decoders.ham_head import LightHamHead
        from utils.init_func import init_weight

        norm_layer: type[nn.Module]
        if cfg.use_syncbn and hasattr(nn, "SyncBatchNorm"):
            norm_layer = nn.SyncBatchNorm
        else:
            norm_layer = nn.BatchNorm2d

        self.backbone = DFormerv2_S(drop_path_rate=cfg.drop_path_rate, norm_cfg=cfg.norm_cfg)
        self.encoder = self.backbone.layers  # expose encoder stack for clarity

        self.decode_head = LightHamHead(
            in_channels=cfg.channels[1:],
            in_index=[1, 2, 3],
            channels=cfg.decoder_embed_dim,
            ham_channels=cfg.decoder_embed_dim,
            num_classes=cfg.num_classes,
            norm_cfg=cfg.norm_cfg,
            align_corners=False,
        )

        self.decoder = self.decode_head  # alias for readability
        self.head = self.decode_head.cls_seg
        self.align_corners = False

        self._init_weights(init_weight, norm_layer)

    def _init_weights(self, init_weight, norm_layer: type[nn.Module]) -> None:
        pretrained = self.cfg.pretrained_path
        if pretrained is not None and not os.path.exists(pretrained):
            raise FileNotFoundError(f"Pretrained weights not found: {pretrained}")

        self.backbone.init_weights(pretrained)
        init_weight(
            self.decode_head,
            nn.init.kaiming_normal_,
            norm_layer,
            self.cfg.bn_eps,
            self.cfg.bn_momentum,
            mode="fan_in",
            nonlinearity="relu",
        )

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if depth is None:
            depth = torch.zeros(rgb.size(0), 1, rgb.size(2), rgb.size(3), device=rgb.device, dtype=rgb.dtype)

        features = self.backbone(rgb, depth)
        logits = self.decode_head(features)
        logits = F.interpolate(logits, size=rgb.shape[2:], mode="bilinear", align_corners=self.align_corners)
        return logits


def demo_forward_pass() -> Tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass with random tensors to demonstrate the model wiring."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFormerV2_Small().to(device)
    model.eval()

    rgb = torch.randn(1, 3, 480, 640, device=device)
    depth = torch.randn(1, 1, 480, 640, device=device)

    with torch.no_grad():
        logits = model(rgb, depth)

    return logits.cpu(), rgb.cpu()


def main() -> None:
    logits, rgb = demo_forward_pass()
    print("Input RGB shape:", tuple(rgb.shape))
    print("Output logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
