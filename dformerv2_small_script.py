"""Utility script to assemble a DFormerv2 Small network using the project's builder."""

import copy
import importlib.util
import sys
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_mmengine_stubs() -> None:
    """Provide light-weight fallbacks for mmengine/mmcv/mmseg if missing.

    The decoding heads shipped with the project rely on a couple of helper
    classes from the OpenMMLab stack. The execution environment used for the
    script might not have those heavy dependencies installed.  When they are
    missing we provide very small stand-ins that mimic the interfaces required
    by the shipped heads so that the model can still be instantiated for quick
    experiments.
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
                # mimic mmengine behaviour by applying init_cfg if possible
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

    # mmseg is present in the environment shipped with the repo, but the ops
    # submodule depends on mmcv.  When mmcv is replaced with the stub above we
    # also provide a very small replacement for the resize helper used by the
    # decode heads.
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


def _build_config(load_pretrained: bool = False):
    from local_configs.NYUDepthv2.DFormerv2_S import config as base_config

    cfg = copy.deepcopy(base_config)
    if not load_pretrained:
        cfg.pretrained_model = None
    return cfg


class DFormerV2_Small(nn.Module):
    """Thin wrapper around the project's EncoderDecoder for inference/demo."""

    def __init__(self, load_pretrained: bool = False) -> None:
        super().__init__()
        _ensure_mmengine_stubs()
        from models.builder import EncoderDecoder

        cfg = _build_config(load_pretrained=load_pretrained)
        self.cfg = cfg
        self.model = EncoderDecoder(cfg=cfg, criterion=None)
        # Explicitly initialise the decoder since we skipped the criterion.
        self.model.init_weights(cfg, pretrained=cfg.pretrained_model)

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        if depth is None:
            depth = torch.zeros(rgb.size(0), 1, rgb.size(2), rgb.size(3), device=rgb.device, dtype=rgb.dtype)
        return self.model(rgb, depth)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DFormerV2_Small().to(device)
    model.eval()

    rgb = torch.randn(1, 3, 480, 640, device=device)
    depth = torch.randn(1, 1, 480, 640, device=device)

    with torch.no_grad():
        output = model(rgb, depth)
    print("Output shape:", tuple(output.shape))


if __name__ == "__main__":
    main()
