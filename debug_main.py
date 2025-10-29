"""Utility entry point for launching DFormer tasks programmatically.

This module mirrors the behaviour of the shell utilities (``train.sh``,
``infer.sh`` and ``eval.sh``) but exposes Python functions so experiments can
be launched from an IDE or a debugger without typing long command lines.

Example
-------
To kick off a quick training session, adjust the parameters inside ``main``
and run the module directly::

    if __name__ == "__main__":
        main(
            mode="train",
            config="local_configs.NYUDepthv2.DFormerv2_S",
            gpus=1,
            use_seed=False,
        )

The helper functions can also be imported and reused in other scripts::

    from debug_main import launch_train

    launch_train(config="local_configs.NYUDepthv2.DFormer_Base", gpus=4)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent


def _build_env(
    *,
    cuda_visible_devices: Optional[str] = None,
    torchdynamo_verbose: Optional[int] = None,
    extra_pythonpath: Optional[Sequence[Path]] = None,
    overrides: Optional[Mapping[str, str]] = None,
) -> MutableMapping[str, str]:
    """Construct the environment for a ``torchrun`` invocation.

    Parameters
    ----------
    cuda_visible_devices:
        Optional CUDA device specification (e.g. ``"0,1"``). ``None`` leaves
        the current configuration untouched.
    torchdynamo_verbose:
        Optional verbosity flag for TorchDynamo. When ``None`` the variable is
        not modified.
    extra_pythonpath:
        Additional paths to append to ``PYTHONPATH``. The repository expects
        the project root and its parent to be present, mirroring the shell
        scripts shipped for Linux.
    overrides:
        Arbitrary environment variable overrides.
    """

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if torchdynamo_verbose is not None:
        env["TORCHDYNAMO_VERBOSE"] = str(torchdynamo_verbose)

    search_paths: List[str] = []
    if extra_pythonpath:
        search_paths.extend(str(path) for path in extra_pythonpath)
    if search_paths:
        existing = env.get("PYTHONPATH")
        if existing:
            search_paths.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(search_paths)

    if overrides:
        env.update(overrides)
    return env


def _torchrun(command: Sequence[str], env: Optional[Mapping[str, str]] = None) -> None:
    """Execute ``torchrun`` with ``command`` and raise on failure."""

    cmdline = [sys.executable, "-m", "torch.distributed.run", *command]
    subprocess.run(cmdline, check=True, env=dict(env) if env is not None else None)


def launch_train(
    *,
    config: str = "local_configs.NYUDepthv2.DFormerv2_S",
    gpus: int = 2,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29158,
    cuda_visible_devices: str = "0,1",
    torchdynamo_verbose: int = 1,
    sliding: bool = False,
    compile_enabled: bool = False,
    syncbn: bool = True,
    mst: bool = True,
    compile_mode: str = "default",
    amp: bool = False,
    val_amp: bool = True,
    pad_sunrgbd: bool = True,
    use_seed: bool = False,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Launch the training pipeline with sensible defaults.

    The defaults match the original ``train.sh`` helper script.
    """

    env = _build_env(
        cuda_visible_devices=cuda_visible_devices,
        torchdynamo_verbose=torchdynamo_verbose,
        extra_pythonpath=[ROOT.parent, ROOT],
    )

    args = [
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--nproc_per_node={gpus}",
        f"--master_port={master_port}",
        "utils/train.py",
        f"--config={config}",
        f"--gpus={gpus}",
        "--sliding" if sliding else "--no-sliding",
        "--compile" if compile_enabled else "--no-compile",
        "--syncbn" if syncbn else "--no-syncbn",
        "--mst" if mst else "--no-mst",
        f"--compile_mode={compile_mode}",
        "--amp" if amp else "--no-amp",
        "--val_amp" if val_amp else "--no-val_amp",
        "--pad_SUNRGBD" if pad_sunrgbd else "--no-pad_SUNRGBD",
        "--use_seed" if use_seed else "--no-use_seed",
    ]

    if extra_args:
        args.extend(extra_args)

    _torchrun(args, env)


def launch_infer(
    *,
    config: str = "local_configs.NYUDepthv2.DFormer_Large",
    checkpoint_path: str = "checkpoints/trained/NYUv2_DFormer_Large.pth",
    output_dir: str = "output",
    gpus: int = 2,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29958,
    cuda_visible_devices: Optional[str] = None,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Run multi-GPU inference using ``torchrun``."""

    env = _build_env(
        cuda_visible_devices=cuda_visible_devices,
        extra_pythonpath=[ROOT.parent, ROOT],
    )

    args = [
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--nproc_per_node={gpus}",
        f"--master_port={master_port}",
        "utils/infer.py",
        f"--config={config}",
        f"--continue_fpath={checkpoint_path}",
        f"--save_path={output_dir}",
        f"--gpus={gpus}",
    ]

    if extra_args:
        args.extend(extra_args)

    _torchrun(args, env)


def launch_eval(
    *,
    config: str = "local_configs.NYUDepthv2.DFormerv2_S",
    checkpoint_path: str = "checkpoints/trained/DFormerv2_Small_NYU.pth",
    gpus: int = 8,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str = "127.0.0.1",
    master_port: int = 29158,
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7",
    torchdynamo_verbose: int = 1,
    sliding: bool = True,
    compile_enabled: bool = False,
    syncbn: bool = True,
    mst: bool = True,
    compile_mode: str = "reduce-overhead",
    amp: bool = True,
    pad_sunrgbd: bool = True,
    extra_args: Optional[Iterable[str]] = None,
) -> None:
    """Evaluate a trained checkpoint with defaults mirroring ``eval.sh``."""

    env = _build_env(
        cuda_visible_devices=cuda_visible_devices,
        torchdynamo_verbose=torchdynamo_verbose,
        extra_pythonpath=[ROOT.parent, ROOT],
    )

    args = [
        f"--nnodes={nnodes}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--nproc_per_node={gpus}",
        f"--master_port={master_port}",
        "utils/eval.py",
        f"--config={config}",
        f"--gpus={gpus}",
        "--sliding" if sliding else "--no-sliding",
        "--compile" if compile_enabled else "--no-compile",
        "--syncbn" if syncbn else "--no-syncbn",
        "--mst" if mst else "--no-mst",
        f"--compile_mode={compile_mode}",
        "--amp" if amp else "--no-amp",
        "--pad_SUNRGBD" if pad_sunrgbd else "--no-pad_SUNRGBD",
        f"--continue_fpath={checkpoint_path}",
    ]

    if extra_args:
        args.extend(extra_args)

    _torchrun(args, env)


def main(
    mode: str = "train",
    *,
    config: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "output",
    gpus: Optional[int] = None,
) -> None:
    """Basic front-end for quickly switching between common workflows.

    Parameters
    ----------
    mode:
        One of ``"train"``, ``"infer"`` or ``"eval"``.
    config:
        Optional configuration override. When left as ``None`` the defaults
        defined in each launcher are used.
    checkpoint_path:
        Required for ``"infer"`` and ``"eval"`` when not provided by the
        defaults.
    output_dir:
        Output directory used by inference.
    gpus:
        Override the number of GPUs. If ``None`` the launcher defaults are
        reused.
    """

    mode = mode.lower()
    if mode == "train":
        launch_train(
            config=config or "local_configs.NYUDepthv2.DFormerv2_S",
            gpus=gpus or 2,
        )
    elif mode == "infer":
        launch_infer(
            config=config or "local_configs.NYUDepthv2.DFormer_Large",
            checkpoint_path=checkpoint_path or "checkpoints/trained/NYUv2_DFormer_Large.pth",
            output_dir=output_dir,
            gpus=gpus or 2,
        )
    elif mode == "eval":
        launch_eval(
            config=config or "local_configs.NYUDepthv2.DFormerv2_S",
            checkpoint_path=checkpoint_path or "checkpoints/trained/DFormerv2_Small_NYU.pth",
            gpus=gpus or 8,
        )
    else:
        raise ValueError(f"Unsupported mode '{mode}'.")


if __name__ == "__main__":
    main()
