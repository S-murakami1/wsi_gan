from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer

from utils.multi_h5_sampling import MultiH5PatchSampler


@dataclass
class TrainingState:
    G_xy: nn.Module
    G_yx: nn.Module
    D_x: nn.Module
    D_y: nn.Module
    optimizer_G: Optimizer
    optimizer_D: Optimizer
    sampler_x: MultiH5PatchSampler
    sampler_y: MultiH5PatchSampler


@dataclass
class ResumeInfo:
    start_step: int
    hist_step: list[int]
    hist_loss_G: list[float]
    hist_loss_D: list[float]


def save_training_checkpoint(
    path: Path,
    step: int,
    state: TrainingState,
    hist_step: list[int],
    hist_loss_G: list[float],
    hist_loss_D: list[float],
    seed: int | None = None,
) -> None:
    payload: dict[str, object] = {
        "step": step,
        "G_xy": state.G_xy.state_dict(),
        "G_yx": state.G_yx.state_dict(),
        "D_x": state.D_x.state_dict(),
        "D_y": state.D_y.state_dict(),
        "optimizer_G": state.optimizer_G.state_dict(),
        "optimizer_D": state.optimizer_D.state_dict(),
        "sampler_x": state.sampler_x.state_dict(),
        "sampler_y": state.sampler_y.state_dict(),
        "hist_step": hist_step,
        "hist_loss_G": hist_loss_G,
        "hist_loss_D": hist_loss_D,
    }
    if seed is not None:
        payload["seed"] = seed
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    device: torch.device,
    state: TrainingState,
    seed: int | None = None,
) -> tuple[int, list[int], list[float], list[float]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state.G_xy.load_state_dict(ckpt["G_xy"])
    state.G_yx.load_state_dict(ckpt["G_yx"])
    state.D_x.load_state_dict(ckpt["D_x"])
    state.D_y.load_state_dict(ckpt["D_y"])
    state.optimizer_G.load_state_dict(ckpt["optimizer_G"])
    state.optimizer_D.load_state_dict(ckpt["optimizer_D"])

    if "sampler_x" in ckpt and "sampler_y" in ckpt:
        state.sampler_x.load_state_dict(ckpt["sampler_x"])
        state.sampler_y.load_state_dict(ckpt["sampler_y"])
    else:
        logger.warning(
            "checkpoint has no sampler state; samplers will start from a fresh epoch"
        )

    ckpt_seed = ckpt.get("seed")
    if ckpt_seed is not None and seed is not None and int(ckpt_seed) != seed:
        logger.warning(
            "checkpoint seed {} differs from current seed {}",
            ckpt_seed,
            seed,
        )

    step = int(ckpt["step"])
    hist_step = [int(s) for s in ckpt.get("hist_step", [])]
    hist_loss_G = [float(v) for v in ckpt.get("hist_loss_G", [])]
    hist_loss_D = [float(v) for v in ckpt.get("hist_loss_D", [])]
    return step, hist_step, hist_loss_G, hist_loss_D


def resolve_resume(
    resume: Path | None,
    device: torch.device,
    state: TrainingState,
    total_steps: int,
    seed: int | None = None,
) -> ResumeInfo | None:
    if resume is None:
        return ResumeInfo(1, [], [], [])

    if not resume.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume}")

    last_step, hist_step, hist_loss_G, hist_loss_D = load_training_checkpoint(
        resume, device, state, seed=seed
    )
    start_step = last_step + 1
    logger.info(
        "resumed from {} | last_step={} | next_step={} | "
        "x sampler epoch={} pos={}/{} | y sampler epoch={} pos={}/{} | "
        "loss history={} points",
        resume.resolve(),
        last_step,
        start_step,
        state.sampler_x.epoch,
        state.sampler_x.position,
        state.sampler_x.total_patches,
        state.sampler_y.epoch,
        state.sampler_y.position,
        state.sampler_y.total_patches,
        len(hist_step),
    )
    if start_step > total_steps:
        logger.success(
            "training already complete | last_step={} total_steps={}",
            last_step,
            total_steps,
        )
        return None

    return ResumeInfo(start_step, hist_step, hist_loss_G, hist_loss_D)
