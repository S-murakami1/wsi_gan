import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger

from config import config
from discriminator import PatchDiscriminator
from generator import ResidualGenerator
from losses import discriminator_loss, generator_gan_loss, cycle_consistency_loss
from utils.multi_h5_sampling import (
    PatchIndexCatalog,
    RandomMultiH5PatchSampler,
    list_h5_paths,
)


def tensor_minus1_1_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    x = t.detach().float().cpu().clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5 * 255.0
    return x.squeeze(0).permute(1, 2, 0).numpy().round().astype(np.uint8)


def save_generated_preview(
    out_dir: Path,
    step: int,
    x: torch.Tensor,
    y: torch.Tensor,
    fake_y: torch.Tensor,
    fake_x: torch.Tensor,
) -> Path:
    sub = out_dir / f"step_{step:06d}"
    sub.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(x[0:1])).save(sub / "x_real.png")
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(y[0:1])).save(sub / "y_real.png")
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(fake_y[0:1])).save(sub / "fake_y_Gxy.png")
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(fake_x[0:1])).save(sub / "fake_x_Gyx.png")
    return sub


def save_training_checkpoint(
    path: Path,
    step: int,
    G_xy: torch.nn.Module,
    G_yx: torch.nn.Module,
    D_x: torch.nn.Module,
    D_y: torch.nn.Module,
    optimizer_G: torch.optim.Optimizer,
    optimizer_D: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "step": step,
            "G_xy": G_xy.state_dict(),
            "G_yx": G_yx.state_dict(),
            "D_x": D_x.state_dict(),
            "D_y": D_y.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
        },
        path,
    )


def save_loss_curve_png(
    path: Path,
    steps: list[int],
    loss_G_hist: list[float],
    loss_D_hist: list[float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(steps, loss_G_hist, label="loss_G", alpha=0.9, linewidth=0.8)
    ax.plot(steps, loss_D_hist, label="loss_D", alpha=0.9, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.35)
    ax.set_title("Training losses (updated each save_every)")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


def _lr_schedule_multiplier(step: int, total_steps: int, kind: str, end_ratio: float) -> float:
    k = (kind or "none").strip().lower()
    if k == "none" or total_steps <= 1:
        return 1.0
    if k not in {"linear", "cosine"}:
        raise ValueError(f"Unknown lr_schedule: {kind!r} (use none, linear, cosine)")
    t = (step - 1) / float(total_steps - 1)
    if k == "linear":
        return 1.0 + (end_ratio - 1.0) * t
    return end_ratio + (1.0 - end_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


def train(h5_dir_x: Path, h5_dir_y: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path.cwd() / config.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    h5_paths_x = list_h5_paths(h5_dir_x, config.h5_glob)
    h5_paths_y = list_h5_paths(h5_dir_y, config.h5_glob)
    catalog_x = PatchIndexCatalog(h5_paths_x, config.patches_key)
    catalog_y = PatchIndexCatalog(h5_paths_y, config.patches_key)
    lr_sched = config.lr_schedule
    lr_end_ratio = float(config.lr_schedule_end_ratio)

    logger.info(
        "train start | device={} | batch_size={} | steps={} | save_every={} | "
        "g_updates_per_step={} | d_updates_per_step={} | cwd={} | out={} | h5_dir_x={} ({} files, {} patches) | "
        "h5_dir_y={} ({} files, {} patches) | h5_max_open={} | lr_schedule={} | lr_end_ratio={}",
        device,
        config.batch_size,
        config.total_steps,
        config.save_every,
        config.g_updates_per_step,
        config.d_updates_per_step,
        Path.cwd(),
        out_root.resolve(),
        h5_dir_x.resolve(),
        len(h5_paths_x),
        catalog_x.total_patches,
        h5_dir_y.resolve(),
        len(h5_paths_y),
        catalog_y.total_patches,
        config.h5_max_open_files,
        lr_sched,
        lr_end_ratio,
    )

    sampler_x = RandomMultiH5PatchSampler(
        catalog_x, config.patches_key, device, config.h5_max_open_files
    )
    sampler_y = RandomMultiH5PatchSampler(
        catalog_y, config.patches_key, device, config.h5_max_open_files
    )

    try:
        G_xy = ResidualGenerator().to(device)
        G_yx = ResidualGenerator().to(device)
        D_x = PatchDiscriminator().to(device)
        D_y = PatchDiscriminator().to(device)

        optimizer_G = torch.optim.Adam(
            list(G_xy.parameters()) + list(G_yx.parameters()),
            lr=config.lr_G,
            betas=(config.beta1, config.beta2),
        )
        optimizer_D = torch.optim.Adam(
            list(D_x.parameters()) + list(D_y.parameters()),
            lr=config.lr_D,
            betas=(config.beta1, config.beta2),
        )

        base_lr_G = float(config.lr_G)
        base_lr_D = float(config.lr_D)

        G_xy.train()
        G_yx.train()
        D_x.train()
        D_y.train()

        hist_step: list[int] = []
        hist_loss_G: list[float] = []
        hist_loss_D: list[float] = []
        loss_plot_path = out_root / config.loss_curve_png

        bs = max(1, int(config.batch_size))
        g_steps = max(1, int(config.g_updates_per_step))
        d_steps = max(1, int(config.d_updates_per_step))
        pbar = tqdm(range(1, config.total_steps + 1), desc="train", unit="step")
        for step in pbar:
            m = _lr_schedule_multiplier(step, config.total_steps, lr_sched, lr_end_ratio)
            optimizer_G.param_groups[0]["lr"] = base_lr_G * m
            optimizer_D.param_groups[0]["lr"] = base_lr_D * m
            
            # Generator training
            loss_G_last: torch.Tensor | None = None
            for _ in range(g_steps):
                x = sampler_x.sample_batch(bs)
                y = sampler_y.sample_batch(bs)
                fake_y = G_xy(x)
                rec_x = G_yx(fake_y)
                fake_x = G_yx(y)
                rec_y = G_xy(fake_x)
                loss_gan_xy = generator_gan_loss(D_y, fake_y)
                loss_gan_yx = generator_gan_loss(D_x, fake_x)
                loss_cycle_x = cycle_consistency_loss(x, rec_x)
                loss_cycle_y = cycle_consistency_loss(y, rec_y)
                loss_G = (
                    config.lambda_gan * (loss_gan_xy + loss_gan_yx)
                    + config.lambda_cycle * (loss_cycle_x + loss_cycle_y)
                )
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()
                loss_G_last = loss_G

            # Discriminator training
            loss_D_last: torch.Tensor | None = None
            for _ in range(d_steps):
                x = sampler_x.sample_batch(bs)
                y = sampler_y.sample_batch(bs)
                fake_y = G_xy(x)
                fake_x = G_yx(y)
                loss_D_y = discriminator_loss(D_y, y, fake_y)
                loss_D_x = discriminator_loss(D_x, x, fake_x)
                loss_D = config.lambda_discriminator * (loss_D_x + loss_D_y)
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()
                loss_D_last = loss_D

            # Save Checkpoint and Loss Curve
            assert loss_G_last is not None and loss_D_last is not None
            g_val = float(loss_G_last.detach().cpu())
            d_val = float(loss_D_last.detach().cpu())
            hist_step.append(step)
            hist_loss_G.append(g_val)
            hist_loss_D.append(d_val)
            pbar.set_postfix(G=g_val, D=d_val, refresh=False)

            if step % config.save_every == 0:
                with torch.no_grad():
                    fx = G_yx(y)
                    fy = G_xy(x)
                sub = save_generated_preview(out_root, step, x, y, fy, fx)
                ckpt_path = sub / "checkpoint.pt"
                save_training_checkpoint(
                    ckpt_path,
                    step,
                    G_xy,
                    G_yx,
                    D_x,
                    D_y,
                    optimizer_G,
                    optimizer_D,
                )
                save_loss_curve_png(loss_plot_path, hist_step, hist_loss_G, hist_loss_D)
                logger.info(
                    "step {} | loss_G={:.6f} loss_D={:.6f} | previews={} | checkpoint={} | loss_plot={}",
                    step,
                    g_val,
                    d_val,
                    sub.resolve(),
                    ckpt_path.resolve(),
                    loss_plot_path.resolve(),
                )

    finally:
        sampler_x.close()
        sampler_y.close()
        logger.success("train finished | steps={}", config.total_steps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train patch CycleGAN on two HDF5 patch directories.",
    )
    p.add_argument(
        "--h5-dir-x",
        type=Path,
        required=True,
        help="HDF5 directory for domain X",
    )
    p.add_argument(
        "--h5-dir-y",
        type=Path,
        required=True,
        help="HDF5 directory for domain Y",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.h5_dir_x, args.h5_dir_y)
