from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from loguru import logger

from config import config
from generator import ResidualGenerator
from discriminator import PatchDiscriminator
from losses import discriminator_loss, generator_gan_loss, cycle_consistency_loss


class H5PatchSampler:

    def __init__(self, h5_path: Path, patches_key: str, device: torch.device):
        self._device = device
        self._f = h5py.File(h5_path, "r")
        self._ds = self._f[patches_key]
        self.n = self._ds.shape[0]

    def sample(self) -> torch.Tensor:
        i = int(torch.randint(0, self.n, (1,)).item())
        patch = np.asarray(self._ds[i], dtype=np.uint8)
        t = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        t = t * 2.0 - 1.0
        return t.to(self._device)

    def close(self) -> None:
        self._f.close()


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
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(x)).save(sub / "x_real.png")
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(y)).save(sub / "y_real.png")
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(fake_y)).save(sub / "fake_y_Gxy.png")
    Image.fromarray(tensor_minus1_1_to_uint8_hwc(fake_x)).save(sub / "fake_x_Gyx.png")
    return sub


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path.cwd() / config.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    patho2_h5 = Path(config.patho2_patches_h5)
    morph_h5 = Path(config.morph_patches_h5)

    logger.info(
        "train start | device={} | steps={} | save_every={} | cwd={} | out={} | patho2={} | morph={}",
        device,
        config.total_steps,
        config.save_every,
        Path.cwd(),
        out_root.resolve(),
        patho2_h5,
        morph_h5,
    )

    sampler_x = H5PatchSampler(patho2_h5, config.patches_key, device)
    sampler_y = H5PatchSampler(morph_h5, config.patches_key, device)

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

        G_xy.train()
        G_yx.train()
        D_x.train()
        D_y.train()

        pbar = tqdm(range(1, config.total_steps + 1), desc="train", unit="step")
        for step in pbar:
            x = sampler_x.sample()
            y = sampler_y.sample()

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

            loss_D_y = discriminator_loss(D_y, y, fake_y)
            loss_D_x = discriminator_loss(D_x, x, fake_x)
            loss_D = config.lambda_discriminator * (loss_D_x + loss_D_y)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            pbar.set_postfix(
                G=float(loss_G.detach().cpu()),
                D=float(loss_D.detach().cpu()),
                refresh=False,
            )

            if step % config.save_every == 0:
                with torch.no_grad():
                    fx = G_yx(y)
                    fy = G_xy(x)
                sub = save_generated_preview(out_root, step, x, y, fy, fx)
                logger.info(
                    "step {} | loss_G={:.6f} loss_D={:.6f} | saved {}",
                    step,
                    float(loss_G.detach().cpu()),
                    float(loss_D.detach().cpu()),
                    sub.resolve(),
                )

    finally:
        sampler_x.close()
        sampler_y.close()
        logger.success("train finished | steps={}", config.total_steps)


if __name__ == "__main__":
    train()
