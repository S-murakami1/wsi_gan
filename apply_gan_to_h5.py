"""Apply a saved generator to ``cache/512/patches`` and write ``cache/512/gan/patches``.

Then save a downsampled montage of all transformed patches (one PNG per HDF5).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from loguru import logger

from config import config
from generator import ResidualGenerator


DST_KEY_DEFAULT = "cache/512/gan/patches"


def uint8_nhwc_to_model_input(x: np.ndarray) -> torch.Tensor:
    """(N, H, W, C) uint8 -> (N, 3, H, W) float in [-1, 1]."""
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    return t * 2.0 - 1.0


def model_output_to_uint8_nhwc(t: torch.Tensor) -> np.ndarray:
    """(N, 3, H, W) float -> (N, H, W, C) uint8."""
    x = t.detach().float().cpu().clamp(-1.0, 1.0)
    x = (x + 1.0) * 0.5 * 255.0
    return x.round().byte().permute(0, 2, 3, 1).numpy()


def ensure_dataset_at_key(
    h5: h5py.File,
    dataset_key: str,
    n: int,
    h: int,
    w: int,
    c: int,
    chunks: tuple[int, ...] | None,
) -> h5py.Dataset:
    """Create or replace dataset at ``dataset_key`` (must end with ``/patches``)."""
    parts = dataset_key.strip("/").split("/")
    if len(parts) < 2 or parts[-1] != "patches":
        raise ValueError(f"dataset_key must end with /patches, got {dataset_key!r}")
    grp = h5
    for name in parts[:-1]:
        grp = grp.require_group(name)
    leaf = parts[-1]
    if leaf in grp:
        del grp[leaf]
    return grp.create_dataset(
        leaf,
        shape=(n, h, w, c),
        dtype=np.uint8,
        chunks=chunks,
        compression="gzip",
        compression_opts=4,
    )


def build_montage(
    thumbs: list[np.ndarray],
    ncol: int,
    max_side: int,
) -> Image.Image:
    """Grid of RGB uint8 thumbnails; resize so longest edge <= ``max_side``."""
    if not thumbs:
        raise ValueError("empty thumbs")
    n = len(thumbs)
    nrow = math.ceil(n / ncol)
    t = int(thumbs[0].shape[0])
    canvas = np.zeros((nrow * t, ncol * t, 3), dtype=np.uint8)
    for i, im in enumerate(thumbs):
        r, c = divmod(i, ncol)
        canvas[r * t : (r + 1) * t, c * t : (c + 1) * t] = im
    pil = Image.fromarray(canvas, mode="RGB")
    w, h = pil.size
    m = max(w, h)
    if m > max_side:
        pil = pil.resize((int(w * max_side / m), int(h * max_side / m)), Image.Resampling.LANCZOS)
    return pil


def process_one_h5(
    h5_path: Path,
    G: torch.nn.Module,
    device: torch.device,
    src_key: str,
    dst_key: str,
    batch_size: int,
    thumb_size: int,
    montage_max_side: int,
) -> None:
    with h5py.File(h5_path, "r+") as f:
        if src_key not in f:
            raise KeyError(f"{h5_path}: missing source dataset {src_key!r}")
        src = f[src_key]
        if len(src.shape) != 4 or src.shape[-1] != 3:
            raise ValueError(f"{h5_path}: expected (N,H,W,3), got shape {src.shape}")
        n, height, width, c = src.shape
        logger.info("{} | patches={} shape={}", h5_path.name, n, (height, width, c))

        chunk = getattr(src, "chunks", None)
        if chunk is not None and len(chunk) == 4:
            out_chunks = chunk
        else:
            out_chunks = (min(8, n), height, width, c)

        out_ds = ensure_dataset_at_key(f, dst_key, n, height, width, c, out_chunks)

        thumbs: list[np.ndarray] = []
        G.eval()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = np.asarray(src[start:end], dtype=np.uint8)
                xb = uint8_nhwc_to_model_input(batch).to(device)
                yb = G(xb)
                out_hwc = model_output_to_uint8_nhwc(yb)
                out_ds[start:end] = out_hwc

                small = torch.nn.functional.interpolate(
                    yb,
                    size=(thumb_size, thumb_size),
                    mode="bilinear",
                    align_corners=False,
                )
                for k in range(small.shape[0]):
                    sk = model_output_to_uint8_nhwc(small[k : k + 1])[0]
                    thumbs.append(sk)

        f.flush()

    ncol = max(1, math.ceil(math.sqrt(n)))
    montage_path = h5_path.with_name(f"{h5_path.stem}_gan_montage.png")
    img = build_montage(thumbs, ncol=ncol, max_side=montage_max_side)
    img.save(montage_path)
    logger.success("montage saved | {}", montage_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=getattr(config, "gan_checkpoint_path", None),
        help="checkpoint.pt from training (contains G_xy / G_yx)",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=getattr(config, "gan_export_h5_dir", None) or config.patho2_h5_dir,
        help="Directory containing HDF5 slide files",
    )
    parser.add_argument("--glob", type=str, default=config.h5_glob)
    parser.add_argument("--src-key", type=str, default=config.patches_key)
    parser.add_argument("--dst-key", type=str, default=DST_KEY_DEFAULT)
    parser.add_argument(
        "--which",
        choices=("xy", "yx"),
        default=getattr(config, "gan_export_which", "xy"),
        help="xy=G_xy (patho2->morph style), yx=G_yx (morph->patho2 style)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(getattr(config, "gan_export_batch_size", 8)),
    )
    parser.add_argument(
        "--thumb",
        type=int,
        default=int(getattr(config, "gan_montage_thumb", 48)),
        help="Thumbnail pixel size (square) for montage cells",
    )
    parser.add_argument(
        "--montage-max-side",
        type=int,
        default=int(getattr(config, "gan_montage_max_side", 4096)),
        help="Max width/height of final montage PNG (downsampled if larger)",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.checkpoint is None or not Path(args.checkpoint).is_file():
        raise SystemExit(
            "Pass --checkpoint path/to/checkpoint.pt or set config.gan_checkpoint_path"
        )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    key = "G_xy" if args.which == "xy" else "G_yx"
    if key not in ckpt:
        raise KeyError(f"Checkpoint missing {key}; keys: {list(ckpt.keys())}")

    G = ResidualGenerator().to(device)
    G.load_state_dict(ckpt[key])
    G.eval()

    paths = sorted(args.h5_dir.glob(args.glob))
    if not paths:
        raise FileNotFoundError(f"No HDF5 under {args.h5_dir} with {args.glob!r}")

    logger.info(
        "apply {} | checkpoint={} | files={} | src={} -> dst={} | device={} | bs={}",
        key,
        args.checkpoint.resolve(),
        len(paths),
        args.src_key,
        args.dst_key,
        device,
        args.batch_size,
    )

    for p in paths:
        process_one_h5(
            p,
            G,
            device,
            args.src_key,
            args.dst_key,
            batch_size=max(1, args.batch_size),
            thumb_size=max(8, args.thumb),
            montage_max_side=max(256, args.montage_max_side),
        )


if __name__ == "__main__":
    main()
