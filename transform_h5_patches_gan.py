import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm

from config import config
from generator import ResidualGenerator


DEFAULT_CHECKPOINT = Path("train_outputs/step_010000/checkpoint.pt")
DEFAULT_WHICH = "xy"
DEFAULT_BATCH_SIZE = 8
DEFAULT_THUMB = 48
DEFAULT_MONTAGE_MAX_SIDE = 4096


def uint8_nhwc_to_model_input(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    return t * 2.0 - 1.0


def model_output_to_uint8_nhwc(t: torch.Tensor) -> np.ndarray:
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


def load_coordinates_xy(h5: h5py.File, coord_key: str, n: int) -> np.ndarray | None:
    if coord_key not in h5:
        return None
    arr = np.asarray(h5[coord_key][...])
    if arr.shape[0] != n:
        raise ValueError(
            f"{coord_key}: row count {arr.shape[0]} != patches {n}"
        )
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{coord_key}: expected (N, 2+) for x,y columns, got shape {arr.shape}")
    return arr[:, :2].astype(np.float64, copy=False)


def _paste_thumb_rgb(canvas: np.ndarray, thumb: np.ndarray, y: int, x: int) -> None:
    th, tw = int(thumb.shape[0]), int(thumb.shape[1])
    H, W = canvas.shape[0], canvas.shape[1]
    y1, x1 = max(0, y), max(0, x)
    y2, x2 = min(H, y + th), min(W, x + tw)
    if y1 >= y2 or x1 >= x2:
        return
    sy1, sx1 = y1 - y, x1 - x
    sy2 = sy1 + (y2 - y1)
    sx2 = sx1 + (x2 - x1)
    canvas[y1:y2, x1:x2] = thumb[sy1:sy2, sx1:sx2]


def build_spatial_montage(
    thumbs: list[np.ndarray],
    coords_xy: np.ndarray,
    patch_h: int,
    patch_w: int,
    max_side: int,
) -> Image.Image:
    if not thumbs:
        raise ValueError("empty thumbs")
    n = len(thumbs)
    if coords_xy.shape[0] != n:
        raise ValueError("coords length mismatch")
    ts = int(thumbs[0].shape[0])
    xy = coords_xy
    xmin, ymin = float(xy[:, 0].min()), float(xy[:, 1].min())
    xmax, ymax = float(xy[:, 0].max()), float(xy[:, 1].max())
    span_w = xmax - xmin + float(patch_w)
    span_h = ymax - ymin + float(patch_h)
    scale_w = ts / float(patch_w)
    scale_h = ts / float(patch_h)
    canvas_w = max(1, int(math.ceil(span_w * scale_w)))
    canvas_h = max(1, int(math.ceil(span_h * scale_h)))
    canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)

    for i in range(n):
        x0 = int(math.floor((xy[i, 0] - xmin) * scale_w))
        y0 = int(math.floor((xy[i, 1] - ymin) * scale_h))
        _paste_thumb_rgb(canvas, thumbs[i], y0, x0)

    pil = Image.fromarray(canvas, mode="RGB")
    w, h = pil.size
    m = max(w, h)
    if m > max_side:
        pil = pil.resize((int(w * max_side / m), int(h * max_side / m)), Image.Resampling.LANCZOS)
    return pil


def build_montage(
    thumbs: list[np.ndarray],
    ncol: int,
    max_side: int,
) -> Image.Image:
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
    coord_key: str | None,
) -> None:
    with h5py.File(h5_path, "r+") as f:
        if src_key not in f:
            raise KeyError(f"{h5_path}: missing source dataset {src_key!r}")
        src = f[src_key]
        if len(src.shape) != 4 or src.shape[-1] != 3:
            raise ValueError(f"{h5_path}: expected (N,H,W,3), got shape {src.shape}")
        n, height, width, c = src.shape
        logger.info("{} | patches={} shape={}", h5_path.name, n, (height, width, c))

        coords_xy: np.ndarray | None = None
        if coord_key:
            coords_xy = load_coordinates_xy(f, coord_key, n)
            if coords_xy is not None:
                logger.info("{} | spatial montage from {}", h5_path.name, coord_key)
            else:
                logger.info(
                    "{} | no {} — using index grid montage",
                    h5_path.name,
                    coord_key,
                )

        chunk = getattr(src, "chunks", None)
        if chunk is not None and len(chunk) == 4:
            out_chunks = chunk
        else:
            out_chunks = (min(8, n), height, width, c)

        out_ds = ensure_dataset_at_key(f, dst_key, n, height, width, c, out_chunks)

        thumbs_before: list[np.ndarray] = []
        thumbs_after: list[np.ndarray] = []
        G.eval()
        n_batch = max(1, (n + batch_size - 1) // batch_size)
        with torch.no_grad():
            for start in tqdm(
                range(0, n, batch_size),
                desc=h5_path.name,
                unit="batch",
                total=n_batch,
                leave=False,
            ):
                end = min(start + batch_size, n)
                batch = np.asarray(src[start:end], dtype=np.uint8)
                batch_t = (
                    torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0).to(device)
                )
                small_before = torch.nn.functional.interpolate(
                    batch_t,
                    size=(thumb_size, thumb_size),
                    mode="bilinear",
                    align_corners=False,
                )
                small_before_u8 = (
                    small_before.clamp(0.0, 1.0).mul(255.0).round().byte().permute(0, 2, 3, 1)
                )

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
                    thumbs_before.append(small_before_u8[k].cpu().numpy())
                    sk = model_output_to_uint8_nhwc(small[k : k + 1])[0]
                    thumbs_after.append(sk)

        f.flush()

    def _save_montage(thumbs: list[np.ndarray], suffix: str) -> Path:
        path = h5_path.with_name(f"{h5_path.stem}_gan_montage{suffix}.png")
        if coords_xy is not None:
            img = build_spatial_montage(
                thumbs, coords_xy, patch_h=height, patch_w=width, max_side=montage_max_side
            )
        else:
            ncol = max(1, math.ceil(math.sqrt(n)))
            img = build_montage(thumbs, ncol=ncol, max_side=montage_max_side)
        img.save(path)
        return path

    before_path = _save_montage(thumbs_before, "_before")
    after_path = _save_montage(thumbs_after, "")
    logger.success("montages saved | before={} | after={}", before_path.resolve(), after_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="checkpoint.pt from training (contains G_xy / G_yx)",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        required=True,
        help="Directory containing HDF5 slide files",
    )
    parser.add_argument("--glob", type=str, default=config.h5_glob)
    parser.add_argument(
        "--which",
        choices=("xy", "yx"),
        default=DEFAULT_WHICH,
        help="xy=G_xy (patho2->morph style), yx=G_yx (morph->patho2 style)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--thumb",
        type=int,
        default=DEFAULT_THUMB,
        help="Thumbnail pixel size (square) for montage cells",
    )
    parser.add_argument(
        "--montage-max-side",
        type=int,
        default=DEFAULT_MONTAGE_MAX_SIDE,
        help="Max width/height of final montage PNG (downsampled if larger)",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint
    src_key = config.patches_key
    dst_key = config.gan_patches_key
    coord_key_cfg = (config.coordinates_key or "").strip() or None

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
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
        checkpoint.resolve(),
        len(paths),
        src_key,
        dst_key,
        device,
        args.batch_size,
    )

    for p in tqdm(paths, desc="h5 files", unit="file"):
        process_one_h5(
            p,
            G,
            device,
            src_key,
            dst_key,
            batch_size=max(1, args.batch_size),
            thumb_size=max(8, args.thumb),
            montage_max_side=max(256, args.montage_max_side),
            coord_key=coord_key_cfg,
        )


if __name__ == "__main__":
    main()
