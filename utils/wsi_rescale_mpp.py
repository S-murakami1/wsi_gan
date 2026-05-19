import argparse
from pathlib import Path

import pyvips
from loguru import logger
from tqdm import tqdm

DEFAULT_INPUT_MPP = 0.2535
DEFAULT_TARGET_MPP = 0.22636725823976819 * 2  # Hamamatsu level1 ≈ 0.4527 μm/pixel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rescale WSI slides to a target MPP and write tiled pyramid TIFF.",
    )
    p.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing source slide files (see --glob)",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Directory for output .tif files",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.svs",
        help="Glob pattern for input slides under input-dir",
    )
    p.add_argument(
        "--input-mpp",
        type=float,
        default=DEFAULT_INPUT_MPP,
        help="Source slide resolution in μm/pixel (MPP at level 0)",
    )
    p.add_argument(
        "--target-mpp",
        type=float,
        default=DEFAULT_TARGET_MPP,
        help="Desired output resolution in μm/pixel",
    )
    return p.parse_args()


def rescale_slide_to_tiff(
    slide_path: Path,
    out_path: Path,
    scale: float,
    pixels_per_mm: float,
) -> None:
    try:
        img = pyvips.Image.new_from_file(str(slide_path), level=0)
    except pyvips.Error as e:
        if "does not support optional argument level" in str(e):
            img = pyvips.Image.new_from_file(str(slide_path), page=0)
        else:
            raise
    resized = img.resize(scale)
    if resized.bands > 3:
        resized = resized.extract_band(0, n=3)
    elif resized.bands == 1:
        resized = resized.bandjoin([resized, resized, resized])

    resized.tiffsave(
        str(out_path),
        tile=True,
        tile_width=512,
        tile_height=512,
        compression="jpeg",
        Q=90,
        pyramid=True,
        bigtiff=True,
        xres=pixels_per_mm,
        yres=pixels_per_mm,
    )


def run_rescale(
    input_dir: Path,
    output_dir: Path,
    glob_pattern: str,
    input_mpp: float,
    target_mpp: float,
) -> None:
    if input_mpp <= 0 or target_mpp <= 0:
        raise ValueError("input_mpp and target_mpp must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    scale = input_mpp / target_mpp
    pixels_per_mm = 1000.0 / target_mpp

    slide_paths = sorted(input_dir.glob(glob_pattern))
    if not slide_paths:
        raise FileNotFoundError(
            f"No files matching {glob_pattern!r} under {input_dir.resolve()}"
        )

    logger.info(
        "resize WSI | input={} | output={} | input_mpp={} | target_mpp={} | scale={:.6f} | files={}",
        input_dir.resolve(),
        output_dir.resolve(),
        input_mpp,
        target_mpp,
        scale,
        len(slide_paths),
    )

    for slide_path in tqdm(slide_paths, desc="rescale WSI", unit="file"):
        out_path = output_dir / f"{slide_path.stem}.tif"
        rescale_slide_to_tiff(slide_path, out_path, scale, pixels_per_mm)
        logger.success("saved: {}", out_path.resolve())


def main() -> None:
    args = parse_args()
    try:
        run_rescale(
            args.input_dir,
            args.output_dir,
            args.glob,
            args.input_mpp,
            args.target_mpp,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
