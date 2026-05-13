from pathlib import Path

from sklearn.utils import Bunch

config = Bunch()

# Data: one directory per domain; all matching HDF5 files are used (uniform random patch)
config.patho2_h5_dir = Path("/mnt/d/Patho2")
config.morph_h5_dir = Path("/mnt/e/DLBCL-h5")
config.h5_glob = "*.h5"
config.patches_key = "cache/512/patches"
config.h5_max_open_files = 64

# Output
config.output_subdir = "train_outputs"
config.loss_curve_png = "loss_curve.png"

# Training loop
config.total_steps = 100_000
config.save_every = 100
config.batch_size = 2
# Per outer step: run this many G updates, then this many D updates.
config.generator_steps = 2
config.discriminator_steps = 1

# Optimizer (Adam)
config.lr_G = 2e-4
config.lr_D = 2e-4
config.beta1 = 0.5
config.beta2 = 0.999

# Loss weights (CycleGAN)
config.lambda_cycle = 5.0
config.lambda_gan = 1.0
config.lambda_discriminator = 1.0

# apply_gan_to_h5.py (optional defaults; CLI overrides)
config.gan_checkpoint_path = Path("./train_outputs/step_000100/checkpoint.pt")
config.gan_export_h5_dir = None  # None -> use patho2_h5_dir
config.gan_export_which = "xy"  # "xy" = G_xy, "yx" = G_yx
config.gan_export_batch_size = 8
config.gan_montage_thumb = 48
config.gan_montage_max_side = 4096
