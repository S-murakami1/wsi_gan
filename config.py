from sklearn.utils import Bunch

config = Bunch()

# Data (train.py: --h5-dir-x / --h5-dir-y; transform_h5_patches_gan.py: --h5-dir)
config.h5_glob = "*.h5"
config.patches_key = "cache/512/patches"
config.gan_patches_key = (
    "cache/512/gan/patches"  # Output key for transform_h5_patches_gan.py
)
config.coordinates_key = (
    "cache/512/coordinates"  # Coordinates key for transform_h5_patches_gan.py
)
config.h5_max_open_files = 64
config.seed = 42

# Output
config.output_subdir = "train_outputs"
config.loss_curve_png = "loss_curve.png"

# Training loop
config.total_steps = 100000
config.save_every = 1000
config.batch_size = 2

# Updates per iteration
config.g_updates_per_step = 5
config.d_updates_per_step = 1

# Optimizer (Adam)
config.lr_G = 1e-4
config.lr_D = 1e-4
config.beta1 = 0.5
config.beta2 = 0.999

# LR schedule
config.lr_schedule = "linear"  # "none" | "linear" | "cosine"
config.lr_schedule_end_ratio = 0.01  # End ratio for linear schedule

# Loss weights (CycleGAN)
config.lambda_cycle = 5.0
config.lambda_gan = 1.0
config.lambda_discriminator = 1.0
