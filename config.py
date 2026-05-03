from pathlib import Path

from sklearn.utils import Bunch

config = Bunch()

# Data paths
config.patho2_patches_h5 = Path("/mnt/d/Patho2/20-0494_1_1_patches.h5")
config.morph_patches_h5 = Path("/mnt/e/DLBCL-h5/13952_0_patches.h5")
config.patches_key = "cache/512/patches"

# Output
config.output_subdir = "train_outputs"

# Training loop
config.total_steps = 100_000
config.save_every = 100

# Optimizer (Adam)
config.lr_G = 2e-4
config.lr_D = 2e-4
config.beta1 = 0.5
config.beta2 = 0.999

# Loss weights (CycleGAN)
config.lambda_cycle = 5.0
config.lambda_gan = 1.0
config.lambda_discriminator = 1.0
