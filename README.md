# wsi_gan
<img width="808" height="504" alt="GAN_readme" src="https://github.com/user-attachments/assets/6ed02f96-c5eb-4ff2-91d8-5e8dea1b2734" />

## Setup
```
git clone git@github.com:S-murakami1/wsi_gan.git
cd wsi-gan
uv sync
```
## Training
```
uv run python train.py --h5-dir-x /path/to/domain_x --h5-dir-y /path/to/domain_y
```
## Apply generator to HDF5 patches
After training, run `transform_h5_patches_gan.py` to write transformed patches into each HDF5 and save preview montages.
```
uv run python transform_h5_patches_gan.py --checkpoint train_outputs/step_000100/checkpoint.pt --h5-dir /path/to/h
```
