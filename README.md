# wsi_gan
<img width="808" height="504" alt="GAN_readme" src="https://github.com/user-attachments/assets/6ed02f96-c5eb-4ff2-91d8-5e8dea1b2734" />

## Setup
```
git clone git@github.com:S-murakami1/wsi_gan.git
cd wsi-gan
uv sync
```
## Preprocessing

Before training, preprocess whole-slide images using the following toolbox:

https://github.com/technoplasm/wsi-toolbox/

This toolbox is used for whole-slide image patch extraction and HDF5 generation.
## Training

- Place the `.h5` files you want to train on into `./h5_x` and `./h5_y`.
- Set the HDF5 dataset key in `config.py` to match the slides containing image patches  
(default: `config.patches_key = "cache/512/patches"`).

Training reads that key from every `.h5` file under `--h5-dir-x` and `--h5-dir-y`.

```bash
uv run python train.py --h5-dir-x ./h5_x --h5-dir-y ./h5_y
```
## Apply generator to HDF5 patches (domain X → Y)
- Set the HDF5 dataset key in `config.py` to match the slides containing image patches  
(default: `config.patches_key = "cache/512/patches"`).
- Set the HDF5 dataset key in `config.py` to match the slides containing coordinate information  
(default: `config.coordinates_key = "cache/512/coordinates"`).
- Set the HDF5 dataset key in `config.py` for storing GAN-generated image patches  
  (default: `config.gan_patches_key = "cache/512/gan/patches"`).


```bash
uv run python transform_h5_patches_gan.py --checkpoint train_outputs/step_000100/checkpoint.pt --h5-dir ./h5_x --which xy
```
`--which xy` → **G_xy** (X→Y).  
`--which yx` → **G_yx** (Y→X).
