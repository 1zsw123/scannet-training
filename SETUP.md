# Data Acquisition + Environment Setup

## 1. Environment

The codebase is the C3G C3D-blur stack. We assume the user has C3G already
working with DA3-GIANT + ReSplat. If not, follow C3G's own README first.

Key packages (all already pinned by C3G):

```
torch >= 2.4
pytorch-lightning >= 2
hydra-core
omegaconf
lpips
basicsr (for VDiff)
gsplat (CUDA-built)
diff-gaussian-rasterization
mmengine (basicsr dependency)
einops, jaxtyping, wandb, dacite, opencv-python, pillow, tqdm
```

**TORCH_EXTENSIONS_DIR**: gsplat / diff-gauss build CUDA kernels at first
launch; on cluster filesystems with strict quotas, set this to a writeable
scratch path:

```bash
export TORCH_EXTENSIONS_DIR=/scratch-shared/$USER/torch_extensions
```

**Locale**: gsplat's JIT compile reads `.cu` files with locale-default
encoding; on `LANG=C` systems it crashes on UTF-8. Set:

```bash
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
```

## 2. Data: I2-SLAM ScanNet (color + depth)

Source: I2-SLAM dataset release (the authors' Google Drive). 4 scenes:
`scene0024_01`, `scene0031_00`, `scene0736_00`, `scene0785_00`.

Per-scene structure (after extraction):

```
i2slam/I2-SLAM_dataset/rgbd/scannet/<scene>/gt/
    color_NNNNN.png    (640×480 RGB, the BLURRY input)
    color_NNNNN.npy    (3, 480, 640) float32 [0,1]   — same content as PNG, fast-load
    depth_NNNNN.npy    (1, 480, 640) float32 metres — 0 = invalid hole
```

Frame counts: 0024_01 = 300, 0031_00 = 300, 0736_00 = 400, 0785_00 = 500.

Side index file `rgbd/sharp_frames.txt` lists the i2slam-curated sharp
indices per scene (only ~6–9 per scene; we don't currently use it for
splits).

## 3. Data: VDiff S3 pretrained checkpoint

Source: official VDiff release (`Replica_S3_models/net_g_400000.pth`,
~50 MB). The `infer_scannet_i2slam.py` script depends on this and on
the BasicSR + VDiff `S3_arch` from `https://github.com/.../VDiff`.

```bash
mkdir -p path/to/vdiff_training_states/Replica_S3_models
# download net_g_400000.pth into this directory
```

## 4. Data: ScanNet `.sens` (for poses)

ScanNet requires a Terms of Use agreement. Run their official downloader:

```bash
# Fetch downloader (URL from https://github.com/ScanNet/ScanNet)
curl -L -o download-scannet.py http://kaldir.vc.in.tum.de/scannet/download-scannet.py

# Per-scene .sens download (only need the .sens — not the full release).
# 2 of our scenes are in the public release, 2 in the test split.
printf '\n\n' | python download-scannet.py --id scene0024_01 --type .sens -o data
printf '\n\n' | python download-scannet.py --id scene0031_00 --type .sens -o data
printf '\n\n' | python download-scannet.py --id scene0736_00 --type .sens -o data
printf '\n\n' | python download-scannet.py --id scene0785_00 --type .sens -o data
```

`scene0024_01`/`0031_00` land under `data/scans/`, `scene0736_00`/`0785_00`
under `data/scans_test/`.

Total .sens disk usage ≈ 7 GB. Once you've extracted poses (next step) you
can delete the `.sens` files.

## 5. Data: extract per-frame poses from `.sens`

Pre-extracted output files are included in this release under
`data_meta/per_scene/<scene>/`:

- `poses.npy` — (M, 4, 4) float32 c2w aligned to i2slam frame indices
- `intrinsic_color.txt` — 3×3 K at 1296×968 native ScanNet RGB
- `intrinsic_depth.txt` — 3×3 K at 640×480 (matches i2slam color resolution)
- `extrinsic_color.txt`, `extrinsic_depth.txt` — color↔base / depth↔base
  rigid extrinsics (identity for ScanNet StructureSensor)
- `pose_meta.txt` — human-readable summary

If you want to re-extract from scratch:

```bash
python scripts/extract_poses.py \
    --sens data/scans/scene0024_01/scene0024_01.sens \
    --out path/to/i2slam/.../scannet/scene0024_01/poses.npz
# (script outputs .npz; the dataset class wants per-key .npy + .txt — see
#  README §Quick-recipe for the simple loop that copies them in.)
```

The "i2slam frame index = ScanNet frame index" identity has been verified
via image-hash matching (`scripts/match_indices.py`). For each scene, only
the first M frames of the original ScanNet recording are used (M = 300, 400,
or 500); the rest of the trajectory is unused.

## 6. Pretrained C3G init checkpoint

This is the C3G stage-4 GoPro defocus fine-tuned model (built on top of a
Replica + ScanNet++ + RSBlur curriculum). It serves as our init.

| field | value |
|---|---|
| size | ~5 GB (full optimizer state) |
| training history | Replica → ScanNet++ → RSBlur → GoPro_defocus |
| pose convention | use_pred_pose (DA3 self-predicts; GT poses ignored) |
| baseline val/PSNR on our ScanNet i2slam val set | 21.17 (pure GT path) / 23.16 (pred-pose path) |

Stored at our institution: please request from authors. Once you have it,
update `checkpointing.load:` in
`config/training/finetune_dav3_scannet_i2slam.yaml`.

## 7. Pretrained DA3-GIANT + ReSplat

Both should already be downloaded for any C3G installation:

| | path | size |
|---|---|---|
| DA3-GIANT | `da3_pretrained/DA3-GIANT/{config.json, model.safetensors}` | ~5 GB |
| ReSplat (DL3DV-trained) | `resplat_pretrained/resplat-base-dl3dv-256x448-view8-1934a04c.pth` | ~1.5 GB |

Update the absolute paths in `finetune_dav3_scannet_i2slam.yaml` if your
install layout differs.

## 8. Verification before training

After all data is in place, run a one-batch sanity check:

```python
# python -c "..." or in a notebook
import sys; sys.path.insert(0, 'src')
from omegaconf import OmegaConf
from dacite import from_dict, Config
from pathlib import Path
from src.dataset.dataset_scannet_i2slam import DatasetScannetI2Slam, ScannetI2SlamCfg

y = OmegaConf.load('config/dataset/scannet_i2slam.yaml')
y = OmegaConf.merge(OmegaConf.create({
    'background_color':[0,0,0], 'cameras_are_circular':False,
    'overfit_to_scene':None,
    'view_sampler':{'name':'bounded','num_context_views':6,'num_target_views':6,
                     'min_distance_between_context_views':1,
                     'max_distance_between_context_views':1,
                     'min_distance_to_context_views':0,
                     'warm_up_steps':0,
                     'initial_min_distance_between_context_views':1,
                     'initial_max_distance_between_context_views':1},
}), y)
cfg = from_dict(ScannetI2SlamCfg, OmegaConf.to_container(y, resolve=True),
                config=Config(strict=False, cast=[Path]))

class FakeVS: num_context_views = 6
ds = DatasetScannetI2Slam(cfg, 'train', FakeVS())
print('scenes:', list(ds._scenes.keys()))
sample = next(iter(ds))
print('ctx.image:', sample['context']['image'].shape)        # [6, 3, 336, 448]
print('ctx.sharp_image (vdiff):', sample['context']['sharp_image'].shape)
print('ctx.extrinsics:', sample['context']['extrinsics'].shape)  # [6, 4, 4]
print('tgt.image (vdiff):', sample['target']['image'].shape)
print('tgt.depth:', sample['target']['depth'].shape,
      'holes_pct=', (sample['target']['depth']==0).float().mean().item()*100, '%')
```

If all four lines print without errors and the depth holes are <10%, you
are ready to train.

## 9. Running on a cluster (notes)

- bs=1 + accumulate_grad_batches=4: with 4 GPUs effective bs=16; with
  3 GPUs eff bs=12; with 2 GPUs eff bs=8. We did not retune lr per
  effective-bs.
- Each ckpt is ~1.5 GB if `save_weights_only: true`, ~4.5 GB otherwise.
  On strict-quota systems set `save_top_k: 1` and `every_n_train_steps:
  2000` (already in our config).
- wandb media images can pile up small files quickly — disable image
  logging if you're inode-constrained: `wandb.mode=offline` or pipe through
  `WANDB_DISABLE_CODE=true`, or just `rm -rf */wandb` periodically.
