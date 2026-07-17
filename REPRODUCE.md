# REPRODUCE — full dependency & artifact manifest

This document is the single source of truth for reproducing the ScanNet i2slam
fine-tune of the C3G **DA3 + ReSplat + DiffusionHead** pipeline on a fresh
machine. It fills the gaps identified in a reproduction dry-run (missing C3G
config skeleton, unknown Nova3R version, missing external repos, checkpoint
provenance).

## 0. Honest status of each dependency class

| Class | Status | Where |
|---|---|---|
| **Source code** | ✅ 100% reproducible | all public repos, exact commits in §1 |
| **C3G config skeleton** | ✅ included in this repo | [`c3g_overlay/config/`](c3g_overlay/config/) (119 files) |
| **Nova3R** | ✅ public (was thought missing) | `wrchen530/nova3r` @ `b2818ea4` — verified public |
| **DA3-GIANT / ReSplat weights** | ✅ public, downloadable | HuggingFace (see §4) — use **original** DA3-GIANT, not 1.1 |
| **VDiff S3 ckpt (`net_g_400000.pth`)** | ⚠️ public repo, exact ckpt via release | `Chen-Rao/VD-Diff` (§4) |
| **GoPro init ckpt (`global_step_3999.ckpt`)** | ❌ **NOT available** | author cluster scratch was purged — needs a backup, or use fresh-init path (§4) |
| **ScanNet best ckpts (24.80 / 27.08 dB)** | ❌ **NOT available** | same — purged |
| **Prepared i2slam RGB/depth/vdiff data** | ❌ **NOT available** (only pose metadata survives) | RGB/depth re-fetch from I2-SLAM authors; vdiff regenerate (§5) |
| **Pose metadata** | ✅ included in this repo | [`data_meta/per_scene/`](data_meta/per_scene/) |

> **Bottom line:** you can reproduce the *pipeline* end-to-end from scratch, but
> the author's exact trained artifacts (GoPro init checkpoint, the 24.80/27.08
> ScanNet checkpoints) and the prepared RGB/depth/vdiff frames are **no longer on
> the source cluster**. To match the published numbers exactly you must obtain
> those from a personal backup. Otherwise follow the fresh-init path in §4.

## 1. Repositories & exact commits

All source is public. Clone each at the pinned commit for exact reproduction.

| Component | Repo | Commit |
|---|---|---|
| C3G (base framework) | `https://github.com/cvlab-kaist/C3G` | `39242766ca5dffc45736113334d879a67d165228` |
| C3G author delta (DA3+ReSplat+DiffHead line) | `https://github.com/1zsw123/DAV3-RESPLAT-DIFF` | overlay of `src/` on top of C3G base¹ |
| **Nova3R** (DiffusionHead backbone) | `https://github.com/wrchen530/nova3r` | `b2818ea4928f169761573c8b3182730405de174d` |
| Depth-Anything-3 | `https://github.com/ByteDance-Seed/Depth-Anything-3` | `41736238f5bced4debf3f2a12375d2466874866d` |
| ReSplat | `https://github.com/cvg/resplat` | `cc4594af97a2e559e98f5e307d80535569ae7007` |
| mip-splatting | `https://github.com/autonomousvision/mip-splatting` | `dda02ab5ecf45d6edb8c540d9bb65c7e451345a9` |
| VD-Diff (pseudo-GT generator) | `https://github.com/Chen-Rao/VD-Diff` | `e1ec1c6722f84d728d65c11277f45d01c8dfe269` |

¹ On the source machine the C3G work exists as **uncommitted modifications** on top
of the base commit above (no author fork commit). The `src/` tree of those changes
is published as `1zsw123/DAV3-RESPLAT-DIFF`; combine it with this repo's overlay
files (`code/`, `configs/`, `c3g_overlay/config/`).

### CUDA rasterizer submodules
The three C3G rasterizers (`diff_gaussian_rasterization_w_pose`,
`diff_gaussian_rasterization_w_feature_detach`, `lang_seg`) are **git submodules of
the base C3G repo**. Get them with:
```bash
git clone --recursive https://github.com/cvlab-kaist/C3G
# or, in an existing clone:
git submodule update --init --recursive
```

## 2. Target directory layout

```
C3G/
├── config/                     ← replace with c3g_overlay/config/ from this repo
├── src/                        ← from DAV3-RESPLAT-DIFF, + this repo's code/ overlay
│   ├── main.py
│   ├── dataset/
│   │   ├── dataset_scannet_i2slam.py   ← this repo: code/
│   │   └── __init__.py                 ← merge per INTEGRATION_PATCH.md
│   └── model/encoder/encoder_da3_resplat.py  ← this repo: code/ (OVERWRITES upstream)
├── submodules/                 ← C3G submodules (rasterizers)
├── Depth-Anything-3/           ← clone @ 41736238
├── resplat/                    ← clone @ cc4594af
├── nova3r/                     ← clone @ b2818ea4
├── mip-splatting/              ← clone @ dda02ab5
├── weights/                    ← §4 (NOT in git — download / restore)
│   ├── DA3-GIANT/{config.json,model.safetensors}
│   ├── resplat-base-dl3dv-256x448-view8-1934a04c.pth
│   ├── global_step_3999.ckpt   (GoPro init — see §4)
│   └── net_g_400000.pth        (VDiff S3)
└── data/i2slam_scannet/        ← §5 (NOT in git)
```

## 3. Config integration

Copy `c3g_overlay/config/` over your C3G clone's `config/` (it is the complete,
Hydra-resolvable tree — 119 files including `main.yaml`, `dataset/base_dataset.yaml`,
`model/encoder/noposplat.yaml`, `model/encoder/backbone/croco.yaml`,
`model/decoder/splatting_cuda.yaml`, and `loss/{mse_l1,lpips_v3,ssim,depth}.yaml`).

**Gotchas:**
- There is **no** `config/model/encoder/da3_resplat.yaml`. The encoder base is
  `noposplat` (`override /model/encoder: noposplat`) and the DA3+ReSplat encoder is
  selected **inline** in the training yaml via `model.encoder.name: da3_resplat`.
  Do not go looking for a standalone da3_resplat encoder config — it doesn't exist.
- Entry config: `config/training/finetune_dav3_scannet_i2slam.yaml`
  (this repo also ships the flat copies under `configs/` per `INTEGRATION_PATCH.md`;
  the `c3g_overlay/config/` tree already contains them in the correct locations).
- Edit the three hardcoded absolute paths in that yaml before running:
  `model.encoder.da3_checkpoint`, `model.encoder.resplat_checkpoint`,
  `checkpointing.load`.

The ScanNet entry config resolves to (verified against source):
encoder `da3_resplat` (DA3 + ReSplat both **frozen**, `use_pred_pose: false` → pure
GT-pose path); DiffusionHead `train/infer_num_steps=3`, `use_cam_cond=false`,
`use_vggt_priors=true`, `use_cam_pred=false`; `refiner: null`; losses
`[mse_l1, lpips_v3, ssim, depth]` with `loss.depth.weight=0.0`; lr `2e-5`,
`max_steps 20000`, `val_check_interval 500`, 4×GPU DDP, `bf16-mixed`.

## 4. Checkpoints & weights

| File | Size | Source | Notes |
|---|---|---|---|
| `DA3-GIANT/{config.json,model.safetensors}` | ~5 GB | HuggingFace `depth-anything/...` (ByteDance-Seed release) | **Use the original GIANT, NOT DA3-GIANT-1.1.** Code pinned to DA3 commit `41736238`. |
| `resplat-base-dl3dv-256x448-view8-1934a04c.pth` | ~1.5 GB | ReSplat official release / HF | Must contain `ckpt["state_dict"]` with encoder params prefixed `encoder.`. Verify the `1934a04c` sha prefix. |
| `net_g_400000.pth` (VDiff S3) | ~50 MB | `Chen-Rao/VD-Diff` release | Needs the matching `basicsr.archs.S3_arch`; a stock PyPI basicsr may not include it — install the VD-Diff fork. Only needed if you must regenerate vdiff frames. |
| `global_step_3999.ckpt` (GoPro init) | ~5 GB | ❌ author cluster (purged) | Training history Replica→ScanNet++→RSBlur→GoPro-defocus. **No public source.** See fresh-init options below. |
| ScanNet best `psnr_27.080` / `psnr_24.80x` | ~1.5 GB ea | ❌ author cluster (purged) | Only needed to *validate* the author's result without retraining. |

**If you don't have the GoPro init checkpoint** (the common case):
1. **Fresh init** — start the DiffusionHead from scratch. Trains, but will NOT
   reproduce the 21.17 baseline; results not comparable to the paper numbers.
2. **Substitute init** — load a public C3G / Replica-SOTA checkpoint. Reduces the
   single-view→multi-view distribution shift the EMPIRICAL_LOG flags, but is a
   different init, so numbers differ.
3. **Restore from backup** — the only path to exact-number reproduction.

## 5. Data

Per-scene target layout (place under `data/i2slam_scannet/<scene>/`):
```
<scene>/
├── gt/color_NNNNN.png     640×480 RGB — the BLURRY input
├── gt/depth_NNNNN.npy     (1,480,640) or (480,640) float32 metres, 0=hole
├── vdiff/color_NNNNN.png  640×480 pseudo-sharp GT, frame-index aligned to gt/
├── poses.npy              (M,4,4) float32 c2w   ← provided in data_meta/per_scene/
├── intrinsic_color.txt, intrinsic_depth.txt     ← provided
└── extrinsic_color.txt, extrinsic_depth.txt     ← provided
```
Frames: `scene0024_01`=300, `scene0031_00`=300, `scene0736_00`=400 (train);
`scene0785_00`=500 (val).

| Piece | Status | How to get it |
|---|---|---|
| `poses.npy` + intrinsics/extrinsics | ✅ in this repo | `data_meta/per_scene/<scene>/` — copy into each scene dir |
| `gt/color_*.png`, `gt/depth_*.npy` | ❌ not in repo (~12 GB) | I2-SLAM dataset release (authors' Google Drive), 4 scenes |
| `vdiff/color_*.png` | ❌ not in repo | Regenerate: `scripts/infer_scannet_i2slam.py` with the VDiff S3 ckpt (§4), or obtain precomputed |
| ScanNet `.sens` | not required | only needed to *re-extract* poses; metadata already provided (see `scripts/extract_poses.py`, `SETUP.md §4`) |

## 6. Environment

Use [`environment.yml`](environment.yml) (conda). **Python 3.11 + torch 2.4.0+cu121**
— do NOT use Python 3.13 (breaks several CUDA extensions / basicsr). Then build the
CUDA extensions (each is a distinct, non-interchangeable ABI):
- C3G `diff_gaussian_rasterization_w_pose` (pose-delta)
- C3G `diff_gaussian_rasterization_w_feature_detach` (feature rendering)
- ReSplat's `gsplat` (prebuild a `.so` and cache it to skip 5–10 min JIT per launch)
- mip-splatting `diff-gaussian-rasterization` (different `kernel_size` ABI) — needed
  even with `refiner: null`, because `main.py` imports `MipSplattingRefiner` at module
  top level. **Option:** make that a lazy import so `refiner=null` runs never load the
  extension (small code change; avoids building mip-splatting for a ScanNet-only run).

Set before launch:
```bash
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR=/big/scratch/torch_extensions
export TMPDIR=/big/scratch/tmp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

## 7. Launch

```bash
cd /path/to/C3G
torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    -m src.main +training=finetune_dav3_scannet_i2slam \
    trainer.num_nodes=1 trainer.devices=4 \
    wandb.mode=online wandb.name=my_run
```
Expected startup (per `INTEGRATION_PATCH.md`): weights-only load message ×4 ranks →
wandb URL → `[CHECK_GT_PATH] STAGE 1..5` ×4 → first `val/psnr ≈ 21.17` (pure-GT path
baseline; **only if you have the GoPro init ckpt**).

**Undocumented env vars:** the `scannet_i2slam` config is the **pure-GT path**
(`use_pred_pose=false`), so the `INJECT_GT_INTRINSICS/EXTRINSICS`, `SMOOTH_CTX_*`,
`DISABLE_POSE_FILTER`, `ANISO_*`, `BORDER_MASK_*` env knobs are **not used** here —
they only matter for the RSBlur pred-pose runs. Leave them unset.

## 8. Known blocker (before you burn GPU hours)

Every fine-tune variant peaks at the **first val event (step 124–250) then declines**
(pure-GT path degrades ~3 dB gracefully; mixed-coord+high-lr crashes 12+ dB). See
[`EMPIRICAL_LOG.md`](EMPIRICAL_LOG.md). Leading untested root-cause: `_normalise_poses`
clamps median translation at `0.1`, but ScanNet's 6-consecutive-frame baseline is
~0.025 m, so the clamp engages and scales translations ×10 (OOD for DA3). Candidate
fix: lower the clamp to `1e-4`, and/or raise `dataset.scannet_i2slam.frame_stride`
from 1 → 8 so the context baseline (~0.4 m) matches the Replica training distribution.
