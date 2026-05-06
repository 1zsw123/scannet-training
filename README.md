# ScanNet i2slam Fine-tune for C3G (DA3 + ReSplat + DiffusionHead)

Fine-tune the C3G blur-restoration pipeline on the I2-SLAM ScanNet subset
(4 scenes with real handheld motion blur). Sharp pseudo-GT supervision via
VDiff-deblurred targets; depth supervision via ScanNet StructureSensor depth.

## Quick links

| | |
|---|---|
| Upstream codebase | C3G (DA3 + ReSplat + DiffusionHead) |
| Forked from | `finetune_dav3_gopro_defocus_l1.yaml` (GoPro single-view image-deblur fine-tune) |
| Datasets | I2-SLAM ScanNet (4 scenes) + ScanNet `.sens` (for poses) + VDiff pseudo-GT |
| Init checkpoint | GoPro fine-tuned C3G: `exp_gopro_planB_l2img_depth02_from_scannetpp20k/.../latest-info/global_step_3999.ckpt` |

## What this delivers

- New PyTorch dataset class for I2-SLAM ScanNet with VDiff pseudo-GT + ScanNet GT pose/depth.
- Patches to `encoder_da3_resplat.py` adding (a) `[CHECK_GT_PATH]` debug prints
  in the GT-pose forward and (b) optional `INJECT_GT_INTRINSICS` /
  `INJECT_GT_EXTRINSICS` env-var hooks (ray-based R re-derivation per the
  homography H = K · R formulation).
- Hydra configs (`scannet_i2slam.yaml`, `finetune_dav3_scannet_i2slam.yaml`).
- VDiff inference script tailored to the I2-SLAM directory layout.
- Pose extraction utilities for ScanNet `.sens` files.
- Per-scene preprocessed metadata (`poses.npy`, intrinsics, extrinsics).

## Repo layout

```
scannet_i2slam_release/
├── README.md                     ← this file
├── SETUP.md                      ← data acquisition + env setup
├── EMPIRICAL_LOG.md              ← results table for every variant tried
├── code/
│   ├── dataset_scannet_i2slam.py     drop-in dataset class (place under src/dataset/)
│   ├── encoder_da3_resplat.py        patched encoder (replaces upstream)
│   └── dataset_init_reference.py     reference src/dataset/__init__.py with the new dataset registered
├── configs/
│   ├── scannet_i2slam.yaml           dataset config (place under config/dataset/)
│   └── finetune_dav3_scannet_i2slam.yaml   training config (place under config/training/)
├── scripts/
│   ├── infer_scannet_i2slam.py       run VDiff S3 on each scene's gt/color_*.png → vdiff/color_*.png
│   ├── extract_poses.py              ScanNet .sens → poses.npy (+ intrinsics)
│   ├── decode_sens_frames.py         decode arbitrary frames from a .sens (debug)
│   └── match_indices.py              verify i2slam-frame ↔ scannet-frame mapping (sanity)
├── data_meta/
│   ├── sharp_frames.txt              sharp frame indices per scene (from i2slam release)
│   └── per_scene/
│       ├── scene0024_01/
│       │   ├── poses.npy             (M, 4, 4) float32 c2w aligned to i2slam frames
│       │   ├── intrinsic_color.txt   3×3 K at 1296×968 native ScanNet RGB
│       │   ├── intrinsic_depth.txt   3×3 K at 640×480 (matches i2slam color png)
│       │   ├── extrinsic_color.txt   color↔base extrinsic (identity for ScanNet)
│       │   ├── extrinsic_depth.txt   depth↔base extrinsic
│       │   └── pose_meta.txt         human-readable summary (frame counts, mapping, depth_shift)
│       ├── scene0031_00/  …
│       ├── scene0736_00/  …
│       └── scene0785_00/  …
└── (you fetch:)
    ├── i2slam dataset (color + depth)        small, ~12 GB
    ├── ScanNet .sens (4 scenes)              ~7 GB total
    ├── VDiff S3 ckpt                         ~50 MB
    ├── DA3-GIANT weights                     ~5 GB
    ├── ReSplat pretrained                    ~1.5 GB
    └── GoPro init ckpt                       ~5 GB
```

## End-to-end recipe

```bash
# 1. Clone C3G + apply this overlay (see SETUP.md for details)
git clone <C3G repo>
cd C3G
cp -r path/to/scannet_i2slam_release/code/dataset_scannet_i2slam.py  src/dataset/
cp -r path/to/scannet_i2slam_release/code/encoder_da3_resplat.py     src/model/encoder/
# manually merge dataset_init_reference.py into src/dataset/__init__.py
cp path/to/scannet_i2slam_release/configs/scannet_i2slam.yaml         config/dataset/
cp path/to/scannet_i2slam_release/configs/finetune_dav3_scannet_i2slam.yaml  config/training/

# 2. Fetch i2slam ScanNet (color + depth)  → see SETUP.md §Data
# 3. Fetch ScanNet .sens via official download script    → see SETUP.md §ScanNet
# 4. Run pose extraction    → produces per_scene/poses.npy + intrinsics.txt  (already done; this directory has the output)
python scripts/extract_poses.py --sens path/to/scene0024_01.sens --out per_scene/scene0024_01/poses.npz

# 5. Run VDiff inference (4 scenes, single GPU each, ~3-10 min per scene)
python scripts/infer_scannet_i2slam.py \
    --ckpt path/to/Replica_S3_models/net_g_400000.pth \
    --data_root path/to/i2slam/I2-SLAM_dataset/rgbd/scannet \
    --num_frame 7

# 6. Place poses + intrinsics into the i2slam scene directories so the dataset can read them
for s in scene0024_01 scene0031_00 scene0736_00 scene0785_00; do
    cp per_scene/$s/poses.npy           path/to/i2slam/.../scannet/$s/
    cp per_scene/$s/intrinsic_color.txt path/to/i2slam/.../scannet/$s/
    cp per_scene/$s/intrinsic_depth.txt path/to/i2slam/.../scannet/$s/
    cp per_scene/$s/extrinsic_color.txt path/to/i2slam/.../scannet/$s/
    cp per_scene/$s/extrinsic_depth.txt path/to/i2slam/.../scannet/$s/
done

# 7. Edit the load: line in finetune_dav3_scannet_i2slam.yaml to point to your GoPro init ckpt

# 8. Launch
torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    -m src.main +training=finetune_dav3_scannet_i2slam \
    trainer.num_nodes=1 trainer.devices=4 \
    wandb.mode=online wandb.name=my_run
```

## Key configuration knobs

The training config has a few points where someone may want to deviate from
the defaults; see `EMPIRICAL_LOG.md` for what each setting did empirically.

| field | value | what it controls |
|---|---|---|
| `model.encoder.use_pred_pose` | `false` (current) / `true` | Whether DA3 forward receives our GT poses (`false` → `_forward_gt_pose` path) or self-predicts (`true` → `_forward_pred_pose`). |
| `optimizer.lr` | `5e-6` | Walked between `1e-7` (slow but stable in our window) and `2e-5` (highest peak then crash). |
| `loss.depth.weight` | `0.0` | Disabled because depth loss exhibited 20× per-batch variance from holes + scale-invariant normalisation — slowed the crash but didn't prevent it. |
| `dataset.scannet_i2slam.num_context_views` | `6` | Frames per multi-view bundle (consecutive 6 frames). |
| `dataset.scannet_i2slam.frame_stride` | `1` | 1 = consecutive (small baseline). Increase to 5–10 if you want larger ctx baselines (~Replica scale). |
| `INJECT_GT_INTRINSICS` env var | unset | When set to `1` AND `use_pred_pose=true`, replaces DA3-pred K with GT K after the DA3 forward. |
| `INJECT_GT_EXTRINSICS` env var | unset | When set to `1` (paired with the above), runs the ray-based R re-derivation: `R_new = orth(K_gt⁻¹ · K_pred · R_pred)`. |

## Notable caveats

1. **Crash pattern**: every training variant we tried peaks at val_check #1
   (step 124–250) then degrades. Pure GT path declines gracefully (~3 dB
   over 250 steps); injection path with high lr crashes hard (12+ dB). See
   `EMPIRICAL_LOG.md`.

2. **`_normalise_poses` clamp**: in `encoder_da3_resplat.py`, the median
   normalisation clamps at `0.1`. ScanNet i2slam's 6-consecutive-frame ctx
   baseline is ~0.025 m, so the clamp engages and translations are
   effectively scaled ×10 instead of true median-normalised. This puts DA3
   at a different input distribution than Replica training. Investigated
   but not yet fixed (see EMPIRICAL_LOG §root cause investigation).

3. **i2slam frame ↔ ScanNet frame identity mapping**: empirically verified
   via image-hash matching (`scripts/match_indices.py`). The first
   M = (300|400|500) frames of the original ScanNet recording exactly match
   i2slam color_00000…color_<M-1>. Pose mapping is therefore trivial
   `pose[i] = original_camera_to_world[i]`.

4. **GT pose convention**: ScanNet `.sens` stores `camera_to_world` (c2w),
   verified via trajectory-smoothness test: under c2w mean inter-frame
   step is 13.8 mm (consistent with handheld 30 fps); under w2c
   interpretation it would be 48 mm (inconsistent). Use as c2w directly.

5. **Disk-quota**: training writes 4–6 GB ckpts plus wandb media. On
   GPFS-quota systems clean old experiments aggressively; consider
   `save_weights_only: true` (set in our config) and
   `every_n_train_steps: 2000`.

## Cite / contact

Internal research code; not currently associated with a publication.
