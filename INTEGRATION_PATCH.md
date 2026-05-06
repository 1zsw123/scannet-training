# Integration patch instructions

This release is an **overlay** on top of an existing C3G installation,
not a standalone repo. Drop the files into the right places.

## Files to copy verbatim

```bash
# Inside your C3G clone:
cp release/code/dataset_scannet_i2slam.py    src/dataset/
cp release/code/encoder_da3_resplat.py       src/model/encoder/    # OVERWRITES upstream!
cp release/configs/scannet_i2slam.yaml       config/dataset/
cp release/configs/finetune_dav3_scannet_i2slam.yaml  config/training/
```

## File you must merge (don't overwrite blindly)

`src/dataset/__init__.py` needs three additions to register the new
dataset. Reference is in `release/code/dataset_init_reference.py`. The
diff is:

```python
# 1. Add an import line near the other dataset imports:
from .dataset_scannet_i2slam import (
    DatasetScannetI2Slam,
    ScannetI2SlamCfg,
    DatasetScannetI2SlamCfgWrapper,
)

# 2. Inside DATASETS dict:
DATASETS: dict[str, Dataset] = {
    ...
    "scannet_i2slam": DatasetScannetI2Slam,
}

# 3. Extend DatasetCfgWrapper and DatasetCfg unions:
DatasetCfgWrapper = ... | DatasetScannetI2SlamCfgWrapper
DatasetCfg = ... | ScannetI2SlamCfg
```

## What the patched encoder changes

`code/encoder_da3_resplat.py` adds the following on top of upstream
(diff is small, ~80 lines, all guarded by env vars or a fresh attribute
flag — does not affect existing pipelines that don't set them):

1. **`_forward_gt_pose`**: 5 stages of `[CHECK_GT_PATH]` print on first
   call (gated by `_gt_path_logged` attribute → only fires once).
   Pure debug; no behavioural change.

2. **`_forward_pred_pose`**: optional GT injection at the end of the
   function, gated by env vars:
   - `INJECT_GT_INTRINSICS=1` → after DA3 predicts K, replace it with
     the GT K from `context["intrinsics"]` (one-time print).
   - `INJECT_GT_EXTRINSICS=1` (paired with the above) → ray-based R
     re-derivation: `R_new = orth(K_gt⁻¹ · K_pred · R_pred)`,
     keeps DA3's t_pred, recomposes c2w (one-time print with R-angular
     diff and SVD singular values).
   - `INJECT_GT_EXTRINSICS=1` alone (without `INJECT_GT_INTRINSICS=1`)
     → fallback SE(3) substitution (legacy behaviour from before the
     ray-based reformulation).

If you don't set these env vars, behaviour matches upstream exactly.

## Versions / dependencies tested

```
Python 3.11
torch 2.4.0+cu121
pytorch-lightning 2.x
basicsr (VDiff fork)
gsplat (CUDA-built; we used a pre-built .so cached at
        /scratch-shared/$USER/tmp/torch_extensions/gsplat_cuda/gsplat_cuda.so
        to avoid 5–10 min of JIT compile per launch)
DA3-GIANT (HuggingFace-style local dir with config.json + model.safetensors)
ReSplat: resplat-base-dl3dv-256x448-view8-1934a04c.pth
```

OS: Linux (RHEL9-ish; tested on Snellius cluster H100 nodes).

## Sanity-check launch

```bash
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR=/scratch-shared/$USER/torch_extensions
export TMPDIR=/scratch-shared/$USER/tmp                # avoid quota in /tmp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

cd /path/to/your/C3G

torchrun --nnodes=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    -m src.main +training=finetune_dav3_scannet_i2slam \
    trainer.num_nodes=1 trainer.devices=4 \
    wandb.mode=online wandb.name=test_run
```

You should see, in order:
1. `Loaded weights-only from .../global_step_3999.ckpt, starting from step 0.`
   (×4 ranks)
2. `wandb: 🚀 View run at https://wandb.ai/...`
3. `[CHECK_GT_PATH] STAGE 1 …` through `STAGE 5` (×4 ranks, interleaved)
4. `│           val/psnr            │      21.169...`  ← baseline val/PSNR

If baseline = 21.17 you're aligned with us. If it differs, either
checkpoint is different or pose data has been shifted.
