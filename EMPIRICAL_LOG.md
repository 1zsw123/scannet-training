# Empirical log — ScanNet i2slam fine-tune

Sequential log of variants tried, baseline + peak val/PSNR, what we learnt.
The val set is `scene0785_00` × 8 sliding windows × 6 views = 48 frames per
val event (`val_check_interval: 500`).

## Results table

| variant | path / inject | lr | depth-loss | init ckpt | baseline val/PSNR | peak val/PSNR | crash? | takeaway |
|---|---|---|---|---|---|---|---|---|
| v3 (gcn139 / v11 retry) | pred-pose, no inject | 2e-5 | 0.2 | GoPro 23.16 | 23.20 | (run interrupted before any val update) | n/a | reference clean baseline path |
| v6 / v9 / v18 / v20 | pure GT path (`use_pred_pose: false`), no inject | 2e-5 | 0.2 | GoPro 23.16 | **21.17** | (no train-time updates seen) | n/a | reference clean GT path; 2-dB lower than pred-pose, not catastrophic |
| v17 | pred-pose + INJECT_GT_INTRINSICS + INJECT_GT_EXTRINSICS (SE(3) substitution) | 1e-7 | 0.2 | GoPro 23.16 | 23.08 | (slow) | no | mixed-coord starting point ~ pred-pose; 1e-7 is glacial |
| v19 / v21 | pred-pose + INJECT_GT_INTRINSICS + ray-based R | 1e-7 | 0.2 | GoPro 23.16 | 23.16 | 23.30 @ step 374 | no | ray-based R is K-dependent; angular diff 0.05° (K_pred vs K_gt only differ ~2%) |
| **v22** | mixed-coord (INJECT_K + INJECT_E ray-based) | **2e-5** | 0.2 | GoPro 23.16 | 23.16 | **27.08 @ step 124** ← saved | **YES → 13.6 by step 250** | best peak observed; lr 2e-5 also fastest crash |
| v23 / v23-resume | mixed-coord | 5e-6 | 0.2 | various | 23.16 / 23.50 | 23.50 / **24.66** | yes (but milder, val ~20 by step 450) | mid-lr → mid peak gain (+1 dB) → softer crash |
| v24 | mixed-coord, **depth=0** | 5e-6 | **0.0** | v23-resume 24.66 | 24.66 | 24.90 @ step 124 | yes | disabling depth slowed crash but did not prevent it; rules out depth as primary trigger |
| **v25** | **pure GT path, no inject**, lr 2e-5 | 2e-5 | 0 | GoPro 21.17 | 21.17 | **24.80 @ step 124** | partial (declined to 21.6 by step 374, NOT to 13) | confirmed mixed-coord amplifies the crash; pure GT is graceful decline |

## Strongest empirical observations

1. **Universal "peak at first val event then decline"**: every variant
   peaks at the first or second val (step 124–250), regardless of lr / inject /
   depth. Different setups change the magnitude of peak gain and severity
   of subsequent decline, but the timing pattern is consistent.

2. **Mixed-coord (inject) + high lr = catastrophic crash**: v22 is the
   clearest example (peak +3.92 dB at step 124, then –13.5 dB by step 250).

3. **Pure-GT path (use_pred_pose=false) declines gracefully**: v25 with
   the same lr=2e-5 hit peak +3.63 dB at step 124, and "only" returned to
   baseline by step 374 — never crashed below.

4. **Depth disabling alone does not stabilise**: v24 with depth=0 still
   crashed (mixed-coord + lr 5e-6). Rules out depth-loss spikes as the
   primary destabiliser, even though they were our first hypothesis.

5. **GoPro ckpt was trained on essentially single-view data**: GoPro
   defocus has 6 crops of the SAME image with c2w ≈ identity; the
   DiffHead never saw multi-view geometry during fine-tuning. Loading
   it onto ScanNet (real consecutive frames with translation) is a
   distribution shift on top of any pose injection.

## Best ckpts saved (as of release date)

```
# Highest peak (mixed-coord, crashed after)
exp_finetune_dav3_scannet_i2slam_2gpu_lr2e5/2026-05-06_15-46-28/
    checkpoints/best-psnr-val/psnr_27.080-info/global_step_124.ckpt

# Pure-GT peak (more stable but lower peak)
exp_finetune_dav3_scannet_i2slam_v25_pureGT_lr2e5/2026-05-06_19-01-XX/
    checkpoints/best-psnr-val/psnr_24.80X-info/global_step_124.ckpt

# Mid-lr ladder (intermediate)
exp_finetune_dav3_scannet_i2slam_v23_resume3gpu/2026-05-06_16-40-15/
    checkpoints/best-psnr-val/psnr_24.656-info/global_step_249.ckpt
```

## Open hypotheses for the crash root cause (untested)

1. **`_normalise_poses` clamp at 0.1**: ScanNet i2slam 6-consecutive-frame
   ctx baseline is 0.025 m; the clamp engages and effectively scales
   translations ×10 instead of normalising. DA3 sees an OOD pose scale
   relative to its training distribution. Fix: set `clamp(min=1e-4)` or
   remove. Untested whether this fixes the crash.

2. **AdamW second-moment accumulation**: even when first-moment averages
   to ~0 across noisy batches, the v² term accumulates monotonically →
   effective lr inflates over steps → DiffHead drifts out of GoPro's
   training basin. Fix candidates: lower `eps`, periodic optimizer reset,
   or just lr=1e-7.

3. **DiffHead capacity vs ScanNet diversity**: only ~1200 unique frames
   across 4 scenes, vs DiffHead which is small but still has many params;
   easy to overfit to spurious patterns then collapse.

4. **ScanNet motion pattern OOD for ReSplat**: ReSplat trained on DL3DV /
   Replica with non-trivial baselines; ScanNet's mm-scale baselines
   degenerate triangulation.

## Suggested next experiments

a. Lift the `_normalise_poses` clamp from 0.1 → 1e-4 and re-run v25 setup
   (pure GT, lr 2e-5). Test whether peak holds past step 124.

b. Increase `frame_stride: 1 → 8` so 6-frame baseline is ~0.4 m
   (matching Replica training distribution). Same v25 setup.

c. Switch init ckpt from GoPro (single-view) to a Replica SOTA ckpt
   (true multi-view trained). Reduces DiffHead distribution shift.

d. Stick with v22's 27.08 ckpt as the artefact for downstream evaluation;
   skip further fine-tuning experiments.
