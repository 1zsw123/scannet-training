"""
VD-Diff S3 inference on I2-SLAM ScanNet scenes.

Reads gt/color_NNNNN.png (640x480), writes deblurred frames to vdiff/color_NNNNN.png.

Usage:
    python scripts/infer_scannet_i2slam.py \
        --ckpt /scratch-shared/qzhang1/vdiff_training_states/Replica_S3_models/net_g_400000.pth \
        --data_root /scratch-shared/qzhang1/datasets/i2slam_scannet/I2-SLAM_dataset/rgbd/scannet \
        --scenes scene0024_01 scene0031_00 scene0736_00 scene0785_00 \
        --num_frame 7
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from basicsr.archs.S3_arch import S3_arch


def load_model(ckpt_path: str, device: torch.device) -> S3_arch:
    model = S3_arch(
        num_feat=64,
        propagation_blocks=15,
        num_blocks=[7, 2],
        heads=[4, 8],
        use_cross_attention=False,
        cross_attention_heads=2,
        bias=False,
        n_denoise_res=1,
        linear_start=0.1,
        linear_end=0.99,
        timesteps=4,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def load_frames(img_dir: Path):
    paths = sorted(img_dir.glob("color_*.png"))
    if not paths:
        return None, []
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = torch.from_numpy(np.array(img)).float() / 255.0
        frames.append(t.permute(2, 0, 1))
    return torch.stack(frames, dim=0), [p.name for p in paths]


@torch.no_grad()
def infer_scene(model: S3_arch, frames: torch.Tensor, num_frame: int, device: torch.device) -> torch.Tensor:
    T_orig, C, H, W = frames.shape
    assert H % 8 == 0 and W % 8 == 0, f"H={H}, W={W} must be % 8"

    if T_orig < num_frame:
        pad = num_frame - T_orig
        frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0)

    T = frames.shape[0]
    output = torch.zeros(T, C, H, W)
    count = torch.zeros(T, 1, 1, 1, dtype=torch.float32)

    starts = list(range(0, T - num_frame + 1, num_frame))
    if T % num_frame != 0:
        starts.append(T - num_frame)

    for start in starts:
        clip = frames[start:start + num_frame]
        lq = clip.unsqueeze(0).to(device)
        pred = model(lq)
        pred = pred.squeeze(0).cpu().clamp(0, 1)
        output[start:start + num_frame] += pred
        count[start:start + num_frame] += 1

    return (output / count.to(output.device))[:T_orig]


def save_frames(out_dir: Path, names, frames: torch.Tensor):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in zip(names, frames):
        arr = (t.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True,
                   help="parent dir holding scene*/gt/color_*.png")
    p.add_argument("--scenes", type=str, nargs="+", default=None,
                   help="scene subdir names; default: all subdirs of data_root")
    p.add_argument("--in_subdir", type=str, default="gt",
                   help="subdirectory of each scene that holds blurry color_*.png")
    p.add_argument("--out_subdir", type=str, default="vdiff",
                   help="output subdirectory (created next to in_subdir)")
    p.add_argument("--num_frame", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[infer_scannet_i2slam] loading {args.ckpt}", flush=True)
    model = load_model(args.ckpt, device)

    data_root = Path(args.data_root)
    if args.scenes is None:
        scenes = sorted(d.name for d in data_root.iterdir() if (d / args.in_subdir).is_dir())
    else:
        scenes = list(args.scenes)
    print(f"[infer_scannet_i2slam] {len(scenes)} scene(s): {scenes}", flush=True)

    grand_t0 = time.time()
    for scene in scenes:
        in_dir = data_root / scene / args.in_subdir
        out_dir = data_root / scene / args.out_subdir
        if not in_dir.exists():
            print(f"  [skip] {scene}: {in_dir} missing", flush=True)
            continue
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            print(f"  [skip] {scene}: {out_dir} non-empty (use --overwrite to redo)", flush=True)
            continue

        t0 = time.time()
        frames, names = load_frames(in_dir)
        if frames is None:
            print(f"  [skip] {scene}: no color_*.png in {in_dir}", flush=True)
            continue
        T, C, H, W = frames.shape
        print(f"  [{scene}] T={T} {C}x{H}x{W}  load={time.time()-t0:.1f}s", flush=True)

        t1 = time.time()
        out = infer_scene(model, frames, args.num_frame, device)
        torch.cuda.synchronize()
        dt_infer = time.time() - t1
        peak = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()

        t2 = time.time()
        save_frames(out_dir, names, out)
        dt_save = time.time() - t2
        print(f"  [{scene}] infer={dt_infer:.1f}s  save={dt_save:.1f}s  peak={peak:.1f}GB  -> {out_dir}", flush=True)

    print(f"[infer_scannet_i2slam] done in {time.time()-grand_t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
