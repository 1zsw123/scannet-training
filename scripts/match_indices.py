"""Test which subsampling rule maps i2slam color_NNNNN.png -> original frame.

For each i2slam frame index i (0, mid, last), compute candidate orig indices
under several hypotheses (linspace, fixed-stride, ...), decode the original
.sens color frame at that index, resize to 640x480, and compare with i2slam.
The best match (lowest L1) wins.
"""
from __future__ import annotations
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Reuse decode utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from decode_sens_frames import open_sens, seek_to_frame, read_color_frame


def best_orig_indices_linspace(N_orig: int, M_i2slam: int) -> np.ndarray:
    return np.linspace(0, N_orig - 1, M_i2slam).round().astype(int)


def best_orig_indices_floor(N_orig: int, M_i2slam: int) -> np.ndarray:
    # i * (N // M)
    stride = N_orig // M_i2slam
    return np.arange(M_i2slam) * stride


def best_orig_indices_div(N_orig: int, M_i2slam: int) -> np.ndarray:
    # i * N / M, floor
    return (np.arange(M_i2slam) * N_orig // M_i2slam).astype(int)


def load_i2slam_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))   # H W 3 uint8


def downsize(arr: np.ndarray, target_wh) -> np.ndarray:
    img = Image.fromarray(arr).resize(target_wh, Image.LANCZOS)
    return np.array(img)


def cmp(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def test_mapping(sens_path: Path, i2slam_dir: Path, M_i2slam: int):
    f, hdr = open_sens(sens_path)
    N_orig = hdr["num_frames"]
    print(f"sens: N_orig={N_orig}  i2slam M={M_i2slam}")

    candidates = {
        "linspace": best_orig_indices_linspace(N_orig, M_i2slam),
        "floor_stride": best_orig_indices_floor(N_orig, M_i2slam),
        "i*N//M":     best_orig_indices_div(N_orig, M_i2slam),
    }
    print("Candidate first/mid/last orig indices:")
    for name, arr in candidates.items():
        print(f"  {name}:  i=0->{arr[0]}, i={M_i2slam//2}->{arr[M_i2slam//2]}, i={M_i2slam-1}->{arr[-1]}")

    # Probe i = 0, mid, last in i2slam
    i_probes = [0, M_i2slam // 2, M_i2slam - 1]
    # Collect all unique candidate orig indices
    needed_orig = sorted({int(arr[i]) for arr in candidates.values() for i in i_probes})
    print(f"Decoding {len(needed_orig)} unique orig frames: {needed_orig}")

    decoded = {}
    cur = 0
    for o in needed_orig:
        seek_to_frame(f, o - cur)
        cur = o
        img = read_color_frame(f, hdr["color_compression"])
        cur += 1
        decoded[o] = img
    f.close()
    print("Decoded.")

    # i2slam pngs at probe indices
    i2 = {i: load_i2slam_png(i2slam_dir / f"color_{i:05d}.png") for i in i_probes}
    target_wh = (i2[0].shape[1], i2[0].shape[0])  # (W, H)

    # For each candidate hypothesis, score the average L1 across probes
    scores = {}
    for name, arr in candidates.items():
        diffs = []
        for i in i_probes:
            o = int(arr[i])
            d = downsize(decoded[o], target_wh)
            diffs.append(cmp(d, i2[i]))
        scores[name] = (np.mean(diffs), diffs)
    for name, (avg, diffs) in sorted(scores.items(), key=lambda x: x[1][0]):
        per = "  ".join(f"i={i}->orig={int(candidates[name][i]):d}: L1={d:.2f}"
                        for i, d in zip(i_probes, diffs))
        print(f"  [{name}] avg L1 = {avg:.2f}    {per}")

    return scores, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", type=str, required=True)
    ap.add_argument("--i2slam_dir", type=str, required=True,
                    help="dir holding color_NNNNN.png files")
    ap.add_argument("--num_i2slam", type=int, required=True)
    args = ap.parse_args()

    test_mapping(Path(args.sens), Path(args.i2slam_dir), args.num_i2slam)


if __name__ == "__main__":
    main()
