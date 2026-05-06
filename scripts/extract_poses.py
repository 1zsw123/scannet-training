"""Pose-only extractor for ScanNet .sens files (Python 3, no image decode).

Reads the .sens header and per-frame metadata, returns all camera_to_world
matrices as a (N, 4, 4) float32 array. Skips color/depth payload entirely
(seeks past it), so it's fast: ~seconds per .sens regardless of file size.

Also returns the camera intrinsics + image dimensions.
"""
from __future__ import annotations
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def parse_sens(path: Path):
    with open(path, "rb") as f:
        version = struct.unpack("I", f.read(4))[0]
        assert version == 4, f"Unexpected .sens version {version}"
        strlen = struct.unpack("Q", f.read(8))[0]
        sensor_name = f.read(strlen).decode("utf-8", errors="replace")

        intrinsic_color = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
        extrinsic_color = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
        intrinsic_depth = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
        extrinsic_depth = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()

        color_compression = struct.unpack("i", f.read(4))[0]
        depth_compression = struct.unpack("i", f.read(4))[0]
        color_w = struct.unpack("I", f.read(4))[0]
        color_h = struct.unpack("I", f.read(4))[0]
        depth_w = struct.unpack("I", f.read(4))[0]
        depth_h = struct.unpack("I", f.read(4))[0]
        depth_shift = struct.unpack("f", f.read(4))[0]
        num_frames = struct.unpack("Q", f.read(8))[0]

        poses = np.empty((num_frames, 4, 4), dtype=np.float32)
        ts_color = np.empty(num_frames, dtype=np.uint64)
        ts_depth = np.empty(num_frames, dtype=np.uint64)
        for i in range(num_frames):
            poses[i] = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4)
            ts_color[i] = struct.unpack("Q", f.read(8))[0]
            ts_depth[i] = struct.unpack("Q", f.read(8))[0]
            csz = struct.unpack("Q", f.read(8))[0]
            dsz = struct.unpack("Q", f.read(8))[0]
            f.seek(csz + dsz, 1)  # skip color + depth payload

    return {
        "sensor_name": sensor_name,
        "intrinsic_color": intrinsic_color,
        "extrinsic_color": extrinsic_color,
        "intrinsic_depth": intrinsic_depth,
        "extrinsic_depth": extrinsic_depth,
        "color_size": (color_w, color_h),
        "depth_size": (depth_w, depth_h),
        "depth_shift": depth_shift,
        "num_frames": num_frames,
        "poses": poses,        # (N, 4, 4)  camera_to_world
        "ts_color": ts_color,  # (N,)
        "ts_depth": ts_depth,  # (N,)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", type=str, required=True)
    ap.add_argument("--out", type=str, required=True,
                    help="output .npz path (poses + intrinsics)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    info = parse_sens(Path(args.sens))
    print(f"  sensor:        {info['sensor_name']}")
    print(f"  num_frames:    {info['num_frames']}")
    print(f"  color_size:    {info['color_size']}")
    print(f"  depth_size:    {info['depth_size']}")
    print(f"  depth_shift:   {info['depth_shift']}")
    print(f"  intrinsic_color:\n{info['intrinsic_color']}")
    print(f"  intrinsic_depth:\n{info['intrinsic_depth']}")

    np.savez(out_path,
             poses=info["poses"],
             ts_color=info["ts_color"],
             ts_depth=info["ts_depth"],
             intrinsic_color=info["intrinsic_color"],
             extrinsic_color=info["extrinsic_color"],
             intrinsic_depth=info["intrinsic_depth"],
             extrinsic_depth=info["extrinsic_depth"],
             color_size=np.array(info["color_size"], dtype=np.int32),
             depth_size=np.array(info["depth_size"], dtype=np.int32),
             depth_shift=np.float32(info["depth_shift"]),
             num_frames=np.int64(info["num_frames"]))
    print(f"  -> wrote {out_path}")


if __name__ == "__main__":
    main()
