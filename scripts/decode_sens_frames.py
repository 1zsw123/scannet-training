"""Decode specific color frames from a .sens file (Python 3).

Used to verify the index mapping between i2slam color_NNNNN.png and the
original ScanNet recording.
"""
from __future__ import annotations
import argparse
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def open_sens(path: Path):
    """Returns (file_handle_at_first_frame, header_dict, header_offset_after_header_bytes)."""
    f = open(path, "rb")
    version = struct.unpack("I", f.read(4))[0]
    assert version == 4
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

    header = dict(
        sensor=sensor_name,
        num_frames=num_frames,
        color_size=(color_w, color_h),
        depth_size=(depth_w, depth_h),
        color_compression=color_compression,    # 2 = jpeg
        depth_compression=depth_compression,
    )
    return f, header


def seek_to_frame(f, target_idx):
    """Skip frames until reaching `target_idx`. Stream-only; one direction.
    Caller must call open_sens first to advance to first frame."""
    for _ in range(target_idx):
        # cam_to_world (16 floats) + ts_color + ts_depth + color_size + depth_size + payloads
        f.read(64 + 8 + 8)  # pose + ts_color + ts_depth
        csz = struct.unpack("Q", f.read(8))[0]
        dsz = struct.unpack("Q", f.read(8))[0]
        f.seek(csz + dsz, 1)


def read_color_frame(f, color_compression: int) -> np.ndarray:
    """Read the next frame's color image; returns HWC uint8 RGB."""
    f.read(64)        # pose
    f.read(8)         # ts_color
    f.read(8)         # ts_depth
    csz = struct.unpack("Q", f.read(8))[0]
    dsz = struct.unpack("Q", f.read(8))[0]
    color_data = f.read(csz)
    f.seek(dsz, 1)    # skip depth payload
    if color_compression == 2:    # jpeg
        img = Image.open(BytesIO(color_data)).convert("RGB")
        return np.array(img)
    elif color_compression == 1:  # png
        img = Image.open(BytesIO(color_data)).convert("RGB")
        return np.array(img)
    elif color_compression == 0:  # raw
        # raw RGB at color_size
        raise NotImplementedError("raw not handled")
    raise ValueError(f"Unknown color_compression={color_compression}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", type=str, required=True)
    ap.add_argument("--indices", type=int, nargs="+", required=True,
                    help="zero-based original frame indices to decode")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f, hdr = open_sens(Path(args.sens))
    print(f"sens: num_frames={hdr['num_frames']}  color_size={hdr['color_size']}  compression={hdr['color_compression']}")

    indices = sorted(set(args.indices))
    cur = 0
    try:
        for idx in indices:
            seek_to_frame(f, idx - cur)
            cur = idx
            img = read_color_frame(f, hdr["color_compression"])
            cur += 1
            out = out_dir / f"orig_{idx:06d}.jpg"
            Image.fromarray(img).save(out)
            print(f"  decoded orig frame {idx} ({img.shape[1]}x{img.shape[0]})  -> {out}")
    finally:
        f.close()


if __name__ == "__main__":
    main()
