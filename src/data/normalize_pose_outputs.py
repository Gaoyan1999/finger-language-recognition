import argparse
import os
from typing import Tuple

import numpy as np


def normalize_npz_pair(in_path: str, out_path: str) -> Tuple[int, int]:
    """
    Returns: (num_files_with_any_clip, num_files_total_placeholder)
    """
    data = np.load(in_path, allow_pickle=True)
    keypoints = data["keypoints"]
    mask = data["mask"]

    keypoints = np.asarray(keypoints, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    # Clip x/y to [0,1]; keep score as-is.
    keypoints[..., 0] = np.clip(keypoints[..., 0], 0.0, 1.0)
    keypoints[..., 1] = np.clip(keypoints[..., 1], 0.0, 1.0)
    # Make mask strictly 0/1.
    mask = (mask > 0.5).astype(np.float32)

    # Copy rest keys
    out_dict = {}
    for k in data.files:
        if k == "keypoints":
            out_dict[k] = keypoints
        elif k == "mask":
            out_dict[k] = mask
        else:
            out_dict[k] = data[k]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **out_dict)
    return 0, 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="data/pose_outputs (raw)")
    ap.add_argument("--out_dir", required=True, help="data/pose_outputs_norm")
    ap.add_argument("--scopes", nargs="*", default=["mediapipe_full_pose"])
    args = ap.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir

    total = 0
    for scope in args.scopes:
        src = os.path.join(in_dir, scope)
        dst = os.path.join(out_dir, scope)
        if not os.path.exists(src):
            continue
        for fn in os.listdir(src):
            if not fn.endswith(".npz"):
                continue
            total += 1
            in_path = os.path.join(src, fn)
            out_path = os.path.join(dst, fn)
            normalize_npz_pair(in_path, out_path)

    print(f"[normalize_pose_outputs] normalized={total}")


if __name__ == "__main__":
    main()

