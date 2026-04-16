import argparse
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to .npz file")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    print(f"file: {args.npz}")
    print("keys:", list(d.files))
    for k in d.files:
        v = d[k]
        if isinstance(v, np.ndarray):
            print(f"- {k}: dtype={v.dtype}, shape={v.shape}")
            if v.ndim >= 1 and v.size > 0 and k in ("keypoints", "mask", "sampled_frame_indices", "bbox_used", "bbox_expanded"):
                flat = v.reshape(-1)
                preview = flat[: min(8, flat.shape[0])]
                print(f"  preview={preview}")
        else:
            print(f"- {k}: type={type(v)}")


if __name__ == "__main__":
    main()

