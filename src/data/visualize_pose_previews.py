import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def draw_hand_skeleton(
    img_bgr: np.ndarray,
    keypoints: np.ndarray,
    mask: np.ndarray,
    color_left: Tuple[int, int, int],
    color_right: Tuple[int, int, int],
    radius: int = 2,
    thickness: int = 1,
) -> np.ndarray:
    """
    keypoints: [42,3] where x/y are in [0,1] relative to image
    mask: [42]
    """
    out = img_bgr.copy()
    mp_hands = mp.solutions.hands
    connections = list(mp_hands.HAND_CONNECTIONS)

    # points
    for j in range(42):
        if mask[j] < 0.5:
            continue
        x = int(round(keypoints[j, 0] * out.shape[1]))
        y = int(round(keypoints[j, 1] * out.shape[0]))
        x = max(0, min(out.shape[1] - 1, x))
        y = max(0, min(out.shape[0] - 1, y))
        col = color_left if j < 21 else color_right
        cv2.circle(out, (x, y), radius, col, -1)

    # connections
    for a, b in connections:
        # left hand
        ja, jb = int(a), int(b)
        if mask[ja] >= 0.5 and mask[jb] >= 0.5:
            xa = int(round(keypoints[ja, 0] * out.shape[1]))
            ya = int(round(keypoints[ja, 1] * out.shape[0]))
            xb = int(round(keypoints[jb, 0] * out.shape[1]))
            yb = int(round(keypoints[jb, 1] * out.shape[0]))
            cv2.line(out, (xa, ya), (xb, yb), color_left, thickness)
        # right hand
        ja2, jb2 = ja + 21, jb + 21
        if mask[ja2] >= 0.5 and mask[jb2] >= 0.5:
            xa = int(round(keypoints[ja2, 0] * out.shape[1]))
            ya = int(round(keypoints[ja2, 1] * out.shape[0]))
            xb = int(round(keypoints[jb2, 0] * out.shape[1]))
            yb = int(round(keypoints[jb2, 1] * out.shape[0]))
            cv2.line(out, (xa, ya), (xb, yb), color_right, thickness)
    return out


def draw_full_skeleton(
    img_bgr: np.ndarray,
    keypoints: np.ndarray,
    mask: np.ndarray,
    radius: int = 2,
    thickness: int = 1,
) -> np.ndarray:
    """
    keypoints: [75,3] = body[0:33] + hands[33:75]?? (we store body then hands)
    mask: [75]
    """
    out = img_bgr.copy()
    mp_pose = mp.solutions.pose
    body_connections = list(mp_pose.POSE_CONNECTIONS)

    color_body = (0, 0, 255)  # red in BGR
    color_left = (0, 255, 0)  # green
    color_right = (255, 0, 0)  # blue

    # body points
    body_kp = keypoints[0:33]
    body_mask = mask[0:33]
    for j in range(33):
        if body_mask[j] < 0.5:
            continue
        x = int(round(body_kp[j, 0] * out.shape[1]))
        y = int(round(body_kp[j, 1] * out.shape[0]))
        x = max(0, min(out.shape[1] - 1, x))
        y = max(0, min(out.shape[0] - 1, y))
        cv2.circle(out, (x, y), radius, color_body, -1)

    # body connections
    for a, b in body_connections:
        ia, ib = int(a), int(b)
        if body_mask[ia] < 0.5 or body_mask[ib] < 0.5:
            continue
        xa = int(round(body_kp[ia, 0] * out.shape[1]))
        ya = int(round(body_kp[ia, 1] * out.shape[0]))
        xb = int(round(body_kp[ib, 0] * out.shape[1]))
        yb = int(round(body_kp[ib, 1] * out.shape[0]))
        cv2.line(out, (xa, ya), (xb, yb), color_body, thickness)

    # hands (reuse existing function)
    hands_kp = keypoints[33:75]
    hands_mask = mask[33:75]
    out = draw_hand_skeleton(
        out,
        keypoints=hands_kp,
        mask=hands_mask,
        color_left=color_left,
        color_right=color_right,
        radius=radius,
        thickness=thickness,
    )
    return out


def make_grid(frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
    assert len(frames) == rows * cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        r = i // cols
        c = i % cols
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = fr
    return grid


def compute_pose_stats_full(keypoints: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    keypoints: [T,75,3] = body(33)+hands(42) stored after normalization
    mask: [T,75]
    """
    body_mask = mask[:, 0:33]
    hands_mask = mask[:, 33:75]
    left_mask = hands_mask[:, 0:21]
    right_mask = hands_mask[:, 21:42]
    left_present = (left_mask.sum(axis=1) > 0)
    right_present = (right_mask.sum(axis=1) > 0)
    both_present = left_present & right_present

    body_present = body_mask.sum(axis=1) > 0
    body_detect_rate = float(body_present.mean()) if body_present.size else 0.0

    hand_detect_rate = float(both_present.mean()) if both_present.size else 0.0
    missing_ratio = float(1.0 - mask.mean()) if mask.size else 1.0
    valid_scores = keypoints[..., 2][mask > 0.5]
    avg_score = float(valid_scores.mean()) if valid_scores.size else 0.0
    return {
        "hand_detect_rate": hand_detect_rate,
        "body_detect_rate": body_detect_rate,
        "missing_ratio": missing_ratio,
        "avg_score": avg_score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_manifest", required=True, help="data/preprocess_manifests/demo_instances_top20x3.json")
    ap.add_argument("--pose_outputs_norm_dir", required=True, help="data/pose_outputs_norm")
    ap.add_argument("--previews_out_dir", required=True, help="data/pose_previews")
    ap.add_argument("--input_size_override", type=int, default=None, help="Optional debug override")
    ap.add_argument("--grid_cols", type=int, default=4)
    args = ap.parse_args()

    with open(args.demo_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    instances = manifest["instances"]

    os.makedirs(args.previews_out_dir, exist_ok=True)
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    # Stats accumulator
    stats_rows: List[Dict[str, Any]] = []

    # Cache video captures per path to reduce overhead (small number of videos)
    cap_cache: Dict[str, cv2.VideoCapture] = {}

    def get_cap(video_path: str) -> cv2.VideoCapture:
        if video_path not in cap_cache:
            cap_cache[video_path] = cv2.VideoCapture(video_path)
        return cap_cache[video_path]

    for i, inst in enumerate(instances):
        gloss = inst["gloss"]
        instance_id = int(inst["instance_id"])
        local_video_path = inst["local_video_path"]

        full_npz = os.path.join(args.pose_outputs_norm_dir, "mediapipe_full_pose", f"{gloss}__{instance_id}.npz")
        if not os.path.exists(full_npz):
            print(f"[viz] missing npz for {gloss} {instance_id}")
            continue

        full_data = np.load(full_npz, allow_pickle=True)
        sampled = full_data["sampled_frame_indices"].astype(np.int64)
        input_size = int(full_data["input_size"][0]) if full_data["input_size"].size else int(args.input_size_override or 256)
        bbox_expanded_full = full_data["bbox_expanded"].astype(np.int32)

        keypoints_full = full_data["keypoints"]
        mask_full = full_data["mask"]

        T = int(keypoints_full.shape[0])
        cols = int(args.grid_cols)
        rows = int(math.ceil(T / cols))
        total_cells = rows * cols

        # --- Render full-pose grid only ---
        frames_out: List[np.ndarray] = []
        cap = get_cap(local_video_path)
        for t in range(T):
            idx0 = int(sampled[t])
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx0)
            ok, frame = cap.read()
            if not ok or frame is None:
                # fallback blank
                frames_out.append(np.zeros((input_size, input_size, 3), dtype=np.uint8))
                continue
            x1, y1, x2, y2 = bbox_expanded_full.tolist()
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                frames_out.append(np.zeros((input_size, input_size, 3), dtype=np.uint8))
                continue
            crop_resized = cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            drawn = draw_full_skeleton(
                crop_resized,
                keypoints=keypoints_full[t],
                mask=mask_full[t],
                radius=2,
                thickness=1,
            )
            frames_out.append(drawn)

        # Pad to total_cells
        while len(frames_out) < total_cells:
            frames_out.append(np.zeros((input_size, input_size, 3), dtype=np.uint8))

        grid = make_grid(frames_out[: total_cells], rows=rows, cols=cols)
        full_preview_dir = os.path.join(args.previews_out_dir, "mediapipe_full_pose", gloss)
        os.makedirs(full_preview_dir, exist_ok=True)
        full_preview_path = os.path.join(full_preview_dir, f"{instance_id}_grid.png")
        cv2.imwrite(full_preview_path, grid)

        full_stats = compute_pose_stats_full(keypoints_full, mask_full)
        stats_rows.append(
            {
                "tool": "mediapipe",
                "scope": "full_pose",
                "gloss": gloss,
                "instance_id": instance_id,
                "video_id": inst["video_id"],
                **full_stats,
            }
        )

        print(f"[viz] {i+1}/{len(instances)} gloss={gloss} instance_id={instance_id}")

    # Release captures
    for _, cap_obj in cap_cache.items():
        try:
            cap_obj.release()
        except Exception:
            pass

    # Write stats CSV
    # previews_out_dir is expected to be: <repo_root>/data/pose_previews
    repo_root = os.path.abspath(os.path.join(args.previews_out_dir, "..", ".."))
    stats_dir = os.path.join(repo_root, "reports", "pose_quality")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "pose_stats.csv")

    # Normalize columns
    fieldnames = [
        "tool",
        "scope",
        "gloss",
        "instance_id",
        "video_id",
        "hand_detect_rate",
        "body_detect_rate",
        "missing_ratio",
        "avg_score",
    ]

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in stats_rows:
            if "body_detect_rate" not in r:
                r["body_detect_rate"] = ""
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[viz] stats_path={stats_path}")


if __name__ == "__main__":
    main()

