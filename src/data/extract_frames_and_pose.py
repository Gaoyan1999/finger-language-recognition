import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


BODY_KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def expand_bbox(bbox_xyxy: List[float], img_w: int, img_h: int, margin_ratio: float) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    dx = bw * margin_ratio / 2.0
    dy = bh * margin_ratio / 2.0
    x1e = max(0, int(round(x1 - dx)))
    y1e = max(0, int(round(y1 - dy)))
    # Treat x2/y2 as exclusive indices for cv2 slicing (frame[y1:y2, x1:x2]).
    x2e = min(img_w, int(round(x2 + dx)))
    y2e = min(img_h, int(round(y2 + dy)))
    # Ensure non-empty crop
    if x2e <= x1e:
        x2e = min(img_w, x1e + 1)
    if y2e <= y1e:
        y2e = min(img_h, y1e + 1)
    return [x1e, y1e, x2e, y2e]


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def sample_frame_indices(
    frame_start_1based: int,
    frame_end: int,
    T: int,
    frame_count: int,
    max_len_when_end_unknown: int,
) -> np.ndarray:
    # WLASL frame indices in metadata typically start at 1
    start0 = int(frame_start_1based) - 1
    start0 = clamp(start0, 0, max(0, frame_count - 1))
    if frame_end is None:
        frame_end = -1
    if frame_end != -1:
        end0 = int(frame_end) - 1
    else:
        end0 = start0 + int(max_len_when_end_unknown) - 1
    end0 = clamp(end0, start0, max(0, frame_count - 1))
    if T <= 1:
        return np.array([start0], dtype=np.int64)

    idx = np.linspace(start0, end0, T).round().astype(np.int64)
    # Keep length exactly T; duplicates are allowed.
    return idx


def read_video_metadata(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False}
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    meta["ok"] = meta["frame_count"] > 0
    return meta


def read_frame_at(cap: cv2.VideoCapture, idx0: int) -> Optional[np.ndarray]:
    # CAP_PROP_POS_FRAMES is 0-based.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def hands_from_mediapipe(
    hands_model: Any,
    rgb_img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      keypoints: float32[K=42,3] with (x_norm, y_norm, score)
      mask: float32[K=42] with 1 valid, 0 missing
    """
    res = hands_model.process(rgb_img)
    keypoints = np.zeros((42, 3), dtype=np.float32)
    mask = np.zeros((42,), dtype=np.float32)

    if not res.multi_hand_landmarks or not res.multi_handedness:
        return keypoints, mask

    # MediaPipe returns lists with aligned ordering: landmarks[i] with handedness[i]
    for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
        label = handedness.classification[0].label  # 'Left'/'Right'
        score_hand = float(handedness.classification[0].score)
        if label == "Left":
            base = 0
        else:
            base = 21
        pts = hand_landmarks.landmark
        for i, lm in enumerate(pts):
            keypoints[base + i, 0] = float(lm.x)
            keypoints[base + i, 1] = float(lm.y)
            keypoints[base + i, 2] = score_hand
            mask[base + i] = 1.0
    return keypoints, mask


def pose_from_mediapipe(
    pose_model: Any,
    rgb_img: np.ndarray,
    body_visibility_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      keypoints: float32[33,3] with (x_norm, y_norm, visibility)
      mask: float32[33]
    """
    res = pose_model.process(rgb_img)
    keypoints = np.zeros((33, 3), dtype=np.float32)
    mask = np.zeros((33,), dtype=np.float32)
    if not res.pose_landmarks:
        return keypoints, mask
    pts = res.pose_landmarks.landmark
    for i, lm in enumerate(pts):
        keypoints[i, 0] = float(lm.x)
        keypoints[i, 1] = float(lm.y)
        # MediaPipe provides visibility in [0,1]; use it as confidence signal.
        vis = float(getattr(lm, "visibility", 0.0) or 0.0)
        keypoints[i, 2] = vis
        mask[i] = 1.0 if vis >= body_visibility_threshold else 0.0
    return keypoints, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_manifest", required=True, help="data/preprocess_manifests/demo_instances_top20x3.json")
    ap.add_argument("--video_index_json", required=True, help="data/preprocess_manifests/demo_video_index.json")
    ap.add_argument("--videos_root", required=True, help="archive/videos")
    ap.add_argument("--out_dir", required=True, help="data/pose_outputs")

    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--max_len_when_end_unknown", type=int, default=64)
    ap.add_argument("--bbox_margin_ratio", type=float, default=0.2)

    ap.add_argument("--input_size", type=int, default=256, help="Resize cropped frame to square input for pose models")
    ap.add_argument("--hand_min_conf", type=float, default=0.5)
    ap.add_argument("--pose_body_visibility_threshold", type=float, default=0.2)

    args = ap.parse_args()

    with open(args.demo_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    instances = manifest["instances"]

    with open(args.video_index_json, "r", encoding="utf-8") as f:
        video_index = json.load(f)
    idx_by_instance_id = {int(x["instance_id"]): x for x in video_index}

    os.makedirs(args.out_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=float(args.hand_min_conf),
        min_tracking_confidence=float(args.hand_min_conf),
    )
    pose_model = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    for i, inst in enumerate(instances):
        instance_id = int(inst["instance_id"])
        gloss = inst["gloss"]
        split = inst.get("split", "unknown")
        local_video_path = inst["local_video_path"]
        bbox = inst["bbox"]
        frame_start = int(inst["frame_start"])
        frame_end = int(inst["frame_end"])

        # video meta (frame_count) from earlier mapping
        meta_entry = idx_by_instance_id.get(instance_id)
        if meta_entry and meta_entry.get("video_meta"):
            frame_count = int(meta_entry["video_meta"].get("frame_count") or 0)
        else:
            frame_count = 0

        if not local_video_path or not os.path.exists(local_video_path):
            print(f"[extract] skip missing video: {local_video_path}")
            continue

        vmeta = read_video_metadata(local_video_path)
        if not frame_count:
            frame_count = int(vmeta.get("frame_count") or 0)
        if frame_count <= 0:
            print(f"[extract] skip unreadable video: {local_video_path}")
            continue

        sampled = sample_frame_indices(
            frame_start_1based=frame_start,
            frame_end=frame_end,
            T=int(args.T),
            frame_count=frame_count,
            max_len_when_end_unknown=int(args.max_len_when_end_unknown),
        )
        sampled_list = sampled.tolist()

        cap = cv2.VideoCapture(local_video_path)
        if not cap.isOpened():
            print(f"[extract] cap open failed: {local_video_path}")
            continue

        # Pre-allocate output tensors
        keypoints_hands = np.zeros((args.T, 42, 3), dtype=np.float32)
        mask_hands = np.zeros((args.T, 42), dtype=np.float32)

        keypoints_full = np.zeros((args.T, 75, 3), dtype=np.float32)  # body33 + hands42
        mask_full = np.zeros((args.T, 75), dtype=np.float32)

        # bbox expanded computed from the first frame's width/height
        # (bbox is provided in original pixel coords, so we need image size)
        ok_first = False
        for t_i, idx0 in enumerate(sampled_list):
            if t_i == 0:
                frame0 = read_frame_at(cap, idx0)
                if frame0 is None:
                    continue
                img_h, img_w = frame0.shape[:2]
                bbox_expanded = expand_bbox(bbox, img_w=img_w, img_h=img_h, margin_ratio=float(args.bbox_margin_ratio))
                ok_first = True
                # Use frame0 for this t_i
                frame = frame0
            else:
                frame = read_frame_at(cap, idx0)
                if frame is None:
                    continue

            if not ok_first:
                continue

            x1, y1, x2, y2 = bbox_expanded
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (int(args.input_size), int(args.input_size)), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            k_h, m_h = hands_from_mediapipe(hands_model, rgb)
            k_b, m_b = pose_from_mediapipe(pose_model, rgb, body_visibility_threshold=float(args.pose_body_visibility_threshold))

            keypoints_hands[t_i] = k_h
            mask_hands[t_i] = m_h

            keypoints_full[t_i, 0:33] = k_b
            mask_full[t_i, 0:33] = m_b
            keypoints_full[t_i, 33:75] = k_h
            mask_full[t_i, 33:75] = m_h

        cap.release()

        full_out_dir = os.path.join(args.out_dir, "mediapipe_full_pose")
        os.makedirs(full_out_dir, exist_ok=True)

        full_out_path = os.path.join(full_out_dir, f"{gloss}__{instance_id}.npz")

        # keypoint names for full pose: body + hands (hands indices are numeric)
        hand_names: List[str] = [f"left_hand_{i}" for i in range(21)] + [f"right_hand_{i}" for i in range(21)]
        full_names = BODY_KEYPOINT_NAMES + hand_names

        np.savez_compressed(
            full_out_path,
            keypoints=keypoints_full,
            mask=mask_full,
            sampled_frame_indices=np.asarray(sampled_list, dtype=np.int64),
            bbox_used=np.asarray(bbox, dtype=np.float32),
            bbox_expanded=np.asarray(bbox_expanded, dtype=np.float32),
            input_size=np.asarray([args.input_size], dtype=np.int32),
            keypoint_names=np.asarray(full_names, dtype=object),
        )

        print(f"[extract] {i+1}/{len(instances)} gloss={gloss} instance_id={instance_id} frames={len(sampled_list)}")


if __name__ == "__main__":
    main()

