import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2


def resolve_video_path(video_id: Any, videos_root: str) -> Tuple[Optional[str], List[str]]:
    vid = "" if video_id is None else str(video_id).strip()
    if not vid:
        return None, []

    stripped = vid.lstrip("0")
    candidates: List[str] = []
    for cand in [vid, stripped]:
        if cand:
            candidates.append(cand)
    for width in (5, 6):
        try:
            n = int(vid)
            candidates.append(str(n).zfill(width))
        except Exception:
            pass

    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)

    tried_paths = [os.path.join(videos_root, f"{c}.mp4") for c in uniq]
    for p in tried_paths:
        if os.path.exists(p):
            return p, tried_paths
    return None, tried_paths


def read_first_frame_ok(video_path: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "cap_not_opened", {}

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return False, "read_failed", {}

    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return True, None, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_manifest", type=str, required=True, help="Path to demo_instances_top20x3.json")
    ap.add_argument("--videos_root", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True, help="Output json mapping")
    ap.add_argument("--out_csv", type=str, required=True, help="Output csv debug log")
    args = ap.parse_args()

    with open(args.demo_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    instances: List[Dict[str, Any]] = manifest["instances"]

    mapping_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for inst in instances:
        instance_id = inst.get("instance_id")
        gloss = inst.get("gloss")
        video_id = inst.get("video_id")
        split = inst.get("split")
        url = inst.get("url")

        bbox = inst.get("bbox")
        local_video_path = inst.get("local_video_path")
        if not local_video_path:
            local_video_path, tried = resolve_video_path(video_id, args.videos_root)
        else:
            tried = [local_video_path]

        if not local_video_path or not os.path.exists(local_video_path):
            row = {
                "instance_id": instance_id,
                "gloss": gloss,
                "video_id": video_id,
                "split": split,
                "url": url,
                "local_video_path": local_video_path,
                "ok": False,
                "error": "video_not_found_local",
                "tried_paths": tried,
                "bbox_valid": isinstance(bbox, list) and len(bbox) == 4,
            }
            mapping_rows.append(row)
            csv_rows.append({k: row.get(k, "") for k in row.keys()})
            continue

        ok, err, meta = read_first_frame_ok(local_video_path)
        row = {
            "instance_id": instance_id,
            "gloss": gloss,
            "video_id": video_id,
            "split": split,
            "url": url,
            "local_video_path": local_video_path,
            "ok": bool(ok),
            "error": None if ok else err,
            "video_meta": meta if ok else {},
            "bbox_valid": isinstance(bbox, list) and len(bbox) == 4,
        }
        mapping_rows.append(row)

        flat = {
            "instance_id": instance_id,
            "gloss": gloss,
            "video_id": video_id,
            "split": split,
            "ok": int(ok),
            "error": None if ok else err,
            "width": meta.get("width", "") if ok else "",
            "height": meta.get("height", "") if ok else "",
            "fps": meta.get("fps", "") if ok else "",
            "frame_count": meta.get("frame_count", "") if ok else "",
            "local_video_path": local_video_path,
        }
        csv_rows.append(flat)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(mapping_rows, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fieldnames = [
        "instance_id",
        "gloss",
        "video_id",
        "split",
        "ok",
        "error",
        "width",
        "height",
        "fps",
        "frame_count",
        "local_video_path",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    ok_count = sum(1 for r in mapping_rows if r.get("ok"))
    print(f"[build_video_index_mapping] ok={ok_count}/{len(mapping_rows)}")
    print(f"[build_video_index_mapping] out_json={args.out_json}")
    print(f"[build_video_index_mapping] out_csv={args.out_csv}")


if __name__ == "__main__":
    main()

