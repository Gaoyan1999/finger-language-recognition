import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def resolve_video_path(video_id: Any, videos_root: str) -> Tuple[Optional[str], List[str]]:
    """
    WLASL video_id is numeric-as-string in metadata, while local archive files often look like 00341.mp4.
    We try several common renderings and return the first existing candidate.
    """
    vid = "" if video_id is None else str(video_id)
    vid = vid.strip()
    if not vid:
        return None, []

    # Common candidates: exact, stripped leading zeros, and zero-padded.
    stripped = vid.lstrip("0")
    candidates = []
    for cand in [vid, stripped]:
        if cand:
            candidates.append(cand)
    for width in (5, 6):
        try:
            n = int(vid)
            candidates.append(str(n).zfill(width))
        except Exception:
            pass
    # de-dup while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    tried_paths = [os.path.join(videos_root, f"{c}.mp4") for c in uniq]
    for p in tried_paths:
        if os.path.exists(p):
            return p, tried_paths
    return None, tried_paths


def bbox_valid(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    # Basic sanity checks; bbox in WLASL is expected to be in pixel coords.
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def clip_len_frames(inst: Dict[str, Any]) -> Optional[int]:
    fs = _as_int(inst.get("frame_start"))
    fe = _as_int(inst.get("frame_end"))
    if fs is None or fs <= 0:
        return None
    if fe is None:
        return None
    if fe == -1:
        return None
    if fe < fs:
        return None
    # WLASL uses 1-based frame indices in practice.
    return fe - fs + 1


@dataclass
class InstanceChoice:
    gloss: str
    split: str
    instance_id: int
    video_id: str
    url: str
    bbox: List[float]
    frame_start: int
    frame_end: int
    source: str
    signer_id: int
    variation_id: int
    local_video_path: Optional[str]

    def to_json(self) -> Dict[str, Any]:
        d = {
            "gloss": self.gloss,
            "split": self.split,
            "instance_id": self.instance_id,
            "video_id": self.video_id,
            "url": self.url,
            "bbox": self.bbox,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "source": self.source,
            "signer_id": self.signer_id,
            "variation_id": self.variation_id,
            "local_video_path": self.local_video_path,
        }
        return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl_json", type=str, required=True)
    ap.add_argument("--videos_root", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument(
        "--samples_per_gloss",
        type=int,
        default=3,
        help="How many instances to keep per gloss. Use <=0 to keep all matched instances.",
    )
    ap.add_argument("--T", type=int, default=16, help="Default temporal sampling length (used for filtering when frame_end is known).")
    ap.add_argument("--min_clip_len_known", type=int, default=16, help="If frame_end!=-1, require clip length >= this number of frames.")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--manifest_filename",
        type=str,
        default="demo_instances_top20x3.json",
        help="Output manifest filename (saved under out_dir).",
    )
    args = ap.parse_args()

    with open(args.wlasl_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data is a list of {gloss: str, instances: [...]}
    gloss_to_count: Dict[str, int] = {}
    gloss_to_instances: Dict[str, List[Dict[str, Any]]] = {}
    for entry in data:
        gloss = entry.get("gloss")
        instances = entry.get("instances") or []
        if not isinstance(gloss, str):
            continue
        gloss_to_count[gloss] = gloss_to_count.get(gloss, 0) + len(instances)
        gloss_to_instances[gloss] = instances

    top_glosses = sorted(gloss_to_count.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
    top_gloss_names = [g for g, _ in top_glosses]

    os.makedirs(args.out_dir, exist_ok=True)
    top20_path = os.path.join(args.out_dir, "top20_glosses.json")
    with open(top20_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"gloss": g, "num_instances": c} for g, c in top_glosses],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Select demo instances: prefer ones with bbox and local mp4 found.
    chosen: List[InstanceChoice] = []
    select_log_rows: List[Dict[str, Any]] = []
    for gloss in top_gloss_names:
        instances = gloss_to_instances.get(gloss, [])
        # stable ordering: use instance_id if present, else keep source order
        def sort_key(inst: Dict[str, Any]) -> Tuple[int, int]:
            iid = _as_int(inst.get("instance_id")) or 0
            fs = _as_int(inst.get("frame_start")) or 0
            return (iid, fs)

        instances_sorted = sorted(instances, key=sort_key)

        cands: List[Dict[str, Any]] = []
        for inst in instances_sorted:
            bbox = inst.get("bbox")
            if not bbox_valid(bbox):
                select_log_rows.append(
                    {
                        "gloss": gloss,
                        "instance_id": inst.get("instance_id"),
                        "video_id": inst.get("video_id"),
                        "split": inst.get("split"),
                        "reason": "bbox_invalid",
                    }
                )
                continue

            clip_len = clip_len_frames(inst)
            if clip_len is not None and clip_len < args.min_clip_len_known:
                select_log_rows.append(
                    {
                        "gloss": gloss,
                        "instance_id": inst.get("instance_id"),
                        "video_id": inst.get("video_id"),
                        "split": inst.get("split"),
                        "reason": f"clip_too_short_known_len<{args.min_clip_len_known}>",
                    }
                )
                continue

            frame_start = _as_int(inst.get("frame_start"))
            if frame_start is None or frame_start <= 0:
                select_log_rows.append(
                    {
                        "gloss": gloss,
                        "instance_id": inst.get("instance_id"),
                        "video_id": inst.get("video_id"),
                        "split": inst.get("split"),
                        "reason": "frame_start_invalid",
                    }
                )
                continue

            local_path, _tried = resolve_video_path(inst.get("video_id"), args.videos_root)
            if local_path is None:
                select_log_rows.append(
                    {
                        "gloss": gloss,
                        "instance_id": inst.get("instance_id"),
                        "video_id": inst.get("video_id"),
                        "split": inst.get("split"),
                        "reason": "video_not_found_local",
                    }
                )
                continue

            cands.append(inst)

        # Choose first N candidates (can be randomized later if you want).
        if len(cands) < args.samples_per_gloss:
            # Soft fallback: relax clip length constraint by allowing shorter clips if video exists.
            # This keeps demo extraction running even if metadata is strict.
            relaxed: List[Dict[str, Any]] = []
            for inst in instances_sorted:
                bbox = inst.get("bbox")
                if not bbox_valid(bbox):
                    continue
                local_path, _tried = resolve_video_path(inst.get("video_id"), args.videos_root)
                if local_path is None:
                    continue
                frame_start = _as_int(inst.get("frame_start"))
                if frame_start is None or frame_start <= 0:
                    continue
                relaxed.append(inst)
            cands = relaxed

        chosen_pool = cands if int(args.samples_per_gloss) <= 0 else cands[: args.samples_per_gloss]
        for inst in chosen_pool:
            frame_end = _as_int(inst.get("frame_end"))
            frame_end = -1 if frame_end is None else frame_end
            instance_id = _as_int(inst.get("instance_id")) or 0
            signer_id = _as_int(inst.get("signer_id")) or 0
            variation_id = _as_int(inst.get("variation_id")) or 0
            bbox = [float(v) for v in inst.get("bbox")]
            local_path, _tried = resolve_video_path(inst.get("video_id"), args.videos_root)

            chosen.append(
                InstanceChoice(
                    gloss=gloss,
                    split=inst.get("split") or "unknown",
                    instance_id=instance_id,
                    video_id=str(inst.get("video_id")),
                    url=inst.get("url") or "",
                    bbox=bbox,
                    frame_start=_as_int(inst.get("frame_start")) or 1,
                    frame_end=frame_end,
                    source=inst.get("source") or "",
                    signer_id=signer_id,
                    variation_id=variation_id,
                    local_video_path=local_path,
                )
            )

    demo_manifest = {
        "top_k": args.top_k,
        "samples_per_gloss": args.samples_per_gloss,
        "T_default": args.T,
        "instances": [c.to_json() for c in chosen],
    }
    manifest_path = os.path.join(args.out_dir, args.manifest_filename)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(demo_manifest, f, ensure_ascii=False, indent=2)

    # CSV log (for quick inspection)
    import csv

    # out_dir is expected to be: <repo_root>/data/preprocess_manifests
    repo_root = os.path.abspath(os.path.join(args.out_dir, "..", ".."))
    csv_path = os.path.join(repo_root, "reports", "pose_quality", "select_log.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = ["gloss", "instance_id", "video_id", "split", "reason"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in select_log_rows:
            # Keep missing keys safe.
            out = {k: row.get(k) for k in fieldnames}
            w.writerow(out)

    print(f"[build_demo_manifest] top20_path={top20_path}")
    print(f"[build_demo_manifest] demo_manifest={manifest_path}")
    print(f"[build_demo_manifest] select_log={csv_path}")
    if int(args.samples_per_gloss) > 0:
        print(f"[build_demo_manifest] chosen_instances={len(chosen)} (expected {args.top_k*args.samples_per_gloss})")
    else:
        print(f"[build_demo_manifest] chosen_instances={len(chosen)} (all matched instances kept)")


if __name__ == "__main__":
    main()

