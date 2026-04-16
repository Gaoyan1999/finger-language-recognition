#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# Top-100 glosses + all instances + full-pose extraction + normalization + previews.
#
# It does NOT rely on .skel_venv*.
# It will create/use .pose_venv and auto-install required packages if missing.
#
# Usage:
#   bash scripts/run_top100_fullpose.sh
#
# Optional overrides:
#   WLASL_JSON="/Users/daniel/Downloads/archive/WLASL_v0.3.json" \
#   VIDEOS_ROOT="/Users/daniel/Downloads/archive/videos" \
#   TOP_K=100 T=16 INPUT_SIZE=256 \
#   bash scripts/run_top100_fullpose.sh

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WLASL_JSON="${WLASL_JSON:-/Users/daniel/Downloads/archive/WLASL_v0.3.json}"
VIDEOS_ROOT="${VIDEOS_ROOT:-/Users/daniel/Downloads/archive/videos}"
TOP_K="${TOP_K:-100}"
T="${T:-16}"
INPUT_SIZE="${INPUT_SIZE:-256}"
MAX_LEN_UNKNOWN="${MAX_LEN_UNKNOWN:-64}"
BBOX_MARGIN_RATIO="${BBOX_MARGIN_RATIO:-0.2}"

MANIFEST_DIR="$REPO_ROOT/data/preprocess_manifests"
MANIFEST_FILE="$MANIFEST_DIR/top${TOP_K}_all_instances.json"
VIDEO_INDEX_JSON="$MANIFEST_DIR/top${TOP_K}_video_index.json"

POSE_OUT_RAW="$REPO_ROOT/data/pose_outputs"
POSE_OUT_NORM="$REPO_ROOT/data/pose_outputs_norm"
POSE_PREVIEWS="$REPO_ROOT/data/pose_previews"
REPORTS_DIR="$REPO_ROOT/reports/pose_quality"
VIDEO_RESOLVE_LOG="$REPORTS_DIR/video_resolve_log_top${TOP_K}.csv"

mkdir -p "$MANIFEST_DIR" "$POSE_OUT_RAW" "$POSE_OUT_NORM" "$POSE_PREVIEWS" "$REPORTS_DIR"

# ---------- Python bootstrap (no .skel_venv dependency) ----------
if command -v python3.12 >/dev/null 2>&1; then
  BOOT_PY="python3.12"
else
  BOOT_PY="python3"
fi

POSE_VENV_DIR="${POSE_VENV_DIR:-$REPO_ROOT/.pose_venv}"
if [[ ! -x "$POSE_VENV_DIR/bin/python" ]]; then
  echo "[bootstrap] creating venv at $POSE_VENV_DIR (via $BOOT_PY)"
  "$BOOT_PY" -m venv "$POSE_VENV_DIR"
fi

PYTHON_BIN="$POSE_VENV_DIR/bin/python"
PIP_BIN="$POSE_VENV_DIR/bin/pip"

echo "[bootstrap] python: $PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U pip >/dev/null

echo "[bootstrap] checking/installing dependencies"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import subprocess
import sys

required = [
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
]
missing_pkgs = [pkg for mod, pkg in required if importlib.util.find_spec(mod) is None]
if missing_pkgs:
    cmd = [sys.executable, "-m", "pip", "install"] + missing_pkgs
    print("[bootstrap] installing:", " ".join(missing_pkgs))
    subprocess.check_call(cmd)
else:
    print("[bootstrap] all required packages already installed")
PY

# ---------- Pipeline ----------
echo "[1/5] Build top-${TOP_K} manifest (all instances per gloss)"
"$PYTHON_BIN" "$REPO_ROOT/src/data/build_demo_manifest.py" \
  --wlasl_json "$WLASL_JSON" \
  --videos_root "$VIDEOS_ROOT" \
  --top_k "$TOP_K" \
  --samples_per_gloss -1 \
  --T "$T" \
  --out_dir "$MANIFEST_DIR" \
  --manifest_filename "$(basename "$MANIFEST_FILE")"

echo "[2/5] Build video index mapping"
"$PYTHON_BIN" "$REPO_ROOT/src/data/build_video_index_mapping.py" \
  --demo_manifest "$MANIFEST_FILE" \
  --videos_root "$VIDEOS_ROOT" \
  --out_json "$VIDEO_INDEX_JSON" \
  --out_csv "$VIDEO_RESOLVE_LOG"

echo "[3/5] Extract full-pose keypoints"
"$PYTHON_BIN" "$REPO_ROOT/src/data/extract_frames_and_pose.py" \
  --demo_manifest "$MANIFEST_FILE" \
  --video_index_json "$VIDEO_INDEX_JSON" \
  --videos_root "$VIDEOS_ROOT" \
  --out_dir "$POSE_OUT_RAW" \
  --T "$T" \
  --input_size "$INPUT_SIZE" \
  --max_len_when_end_unknown "$MAX_LEN_UNKNOWN" \
  --bbox_margin_ratio "$BBOX_MARGIN_RATIO"

echo "[4/5] Normalize outputs (full pose only)"
"$PYTHON_BIN" "$REPO_ROOT/src/data/normalize_pose_outputs.py" \
  --in_dir "$POSE_OUT_RAW" \
  --out_dir "$POSE_OUT_NORM" \
  --scopes mediapipe_full_pose

echo "[5/5] Generate full-pose previews + stats"
"$PYTHON_BIN" "$REPO_ROOT/src/data/visualize_pose_previews.py" \
  --demo_manifest "$MANIFEST_FILE" \
  --pose_outputs_norm_dir "$POSE_OUT_NORM" \
  --previews_out_dir "$POSE_PREVIEWS" \
  --grid_cols 4

echo ""
echo "Done."
echo "Manifest: $MANIFEST_FILE"
echo "Video index: $VIDEO_INDEX_JSON"
echo "Pose outputs (norm): $POSE_OUT_NORM/mediapipe_full_pose"
echo "Previews: $POSE_PREVIEWS/mediapipe_full_pose"
echo "Stats: $REPORTS_DIR/pose_stats.csv"
