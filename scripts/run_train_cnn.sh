#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

POSE_VENV_DIR="${POSE_VENV_DIR:-$REPO_ROOT/.pose_venv}"
PYTHON_BIN="${PYTHON_BIN:-$POSE_VENV_DIR/bin/python}"

MANIFEST="${MANIFEST:-$REPO_ROOT/data/preprocess_manifests/top100_all_instances.json}"
POSE_NPZ_DIR="${POSE_NPZ_DIR:-$REPO_ROOT/data/pose_outputs_norm/mediapipe_full_pose}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports/train_cnn/top100}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
DROPOUT="${DROPOUT:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[train-cnn] python not found at $PYTHON_BIN"
  echo "[train-cnn] create venv first, or set PYTHON_BIN=/path/to/python"
  exit 1
fi

echo "[train-cnn] python=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import subprocess
import sys

required = [
    ("torch", "torch"),
    ("numpy", "numpy"),
]
missing = [pkg for mod, pkg in required if importlib.util.find_spec(mod) is None]
if missing:
    print("[train-cnn] installing:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
else:
    print("[train-cnn] dependencies already installed")
PY

"$PYTHON_BIN" "$REPO_ROOT/src/train/train_cnn_classifier.py" \
  --manifest "$MANIFEST" \
  --pose_npz_dir "$POSE_NPZ_DIR" \
  --out_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --dropout "$DROPOUT" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED"

echo "[train-cnn] done. outputs=$OUT_DIR"
