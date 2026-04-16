# Finger Language Recognition

Sign Language Recognition based on CNN

## Train CNN Classifier

After preprocessing (`top100_all_instances.json` + normalized pose `.npz` files) is ready:

```bash
bash scripts/run_train_cnn.sh
```

Optional overrides:

```bash
MANIFEST="/Users/daniel/Workspace/finger-language-recognition/data/preprocess_manifests/top100_all_instances.json" \
POSE_NPZ_DIR="/Users/daniel/Workspace/finger-language-recognition/data/pose_outputs_norm/mediapipe_full_pose" \
OUT_DIR="/Users/daniel/Workspace/finger-language-recognition/reports/train_cnn/top100" \
EPOCHS=30 BATCH_SIZE=64 LR=0.001 \
bash scripts/run_train_cnn.sh
```