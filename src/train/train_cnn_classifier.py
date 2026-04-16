import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    npz_path: str
    label_id: int
    split: str


class PoseNPZDataset(Dataset):
    def __init__(self, samples: List[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _normalize_keypoints(
        keypoints: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize each frame to a body-centric coordinate system.
        Center: shoulder midpoint when available, otherwise visible body centroid.
        Scale: shoulder distance when available, otherwise mean visible distance.
        """
        xy = keypoints[..., :2].copy()  # [T,75,2]
        z = keypoints[..., 2:3].copy()  # [T,75,1]
        t = xy.shape[0]

        body_xy = xy[:, :33, :]
        body_mask = mask[:, :33]

        # Body centroid fallback.
        body_count = np.maximum(body_mask.sum(axis=1, keepdims=True), 1.0)  # [T,1]
        body_center = (body_xy * body_mask[..., None]).sum(axis=1) / body_count  # [T,2]

        # Prefer shoulder midpoint when both shoulders are visible.
        l_shoulder = xy[:, 11, :]
        r_shoulder = xy[:, 12, :]
        shoulder_valid = (mask[:, 11] > 0.5) & (mask[:, 12] > 0.5)
        shoulder_center = (l_shoulder + r_shoulder) / 2.0
        center = body_center.copy()
        center[shoulder_valid] = shoulder_center[shoulder_valid]

        # Scale by shoulder distance when possible, otherwise by mean visible radius.
        shoulder_dist = np.linalg.norm(l_shoulder - r_shoulder, axis=1)  # [T]
        dist_from_center = np.linalg.norm(xy - center[:, None, :], axis=2)  # [T,75]
        visible_count = np.maximum(mask.sum(axis=1), 1.0)  # [T]
        mean_visible_dist = (dist_from_center * mask).sum(axis=1) / visible_count  # [T]
        scale = np.where(shoulder_valid, shoulder_dist, mean_visible_dist)
        scale = np.maximum(scale, 1e-3).astype(np.float32)  # [T]

        xy = (xy - center[:, None, :]) / scale[:, None, None]
        z = z / scale[:, None, None]
        out = np.concatenate([xy, z], axis=2)  # [T,75,3]
        return out.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        d = np.load(s.npz_path, allow_pickle=True)
        keypoints = np.asarray(d["keypoints"], dtype=np.float32)  # [T,75,3]
        mask = np.asarray(d["mask"], dtype=np.float32)  # [T,75]
        mask = np.clip(mask, 0.0, 1.0)
        mask_3d = mask[..., None]  # [T,75,1]

        # (2) Normalize to body-centric frame, then keep only detected points.
        coords = self._normalize_keypoints(keypoints=keypoints, mask=mask) * mask_3d  # [T,75,3]
        # (3) Add temporal first-order differences as motion features.
        dx = np.zeros_like(coords, dtype=np.float32)
        dx[1:] = coords[1:] - coords[:-1]
        dx = dx * mask_3d
        # (1) Keep mask as explicit feature channels.
        feat = np.concatenate([coords, dx, mask_3d], axis=2)  # [T,75,7]

        # 1D temporal CNN expects [C, T]. Flatten per frame:
        # [T,75,7] -> [T,525] -> [525,T]
        x = feat.reshape(feat.shape[0], -1)
        x = np.transpose(x, (1, 0))
        x_t = torch.from_numpy(x)
        y_t = torch.tensor(s.label_id, dtype=torch.long)
        return x_t, y_t


class PoseTemporalCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_samples(
    manifest_path: str,
    pose_npz_dir: str,
) -> Tuple[List[Sample], Dict[str, int]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    instances = manifest["instances"]
    glosses = sorted({inst["gloss"] for inst in instances})
    gloss_to_id = {g: i for i, g in enumerate(glosses)}

    samples: List[Sample] = []
    skipped_missing = 0
    for inst in instances:
        gloss = inst["gloss"]
        instance_id = int(inst["instance_id"])
        split = str(inst.get("split") or "train")
        npz_path = os.path.join(pose_npz_dir, f"{gloss}__{instance_id}.npz")
        if not os.path.exists(npz_path):
            skipped_missing += 1
            continue
        samples.append(
            Sample(
                npz_path=npz_path,
                label_id=gloss_to_id[gloss],
                split=split,
            )
        )
    print(f"[data] usable_samples={len(samples)} skipped_missing_npz={skipped_missing}")
    return samples, gloss_to_id


def split_samples(samples: List[Sample]) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    train, val, test = [], [], []
    for s in samples:
        sp = s.split.lower()
        if sp == "train":
            train.append(s)
        elif sp == "val":
            val.append(s)
        elif sp == "test":
            test.append(s)
        else:
            train.append(s)
    return train, val, test


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            total_correct += int((pred == y).sum().item())
            total += int(y.size(0))

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to topK manifest json")
    ap.add_argument(
        "--pose_npz_dir",
        required=True,
        help="Path to pose_outputs_norm/mediapipe_full_pose directory",
    )
    ap.add_argument("--out_dir", required=True, help="Directory for checkpoints and logs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    samples, gloss_to_id = build_samples(
        manifest_path=args.manifest,
        pose_npz_dir=args.pose_npz_dir,
    )
    train_s, val_s, test_s = split_samples(samples)
    print(f"[split] train={len(train_s)} val={len(val_s)} test={len(test_s)} num_classes={len(gloss_to_id)}")

    if len(train_s) == 0:
        raise RuntimeError("No training samples found. Check manifest split field and npz files.")

    train_ds = PoseNPZDataset(train_s)
    val_ds = PoseNPZDataset(val_s)
    test_ds = PoseNPZDataset(test_s)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    sample_x, _ = train_ds[0]
    in_channels = int(sample_x.shape[0])
    model = PoseTemporalCNN(
        in_channels=in_channels,
        num_classes=len(gloss_to_id),
        dropout=args.dropout,
    ).to(device)
    print(f"[model] temporal_cnn_1d in_channels={in_channels}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = -1.0
    best_path = os.path.join(args.out_dir, "best_model.pt")
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * y.size(0)
            pred = torch.argmax(logits, dim=1)
            train_correct += int((pred == y).sum().item())
            train_total += int(y.size(0))

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)
        val_loss, val_acc = evaluate(model, val_loader, device) if len(val_ds) > 0 else (0.0, 0.0)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        score_for_best = val_acc if len(val_ds) > 0 else train_acc
        if score_for_best > best_val_acc:
            best_val_acc = score_for_best
            ckpt = {
                "model_state_dict": model.state_dict(),
                "gloss_to_id": gloss_to_id,
                "args": vars(args),
                "best_score": best_val_acc,
            }
            torch.save(ckpt, best_path)
            print(f"[checkpoint] saved best model to {best_path}")

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device) if len(test_ds) > 0 else (0.0, 0.0)
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

    metrics_out = {
        "history": history,
        "best_score": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_size": len(train_s),
        "val_size": len(val_s),
        "test_size": len(test_s),
        "num_classes": len(gloss_to_id),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(gloss_to_id, f, indent=2)

    print(f"[done] outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
