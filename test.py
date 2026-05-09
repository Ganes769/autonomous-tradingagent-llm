#!/usr/bin/env python3
"""
DermaMNIST skin lesion classifier
- Auto-downloads dataset using medmnist
- Trains a compact CNN
- Saves metrics, confusion matrix, learning curves, predictions, best model
- Saves one main comparison graph
"""



from __future__ import annotations

import os
import gc
import json
import time
import random
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import medmnist
from medmnist import INFO


# ----------------------------
# Utility
# ----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def directory_size_bytes(path: str | Path) -> int:
    path = Path(path)
    total = 0
    if not path.exists():
        return 0
    for fp in path.rglob("*"):
        if fp.is_file():
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024 or unit == "TB":
            return f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} B"


def enforce_dir_budget(root: str | Path, budget_mb: int, protected_names=None):
    root = Path(root)
    if not root.exists():
        return
    protected_names = set(protected_names or [])
    budget_bytes = budget_mb * 1024 * 1024
    if directory_size_bytes(root) <= budget_bytes:
        return

    files = []
    for fp in root.rglob("*"):
        if fp.is_file() and fp.name not in protected_names:
            try:
                files.append((fp.stat().st_mtime, fp))
            except OSError:
                pass

    files.sort()
    for _, fp in files:
        if directory_size_bytes(root) <= budget_bytes:
            break
        try:
            fp.unlink()
        except OSError:
            pass


# ----------------------------
# Transforms
# ----------------------------

class TrainTransform:
    def __init__(self, size: int = 28):
        self.size = size

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB").resize((self.size, self.size))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.15:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        angle = random.uniform(-12, 12)
        img = img.rotate(angle)

        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.tensor(arr, dtype=torch.float32)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std


class EvalTransform:
    def __init__(self, size: int = 28):
        self.size = size

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.convert("RGB").resize((self.size, self.size))

        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.tensor(arr, dtype=torch.float32)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std


# ----------------------------
# Dataset wrapper
# ----------------------------

class MedMNISTWrapper(Dataset):
    def __init__(self, base_ds, transform=None):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        if isinstance(label, np.ndarray):
            label = int(label.squeeze())
        elif torch.is_tensor(label):
            label = int(label.squeeze().item())
        else:
            label = int(label)

        x = self.transform(img) if self.transform else img
        meta = {"index": idx}
        return x, label, meta


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), list(metas)


# ----------------------------
# Model
# ----------------------------

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, kernel_size=3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class MediumCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# ----------------------------
# Train / Eval
# ----------------------------

def run_train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0

    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0

    all_probs = []
    all_preds = []
    all_true = []
    all_meta = []

    for xb, yb, meta in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_true.append(yb.cpu())
        all_meta.extend(meta)

    y_prob = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_true).numpy()

    return total_loss / max(n, 1), y_true, y_pred, y_prob, all_meta


def compute_metrics(y_true, y_pred, y_prob, class_names):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    try:
        if len(class_names) == 2:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr")
            )
    except Exception:
        metrics["roc_auc_ovr"] = None

    return metrics


def save_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_learning_curves(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# MAIN COMPARISON GRAPH
# ----------------------------

def save_main_comparison_graph(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="s", label="Validation Loss")
    ax.plot(epochs, history["val_accuracy"], marker="^", label="Validation Accuracy")
    ax.plot(epochs, history["val_f1_macro"], marker="d", label="Validation F1-score")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Main Comparison Graph")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./medmnist_data")
    parser.add_argument("--output_dir", type=str, default="./dermamnist_results")
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--subset_frac", type=float, default=1.0)
    parser.add_argument("--disk_budget_mb", type=int, default=250)
    parser.add_argument("--cleanup_after_run", action="store_true")
    args, _unknown = parser.parse_known_args()

    seed_everything(args.seed)

    output_dir = safe_mkdir(args.output_dir)
    data_root = safe_mkdir(args.data_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_flag = "dermamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    class_names = [str(v) for _, v in sorted(info["label"].items(), key=lambda x: int(x[0]))]
    n_classes = len(class_names)

    print("Downloading/loading DermaMNIST...")
    train_base = DataClass(split="train", root=str(data_root), download=True, size=args.image_size)
    val_base = DataClass(split="val", root=str(data_root), download=True, size=args.image_size)
    test_base = DataClass(split="test", root=str(data_root), download=True, size=args.image_size)

    if args.subset_frac < 1.0:
        def subset_dataset(base_ds, frac):
            n = len(base_ds)
            keep = max(1, int(n * frac))
            indices = np.random.RandomState(args.seed).choice(n, keep, replace=False)
            base_ds.imgs = base_ds.imgs[indices]
            base_ds.labels = base_ds.labels[indices]
            return base_ds

        train_base = subset_dataset(train_base, args.subset_frac)
        val_base = subset_dataset(val_base, args.subset_frac)
        test_base = subset_dataset(test_base, args.subset_frac)

    train_ds = MedMNISTWrapper(train_base, transform=TrainTransform(args.image_size))
    val_ds = MedMNISTWrapper(val_base, transform=EvalTransform(args.image_size))
    test_ds = MedMNISTWrapper(test_base, transform=EvalTransform(args.image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )

    model = MediumCNN(num_classes=n_classes).to(device)

    train_labels = np.array(train_base.labels).squeeze()
    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = len(train_labels) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
    }

    best_val_f1 = -1.0
    best_path = output_dir / "best_model.pt"

    print("Start training...")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true, y_pred, y_prob, _ = eval_epoch(model, val_loader, criterion, device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "medmnist_version": medmnist.__version__,
                },
                best_path,
            )

        enforce_dir_budget(output_dir, args.disk_budget_mb, protected_names=["best_model.pt"])

    total_train_time = time.time() - t0

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, y_true, y_pred, y_prob, metas = eval_epoch(model, test_loader, criterion, device)
    test_metrics = compute_metrics(y_true, y_pred, y_prob, class_names)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    predictions_df = pd.DataFrame({
        "index": [m["index"] for m in metas],
        "y_true": y_true,
        "y_pred": y_pred,
    })
    for i, cname in enumerate(class_names):
        predictions_df[f"prob_{cname}"] = y_prob[:, i]

    predictions_path = output_dir / "test_predictions.csv"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.json"
    cm_path = output_dir / "confusion_matrix.png"
    curves_path = output_dir / "learning_curves.png"
    main_graph_path = output_dir / "main_comparison_graph.png"
    summary_path = output_dir / "run_summary.txt"

    predictions_df.to_csv(predictions_path, index=False)
    save_confusion_matrix(y_true, y_pred, class_names, cm_path)
    save_learning_curves(history, curves_path)
    save_main_comparison_graph(history, main_graph_path)

    final_payload = {
        "dataset": "DermaMNIST",
        "medmnist_version": medmnist.__version__,
        "device": str(device),
        "n_classes": n_classes,
        "class_names": class_names,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "subset_frac": args.subset_frac,
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "test_samples": int(len(test_ds)),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "best_val_f1_macro": float(best_val_f1),
        "total_train_time_sec": float(total_train_time),
        "output_dir_size": human_bytes(directory_size_bytes(output_dir)),
        "data_dir_size": human_bytes(directory_size_bytes(data_root)),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DermaMNIST training completed\n")
        f.write(f"Device: {device}\n")
        f.write(f"Train samples: {len(train_ds)}\n")
        f.write(f"Val samples: {len(val_ds)}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Best val F1-macro: {best_val_f1:.4f}\n")
        f.write(f"Test accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Test F1-macro: {test_metrics['f1_macro']:.4f}\n")
        f.write(f"Test ROC-AUC OVR: {test_metrics['roc_auc_ovr']}\n")
        f.write(f"Main graph saved to: {main_graph_path}\n")
        f.write(f"Output dir size: {human_bytes(directory_size_bytes(output_dir))}\n")
        f.write(f"Data dir size: {human_bytes(directory_size_bytes(data_root))}\n")

    print("\nTraining finished.")
    print(f"Main comparison graph saved at: {main_graph_path}")
    print(json.dumps(final_payload, indent=2))

    enforce_dir_budget(output_dir, args.disk_budget_mb, protected_names=["best_model.pt"])

    if args.cleanup_after_run:
        print("\nCleaning cached dataset files...")
        try:
            shutil.rmtree(data_root)
            print("Dataset cache removed.")
        except Exception as e:
            print(f"Could not remove data root: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()