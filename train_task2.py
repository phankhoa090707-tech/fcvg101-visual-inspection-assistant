import argparse
import json
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import CLASS_NAMES, build_dataloaders
from src.model import CIFAKEResNet, predict_with_gradcam


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def save_training_plots(history: dict, cm: np.ndarray, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="s", label="Val Accuracy")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_training_curves.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        title="Test Set Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_confusion_matrix.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train CIFAKEResNet for FCVG101 Task 2")
    parser.add_argument("--data-root", type=str, default="data/cifake")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap for fast CPU runs")
    parser.add_argument("--save-path", type=str, default="models/cifake_cnn_best.pth")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    train_loader, val_loader, test_loader, summary = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
    )
    print("=== Task 2.1: Data Pipeline Summary ===")
    print(json.dumps(summary, indent=2))

    model = CIFAKEResNet(num_classes=2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CIFAKEResNet | Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    start_time = time.time()

    print("\n=== Task 2.2: Training CIFAKEResNet ===")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": epoch,
                },
                args.save_path,
            )

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")

    checkpoint = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    cm = confusion_matrix(y_true, y_pred)
    metrics_report = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_precision": round(float(precision_score(y_true, y_pred)), 4),
        "test_recall": round(float(recall_score(y_true, y_pred)), 4),
        "test_f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "confusion_matrix": cm.tolist(),
    }

    print("\n=== Task 2.3: Test Set Evaluation Metrics ===")
    print(json.dumps(metrics_report, indent=2))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    with open("outputs/task2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    save_training_plots(history, cm, out_dir="outputs")

    sample_img_path, _ = test_loader.dataset.samples[0]
    sample_bgr = cv2.imread(sample_img_path)
    cam_out = predict_with_gradcam(sample_bgr, model, device)
    cv2.imwrite("outputs/task2_sample_gradcam.png", cam_out["gradcam_overlay_bgr"])
    print("Saved plots and sample Grad-CAM visualization to outputs/.")


if __name__ == "__main__":
    main()
