import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["FAKE", "REAL"]
CLASS_TO_IDX = {"FAKE": 0, "REAL": 1}
IDX_TO_CLASS = {0: "FAKE", 1: "REAL"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def scan_and_clean_split(split_dir: str, verify_images: bool = False) -> tuple[list[tuple[str, int]], dict]:
    """Ingests image paths from REAL/ and FAKE/ folders and removes corrupt/empty files."""
    samples: list[tuple[str, int]] = []
    stats = {"total_scanned": 0, "removed_corrupt_or_empty": 0, "FAKE": 0, "REAL": 0}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_name, label in CLASS_TO_IDX.items():
        class_dir = Path(split_dir) / class_name
        if not class_dir.exists():
            continue
        for entry in sorted(class_dir.iterdir()):
            if not entry.is_file() or entry.suffix.lower() not in valid_exts:
                continue
            stats["total_scanned"] += 1
            if entry.stat().st_size == 0:
                stats["removed_corrupt_or_empty"] += 1
                continue
            if verify_images:
                try:
                    with Image.open(entry) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError):
                    stats["removed_corrupt_or_empty"] += 1
                    continue

            samples.append((str(entry), label))
            stats[class_name] += 1

    return samples, stats


class CIFAKEDataset(Dataset):
    """PyTorch Dataset for CIFAKE real vs. synthetic image classification."""

    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def get_transforms(img_size: int = 32):
    """Returns training (with geometric & photometric augmentation) and validation/inference transforms."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    data_root: str = "data/cifake",
    batch_size: int = 128,
    val_ratio: float = 0.15,
    max_train_samples: int | None = None,
    verify_images: bool = False,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Constructs cleaned, stratified Train, Validation, and Test DataLoaders."""
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    all_train_samples, train_stats = scan_and_clean_split(train_dir, verify_images=verify_images)
    test_samples, test_stats = scan_and_clean_split(test_dir, verify_images=verify_images)

    if len(all_train_samples) == 0 or len(test_samples) == 0:
        raise FileNotFoundError(
            f"CIFAKE images not found under '{data_root}'. Ensure train/REAL, train/FAKE, test/REAL, test/FAKE exist."
        )

    if max_train_samples is not None and max_train_samples < len(all_train_samples):
        labels_all = [lbl for _, lbl in all_train_samples]
        all_train_samples, _ = train_test_split(
            all_train_samples,
            train_size=max_train_samples,
            stratify=labels_all,
            random_state=seed,
        )

    labels = [lbl for _, lbl in all_train_samples]
    train_samples, val_samples = train_test_split(
        all_train_samples,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )

    train_tf, eval_tf = get_transforms(img_size=32)

    train_ds = CIFAKEDataset(train_samples, transform=train_tf)
    val_ds = CIFAKEDataset(val_samples, transform=eval_tf)
    test_ds = CIFAKEDataset(test_samples, transform=eval_tf)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)

    pipeline_summary = {
        "train_folder_stats": train_stats,
        "test_folder_stats": test_stats,
        "num_train_samples": len(train_ds),
        "num_val_samples": len(val_ds),
        "num_test_samples": len(test_ds),
    }

    return train_loader, val_loader, test_loader, pipeline_summary
