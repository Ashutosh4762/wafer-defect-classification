import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class WaferDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int, color_mode: str, extensions):
        self.samples = []

        root = Path(root_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset path not found: {root}")

        self.color_mode = color_mode.lower()
        if self.color_mode not in ("grayscale", "rgb"):
            raise ValueError("color_mode must be 'grayscale' or 'rgb'")

        exts = tuple(e.lower() for e in extensions)

        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if not classes:
            raise ValueError(f"No class folders found in: {root}")

        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        total = 0
        for cls in classes:
            cls_dir = root / cls
            files = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
            total += len(files)
            for p in files:
                self.samples.append((str(p), self.class_to_idx[cls]))

        if total == 0:
            raise ValueError(
                f"No images found under: {root}\nExpected extensions: {exts}"
            )

        channels = 1 if self.color_mode == "grayscale" else 3
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*channels, std=[0.5]*channels),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)

        if self.color_mode == "grayscale":
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        return self.transform(img), label
