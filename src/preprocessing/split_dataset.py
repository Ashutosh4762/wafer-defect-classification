import shutil
import random
from pathlib import Path

from src.utils.config import load_config


def _clear_dir(folder: Path):
    if folder.exists():
        for p in folder.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)


def split_dataset(config_path="config/config.yaml"):
    cfg = load_config(config_path)

    # ✅ ALWAYS split from PROCESSED (your requirement)
    source_root = Path(cfg.get("dataset", {}).get("processed_path", "data/processed"))
    splits_root = Path(cfg.get("dataset", {}).get("splits_path", "data/splits"))

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.7))
    val_ratio = float(split_cfg.get("val", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))
    seed = int(split_cfg.get("seed", 42))

    img_cfg = cfg.get("dataset", {}).get("image", {})
    exts = tuple(e.lower() for e in img_cfg.get("extensions", [".jpg", ".jpeg", ".png"]))

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    if not source_root.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {source_root.resolve()}\n"
            "Run: python -m src.preprocessing.preprocess"
        )

    random.seed(seed)

    # ✅ Clear old splits completely to avoid stale counts
    _clear_dir(splits_root)
    splits_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (splits_root / split).mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in source_root.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in: {source_root.resolve()}")

    print(f"✅ Splitting from PROCESSED: {source_root.resolve()}")
    print(f"✅ Output splits to: {splits_root.resolve()}")
    print(f"✅ Ratios: train={train_ratio}, val={val_ratio}, test={test_ratio} | seed={seed}")

    for cls_dir in classes:
        cls_name = cls_dir.name
        files = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        split_map = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split, split_files in split_map.items():
            out_dir = splits_root / split / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                shutil.copy2(str(f), str(out_dir / f.name))

        print(f"✅ {cls_name}: total={n_total} | train={len(split_map['train'])} | val={len(split_map['val'])} | test={len(split_map['test'])}")

    print("\n✅ Dataset split complete.")


if __name__ == "__main__":
    split_dataset()
