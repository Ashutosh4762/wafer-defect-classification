import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config import load_config


# ---------- helpers ----------
def ensure_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    # Always grayscale
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def aug_flip(img: np.ndarray, mode: str) -> np.ndarray:
    # mode: "h", "v", "hv"
    if mode == "h":
        return cv2.flip(img, 1)
    if mode == "v":
        return cv2.flip(img, 0)
    if mode == "hv":
        return cv2.flip(img, -1)
    return img


def aug_rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def aug_brightness(img: np.ndarray, delta: int) -> np.ndarray:
    return ensure_uint8(img.astype(np.int16) + delta)


def aug_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    return ensure_uint8(img.astype(np.float32) * alpha)


def write_jpg(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ---------- main ----------
def preprocess(config_path="config/config.yaml"):
    cfg = load_config(config_path)

    raw_root = Path(cfg["dataset"]["raw_path"])
    out_root = Path(cfg["dataset"]["processed_path"])

    img_size = int(cfg["dataset"]["image"]["size"])
    exts = tuple(e.lower() for e in cfg["dataset"]["image"]["extensions"])

    pre_cfg = cfg.get("preprocessing", {})
    overwrite = bool(pre_cfg.get("overwrite", False))
    target = int(pre_cfg.get("target_per_class", 70))
    seed = int(pre_cfg.get("seed", 42))

    aug_cfg = pre_cfg.get("augment", {})
    flip_h = bool(aug_cfg.get("flip_horizontal", True))
    flip_v = bool(aug_cfg.get("flip_vertical", True))
    rotate_degs = aug_cfg.get("rotate_degrees", [-10, -5, 5, 10])
    bright_deltas = aug_cfg.get("brightness_delta", [-30, -15, 15, 30])
    contrast_alphas = aug_cfg.get("contrast_alpha", [0.8, 1.2])

    random.seed(seed)
    np.random.seed(seed)

    if not raw_root.exists():
        raise FileNotFoundError(f"RAW path not found: {raw_root.resolve()}")

    out_root.mkdir(parents=True, exist_ok=True)

    classes = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in: {raw_root.resolve()}")

    print(f"✅ RAW:       {raw_root.resolve()}")
    print(f"✅ PROCESSED: {out_root.resolve()}")
    print(f"✅ size={img_size} | grayscale enforced | target_per_class={target} | overwrite={overwrite}")

    for cls_dir in classes:
        cls_name = cls_dir.name
        out_cls = out_root / cls_name

        raw_files = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not raw_files:
            print(f"⚠️ {cls_name}: no raw images found. Skipping.")
            continue

        # rebuild if overwrite
        if overwrite and out_cls.exists():
            for p in out_cls.glob("*.*"):
                if p.is_file():
                    p.unlink()

        out_cls.mkdir(parents=True, exist_ok=True)

        # If already has enough processed images and overwrite is off, skip
        existing = sorted(list(out_cls.glob("*.jpg")))
        if (not overwrite) and len(existing) >= target:
            print(f"✅ {cls_name}: already processed ({len(existing)} images). Skipping.")
            continue

        # shuffle raw files reproducibly
        raw_files = sorted(raw_files)
        random.shuffle(raw_files)

        # If more than target, select target originals (no augmentation needed)
        selected_raw = raw_files[:min(len(raw_files), target)]

        # 1) Save processed originals
        print(f"\n📂 Class: {cls_name} | raw={len(raw_files)} | selected_originals={len(selected_raw)}")
        saved_paths = []

        for i, p in enumerate(tqdm(selected_raw, desc=f"Process {cls_name}")):
            img = cv2.imread(str(p))
            img = to_grayscale(img)
            if img is None:
                continue
            img = resize(img, img_size)
            img = ensure_uint8(img)

            out_path = out_cls / f"orig_{i:04d}.jpg"
            write_jpg(out_path, img)
            saved_paths.append(out_path)

        # 2) If less than target, augment until exactly target
        if len(saved_paths) < target:
            need = target - len(saved_paths)
            print(f"🔧 Augmenting {cls_name}: need {need} more images")

            # load base images from the processed originals
            base_imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in saved_paths]
            base_imgs = [b for b in base_imgs if b is not None]

            if not base_imgs:
                print(f"⚠️ {cls_name}: no valid base images for augmentation. Skipping.")
                continue

            flip_modes = ["none"]
            if flip_h:
                flip_modes.append("h")
            if flip_v:
                flip_modes.append("v")
            if flip_h and flip_v:
                flip_modes.append("hv")

            aug_index = 0
            for _ in tqdm(range(need), desc=f"Augment {cls_name}"):
                img = random.choice(base_imgs).copy()

                # random flip
                fm = random.choice(flip_modes)
                if fm != "none":
                    img = aug_flip(img, fm)

                # random rotation
                deg = float(random.choice(rotate_degs))
                img = aug_rotate(img, deg)

                # random brightness
                delta = int(random.choice(bright_deltas))
                img = aug_brightness(img, delta)

                # random contrast
                alpha = float(random.choice(contrast_alphas))
                img = aug_contrast(img, alpha)

                # ensure final properties
                img = resize(img, img_size)
                img = ensure_uint8(img)

                out_path = out_cls / f"aug_{aug_index:04d}.jpg"
                aug_index += 1
                write_jpg(out_path, img)

        # 3) Enforce EXACT count = target
        all_imgs = sorted(list(out_cls.glob("*.jpg")))
        if len(all_imgs) > target:
            # deterministically delete extras
            for extra in all_imgs[target:]:
                extra.unlink()

        final_count = len(list(out_cls.glob("*.jpg")))
        print(f"✅ {cls_name}: final processed images = {final_count}")

    print("\n✅ Preprocessing complete. Processed dataset is ready for training.")


if __name__ == "__main__":
    preprocess()
