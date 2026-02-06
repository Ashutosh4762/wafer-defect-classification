import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import io
import json
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F

from src.models.mobilenet_v3 import get_mobilenet_v3_small



# -----------------------
# Config (adjust if needed)
# -----------------------
MODEL_PATH = Path("models/mobilenet_best.pth")
LABEL_MAP_PATH = Path("models/label_map.json")
IMG_SIZE = 128
IN_CHANNELS = 1  # grayscale


# -----------------------
# Helpers
# -----------------------
def load_label_map(label_map_path: Path):
    with open(label_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_to_idx, idx_to_class, class_names


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH.resolve()}")

    _, _, class_names = load_label_map(LABEL_MAP_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_mobilenet_v3_small(
        num_classes=len(class_names),
        in_channels=IN_CHANNELS,
        pretrained=False
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device, class_names


def preprocess_pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image -> grayscale -> resize -> tensor [1,1,128,128] float32 in [0,1]
    """
    img = img.convert("L")  # grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0  # [H,W]
    arr = arr[None, None, :, :]  # [1,1,H,W]
    return torch.from_numpy(arr)


@torch.no_grad()
def predict_image(model, device, class_names, img: Image.Image, top_k=3):
    x = preprocess_pil_to_tensor(img).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)  # [C]
    conf, pred_idx = torch.max(probs, dim=0)

    # Top-k
    topk = torch.topk(probs, k=min(top_k, probs.shape[0]))
    topk_items = [(class_names[i], float(topk.values[j])) for j, i in enumerate(topk.indices.tolist())]

    return class_names[int(pred_idx)], float(conf), topk_items


def extract_images_from_zip(zip_bytes: bytes):
    """
    Returns list of (name, PIL.Image)
    """
    images = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            suffix = Path(name).suffix.lower()
            if suffix not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
                continue
            with z.open(info) as f:
                data = f.read()
                try:
                    img = Image.open(io.BytesIO(data))
                    img.load()
                    images.append((Path(name).name, img))
                except Exception:
                    pass
    return images


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Wafer Defect Classifier", layout="wide")
st.title("🧪 Wafer Defect Classifier (MobileNetV3-Small)")
st.caption("Upload images (single/multiple) or a ZIP folder. The app shows prediction + confidence for each image.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Show Top-K classes", min_value=1, max_value=8, value=3)
    st.write("Model:", str(MODEL_PATH))
    st.write("Labels:", str(LABEL_MAP_PATH))

# Load model
try:
    model, device, class_names = load_model()
    st.success(f"✅ Model loaded on: {device} | Classes: {len(class_names)}")
except Exception as e:
    st.error(str(e))
    st.stop()

tabs = st.tabs(["📷 Upload Image(s)", "🗂️ Upload ZIP Folder"])

# --- Tab 1: upload image(s)
with tabs[0]:
    files = st.file_uploader(
        "Upload one or multiple images",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True
    )

    if files:
        st.subheader(f"Results ({len(files)} images)")
        cols_per_row = 3
        rows = (len(files) + cols_per_row - 1) // cols_per_row

        idx = 0
        for _ in range(rows):
            cols = st.columns(cols_per_row)
            for c in cols:
                if idx >= len(files):
                    break
                f = files[idx]
                idx += 1

                try:
                    img = Image.open(f)
                    pred, conf, topk = predict_image(model, device, class_names, img, top_k=top_k)
                except Exception as ex:
                    c.error(f"Failed: {f.name}\n{ex}")
                    continue

                c.image(img, caption=f.name, use_container_width=True)
                c.markdown(f"**Prediction:** `{pred}`")
                c.markdown(f"**Confidence:** `{conf:.3f}`")

                with c.expander("Top-K"):
                    for name, p in topk:
                        c.write(f"- {name}: {p:.3f}")

# --- Tab 2: upload zip
with tabs[1]:
    zip_file = st.file_uploader(
        "Upload a ZIP containing images (works like selecting a folder)",
        type=["zip"],
        accept_multiple_files=False
    )

    if zip_file:
        imgs = extract_images_from_zip(zip_file.read())
        if not imgs:
            st.warning("No readable images found in this ZIP.")
        else:
            st.subheader(f"Results ({len(imgs)} images)")
            cols_per_row = 3
            rows = (len(imgs) + cols_per_row - 1) // cols_per_row

            idx = 0
            for _ in range(rows):
                cols = st.columns(cols_per_row)
                for c in cols:
                    if idx >= len(imgs):
                        break
                    name, img = imgs[idx]
                    idx += 1

                    try:
                        pred, conf, topk = predict_image(model, device, class_names, img, top_k=top_k)
                    except Exception as ex:
                        c.error(f"Failed: {name}\n{ex}")
                        continue

                    c.image(img, caption=name, use_container_width=True)
                    c.markdown(f"**Prediction:** `{pred}`")
                    c.markdown(f"**Confidence:** `{conf:.3f}`")

                    with c.expander("Top-K"):
                        for cls_name, p in topk:
                            c.write(f"- {cls_name}: {p:.3f}")
