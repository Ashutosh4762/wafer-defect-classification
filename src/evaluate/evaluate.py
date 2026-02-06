import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.config import load_config
from src.dataset.wafer_dataset import WaferDataset
from src.models.mobilenet_v3 import get_mobilenet_v3_small


@torch.no_grad()
def main(config_path="config/config.yaml"):
    cfg = load_config(config_path)

    # ---- Dataset config ----
    img_size = int(cfg["dataset"]["image"]["size"])
    extensions = cfg["dataset"]["image"]["extensions"]
    splits_root = cfg["dataset"].get("splits_path", "data/splits")

    # 🔒 ALWAYS GRAYSCALE
    color_mode = "grayscale"
    in_channels = 1

    batch_size = int(cfg.get("training", {}).get("batch_size", 32))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")

    test_root = os.path.join(splits_root, "test")
    if not os.path.exists(test_root):
        raise FileNotFoundError(
            f"Test split not found: {test_root}\n"
            "Run: python -m src.preprocessing.split_dataset"
        )

    # Load label map from training
    label_map_path = "models/label_map.json"
    if not os.path.exists(label_map_path):
        raise FileNotFoundError("models/label_map.json not found. Train the model first.")

    with open(label_map_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Dataset & loader
    test_ds = WaferDataset(test_root, img_size, color_mode, extensions)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = get_mobilenet_v3_small(
        num_classes=len(class_names),
        in_channels=in_channels,
        pretrained=False
    ).to(device)

    weights_path = "models/mobilenet_best.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError("models/mobilenet_best.pth not found.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()
    print(f"\n✅ Test Accuracy: {acc:.4f}\n")

    print("📌 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 8))
    #plt.imshow(cm)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    out_path = "reports/confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    print(f"✅ Saved confusion matrix to: {out_path}")


if __name__ == "__main__":
    main()
