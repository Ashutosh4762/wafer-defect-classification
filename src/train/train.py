# import os
# import json
# import yaml
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader

# from src.utils.config import load_config
# from src.dataset.wafer_dataset import WaferDataset
# from src.models.mobilenet_v3 import get_mobilenet_v3_small


# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# def train_one_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss, correct, total = 0.0, 0, 0
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         logits = model(x)
#         loss = criterion(logits, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * x.size(0)
#         preds = torch.argmax(logits, dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     return total_loss / max(total, 1), correct / max(total, 1)


# @torch.no_grad()
# def eval_one_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss, correct, total = 0.0, 0, 0
#     for x, y in loader:
#         x, y = x.to(device), y.to(device)
#         logits = model(x)
#         loss = criterion(logits, y)

#         total_loss += loss.item() * x.size(0)
#         preds = torch.argmax(logits, dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)

#     return total_loss / max(total, 1), correct / max(total, 1)


# def main(config_path="config/config.yaml"):
#     cfg = load_config(config_path)

#     # ---- Dataset config ----
#     img_size = int(cfg["dataset"]["image"]["size"])
#     extensions = cfg["dataset"]["image"]["extensions"]
#     splits_root = cfg["dataset"].get("splits_path", "data/splits")

#     # IMPORTANT: ALWAYS GRAYSCALE
#     color_mode = "grayscale"

#     # ---- Training config ----
#     train_cfg = cfg.get("training", {})
#     batch_size = int(train_cfg.get("batch_size", 32))
#     epochs = int(train_cfg.get("epochs", 20))
#     lr = float(train_cfg.get("learning_rate", 0.0003))
#     num_workers = int(train_cfg.get("num_workers", 0))
#     seed = int(train_cfg.get("seed", 42))


#     pretrained = bool(cfg["model"].get("pretrained", True))
#     model_name = cfg["model"].get("name", "mobilenet_v3_small")

#     set_seed(seed)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"✅ Device: {device}")

#     train_root = os.path.join(splits_root, "train")
#     val_root = os.path.join(splits_root, "val")

#     if not os.path.exists(train_root) or not os.path.exists(val_root):
#         raise FileNotFoundError(
#             f"Splits not found.\nExpected:\n  {train_root}\n  {val_root}\n"
#             "Run: python -m src.preprocessing.split_dataset"
#         )

#     train_ds = WaferDataset(train_root, img_size, color_mode, extensions)
#     val_ds = WaferDataset(val_root, img_size, color_mode, extensions)

#     num_classes = len(train_ds.class_to_idx)

#     print("✅ Classes:", train_ds.class_to_idx)
#     print("✅ Train samples:", len(train_ds))
#     print("✅ Val samples:", len(val_ds))

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     if model_name != "mobilenet_v3_small":
#         raise ValueError(f"Unsupported model: {model_name}")

#     # IMPORTANT: grayscale => in_channels=1
#     model = get_mobilenet_v3_small(num_classes=num_classes, in_channels=1, pretrained=pretrained).to(device)

#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     os.makedirs("models", exist_ok=True)

#     # Save label map + config snapshot
#     with open("models/label_map.json", "w", encoding="utf-8") as f:
#         json.dump(train_ds.class_to_idx, f, indent=2)

#     with open("models/train_config_snapshot.yaml", "w", encoding="utf-8") as f:
#         yaml.safe_dump(cfg, f, sort_keys=False)

#     best_val_acc = -1.0

#     for epoch in range(1, epochs + 1):
#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
#         val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

#         print(
#             f"Epoch {epoch}/{epochs} | "
#             f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
#             f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
#         )

#         torch.save(model.state_dict(), "models/mobilenet_latest.pth")

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "models/mobilenet_best.pth")

#     print("✅ Training complete.")
#     print("✅ Saved: models/mobilenet_latest.pth and models/mobilenet_best.pth")


# if __name__ == "__main__":
#     main()



# train.py
 
import os
import json
import yaml
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
 
from src.utils.config import load_config
from src.dataset.wafer_dataset import WaferDataset
from src.models.mobilenet_v3 import get_mobilenet_v3_small
 
 
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 
 
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
 
    return total_loss / max(total, 1), correct / max(total, 1)
 
 
@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
 
        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
 
    return total_loss / max(total, 1), correct / max(total, 1)
 
 
def main(config_path="config/config.yaml"):
    cfg = load_config(config_path)
 
    # ---- Dataset config ----
    img_size = int(cfg["dataset"]["image"]["size"])
    extensions = cfg["dataset"]["image"]["extensions"]
    splits_root = cfg["dataset"].get("splits_path", "data/splits")
 
    # IMPORTANT: ALWAYS GRAYSCALE
    color_mode = "grayscale"
 
    # ---- Training config ----
    train_cfg = cfg.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 20))
    lr = float(train_cfg.get("learning_rate", 0.0003))
    num_workers = int(train_cfg.get("num_workers", 0))
    seed = int(train_cfg.get("seed", 42))
 
    pretrained = bool(cfg["model"].get("pretrained", True))
    model_name = cfg["model"].get("name", "mobilenet_v3_small")
 
    set_seed(seed)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device: {device}")
 
    train_root = os.path.join(splits_root, "train")
    val_root = os.path.join(splits_root, "val")
 
    if not os.path.exists(train_root) or not os.path.exists(val_root):
        raise FileNotFoundError(
            f"Splits not found.\nExpected:\n  {train_root}\n  {val_root}\n"
            "Run: python -m src.preprocessing.split_dataset"
        )
 
    train_ds = WaferDataset(train_root, img_size, color_mode, extensions)
    val_ds = WaferDataset(val_root, img_size, color_mode, extensions)
 
    num_classes = len(train_ds.class_to_idx)
 
    print("✅ Classes:", train_ds.class_to_idx)
    print("✅ Train samples:", len(train_ds))
    print("✅ Val samples:", len(val_ds))
 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
    if model_name != "mobilenet_v3_small":
        raise ValueError(f"Unsupported model: {model_name}")
 
    # IMPORTANT: grayscale => in_channels=1
    model = get_mobilenet_v3_small(num_classes=num_classes, in_channels=1, pretrained=pretrained).to(device)
 
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    os.makedirs("models", exist_ok=True)
 
    # Save label map + config snapshot
    with open("models/label_map.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)
 
    with open("models/train_config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
 
    best_val_acc = -1.0
 
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
 
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
 
        torch.save(model.state_dict(), "models/mobilenet_latest.pth")
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/mobilenet_best.pth")
 
    print("✅ Training complete.")
    print("✅ Saved: models/mobilenet_latest.pth and models/mobilenet_best.pth")
 
 
if __name__ == "__main__":
    main()
 