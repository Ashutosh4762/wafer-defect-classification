import os
import torch
import json
import logging
from torchvision import transforms
from PIL import Image
from datetime import datetime
from src.models.mobilenet_v3 import get_mobilenet_v3_small


# ---------------- CONFIG ----------------
MODEL_PATH = "models/mobilenet_best.pth"
LABEL_MAP_PATH = "models/label_map.json"
IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Logging Setup ----------------
os.makedirs("logs", exist_ok=True)

log_filename = f"logs/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

print(f"📝 Logging predictions to: {log_filename}")

# ---------------- Load Label Map ----------------
with open(LABEL_MAP_PATH, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)

# ---------------- Load Model ----------------
model = get_mobilenet_v3_small(
    num_classes=num_classes,
    in_channels=1,
    pretrained=False
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    predicted_class = idx_to_class[pred.item()]
    confidence_score = confidence.item()

    # Log result
    logging.info(
        f"{os.path.basename(image_path)} | "
        f"Pred: {predicted_class} | "
        f"Confidence: {confidence_score:.4f}"
    )

    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_score:.4f}")
    print("-" * 40)


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    test_folder = "data/splits/hack_test" 

    image_paths = []
    for root, _, files in os.walk(test_folder):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                image_paths.append(os.path.join(root, fn))

    print(f"✅ Found {len(image_paths)} images")

    if len(image_paths) == 0:
        print("❌ No images found. Check your folder path and extensions.")
    else:
        for img_path in image_paths:
            predict_image(img_path)

