import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models.mobilenet_v3 import get_mobilenet_v3_small


class Normalize01(nn.Module):
    """
    Quantization-safe normalization inside the graph.
    Assumes input uint8 or float in [0,255] or [0,1].
    We'll feed float32 in [0,1] from preprocessing usually.
    Keeping this as identity-like helps eIQ pipelines.
    """
    def forward(self, x):
        return x


def main():
    # -------- Paths --------
    weights_path = Path("models/mobilenet_best.pth")
    label_map_path = Path("models/label_map.json")
    out_path = Path("models/mobilenet_best.onnx")

    assert weights_path.exists(), f"Missing: {weights_path}"
    assert label_map_path.exists(), f"Missing: {label_map_path}"

    # -------- Load label map to get num classes --------
    class_to_idx = json.loads(label_map_path.read_text(encoding="utf-8"))
    num_classes = len(class_to_idx)

    # -------- Build model (same as training) --------
    model = get_mobilenet_v3_small(
        num_classes=num_classes,
        in_channels=1,      # grayscale
        pretrained=False    # IMPORTANT: False when loading trained weights
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Optional wrapper (keeps graph stable)
    wrapped = nn.Sequential(Normalize01(), model).eval()

    # -------- Dummy input (match your training size) --------
    # NCHW: batch, channels, height, width
    dummy = torch.randn(1, 1, 128, 128, dtype=torch.float32)

    # -------- Export settings --------
    input_names = ["input"]
    output_names = ["logits"]

    # Dynamic batch is usually safe; keep H/W fixed for embedded
    dynamic_axes = {
        "input": {0: "batch"},
        "logits": {0: "batch"}
    }

    torch.onnx.export(
        wrapped,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )



    print("✅ Exported ONNX:", out_path.resolve())
    print("ℹ️ Input shape:  [batch, 1, 128, 128]")
    print("ℹ️ Output logits: [batch, num_classes]")


if __name__ == "__main__":
    main()
