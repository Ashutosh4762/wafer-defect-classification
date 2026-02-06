import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def get_mobilenet_v3_small(num_classes: int, in_channels: int = 1, pretrained: bool = True):
    """
    MobileNetV3-Small adapted for grayscale (1ch) or RGB (3ch) and custom num_classes.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # Adapt first conv for grayscale if needed
    if in_channels == 1:
        old_conv = model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        # Initialize grayscale conv weights from pretrained RGB weights
        if pretrained and old_conv.weight.shape[1] == 3:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        model.features[0][0] = new_conv

    elif in_channels != 3:
        raise ValueError("in_channels must be 1 (grayscale) or 3 (rgb)")

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
