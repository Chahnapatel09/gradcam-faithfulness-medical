import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """ResNet-18 fine-tuned for binary classification.

    Expects 3-channel input (PneumoniaMNIST's grayscale images must be
    replicated to 3 channels upstream, in data.py, to keep the pretrained
    ImageNet weights valid).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
