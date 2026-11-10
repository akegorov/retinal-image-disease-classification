from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def get_backbone(name: str = "resnet18") -> tuple[nn.Module, int]:
    """ImageNet torchvision model without its classifier and the size of its features.

    Add new backbones here: drop the classifier, return (model, feature size).
    """
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    raise ValueError(f"Unknown backbone {name!r}; available: resnet18")
