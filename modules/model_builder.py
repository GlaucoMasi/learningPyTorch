"""
Module to instantiate a TinyVGG PyTorch model
"""
import torch
import torchvision
from torch import nn
from torchinfo import summary

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture

    Replicates, with some differences, the TinyVGG architecture from the CNN explainer website in PyTorch.
    Original architecture at: https://poloclub.github.io/cnn-explainer/.

    Args:
        input_shape: The number of input channels
        hidden_units: The number of hidden units between layers
        output_shape: The number of output units, ofter the number of classes
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            # Using adaptive average to avoid hardcoding input image dimensions in the last linear layer
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(hidden_units*7*7, output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

def create_effnetb0(out_features: int, device: torch.device, print_summary: bool=False):
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=out_features, bias=True)
    ).to(device)

    model.name = "effnetb0"
    if(print_summary):
        summary(model)
    return model

def create_effnetb2(out_features: int, device: torch.device, print_summary: bool=False):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=out_features, bias=True)
    ).to(device)

    model.name = "effnetb2"
    if(print_summary):
        summary(model)
    return model