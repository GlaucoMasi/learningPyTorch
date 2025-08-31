import torch
from torch import nn

# All models should inherit from nn.Module and implement a forward()
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # If requires_grad=True, gradients from gradient descend are calculated automatically
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
