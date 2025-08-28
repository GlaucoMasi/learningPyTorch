import torch
from torch import nn

# Model is underfitting, fails because it tries to split dots using a straight line
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()

        # https://playground.tensorflow.org/
        # Two hidden layers: 5 neurons, 1 neuron
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(x))
    
# Model is underfitting, but using it to fit a straight line proves that it has some learning capabilities
# Even by stacking more linear layers: y2 = m1 * m2 * x + m2 * c1 + c2, which is linear!
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()

        # Three hidden layers: 10 neurons, 10 neurons, 1 neuron
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.layer_2(self.layer_1(x)))
    
# The Rectified Linear-Unit is a non-linear activation function, it turns negatives into 0 and leaves positives as they are
# Also adding the Sigmoid makes (nn.Sigmoid()) it so it isn't needed after, but it's not industry standard
# p.s. The Sigmoid is also a non-linear activation function
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))