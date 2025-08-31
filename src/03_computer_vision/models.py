import torch
from torch import nn

# First baseline model with Flatten layer, that compresses the dimensions of a tensor into a single vector
# Model achieves roughly 80% accuracy, regardless of hidden units
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)


# Introducing non-linear layers
# Shows signs of overfitting: train accuracy is increasing while test accuracy is decreasing
# That means that the model is learning patterns in the training data that aren't generalizing to the test data
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)
    

# Convolutional Neural Network (CNN)
# Really capable at finding patterns in visual data, but is also used in other applications
# "Tiny VGG contains many of the same layers and operations used in state-of-the-art CNNs today, but on a smaller scale" from CNN Explainer
# It follows the typical structure of CNNs:
# Input layer -> [Convolutional layer -> Activation layer -> Pooling layer] -> Output layer
# 1) Convolutional layers: they go over the image with kernels/filters, that could be 3x3 or 5x5 pixels big, instead of having a neuron per single pixel.
# The first looks for lines, the second for shapes, and so on, creating features maps
# IMPORTANT: Conv layers are designed to process batches of data by default. The same is true for basically any layer, even if some work anyway
# 2) Activation layers: they add non-linearity to the model. ReLU is an example of such layer
# 3) Pooling layers: they reduce the spatial dimensions of features maps while retaining import informations.
# Max pooling, for example, takes the maximum value from small regions, making the network more robust to small traslations and reducing computational load
# The performance-speed tradeoff: model is more accurate but takes a lot more to train
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3, # The size of of the convolutional filter
                stride=1, # How far the filter moves with each step, 1 is the default value
                padding=1 # How pixels are added to each side of the final map, default is 0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2 # Default value is the same as kernel_size, to actually reduce the shape of the map
            )
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # Now the tensor is of shape [hidden_units*(lenght/4)*(width/4)]
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        # x.shape: [batches, 1, 28, 28]
        x = self.block_1(x)
        # x.shape: [batches, 10, 14, 14] (Pooling layers trasforms 2x2s into 1x1s)
        x = self.block_2(x)
        # x.shape: [batches, 10, 7, 7] (Same as above)
        x = self.classifier(x)
        # x.shape = [batches, 10]
        return x