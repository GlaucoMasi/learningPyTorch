import sys
from pathlib import Path

import torch
from torch import nn

from torchvision import datasets
from torchvision.transforms import ToTensor

from tqdm.auto import tqdm

from timeit import default_timer as timer


import models
from train import eval_model
from train import perform_step

venv_dir = Path(sys.prefix)
project_root = venv_dir.parent
data_dir = project_root/"data"
# FashionMNIST is a database containing grayscale images of 10 different kinds of clothing
# PyTorch has a bunch of common computer vision datasets, including torchvision.datasets.FashionMNIST()

train_data = datasets.FashionMNIST(
    root=data_dir,                # where to download data to
    train=True,                 # get training data
    download=True,              # download if it doesn't exist on disk
    transform=ToTensor(),       # trasformation applied to images, PIL format -> Torch tensor
    target_transform=None       # no trasformation on labels
)

test_data = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=ToTensor()
)

# There are 60000 training samples and 20000 test samples
# The shape of each image tensor is [1, 28, 28] => 28 height, 28 weight and 1 color channel => grayscale
# This order is reffered to as CHW, HWC is also a valid option
# In case of batches, N stands for number of images in NCHW. PyTorch accepts NCHW bu CHWN performs better and is considered best practice

# 10 classes => multi-class classification
class_names = train_data.classes

# Plotting some images
# fig = plt.figure(figsize=(9, 9))
# rows, cols = 4, 4
# for i in range(1, rows*cols+1):
#     idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title(class_names[label])
#     plt.axis(False)

# Data from databases is fed to models with DataLoaders, that split it into chunks
# The chunks are called batches, they are more computationally efficient and you get to make more descends per epoch
from torch.utils.data import DataLoader

BATCH_SIZE = 32
EPOCHS = 8 

train_dataloader = DataLoader(train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)       # Shuffles data after every epoch

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle = False)     # Not needed in test dataloader




# Every image is 28x28 = 784 pixels/features
modelV0 = models.FashionMNISTModelV0(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fnV0 = nn.CrossEntropyLoss()
optimizerV0 = torch.optim.SGD(params=modelV0.parameters(), lr=0.1)
def accuracy_fnV0(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true))*100
    return acc

startV0 = timer()
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}")
    perform_step(
        model=modelV0,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fnV0,
        optimizer=optimizerV0,
        accuracy_fn=accuracy_fnV0
    )
endV0 = timer()



# Stardand practice: keeping all settings from last model except for one change
modelV1 = models.FashionMNISTModelV1(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fnV1 = nn.CrossEntropyLoss()
optimizerV1 = torch.optim.SGD(params=modelV1.parameters(), lr=0.1)
def accuracy_fnV1(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true))*100
    return acc

startV1 = timer()
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}")
    perform_step(
        model=modelV1,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fnV1,
        optimizer=optimizerV1,
        accuracy_fn=accuracy_fnV1
    )
endV1 = timer()



# CNNs think in images, not pixels! Therefore the input_shape is about color channels, 1 if they images are grayscale
modelV2 = models.FashionMNISTModelV2(
    input_shape=1,
    hidden_units=10,
    output_shape=len(class_names)
)

loss_fnV2 = nn.CrossEntropyLoss()
optimizerV2 = torch.optim.SGD(params=modelV2.parameters(), lr=0.1)
def accuracy_fnV2(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true))*100
    return acc

startV2 = timer()
for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}")
    perform_step(
        model=modelV2,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fnV2,
        optimizer=optimizerV2,
        accuracy_fn=accuracy_fnV2
    )
endV2 = timer()

print(eval_model(modelV0, test_dataloader, loss_fnV0, accuracy_fnV0, startV0, endV0))
print(eval_model(modelV1, test_dataloader, loss_fnV1, accuracy_fnV1, startV1, endV1))
print(eval_model(modelV2, test_dataloader, loss_fnV2, accuracy_fnV2, startV2, endV2))


models_dir = project_root/"trained_models"
model_name = modelV2.__class__.__name__+".pth"
model_save_path = models_dir / model_name
torch.save(obj=modelV2.state_dict(), f=model_save_path)