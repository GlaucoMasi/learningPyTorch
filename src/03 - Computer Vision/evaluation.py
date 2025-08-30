import train
import models

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path


BATCH_SIZE = 32

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle = False)

class_names = test_data.classes



# Load model
modelV2 = models.FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=10)
SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = SCRIPT_DIR/"trained_models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / modelV2.__class__.__name__
modelV2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loss_fnV2 = nn.CrossEntropyLoss()
def accuracy_fnV2(y_pred, y_true):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true))*100
    return acc
print(train.eval_model(
    model=modelV2, 
    dataloader=test_dataloader, 
    loss_fn=loss_fnV2,
    accuracy_fn=accuracy_fnV2,
))



# Let's create a confusion model of the best performing model, the V2

# 1) Make predictions
y_preds = []
modelV2.eval()
with torch.inference_mode():
    for X, y in tqdm(test_dataloader, desc="Making predictions"):
        y_logits = modelV2(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        y_preds.append(y_pred)

    y_pred_tensor = torch.cat(y_preds)

# 2) Make a confusion matrix
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

# 3) Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=class_names,
    figsize=(10, 7)
)


# We get some cool insights:
# - the model confuses shirts with t-shirts, tops, pullovers and coats
# - the model confuses ankle boots, sneakers and sandals
# - the model confuses coats for pullovers, but not the opposite