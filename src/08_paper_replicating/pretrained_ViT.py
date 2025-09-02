import torch
import torchvision
from torch import nn
from torchinfo import summary

from modules import utils, data_setup, engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = utils.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")
test_dir = data_path / "test"
train_dir = data_path / "train"

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=pretrained_vit_weights.transforms(),
    batch_size=32
)

pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
summary(
    model=pretrained_vit,
    input_size=(32, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    params=pretrained_vit.parameters(),
    lr=1e-3
)

results = engine.train(
    model=pretrained_vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=device,
    epochs=10
)

utils.save_model(pretrained_vit, "Pretrained_ViT-B16.pth")

# Some insights: we are getting roughly the same accuracy as the EffNetB2, which is 11 times smaller. But it was trained on double the amount of data. Worth it?