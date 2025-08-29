"""
Trains and saves a PyTorch image classification model using device-agnostic code
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from torchvision import transforms
import data_setup, engine, model_builder, utils

parser = argparse.ArgumentParser(description="Trains and saves a PyTorch image classification model using device-agnostic code")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epoches for the training and testing")
parser.add_argument("--batch_size", type=int, default=32, help="Dimension of every batch")
parser.add_argument("--hidden_units", type=int, default=10, help="Hidden units for every model layer, except for input and output")
parser.add_argument("--lr", type=float, default=0.001, help="Starting learning rate for the training")
args = parser.parse_args()

venv_dir = Path(sys.prefix)
project_root = venv_dir.parent
data_dir = project_root/"data"
train_dir = data_dir/"pizza_steak_sushi/train"
test_dir = data_dir/"pizza_steak_sushi/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloder, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform, args.batch_size)

model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=args.hidden_units,
    output_shape=len(class_names)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)

engine.train(
    model,
    train_dataloader,
    test_dataloder,
    loss_fn,
    optimizer,
    scheduler,
    device,
    args.num_epochs
)

models_dir = project_root/"trained_models"
utils.save_model(model, models_dir, "tinyvgg_from_cmd.pth")
