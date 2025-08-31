import torch
from torch import nn
from torchvision import transforms

from modules import data_setup, engine, utils, model_builder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_10_percent = utils.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", destination="pizza_steak_sushi")
data_20_percent = utils.download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip", destination="pizza_steak_sushi_20_percent")

# Use the same testing data for comparison
test_dir = data_10_percent/"test"
train_dir_10_percent = data_10_percent/"train"
train_dir_20_percent = data_20_percent/"train"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BATCH_SIZE = 32

train_dataloader_10_percent, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir_10_percent,
    test_dir,
    transform,
    batch_size=BATCH_SIZE,
)

train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir_20_percent,
    test_dir,
    transform,
    batch_size=BATCH_SIZE,
)

# Interesting: b2 as almost nearly the amount of parameters as b0, but basically the same number of trainable parameters
# effnetb0 = model_builder.create_effnetb0(out_features=len(class_names), device=device, print_summary=True)
# effnetb2 = model_builder.create_effnetb2(out_features=len(class_names), device=device, print_summary=True)

num_epochs = [5, 10]
models = {"effnetb0": model_builder.create_effnetb0,
          "effnetb2": model_builder.create_effnetb2}
train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
                    "data_20_percent": train_dataloader_20_percent}

experiment_number = 0
for dataloader_name, train_dataloader in train_dataloaders.items():
    for epochs in num_epochs:
        for model_name, model_function in models.items():
            experiment_number += 1

            model = model_function(out_features=len(class_names), device=device, print_summary=False)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

            engine.train(
                model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epochs=epochs,
                writer=utils.create_writer(
                    experiment_name=dataloader_name,
                    model_name=model_name,
                    extra=f"{epochs}_epochs"
                )
            )

            utils.save_model(model=model, model_name=f"07_{model_name}_{dataloader_name}_{epochs}_epochs.pth")

# In conclusion, the best model was the b2, with 10 epochs and 20% data, as espected. Bigger model, more data, more training
# But watching the results can still give insights on what factors are the most important