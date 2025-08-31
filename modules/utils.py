"""
Utility module that helps with training, saving and loading a PyTorch model
"""
import os
import sys
import torch
import random
import zipfile
import requests
import torchvision
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter 

def save_model(
    model: torch.nn.Module,
    model_name: str
):
    """Saves a PyTorch model to a target directory

    Args:
        model: PyTorch model to save
        model_name: Name under which the model will be saved, should end in ".pth" of ".pt" as for naming conventions
    """
    venv_dir = Path(sys.prefix)
    project_root = venv_dir.parent
    models_dir = project_root/"trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name doesn't follow naming convention"
    model_save_path = models_dir / model_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def download_data(
    source: str,
    destination: Path,
    remove_source: bool=True
) -> Path:
    """Downloads a zipped dataset from source and unzips to destination

    Args:
        source: A link to a zip archive containing data
        destination: The directory the datasets will be unzipped to
        remove_source: Wheter to remove the source after extracting

    Returns:
        pathlib.Path to the downloaded data directory
    """
    venv_dir = Path(sys.prefix)
    project_root = venv_dir.parent
    data_path = project_root/"data"
    image_path = data_path/destination

    if(image_path.is_dir() is False):
        image_path.mkdir(parents=True, exist_ok=True)

        target_file = Path(source).name
        with open(data_path/target_file, "wb") as f:
            request = requests.get(source)
            f.write(request.content)

        with zipfile.ZipFile(data_path/target_file, "r") as zip_ref:
            zip_ref.extractall(image_path)

        if remove_source:
            os.remove(data_path/target_file)

    return image_path

def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str=None
) -> SummaryWriter:
    """Creates a SummaryWriter instance that saves to a specific directory

    The logs directory is a combination of runs/timestamp/experiment_name/extra
    Where timestamp is the current date in YYYY-MM-DD format

    Args:
    - experiment_name: Name of the experiment
    - model_name: Name of the model
    - extra: Extra informations, defaults to None
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    if extra:
        log_dir = os.path.join(log_dir, extra)
    return SummaryWriter(log_dir=log_dir)

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: Path,
    class_names: List[str],
    device: torch.device,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None
):
    img = Image.open(image_path)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = transform(img).unsqueeze(dim=0).to(device)
        target_pred = model(transformed_image)
    
    target_pred_probs = torch.softmax(target_pred, dim=1)
    target_pred_labels = torch.argmax(target_pred_probs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Model predicts '{class_names[target_pred_labels]}' with {target_pred_probs.max()*100:.1f}% probability")
    plt.axis(False)

def pred_and_plot_images(
    model: torch.nn.Module,
    test_dir: Path,
    class_names: List[str],
    device: torch.device,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
    num_images: int=3
):
    image_path_list = list(Path(test_dir).glob("*/*.jpg"))
    image_random_sample = random.sample(population=image_path_list, k=num_images)

    for image_path in image_random_sample:
        pred_and_plot_image(
            model=model,
            image_path=image_path,
            class_names=class_names,
            device=device,
            image_size=image_size,
            transform=transform
        )