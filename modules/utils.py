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
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Dict, List, Tuple
from safetensors.torch import save_file
from timeit import default_timer as timer
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
    torch.save(obj=model.state_dict(), f=model_save_path, _use_new_zipfile_serialization=True)

    model_size = model_save_path.stat().st_size // (1024*1024)
    return model_size

def save_model_with_savetensors(
    model: torch.nn.Module,
    model_name: str 
):
    venv_dir = Path(sys.prefix)
    project_root = venv_dir.parent
    models_dir = project_root/"trained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".safetensors"), "model_name doesn't follow naming convention"
    model_save_path = models_dir / model_name

    print(f"Saving model to: {model_save_path}")
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(sd, model_save_path)

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

def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def pred_and_store(
    model: torch.nn.Module,
    paths: List[Path],
    transforms: torchvision.transforms,
    class_names: List[str],
    device = torch.device
) -> List[Dict]:
    pred_list = []

    for path in paths:
        pred_dict = {}

        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        start_time = timer()
        
        img = Image.open(path)
        transformed_image = transforms(img).unsqueeze(dim=0).to(device)

        model.to(device)
        model.eval()

        with torch.inference_mode():
            pred_logit = model(transformed_image)
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            pred_dict["pred_prob"] = pred_prob
            pred_dict["pred_class"] = pred_class

            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        pred_dict["correct"] = class_name == pred_class
        pred_list.append(pred_dict)

    return pred_list