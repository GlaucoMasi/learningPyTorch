"""
Utility module that helps with training, saving and loading a PyTorch model
"""
import torch
from pathlib import Path

def save_model(
    model: torch.nn.Module,
    target_dir_path: Path,
    model_name: str
):
    """Saves a PyTorch model to a target directory

    Args:
        model: PyTorch model to save
        target_dir: Directory where the model will be saved
        model_name: Name under which the model will be saved, should end in ".pth" of ".pt" as for naming conventions
    """
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name doesn't follow naming convention"
    model_save_path = target_dir_path / model_name

    print(f"Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)
