"""
Module for creating PyTorch DataLoaders for image classification data
"""
import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count() or 1

def create_dataloaders(
    train_dir: Path,
    test_dir: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """Creates training and testing DataLoaders

    Takes in a training directory and a test directory, and turns them into PyTorch datasets and then into PyTorch DataLoaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transform: torchvision transforms to beperformed on training and testing data
        batch_size: Batch size for the DataLoaders
        num_workers: Number of workers for the DataLoaders

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names), where class_names is a list of the target classes.
    """

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_dataset.classes

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, test_dataloader, class_names
