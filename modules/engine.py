"""
Module contains functions to train and test a PyTorch model
"""
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Performs a single training step on a PyTorch model

    Args:
        model: PyTorch model to be trained
        dataloader: DataLoader instance containing training data
        loss_fn: Loss function to minimize
        optimizer: PyTorch optimizer used to minimize the loss function
        device: Target device to compute on

    Returns:
        A tuple of training loss and accuracy, in the form (train_loss, train_accuracy).
    """

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (torch.eq(y_pred_class, y)).sum().item()/len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return (train_loss, train_acc)

def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Performs a single test step on a PyTorch model

    Args:
        model: PyTorch model to be tested
        dataloader: DataLoader instance containing test data
        loss_fn: Loss function to evaluate model
        device: Target device to compute on

    Returns:
        A tuple of testing loss and accuracy, in the form (test_loss, test_accuracy).
    """
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            test_acc += (torch.eq(y_pred_class, y)).sum().item()/len(y_pred)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return (test_loss, test_acc)

def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device=torch.device("cpu"),
    epochs: int=5,
    writer: SummaryWriter=None
) -> Dict[str, List]:
    """Train and tests a PyTorch model

    Calls train_step() and test_steps() functions to train and test a model for a given number of epoches. Supports learning rate scheduling.
    Calculates, print and store training and testing metrics throughout for monitoring.

    Args:
        model: PyTorch model to be trained
        train_dataloader: DataLoader instance containing training data
        test_dataloader: DataLoader instance containing test data
        loss_fn: Loss function to minimize
        optimizer: PyTorch optimizer used to minimize the loss function
        scheduler: PyTorch learning rate scheduler for the optimizer, not mandatory
        device: Target device to compute on (default is "cpu")
        epochs: Number of epochs for the training (default is 5)
        writer: A SummaryWriter can be given to the function to be reasured

    Returns:
        A dictionary of training and testing loss and training and testing accuracy for each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
    """
    new_writer = writer is None
    if(new_writer):
        writer = SummaryWriter()

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        if(scheduler is not None):
            scheduler.step()

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
            global_step=epoch
        )

        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
            global_step=epoch
        )

        writer.add_graph(model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device))

    if new_writer:
        writer.close()

    return results
