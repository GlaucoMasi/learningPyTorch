import torch
from torch import nn

# Returns a dictionary containing the results of model predicting on the dataloader
def eval_model(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               accuracy_fn,
               start = 0,
               end = 0):
    
    loss, acc = 0, 0
    with torch.inference_mode():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y)

            # Performing on dim=1 because dim=0 is for batch
            acc += accuracy_fn(y_pred=pred.argmax(dim=1), y_true=y)

        # Good practice to keep calculations inside torch.inference_mode()
        loss /= len(dataloader)
        acc /= len(dataloader)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss,
            "model_acc": acc,
            "train_time": f"{(end-start):.3f} seconds"}

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn):
    loss, acc = 0, 0
    for (X, y) in iter(dataloader):
        # We only perform argmax for the accuracy function
        # Softmax is not need as it is monotonic, so the highest value remains the same
        y_pred = model(X)
        curr_loss = loss_fn(y_pred, y)
        loss += curr_loss
        acc += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true=y)

        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

    loss /= len(dataloader)
    acc /= len(dataloader)
    print(f"Train loss: {loss:.5f} | Train accuracy: {acc:.2f}%\n")

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn):
    model.eval()
    loss, acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_pred=y_pred.argmax(dim=1), y_true=y)

        # Good practice to keep calculations inside torch.inference_mode()
        loss /= len(dataloader)
        acc /= len(dataloader)
        print(f"Test loss: {loss:.5f} | Test accuracy: {acc:.2f}%\n")

def perform_step(model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy_fn):
    train_step(
        model=model,
        dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )