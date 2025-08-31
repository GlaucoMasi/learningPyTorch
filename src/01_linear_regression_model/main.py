# Steps:
# 1) Prepare data and split between training and testing
# 2) Building a model
# 3) Training a model on training data
# 4) Making predictions and evaluating a model on test data (inference)
# 5) Saving and loading a model
# 6) Putting it all together

import sys
import torch
from torch import nn
import matplotlib.pyplot as plt
from model import LinearRegressionModel
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def plot_predictions(train_features, train_labels, test_features, test_labels, predictions):
    plt.figure(figsize = (10, 7))
    plt.scatter(train_features, train_labels, c = "b", s = 4, label = "Training data")
    plt.scatter(test_features, test_labels, c = "g", s = 4, label = "Testing data")
    
    if predictions is not None:
        plt.scatter(test_features, predictions, c = "r", s = 4, label = "Testing data")

    plt.legend(prop = {"size" : 14})
    
    plt.savefig(SCRIPT_DIR / "plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    weight = 0.7
    bias = 0.3

    # X = features, y = labels. The model learns to assign a label to a feature
    X = torch.arange(0, 1, 0.02).unsqueeze(1)
    y = weight * X + bias

    train_split = int(0.8 * len(X)) # 80% of data is used for training
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    # To train the model we will use:
    # - torch.nn.L1loss as a Mean absolute error (MAE) Loss function
    # - torch.optim.SGD(params, lr) as a Stochastic gradient descend Optimizer,
    # where params are the target model parameteres to optimize and lr is the learning rate,
    # an hyperparameter that indicates the "size" of each update. It can be scheduled over time
    model_0 = LinearRegressionModel()    
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

    epochs = 1000
    train_loss_values = []
    test_loss_value = []
    epoch_count = []

    # Training loop
    for epoch in range(epochs):
        # Put model in training mode
        model_0.train()

        # 1) Forward pass on training data
        y_pred = model_0(X_train)

        # 2) Calculate the loss
        loss = loss_fn(y_pred, y_train)

        # 3) Zero the optimizer gradients
        optimizer.zero_grad()

        # 4) Perform backpropagation on the loss
        loss.backward()

        # 5) Progress/step the optimizer performing gradient descent
        optimizer.step()

        # Put model in evaluation mode for testing
        model_0.eval()

        # Testing loop
        # This context manager makes forward_passes faster by disabling things
        with torch.inference_mode():
            # 1) Forward pass on test data
            test_pred = model_0(X_test)

            # 2) Calculate loss on test data
            test_loss = loss_fn(test_pred, y_test)

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss)
                test_loss_value.append(test_loss)
                print(f"Epoch: {epoch} | MAE Train loss: {loss} | MAE Test loss: {test_loss} | Weights and bias: {model_0.state_dict()}")

    model_0.eval()
    # Setup context manager for inference
    with torch.inference_mode():
        final_pred = model_0(X_test)
        plot_predictions(X_train, y_train, X_test, y_test, final_pred)

    venv_dir = Path(sys.prefix)
    project_root = venv_dir.parent
    models_dir = project_root/"trained_models"
    model_name = model_0.__class__.__name__+".pth"
    model_save_path = models_dir / model_name

    # Saving and loading the model through his state
    torch.save(obj=model_0.state_dict(), f=model_save_path)

    loaded_model_0 = LinearRegressionModel();
    loaded_model_0.load_state_dict(torch.load(f=model_save_path))

    loaded_model_0.eval()
    with torch.inference_mode():
        loaded_pred = loaded_model_0(X_test)

    print(final_pred == loaded_pred)

if __name__ == "__main__":
    main()